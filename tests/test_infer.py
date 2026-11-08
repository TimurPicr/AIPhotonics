import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from src.infer import DEFAULT_CHECKPOINT, discover_quartets, load_checkpoint, main
from src.io import export_phase_png, load_four_images
from src.refinement import refine_phase


ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def assert_output(path):
    contents = path.read_bytes()
    assert contents[:8] == b"\x89PNG\r\n\x1a\n"
    assert contents[12:16] == b"IHDR"
    assert struct.unpack(">IIBB", contents[16:26]) == (128, 128, 8, 0)
    with Image.open(path) as image:
        image.load()
        assert image.format == "PNG" and image.mode == "L" and image.size == (128, 128)
        assert np.asarray(image).dtype == np.uint8


def test_all_example_images_match_manifest():
    manifest = json.loads((EXAMPLES / "manifest.json").read_text())
    assert manifest["size"] == len(manifest["samples"]) == 8
    assert len({sample["category"] for sample in manifest["samples"]}) == 4
    for sample in manifest["samples"]:
        for source in sample["sources"]:
            path = EXAMPLES / source["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
            assert_output(path)


def test_cli_processes_eight_quartets_with_supplied_weights(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "src.infer", "--input-dir", str(EXAMPLES),
         "--output-dir", str(tmp_path)], cwd=ROOT, text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    paths = sorted(tmp_path.glob("*.png"))
    assert len(paths) == 8
    assert {path.stem for path in paths} == {name for name, _ in discover_quartets(EXAMPLES)}
    for path in paths:
        assert_output(path)
    # Check that the real checkpoint, all four inputs and default refinement are used.
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(DEFAULT_CHECKPOINT)
    assert checkpoint["epochs"] == 259
    name, inputs = discover_quartets(EXAMPLES)[0]
    targets = load_four_images(inputs)
    with torch.inference_mode():
        phase, _ = model(targets.contiguous(memory_format=torch.channels_last))
    phase = refine_phase(phase, targets, steps=10, **checkpoint["loss_config"])
    expected = tmp_path / "expected.png"
    export_phase_png(phase, expected)
    assert (tmp_path / f"{name}.png").read_bytes() == expected.read_bytes()


def test_explicit_image_order_and_disabled_refinement(tmp_path):
    images = [EXAMPLES / "examples_0000" / f"{index}.png" for index in (4, 3, 2, 1)]
    out = tmp_path / "phase.png"
    main(["--images", *map(str, images), "--output", str(out), "--refine-steps", "0"])
    model, _ = load_checkpoint(DEFAULT_CHECKPOINT)
    with torch.inference_mode():
        phase, _ = model(load_four_images(images).contiguous(memory_format=torch.channels_last))
    expected = tmp_path / "expected.png"
    export_phase_png(phase, expected)
    assert out.read_bytes() == expected.read_bytes()
    assert_output(out)


@pytest.mark.parametrize("count", [0, 3, 5])
def test_incomplete_or_oversized_quartets_fail(tmp_path, count):
    inputs = tmp_path / "input"
    inputs.mkdir()
    for index in range(count):
        Image.new("L", (128, 128)).save(inputs / f"{index}.png")
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as error:
        main(["--input-dir", str(inputs), "--output-dir", str(output)])
    assert error.value.code == 1 and not output.exists()


def test_entire_batch_is_validated_before_writing_output(tmp_path):
    for name, size in (("a", (128, 128)), ("b", (64, 64))):
        folder = tmp_path / name
        folder.mkdir()
        for index in range(4):
            Image.new("L", size).save(folder / f"{index}.png")
    output = tmp_path / "results"
    with pytest.raises(SystemExit) as error:
        main(["--input-dir", str(tmp_path), "--output-dir", str(output)])
    assert error.value.code == 1 and not output.exists()


@pytest.mark.parametrize("arguments", [
    ["--refine-steps", "-1"], ["--refine-lr", "nan"], ["--threads", "0"],
])
def test_invalid_options_fail_before_inference(tmp_path, arguments):
    with pytest.raises(SystemExit) as error:
        main(["--input-dir", str(EXAMPLES), "--output-dir", str(tmp_path), *arguments])
    assert error.value.code == 2 and not list(tmp_path.iterdir())


def test_single_output_rejects_multiple_quartets(tmp_path):
    out = tmp_path / "phase.png"
    with pytest.raises(SystemExit) as error:
        main(["--input-dir", str(EXAMPLES), "--output", str(out)])
    assert error.value.code == 1 and not out.exists()


def test_input_images_cannot_be_overwritten():
    with pytest.raises(SystemExit) as error:
        main(["--input-dir", str(EXAMPLES / "examples_0000"),
              "--output", str(EXAMPLES / "examples_0000" / "1.png")])
    assert error.value.code == 1


def test_checkpoint_protocol_is_checked(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"version": 1, "protocol": {}}, path)
    with pytest.raises(ValueError, match="optical protocol"):
        load_checkpoint(path)
