"""Build 5,500 target PNGs: 1,000 / 125 / 250 ordered quartets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version
import torch
import torch.nn.functional as F

from config import CONFIG
from src.dataset_sources import RemoteZip, download
from src.io import export_phase_png, file_hash, load_four_png, load_phase_png, read_png, write_json, write_png
from src.physics import output_pitch_m
from src.reconstruction import normalize_recons, reconstruct_four

GENERATOR = "mixed55k-physical-v2"
FAMILIES = ("smooth", "analytic", "mixed")
COCO_ANNOTATIONS = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGES = "https://s3.amazonaws.com/images.cocodataset.org/train2017/"
VIMEO_URL = "https://www.kaggle.com/api/v1/datasets/download/maiimaii/vimeo-septuplet?datasetVersionNumber=2"


def rejection_reason(error: Exception) -> str:
    return re.sub(r"0x[0-9a-fA-F]+", "<address>", str(error))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def sample_seed(seed: int, category: str, index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{GENERATOR}:{seed}:{category}:{index}".encode()).digest()[:8],
                          "little") % (2**63 - 1)


def make_phase(seed: int, family: str) -> tuple[torch.Tensor, dict]:
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.linspace(-1, 1, 128), torch.linspace(-1, 1, 128), indexing="ij")
    coarse_size = int(rng.choice([4, 8, 16, 32]))
    scale = float(rng.uniform(0.5, 4.0))
    coarse = torch.randn((1, 1, coarse_size, coarse_size), generator=generator)
    smooth = F.interpolate(coarse, size=(128, 128), mode="bicubic", align_corners=False) * scale
    parameters = {"coarse_size": coarse_size, "phase_scale": scale}
    if family == "smooth":
        phase = smooth
    else:
        tilt = rng.uniform(-24, 24, size=2)
        curvature = rng.uniform(-80, 80, size=2)
        angle, frequency, strength = float(rng.uniform(0, math.pi)), float(rng.uniform(1, 18)), float(rng.uniform(0.2, 5))
        analytic = math.pi * (float(tilt[0]) * x + float(tilt[1]) * y +
                               float(curvature[0]) * x.square() + float(curvature[1]) * y.square())
        analytic += strength * torch.sin(math.pi * frequency * (x * math.cos(angle) + y * math.sin(angle)))
        parameters.update(tilt=tilt.tolist(), curvature=curvature.tolist(), angle=angle,
                          frequency=frequency, strength=strength)
        if family == "analytic":
            phase = analytic[None, None]
        elif family == "mixed":
            noise_scale = float(rng.uniform(0.05, 1.5))
            phase = analytic[None, None] + smooth + noise_scale * torch.randn((1, 1, 128, 128), generator=generator)
            parameters["noise_scale"] = noise_scale
        else:
            raise ValueError(f"Unknown phase family: {family}")
    return (phase + math.pi).remainder(2 * math.pi) - math.pi, parameters


def _record(root: Path, path: Path, source_id: str, **metadata) -> dict:
    return {"id": source_id, "path": path.relative_to(root / "train").as_posix(),
            "sha256": file_hash(path), **metadata}


def _cached(root: Path, category: str, index: int) -> dict | None:
    path = root / "train" / category / f"{category}_{index:06d}" / "sample.json"
    if not path.exists():
        return None
    sample = json.loads(path.read_text())
    for item in sample["sources"] + ([sample["phase"]] if "phase" in sample else []):
        png = root / "train" / item["path"]
        if not png.exists() or file_hash(png) != item["sha256"]:
            return None
    return sample


def _save_sample(root: Path, sample: dict) -> dict:
    atomic_json(root / "train" / sample["category"] / sample["id"] / "sample.json", sample)
    return sample


def build_synthetic(root: Path, count: int, seed: int, *, batch_size: int) -> list[dict]:
    samples = []
    started = time.perf_counter()
    for start in range(0, count, batch_size):
        indices = list(range(start, min(start + batch_size, count)))
        pending = [index for index in indices if _cached(root, "synthetic", index) is None]
        phases, descriptions = [], {}
        for index in pending:
            local_seed = sample_seed(seed, "synthetic", index)
            family = FAMILIES[index % len(FAMILIES)]
            phase, parameters = make_phase(local_seed, family)
            descriptions[index] = {"seed": local_seed, "family": family, "parameters": parameters}
            phases.append(phase)
        if pending:
            quantized = []
            for index, phase in zip(pending, phases):
                phase_path = root / "train" / "synthetic" / f"synthetic_{index:06d}" / "phase.png"
                export_phase_png(phase, str(phase_path))
                quantized.append(load_phase_png(str(phase_path)))
            with torch.no_grad():
                values = normalize_recons(reconstruct_four(torch.cat(quantized)))
            for slot, index in enumerate(pending):
                sample_id = f"synthetic_{index:06d}"
                folder = root / "train" / "synthetic" / sample_id
                sources = []
                for plane in range(4):
                    path = folder / f"{plane + 1}.png"
                    write_png(values[slot, plane], path)
                    sources.append(_record(root, path, f"{sample_id}:{plane + 1}",
                                           origin="forward_fresnel_from_quantized_phase", plane=plane + 1))
                _save_sample(root, {"id": sample_id, "category": "synthetic", "sources": sources,
                    "phase": _record(root, folder / "phase.png", f"{sample_id}:phase"),
                    "generation": descriptions[index]})
        samples.extend(_cached(root, "synthetic", index) for index in indices)
        if start == 0 or (start + batch_size) // 100 != start // 100 or len(samples) == count:
            print(f"synthetic {len(samples)}/{count}, {time.perf_counter() - started:.1f}s", flush=True)
    return samples


def prepare_image(payload: bytes, *, box: tuple | None = None) -> tuple[torch.Tensor, dict]:
    with Image.open(io.BytesIO(payload)) as original:
        image = ImageOps.exif_transpose(original).convert("L")
        width, height = image.size
        if box is None:
            side = min(width, height)
            left, top = (width - side) // 2, (height - side) // 2
            box = (left, top, left + side, top + side)
        image = image.crop(box).resize((128, 128), Image.Resampling.LANCZOS)
        array = np.array(image, dtype=np.uint8)
    if array.max() == array.min():
        raise ValueError("Constant source image is unsuitable for the main training set")
    return torch.from_numpy(array.astype(np.float32) / 255), {
        "original_size": [width, height], "crop_box": list(box), "grayscale": "Pillow L",
        "resize": "LANCZOS", "output_size": [128, 128]}


def coco_selection(cache: Path, count: int, seed: int) -> list[dict]:
    path = cache / "coco_train_metadata.json"
    if not path.exists():
        archive = RemoteZip(COCO_ANNOTATIONS, cache / "coco_annotations_index.json")
        metadata = json.loads(archive.read("annotations/captions_train2017.json"))
        licenses = {item["id"]: item for item in metadata["licenses"]}
        images = [{**image, "license": licenses[image["license"]]} for image in metadata["images"]]
        atomic_json(path, {"source": COCO_ANNOTATIONS, "images": images})
    images = sorted(json.loads(path.read_text())["images"], key=lambda image: image["id"])
    if count > len(images):
        raise ValueError("Not enough COCO images")
    order = np.random.default_rng(sample_seed(seed, "coco_selection", 0)).permutation(len(images))[:count]
    return [images[int(index)] for index in order]


def build_random(root: Path, cache: Path, count: int, seed: int, workers: int) -> list[dict]:
    if not count:
        return []
    total_images = count * 4
    # Three disjoint, deterministic reserves per slot handle damaged public JPEGs.
    selection = coco_selection(cache, total_images * 4, seed)
    atomic_json(root / "coco_selection.json", {"seed": seed, "planned_images": selection[:total_images]})

    def build(index):
        cached = _cached(root, "random", index)
        if cached:
            return cached
        sample_id = f"random_{index:06d}"
        previous_path = root / "train" / "random" / sample_id / "sample.json"
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
        sources = []
        for plane in range(1, 5):
            slot = index * 4 + plane - 1
            candidates = ([previous["sources"][plane - 1]["upstream"]] if previous else
                [selection[slot]] + [selection[total_images * (attempt + 1) + slot] for attempt in range(3)])
            rejected = []
            for image in candidates:
                url = COCO_IMAGES + image["file_name"]
                payload, _ = download(url)
                try:
                    tensor, preprocessing = prepare_image(payload)
                    break
                except (OSError, ValueError) as error:
                    rejected.append({"id": image["id"], "reason": rejection_reason(error)})
            else:
                raise ValueError(f"No decodable nonconstant COCO source for slot {slot}: {rejected}")
            original_sha256 = hashlib.sha256(payload).hexdigest()
            if previous and previous["sources"][plane - 1]["original_sha256"] != original_sha256:
                raise ValueError("Previously selected COCO JPEG changed at the source")
            path = root / "train" / "random" / sample_id / f"{plane}.png"
            write_png(tensor, path)
            sources.append(_record(root, path, f"coco2017:train:{image['id']}", url=url,
                original_sha256=original_sha256, upstream=image,
                rejected_candidates=previous["sources"][plane - 1].get("rejected_candidates", []) if previous else rejected,
                preprocessing=preprocessing, plane=plane))
        return _save_sample(root, {"id": sample_id, "category": "random", "sources": sources})

    samples = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sample in pool.map(build, range(count)):
            samples.append(sample)
            if len(samples) % 50 == 0 or len(samples) == count:
                print(f"random {len(samples)}/{count}", flush=True)
    atomic_json(root / "coco_selection.json", {"seed": seed,
        "images": [source["upstream"] for sample in samples for source in sample["sources"]],
        "rejected_candidates": [item for sample in samples for source in sample["sources"]
                                for item in source.get("rejected_candidates", [])]})
    return samples


def vimeo_selection(archive: RemoteZip, count: int, seed: int) -> list[tuple[str, list[str]]]:
    candidates = [name for name in archive.entries if name.endswith("sep_trainlist.txt")]
    if len(candidates) != 1:
        raise ValueError("Vimeo archive must contain one official sep_trainlist.txt")
    train_ids = set(archive.read(candidates[0]).decode().split())
    sequences = {}
    for name in archive.entries:
        parts = name.split("/")
        if len(parts) >= 3 and parts[-1] in {f"im{i}.png" for i in range(1, 8)}:
            sequence_id = "/".join(parts[-3:-1])
            if sequence_id in train_ids:
                sequences.setdefault(sequence_id, []).append(name)
    eligible = sorted(key for key, names in sequences.items() if len(names) == 7)
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} complete training septuplets; need {count}")
    order = np.random.default_rng(sample_seed(seed, "vimeo_selection", 0)).permutation(len(eligible))[:count]
    return [(eligible[int(index)], sorted(sequences[eligible[int(index)]])) for index in order]


def build_similar(root: Path, cache: Path, count: int, seed: int, workers: int, url: str) -> list[dict]:
    if not count:
        return []
    archive = RemoteZip(url, cache / "vimeo_index.json")
    base_count = max(count, 2500)
    selection = vimeo_selection(archive, base_count * 4, seed)
    by_id = dict(selection)
    atomic_json(root / "vimeo_selection.json", {"seed": seed, "archive": url,
        "archive_etag": archive.reader.etag, "planned_sequences": [item[0] for item in selection[:count]]})

    def build(index):
        cached = _cached(root, "similar", index)
        if cached:
            return cached
        frames = (1, 2, 3, 4) if index % 2 == 0 else (1, 3, 5, 7)
        sample_id = f"similar_{index:06d}"
        previous_path = root / "train" / "similar" / sample_id / "sample.json"
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
        candidates = ([(previous["sequence_id"], by_id[previous["sequence_id"]])] if previous else
            [selection[index]] + [selection[base_count * (attempt + 1) + index] for attempt in range(3)])
        rejected = []
        for sequence_id, names in candidates:
            chosen_names = [next(name for name in names if name.endswith(f"/im{frame}.png")) for frame in frames]
            payloads = archive.read_many(chosen_names)
            prepared, crop_box = [], None
            try:
                for name in chosen_names:
                    tensor, preprocessing = prepare_image(payloads[name], box=crop_box)
                    if preprocessing["original_size"] != [448, 256]:
                        raise ValueError("Expected original 448x256 Vimeo90K Septuplet frames")
                    crop_box = tuple(preprocessing["crop_box"])
                    prepared.append((tensor, preprocessing))
                break
            except (ValueError, OSError) as error:
                rejected.append({"sequence_id": sequence_id, "reason": rejection_reason(error)})
        else:
            raise ValueError(f"No usable Vimeo septuplet for slot {index}: {rejected}")
        sources = []
        for plane, (frame, name, (tensor, preprocessing)) in enumerate(zip(frames, chosen_names, prepared), 1):
            path = root / "train" / "similar" / sample_id / f"{plane}.png"
            write_png(tensor, path)
            sources.append(_record(root, path, f"vimeo90k:train:{sequence_id}:im{frame}",
                dataset="Vimeo90K Septuplet original", archive_url=url, archive_member=name,
                original_sha256=hashlib.sha256(payloads[name]).hexdigest(), sequence_id=sequence_id,
                frame=frame, plane=plane, preprocessing=preprocessing))
        return _save_sample(root, {"id": sample_id, "category": "similar", "sources": sources,
            "sequence_id": sequence_id, "frame_indices": list(frames),
            "rejected_sequences": previous.get("rejected_sequences", []) if previous else rejected})

    samples = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sample in pool.map(build, range(count)):
            samples.append(sample)
            if len(samples) % 25 == 0 or len(samples) == count:
                print(f"similar {len(samples)}/{count}", flush=True)
    atomic_json(root / "vimeo_selection.json", {"seed": seed, "archive": url,
        "archive_etag": archive.reader.etag, "sequences": [sample["sequence_id"] for sample in samples],
        "rejected_sequences": [item for sample in samples for item in sample.get("rejected_sequences", [])]})
    return samples


def make_manifest(root: Path, samples: list[dict], seed: int) -> dict:
    project = Path(__file__).resolve().parent.parent
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True)
    code_files = ["config.py", "src/build_dataset.py", "src/dataset_sources.py",
                  "src/physics.py", "src/reconstruction.py", "src/io.py"]
    manifest = {"generator": GENERATOR, "split": "train", "seed": seed,
        "size": len(samples), "target_images": len(samples) * 4,
        "z_m": list(CONFIG.z_m), "protocol": CONFIG.snapshot(),
        "dx_out_m": [output_pitch_m(z, 128) for z in CONFIG.z_m],
        "category_quartets": dict(Counter(sample["category"] for sample in samples)),
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "pillow": pillow_version},
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "code_sha256": {name: file_hash(project / name) for name in code_files}, "samples": samples}
    atomic_json(root / "train" / "manifest.json", manifest)
    for category in manifest["category_quartets"]:
        group = [sample for sample in samples if sample["category"] == category]
        atomic_json(root / "train" / f"manifest_{category}.json", {
            **manifest, "size": len(group), "target_images": len(group) * 4,
            "category_quartets": {category: len(group)}, "samples": group})
    return manifest


def verify_dataset(root: Path, *, audit_phase: bool = True) -> dict:
    manifest = json.loads((root / "train" / "manifest.json").read_text())
    if manifest["protocol"] != CONFIG.snapshot() or manifest["size"] != len(manifest["samples"]):
        raise ValueError("Manifest size or optical protocol mismatch")
    plan_path = root / "build_config.json"
    if plan_path.exists():
        expected = {key: value for key, value in json.loads(plan_path.read_text())["counts_quartets"].items() if value}
        if manifest["category_quartets"] != expected:
            raise ValueError("Dataset is incomplete relative to build_config.json")
    synthetic, target_count, hash_count = 0, 0, 0
    category_counts = Counter()
    phase_hashes, coco_ids, sequence_ids, target_paths = set(), set(), set(), set()
    for index, sample in enumerate(manifest["samples"]):
        category_counts[sample["category"]] += 1
        if len(sample["sources"]) != 4:
            raise ValueError("Every training example must contain four ordered targets")
        if [source["plane"] for source in sample["sources"]] != [1, 2, 3, 4]:
            raise ValueError("Source order does not match the four optical planes")
        paths = []
        for source in sample["sources"]:
            path = root / "train" / source["path"]
            if source["path"] in target_paths or file_hash(path) != source["sha256"]:
                raise ValueError(f"Repeated target path or incorrect PNG hash: {path}")
            target_paths.add(source["path"])
            read_png(path)
            paths.append(str(path))
            target_count += 1
            hash_count += 1
            if sample["category"] == "random":
                if source["id"] in coco_ids:
                    raise ValueError("Repeated COCO source")
                coco_ids.add(source["id"])
        if sample["category"] == "similar":
            if sample["sequence_id"] in sequence_ids:
                raise ValueError("Repeated Vimeo sequence")
            sequence_ids.add(sample["sequence_id"])
        if sample["category"] == "synthetic":
            phase = sample["phase"]
            phase_path = root / "train" / phase["path"]
            if file_hash(phase_path) != phase["sha256"] or phase["sha256"] in phase_hashes:
                raise ValueError("Invalid or repeated synthetic phase")
            phase_hashes.add(phase["sha256"])
            synthetic += 1
            if audit_phase:
                with torch.no_grad():
                    recons = normalize_recons(reconstruct_four(load_phase_png(str(phase_path))))
                    recon_codes = recons.clamp(0, 1).mul(255).round()
                target_codes = load_four_png(paths).mul(255).round()
                if ((target_codes - recon_codes).abs() > 1).any():
                    raise ValueError(f"Synthetic phase consistency check failed: {sample['id']}")
        if (index + 1) % 1000 == 0:
            print(f"verified {index + 1}/{manifest['size']}", flush=True)
    if target_count != manifest["target_images"] or dict(category_counts) != manifest["category_quartets"]:
        raise ValueError("Incorrect category/image counts")
    actual_paths = {path.relative_to(root / "train").as_posix()
                    for path in (root / "train").glob("*/*/[1-4].png")}
    if actual_paths != target_paths:
        raise ValueError("Target PNG files do not match the manifest")
    report = {"status": "verified", "quartets": manifest["size"], "target_images": target_count,
        "category_quartets": dict(category_counts), "synthetic_phase_masks": synthetic,
        "phase_audit": audit_phase, "checked_target_hashes": hash_count,
        "unique_coco_sources": len(coco_ids), "unique_vimeo_sequences": len(sequence_ids),
        "manifest_sha256": file_hash(root / "train" / "manifest.json")}
    atomic_json(root / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/train5.5k_physical"))
    parser.add_argument("--cache", type=Path, default=Path("data/external/train55k"))
    parser.add_argument("--category", choices=["all", "synthetic", "random", "similar"], default="all")
    parser.add_argument("--synthetic-quartets", type=int, default=1000)
    parser.add_argument("--random-quartets", type=int, default=125)
    parser.add_argument("--similar-quartets", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vimeo-url", default=VIMEO_URL)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    counts = {"synthetic": args.synthetic_quartets, "random": args.random_quartets,
              "similar": args.similar_quartets}
    if sum(counts.values()) < 1 or min(counts.values()) < 0 or min(args.batch_size, args.threads, args.workers) < 1:
        parser.error("Counts must be nonnegative; execution parameters must be positive")
    torch.set_num_threads(args.threads)
    if args.verify_only:
        print(json.dumps(verify_dataset(args.root), indent=2))
        return
    plan = {"generator": GENERATOR, "seed": args.seed, "counts_quartets": counts,
            "protocol": CONFIG.snapshot(),
            "vimeo_url": args.vimeo_url, "batch_size": args.batch_size}
    plan_path = args.root / "build_config.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Build parameters changed; use a new --root")
    atomic_json(plan_path, plan)
    if args.category in ("all", "synthetic"):
        build_synthetic(args.root, counts["synthetic"], args.seed, batch_size=args.batch_size)
    if args.category in ("all", "random"):
        build_random(args.root, args.cache, counts["random"], args.seed, args.workers)
    if args.category in ("all", "similar"):
        build_similar(args.root, args.cache, counts["similar"], args.seed, args.workers, args.vimeo_url)
    samples = []
    for category, count in counts.items():
        for index in range(count):
            sample = _cached(args.root, category, index)
            if sample:
                samples.append(sample)
    manifest = make_manifest(args.root, samples, args.seed)
    complete = manifest["category_quartets"] == {key: value for key, value in counts.items() if value}
    atomic_json(args.root / "build_status.json", {"complete": complete,
                "expected_quartets": counts, "actual_quartets": manifest["category_quartets"]})
    if complete:
        print(json.dumps(verify_dataset(args.root), indent=2))
    else:
        print(f"Partial build: {manifest['category_quartets']}", flush=True)


if __name__ == "__main__":
    main()
