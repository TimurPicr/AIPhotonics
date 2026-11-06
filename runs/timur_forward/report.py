"""Build a self-contained visual report for the held-out forward test."""
import base64
from html import escape
import json
from pathlib import Path

from PIL import Image

from src.io import file_hash

ROOT = Path('data/forward/test')
RESULTS = Path('outputs/timur_forward')


def picture(path: Path, label: str, caption: str) -> str:
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'L' or image.size != (128, 128):
            raise ValueError(f'Unexpected image format: {path}')
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return (f'<figure><img src="data:image/png;base64,{data}" width="128" height="128" '
            f'alt="{escape(label, quote=True)}"><figcaption>{caption}</figcaption></figure>')


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    summary = json.loads((RESULTS / 'summary.json').read_text())
    aggregate = summary['methods']['forward_model']
    cases, options = [], []
    for index, sample in enumerate(manifest['samples']):
        sample_id = sample['id']
        directory = RESULTS / 'forward_model' / sample_id
        metrics = json.loads((directory / 'metrics.json').read_text())
        assert metrics['metadata']['checkpoint_sha256'] == summary['checkpoint_sha256']
        assert metrics['metadata']['sample_id'] == sample_id
        assert metrics['metadata']['phase_png_quantized']
        inputs, recons = [], []
        for plane, source in enumerate(sample['sources']):
            path = ROOT / source['path']
            assert file_hash(path) == source['sha256']
            z_mm = manifest['z_m'][plane] * 1000
            inputs.append(picture(path, f'{sample_id}: вход, плоскость {plane+1}',
                                  f'z = {z_mm:.1f} мм'))
            pair = metrics['planes'][plane]
            recons.append(picture(directory / f'recon_{plane+1}.png',
                                  f'{sample_id}: реконструкция, плоскость {plane+1}',
                                  f'z = {z_mm:.1f} мм<br><span>SSIM {pair["ssim"]:.4f} · r {pair["pearson"]:.4f}</span>'))
        mask = picture(directory / 'phase.png', f'{sample_id}: предсказанная фазовая маска',
                       '−π <span class="phase-scale" aria-hidden="true"></span> +π')
        hidden = ' hidden' if index else ''
        cases.append(f'''<section class="sample" id="case-{index}"{hidden}>
<div class="case-heading"><h2>{escape(sample_id)}</h2><p>SSIM <b>{metrics['mean']['ssim']:.4f}</b> · Pearson <b>{metrics['mean']['pearson']:.4f}</b></p></div>
<h3>1. Четыре входных изображения</h3><div class="images">{''.join(inputs)}</div>
<div class="phase"><h3>2. Предсказанная фазовая маска</h3>{mask}</div>
<h3>3. Четыре реконструкции из сохранённой маски</h3><div class="images">{''.join(recons)}</div>
<p class="origin">Основа тестового примера: {escape(sample['parent_sample_id'])}.</p></section>''')
        options.append(f'<option value="{index}">{index+1:02d} · {escape(sample_id)}</option>')
    document = '''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forward-test — визуальный отчёт timur</title>
<style>
:root{color-scheme:light;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#17212d;background:#f3f5f8}
*{box-sizing:border-box}body{margin:0}main{max-width:880px;margin:auto;padding:24px 20px 18px}h1{font-size:24px;margin:0 0 8px}h2{font-size:16px;margin:0}h3{font-size:14px;font-weight:600;margin:0 0 9px}p{margin:0}header>p{font-size:13px;line-height:1.5;color:#526172;max-width:760px}.totals{display:flex;flex-wrap:wrap;gap:10px 24px;margin:14px 0;font-size:14px;font-variant-numeric:tabular-nums}.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:18px 0 14px}.toolbar label{font-size:13px}.toolbar select,.toolbar button{font:inherit;font-size:14px;padding:9px 10px;border:1px solid #bdc7d3;border-radius:6px;background:#fff;color:#17212d}.toolbar select{flex:1;min-width:220px}.toolbar button{min-height:40px}.toolbar button:disabled{opacity:.4}#counter{font-size:13px;min-width:46px;text-align:center;font-variant-numeric:tabular-nums}.sample{background:#fff;border:1px solid #dce2e9;border-radius:12px;padding:18px}.case-heading{display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;margin-bottom:16px}.case-heading p{font-size:13px;font-variant-numeric:tabular-nums}.images{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0;text-align:center}img{display:block;width:128px;max-width:100%;margin-inline:auto;height:auto;aspect-ratio:1;image-rendering:pixelated;border-radius:3px}figcaption{font-size:12px;line-height:1.5;margin-top:5px;color:#405063;font-variant-numeric:tabular-nums}figcaption span{font-size:11px}.phase{display:flex;flex-direction:column;align-items:center;gap:4px;margin:17px 0}.phase figure{width:112px}.phase-scale{display:inline-block;width:45px;height:8px;margin:0 5px;background:linear-gradient(90deg,#000,#fff);border:1px solid #aeb8c3}.origin{font-size:11px;color:#667585;margin-top:12px}footer{font-size:11px;color:#667585;line-height:1.5;margin-top:12px;overflow-wrap:anywhere}b{font-weight:650}
@media(max-width:600px){main{padding:16px 12px}.sample{padding:12px}.images{grid-template-columns:repeat(2,minmax(0,1fr))}.toolbar label{width:100%}.toolbar select{min-width:100%;width:100%}.toolbar button{min-height:44px}.case-heading{margin-bottom:12px}}
@media print{body{background:#fff}main{padding:0}.toolbar{display:none}.sample{border:0;padding:10px 0}.sample[hidden]{display:block!important}.sample{break-after:page}footer{display:none}}
</style></head><body><main>
<header><h1>Forward-test · timur</h1>
<p>Цели получены из существующих масок; в центре — фаза, предсказанная обученной моделью.</p></header>
'''
    document += (f'<div class="totals"><span>{len(cases)} тестовых четвёрок</span>'
                 f'<span>Средний SSIM <b>{aggregate["mean"]["ssim"]:.4f}</b></span>'
                 f'<span>Pearson <b>{aggregate["mean"]["pearson"]:.4f}</b></span></div>')
    document += ('<nav class="toolbar" aria-label="Выбор тестового примера"><label for="sample">Тестовая четвёрка</label>'
                 f'<select id="sample">{"".join(options)}</select>'
                 '<button type="button" id="previous">Назад</button><span id="counter" role="status" aria-live="polite"></span>'
                 '<button type="button" id="next">Далее</button></nav>')
    document += ''.join(cases)
    document += (f'<footer>PNG L · 128×128 · амплитуда · нормировка по максимуму каждой плоскости. '
                 f'Фаза: 0 → −π, 255 → +π. Checkpoint: шаг {summary["selected_step"]}, '
                 f'SHA256 {summary["checkpoint_sha256"][:16]}…</footer>')
    document += '''<script>
const choice=document.getElementById('sample');
const cases=Array.from(document.querySelectorAll('.sample'));
const previous=document.getElementById('previous');
const next=document.getElementById('next');
function show(){const index=choice.selectedIndex;cases.forEach((section,i)=>section.hidden=i!==index);previous.disabled=index===0;next.disabled=index===cases.length-1;document.getElementById('counter').textContent=`${index+1} / ${cases.length}`;}
choice.addEventListener('change',show);
previous.addEventListener('click',()=>{if(choice.selectedIndex>0){choice.selectedIndex--;show();}});
next.addEventListener('click',()=>{if(choice.selectedIndex<cases.length-1){choice.selectedIndex++;show();}});
show();
</script><noscript><style>.sample[hidden]{display:block!important}.toolbar{display:none}</style></noscript>
</main></body></html>'''
    path = RESULTS / 'test-report.html'
    path.write_text(document)
    print(f'{path}: {len(cases)} cases, {path.stat().st_size} bytes')


if __name__ == '__main__':
    main()
