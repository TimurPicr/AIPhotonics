"""Collect every persisted experiment metric into one standalone HTML report."""
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

OUT = Path('ALL_EXPERIMENTS.html')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    records = []
    for root in ('data','examples','runs','outputs'):
        for path in sorted(Path(root).rglob('*')):
            if path.suffix not in ('.json','.jsonl'):
                continue
            text = path.read_text()
            value = ([json.loads(line) for line in text.splitlines() if line.strip()]
                     if path.suffix == '.jsonl' else json.loads(text))
            records.append({'path':str(path),'format':path.suffix[1:],'sha256':digest(path),'value':value})
    for path in sorted(Path('checkpoints').rglob('*.pt')):
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        value = {key:value for key,value in checkpoint.items()
                 if key not in ('model_state_dict','optimizer_state_dict','rng_state')}
        records.append({'path':str(path),'format':'checkpoint metadata','sha256':digest(path),'value':value})
    task = Path('TASKS.md')
    line = next(line for line in task.read_text().splitlines() if '**A2.2**' in line)
    pairs = re.findall(r'(\d+\.\d+)\s*→\s*(\d+\.\d+)',line)
    assert len(pairs) == 3
    values = [{'sample':sample,'planes':planes,'steps':100,'seed':42,
               'initial_mse':float(pair[0]),'final_mse':float(pair[1])}
              for (sample,planes),pair in zip([('val_0000','1–4'),('val_0001','1–4'),('val_0000','1')],pairs)]
    records.append({'path':str(task),'format':'A2.2 rounded metric extract','sha256':digest(task),
                    'value':{'source_line':line,'experiments':values}})
    return sorted(records,key=lambda record:record['path'])


def quality_nodes(value, pointer='', step=None):
    if isinstance(value,dict):
        step = value.get('step',step)
        if isinstance(value.get('mean'),dict) and 'ssim' in value['mean'] and 'planes' in value:
            yield pointer,value,step
        for key,child in value.items():
            yield from quality_nodes(child,f'{pointer}/{key}',step)
    elif isinstance(value,list):
        for index,child in enumerate(value):
            yield from quality_nodes(child,f'{pointer}/{index}',step)


def number(value):
    return '—' if value is None else f'{value:.6f}' if isinstance(value,float) else str(value)


def table(headers, rows):
    return ('<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{escape(h)}</th>' for h in headers)+
            '</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{cell}</td>' for cell in row)+'</tr>' for row in rows)+
            '</tbody></table></div>')


def main():
    records = sources()
    catalog = {record['path']:record['value'] for record in records}
    positions = {record['path']:index for index,record in enumerate(records)}
    revision = subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()
    registry = [(record,pointer,node,step) for record in records
                for pointer,node,step in quality_nodes(record['value'])]
    source_link = lambda path: f'<a href="#source-{positions[path]}">{escape(path)}</a>'

    def quality_row(label, value, path, context=''):
        value = value.get('aggregate',value)
        worst = value.get('worst_plane_mean',value.get('worst_plane',{}))
        n = value.get('samples',1)
        return [escape(label),escape(context),number(n),number(value['mean'].get('pearson')),
                number(value['mean'].get('ssim')),number(worst.get('ssim')),
                f"{value.get('valid_pearson_pairs','—')}/{value.get('total_pairs','—')}",
                number(value.get('failure_count')),source_link(path)]

    quality_headers = ['Опыт / метод','Условия','Четвёрки','Pearson','SSIM','Q / худшая плоскость','Валидные пары','Ошибки','Источник']
    sections = []
    sections.append('''<section id="definitions"><h2>Как устроены наборы</h2>
<p><b>Один пример</b> — четыре PNG 128×128, по одному на каждую плоскость, и одна предсказываемая фазовая маска. Для N четвёрок получаем 4N пар «цель — реконструкция»; это не 4N независимых масок.</p>''')
    sections.append(table(['Набор','Назначение','Как используется'],[
        ['Train','Обновление весов','По этому набору вычисляют loss и градиенты. Примеры можно многократно повторять.'],
        ['Validation / val','Выбор модели и настроек','Веса на этих примерах не обновляются; по оценке выбирают checkpoint, loss и архитектуру.'],
        ['Test','Финальная проверка','Оценивается после фиксации checkpoint. Выбор модели по test превращает его в validation.'],
        ['OOD','Проверка изменения распределения','Содержит новые относительно обучения формы, символы или частоты. Просто новый sample ID ещё не делает пример OOD.']]))
    sections.append('''<h3>Исходный MVP: независимые 2D-цели</h3>
<p>Train генерируется на лету с seed=42. Его не отрезали от общего конечного массива. У val/test/OOD отдельные RNG/seeds и фиксированные manifests; исходники/PNG разделены. В полном бюджете основной ветки обучения — 16 000 показов четвёрок, выбранный checkpoint шага 1400 видел 11 200.</p>''')
    dataset_paths = [('Validation','data/val/manifest.json'),('Test','data/test/manifest.json'),
                     ('OOD','data/ood/manifest.json'),('Диагностический train','data/train_diagnostic/manifest.json'),
                     ('Examples','examples/manifest.json'),('Сравнение A3: подвыборка val','data/val/comparison_manifest.json')]
    rows = []
    for label,path in dataset_paths:
        data = catalog[path]
        assert data['size'] == len(data['samples'])
        counts = Counter(sample['category'] for sample in data['samples'])
        rows.append([label,str(data['size']),str(data['size']*4),str(data['seed']),
                     escape(', '.join(f'{key}: {count}' for key,count in sorted(counts.items()))),source_link(path)])
    sections.append(table(['Набор','Четвёрки','Целевые PNG','Seed','Категории','Manifest'],rows))
    sections.append('''<p>OOD: треугольники, символы XYZ789 и частоты 12–20 вместо train 1–7. Examples предназначены для демонстрации. Набор A3 — первые восемь val, а не независимый дополнительный test.</p>
<p class="note">В MVP найден перекос сочетаний категорий: однородные четвёрки составили 35/1024 train-примеров против 50/64 val-примеров. Это дефект согласования смеси данных, а не намеренный OOD-эксперимент.</p>
<h3>Forward-эксперимент timur</h3>
<p>Сначала выбраны 96 уникальных предсказанных масок. Внутри каждой из четырёх категорий происхождения выполнена перестановка seed=42: из 24 масок 16 отправлены в train, 4 в val и 4 в test. Только затем из каждой маски рассчитаны четыре цели. Категория описывает старый вход сети-генератора; новые изображения являются дифракционными реконструкциями.</p>''')
    rows = []
    families, parents = [], []
    for split in ('train','val','test'):
        path = f'data/forward/{split}/manifest.json'
        data = catalog[path]
        families.append({sample['phase_sha256'] for sample in data['samples']})
        parents.append({sample['parent_sample_id'] for sample in data['samples']})
        assert len(families[-1]) == data['size']
        rows.append([split,str(data['size']),str(data['size']*4),
                     'Обновление весов' if split=='train' else 'Выбор checkpoint' if split=='val' else 'Финальная оценка',source_link(path)])
    assert len(set.union(*families)) == sum(map(len,families)) == 96
    assert len(set.union(*parents)) == sum(map(len,parents)) == 96
    sections.append(table(['Набор','Маски / четвёрки','Целевые PNG','Роль','Manifest'],rows))
    sections.append('''<p><b>«16 отложенных масок»</b> в предыдущем отчёте — это 16 test-масок. Их изображения не использовались ни для обновления весов, ни для выбора checkpoint. Ещё 16 val-масок также исключены из градиентного обучения, но используются для выбора лучшей модели. Всего вне train — 32 маски.</p>
<p>Маска и все четыре её изображения остаются в одном split. Производные одного исходника нужно разделять группой, до аугментаций. Совпадение значения seed=42 в manifests не означает совпадения наборов: это одна перестановка с непересекающимися отрезками.</p>
<p>Сохранённые исходные фазы используются для генерации и oracle-проверки. Нейросеть получает четыре изображения; обучение осталось через физический loss, без supervised loss по фазовым меткам. 2000 шагов × batch 8 = 16 000 показов, но только 64 уникальные train-четвёрки: 250 проходов по ним.</p>
<p><b>Отдельный OOD для forward-эксперимента не строился.</b> Маски из прежней папки OOD стали частью новой общей базы. Их старое имя не сохраняет статус OOD относительно нового обучения. Дополнительный arbitrary_control — 64 новые произвольные четвёрки, seed=44002: проверка переноса на прежний тип целей, не набор выбора checkpoint.</p>
<p>Старые test/OOD уже просмотрены в ходе анализа. Для нового подбора по этим выводам необходим свежий финальный контроль; термин «test» всегда относится к конкретному эксперименту.</p></section>''')

    sections.append('''<section id="rules"><h2>Как читать метрики</h2>
<p>Pearson — корреляция по пикселям пары; SSIM — структурное сходство с data_range=1 и окном 7. Q = минимум по четырём плоскостям от их средних SSIM по набору. Checkpoint выбирается по Q, затем среднему Pearson.</p>
<p>Базовый оптический протокол: амплитуда |U|, нормировка по максимуму каждой плоскости, одно-FFT Френель при 532 нм. Отдельные диагностические преобразования выхода подписаны и не подменяют этот протокол.</p>
<p>Основные оценки — по перечитанным PNG после сохранения и чтения фазы. Float-диагностики выделены отдельно. MSE и общий loss не одно и то же при добавлении SSIM. null означает неопределённую метрику, а не ноль. Число валидных пар и ошибок сохраняется. В разных таблицах нельзя смешивать старые произвольные цели и новый forward-набор.</p>
<p>Одна оценка может быть сохранена в журнале, checkpoint и summary. Полный реестр сохраняет все такие источники; это не три независимых эксперимента. Общий «средний балл по всем опытам» не вычисляется.</p></section>''')

    sections.append('<section id="main"><h2>Основное обучение MVP: 64 val</h2>')
    mvp = catalog['runs/mvp/summary.json']
    ssim = catalog['runs/ssim/summary.json']
    interleave = catalog['runs/interleave/summary.json']
    pilot = catalog['checkpoints/pilot_best.pt']
    rows = [quality_row('Начальная сеть',mvp['initial_validation'],'runs/mvp/summary.json','Шаг 0'),
            quality_row('MSE, пилот',pilot['validation'],'checkpoints/pilot_best.pt',f"Бюджет 500; выбран {pilot['step']}"),
            quality_row('Interleave=4',interleave['candidate_validation'],'runs/interleave/summary.json','Бюджет 500; отклонён'),
            quality_row('MSE, контроль',mvp['best_validation'],'runs/mvp/summary.json','Бюджет 1000; выбран 1000'),
            quality_row('MSE+0.1·(1−SSIM)',ssim['candidate_validation'],'runs/ssim/summary.json',f"Бюджет 1000; выбран {ssim['candidate_best_step']}"),
            quality_row('Финальный MVP, CPU',ssim['final']['verified_cpu_validation'],'runs/ssim/summary.json','Бюджет 2000; выбран 1400'),
            quality_row('Тот же checkpoint, Metal',ssim['final']['recorded_mps_validation'],'runs/ssim/summary.json','Проверка другого устройства')]
    sections.append(table(quality_headers,rows))
    sections.append('<h3>Финальные наборы исходного MVP</h3>')
    sections.append(table(quality_headers,[quality_row(name,catalog[path],path,'Выбранный best.pt, PNG') for name,path in [
        ('Test','outputs/test/summary.json'),('OOD','outputs/ood/summary.json'),('Examples','outputs/examples/summary.json')]]))
    sections.append('<h3>Категории test/OOD</h3>')
    rows=[]
    for split in ('test','ood'):
        path=f'outputs/{split}/summary.json'
        rows.extend(quality_row(category,values,path,split) for category,values in catalog[path]['categories'].items())
    sections.append(table(quality_headers,rows)+'</section>')

    sections.append('<section id="comparisons"><h2>Сравнение трёх методов на восьми val</h2>')
    for path,label in [('outputs/comparison/a_summary.json','Исходный отчёт A'),('outputs/comparison/final/summary.json','Локальное повторение C')]:
        data=catalog[path]
        sections.append(f'<h3>{label}</h3>')
        sections.append(table(quality_headers,[quality_row(name,value,path,'8 val, один протокол') for name,value in data['methods'].items()]))
        rows=[]
        for name,value in data['methods'].items():
            timing=value['median_seconds'];key='optimization_seconds' if name=='adam_phase' else 'forward_seconds'
            rows.append([escape(name),number(timing[key]*1000),number(timing['pipeline_seconds']*1000)])
        sections.append(table(['Метод','Медиана операции, мс','Полный путь, мс'],rows))
    sections.append('<p>Adam: 100 шагов, один старт, seed=42. Замеры разных машин между собой не сравниваются; внутри каждого отчёта методы измерены на одном CPU. Операция означает forward сети или оптимизацию Adam.</p>')
    sections.append('<h3>Повторные экспорты и историческая проверка C2</h3>')
    sections.append(table(quality_headers,[quality_row(label,catalog[path],path,context) for label,path,context in [
        ('Initial, 8 val','outputs/comparison/initial/summary.json','Повторный экспорт текущего initial.pt'),
        ('Network, 8 val','outputs/comparison/network/summary.json','Повторный экспорт best.pt'),
        ('Исторический initial, 8 examples','outputs/c2_initial/examples/summary.json','Ранняя версия initial checkpoint; не текущая инициализация')]]))
    sections.append('</section><section id="forward"><h2>Forward-эксперимент timur</h2>')
    forward=catalog['outputs/timur_forward/summary.json']
    mse=catalog['checkpoints/timur_forward/mse_best.pt']
    best=catalog['checkpoints/timur_forward/best.pt']
    log=catalog['runs/timur_forward/metrics.jsonl']
    initial=next(row for row in log if row.get('kind')=='validation' and row['step']==0)
    sections.append('<h3>Выбор на 16 val</h3>')
    sections.append(table(quality_headers,[
        quality_row('Начальная сеть',initial,'runs/timur_forward/metrics.jsonl','Шаг 0'),
        quality_row('Исходный MVP без дообучения',catalog['runs/timur_forward/baseline_validation.json'],'runs/timur_forward/baseline_validation.json','Новые forward-цели'),
        quality_row('Forward: MSE',mse['validation'],'checkpoints/timur_forward/mse_best.pt',f"Бюджет 500; выбран {mse['step']}"),
        quality_row('Forward: MSE+SSIM',best['validation'],'checkpoints/timur_forward/best.pt',f"Бюджет 2000; выбран {best['step']}")]))
    sections.append('<h3>Одинаковый новый test: 16 масок</h3>')
    sections.append(table(quality_headers,[quality_row(name,value,'outputs/timur_forward/summary.json','Forward-test, PNG') for name,value in forward['methods'].items()]))
    sections.append('<h3>Перенос на 64 свежие произвольные цели</h3>')
    sections.append(table(quality_headers,[quality_row(name,value,'outputs/timur_forward/summary.json','arbitrary_control, seed=44002') for name,value in forward['arbitrary_control']['methods'].items()]))
    changes=forward['paired_change_summary']
    sections.append(f'''<p>Парный прирост на forward-test: ΔSSIM {changes['ssim_delta']['mean']:.6f}, ΔPearson {changes['pearson_delta']['mean']:.6f}; обе метрики выросли на всех 16 четвёрках. 95% bootstrap-интервал ΔSSIM: {changes['ssim_delta']['paired_bootstrap_95ci'][0]:.6f}…{changes['ssim_delta']['paired_bootstrap_95ci'][1]:.6f}. Это неопределённость выборки, не разных запусков обучения.</p>
<p class="note">Высокий forward-SSIM относится к реконструкциям старых масок, которые стали новыми целями. Эти изображения не обязаны содержать старые буквы или фигуры. На произвольном контроле SSIM снизился; универсального улучшения исходной задачи не подтверждено.</p></section>''')

    sections.append('<section id="diagnostics"><h2>Диагностики обучения и оптимизации</h2>')
    diagnostics=catalog['runs/diagnostics/summary.json']
    smoke=diagnostics['smoke'];overfit=diagnostics['overfit']
    sections.append(table(['Контроль','Шаги','Показы','MSE до','MSE после','Дополнительно'],[
        ['FFT/backward smoke, 1 четвёрка',str(smoke['steps']),str(smoke['samples_seen']),number(smoke['initial_mse_cpu_check']),number(smoke['final_mse_cpu_check']),f"Минимум нормы градиента {smoke['minimum_gradient_norm']:.6f}"],
        ['Переобучение четырёх четвёрок',str(overfit['steps']),str(overfit['samples_seen']),number(float(np.mean(overfit['initial_float_mse_by_plane']))),number(float(np.mean(overfit['final_float_mse_by_plane']))),f"Время {overfit['seconds']:.3f} с"]]))
    sections.append(table(quality_headers,[quality_row(label,overfit[key],'runs/diagnostics/summary.json','4 диагностические train-четвёрки, PNG') for label,key in [('До переобучения','initial_png'),('После переобучения','final_png')]]))
    sections.append('<h3>Ранний A2: значения, сохранённые только в TASKS.md</h3>')
    sections.append(table(['Пример','Оптимизированные плоскости','Шаги','MSE до','MSE после'],[[escape(row['sample']),row['planes'],str(row['steps']),number(row['initial_mse']),number(row['final_mse'])] for row in catalog['TASKS.md']['experiments']]))
    sections.append('<p>Эти MSE округлены в исходном документе; отсутствующие SSIM/Pearson не достраиваются.</p><h3>Диагностика причин низкого качества: 64 val, float</h3>')
    quality=catalog['runs/quality_diagnosis/summary.json']
    names=['network_float','network_png_phase_float_recons','diagnostic_only_squared_amplitude','diagnostic_only_zero_reconstruction','diagnostic_only_same_phase_for_all_inputs','diagnostic_only_wrong_target_within_category']
    sections.append(table(quality_headers,[quality_row(name,quality[name],'runs/quality_diagnosis/summary.json','Диагностика, не официальный PNG-score') for name in names]))
    sections.append('<p>Нулевая реконструкция имеет неопределённый Pearson и не является допустимым оптическим решением. Метрики после возведения амплитуды в квадрат или подбора усиления не подменяют исходный протокол.</p>')
    extra=[(key,value) for key,value in quality.items() if isinstance(value,(int,float)) and key!='samples']
    sections.append(table(['Показатель диагностики','Значение'],[[escape(key),number(value)] for key,value in extra]))
    for key in ('sparse_targets','normalization_diagnostic','category_mixture'):
        sections.append(f'<h4>{escape(key)}</h4>'+table(['Показатель','Значение'],[[escape(name),number(value)] for name,value in quality[key].items()]))
    sections.append('<h3>Десять контролей Adam по 1000 шагов: float</h3>')
    rows=[]
    for row in catalog['runs/quality_diagnosis/optimization.json']['experiments']:
        one=len(row['planes_fitted'])==1
        values=row['metrics']['planes'][row['planes_fitted'][0]-1] if one else row['metrics']['mean']
        rows.append([row['sample_id'],','.join(map(str,row['planes_fitted'])),row['initialization'],str(row['seed']),
                     number(values['pearson']),number(values['ssim']),number(row['float_mse']),number(row['seconds'])])
    sections.append(table(['Пример','Обучаемые плоскости','Старт','Seed','Pearson','SSIM','MSE всех четырёх','Секунды'],rows))
    sections.append('<p>В строках с одной оптимизированной плоскостью Pearson/SSIM относятся только к ней. Сохранённый float_mse в этом диагностическом файле рассчитан по всем четырём плоскостям; это другая величина. Полные реконструкционные метрики остальных плоскостей остаются в исходных записях.</p></section>')

    sections.append('<section id="telemetry"><h2>Все журналы обучения</h2>')
    rows=[];log_count=0
    for record in records:
        if record['format']!='jsonl':continue
        data=record['value'];steps=[row for row in data if row.get('kind')=='train']
        log_count+=len(data)
        rows.append([source_link(record['path']),str(len(data)),str(len(steps)),f"{steps[0]['step']}…{steps[-1]['step']}",
                     number(steps[0]['loss']),number(steps[-1]['loss']),number(min(row['gradient_norm'] for row in steps)),
                     number(float(np.median([row['step_seconds'] for row in steps])))])
    sections.append(table(['Файл','Все записи','Train-записи','Диапазон шагов','Первый loss','Последний loss','Минимум нормы градиента','Медиана шага, с'],rows))
    sections.append('<p>Loss до и после смены MSE на MSE+SSIM имеет разные определения. Таймер шага не включает весь конвейер; значения разных устройств не являются общим speed benchmark. Внизу включена каждая строка всех журналов, без прореживания.</p></section>')

    audit = catalog['outputs/timur_forward/wavelength_audit.json']
    sections.append('<section><h2>Физика и проверка данных</h2>')
    sections.append(table(['Проверка','Результат'],[
        ['Требуемая длина волны',str(audit['required_wavelength_nm'])+' нм'],
        ['Сохранённые фазы результатов',str(audit['output_phase_cases'])],
        ['Forward-маски датасета',str(audit['forward_dataset_cases'])],
        ['Повторно рассчитанные оптические PNG',str(audit['verified_optical_pngs'])],
        ['Максимальное отклонение независимого NumPy-расчёта',f"{audit['independent_numpy_max_abs_error']:.9g}"],
        ['Совпадения с неверным контролем 638 нм',str(audit['negative_control_exact_matches'])],
        ['Проверенные checkpoints',str(len(audit['checkpoints']))]]))
    sections.append('<p>Исходные синтетические рисунки не зависят от λ; forward-цели и реконструкции проверены на 532 нм. Полные проверки воспроизводимости сохранены среди исходных JSON.</p></section>')

    sections.append(f'<section id="registry"><h2>Полный реестр агрегатов и отдельных примеров</h2><details><summary>{len(registry)} записей: среднее, каждая плоскость, категории, checkpoint и все validation-шаги</summary>')
    rows=[]
    for record,pointer,node,step in registry:
        worst=node.get('worst_plane_mean',node.get('worst_plane',{}))
        domain = ('PNG-фаза / float' if 'png_phase_float_recons' in pointer
                  else 'float' if 'quality_diagnosis' in record['path'] or 'float_phase_float_recons' in pointer
                  else 'PNG')
        planes='<br>'.join(f"{plane['plane']}: {number(plane.get('pearson'))} / {number(plane.get('ssim'))}" for plane in node['planes'])
        rows.append([source_link(record['path'])+'<br><code>'+escape(pointer or '/')+'</code>',domain,
                     number(step),number(node.get('samples',1)),number(node['mean'].get('pearson')),
                     number(node['mean'].get('ssim')),number(worst.get('ssim')),planes,
                     f"{node.get('valid_pearson_pairs','—')}/{node.get('total_pairs','—')}",number(node.get('failure_count'))])
    sections.append(table(['Источник / JSON-путь','Домен','Шаг','N','Pearson','SSIM','Q / min SSIM','Плоскость: Pearson / SSIM','Валидные пары','Ошибки'],rows))
    sections.append('</details><p>Форматирование таблиц округляет до шести знаков. В следующих блоках сохранены все исходные значения, включая полную точность, time, MSE, loss, gradient_norm, ошибки, версии, протоколы и хеши.</p></section>')

    sections.append('<section id="raw"><h2>Полные исходные данные в этом же файле</h2><p>JSON/JSONL включены целиком. Из checkpoints включены все метаданные и validation; массивы весов, optimizer и RNG не являются метриками и сюда не копируются.</p>')
    for index,record in enumerate(records):
        encoded=json.dumps(record['value'],ensure_ascii=False,indent=2,allow_nan=False)
        sections.append(f'<details id="source-{index}"><summary><code>{escape(record["path"])}</code> · {escape(record["format"])}</summary>'
                        f'<p class="hash">SHA256 {record["sha256"]}</p><pre>{escape(encoded)}</pre></details>')
    sections.append('</section>')
    head='''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Данные и метрики всех экспериментов</title>
<style>:root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#18222e;background:#f5f7fa;color-scheme:light}*{box-sizing:border-box}body{margin:0}main{max-width:1240px;padding:28px 22px;margin:auto}h1{font-size:27px;margin:0 0 12px}h2{font-size:21px;margin:30px 0 12px}h3{font-size:16px;margin:22px 0 10px}h4{font-size:14px;margin:15px 0 8px}p{font-size:14px;line-height:1.6;margin:9px 0}nav{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}a{color:#175aaa;text-decoration:underline;text-underline-offset:3px}nav a{font-size:13px}.meta,.hash{font-size:12px;color:#586779;overflow-wrap:anywhere}.note{padding:12px 14px;border-left:3px solid #bd8a27;background:#fff5dd}.table-wrap{overflow-x:auto;margin:12px 0;background:#fff;border:1px solid #dde3eb;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;vertical-align:top;padding:9px 10px;border-bottom:1px solid #e5eaf0;font-variant-numeric:tabular-nums}th{font-weight:650;background:#edf2f7}td{overflow-wrap:anywhere}td:not(:first-child){min-width:70px}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11px;overflow-wrap:anywhere}details{margin:10px 0;background:#fff;border:1px solid #dce3eb;border-radius:8px;padding:12px}summary{cursor:pointer;font-size:13px;line-height:1.5}pre{overflow:auto;max-height:580px;background:#f3f6fa;padding:12px;border-radius:5px;font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;word-break:break-word}section{scroll-margin-top:16px}footer{font-size:12px;color:#586779;margin:26px 0}@media(max-width:600px){main{padding:18px 12px}h1{font-size:23px}h2{font-size:18px}th,td{padding:8px}table{min-width:780px}summary code{font-size:10px}}@media print{body{background:#fff}main{max-width:none}nav,#registry,#raw{display:none}.table-wrap{overflow:visible}table{font-size:9px}section{break-inside:auto}h2,h3{break-after:avoid}}</style></head><body><main><h1>Данные и метрики всех экспериментов</h1>'''
    overview=(f'<p class="meta">Ветка timur · снимок источников {escape(revision)} · λ = 532 нм · '
              f'{len(records)} источников · {log_count} строк журналов · {len(registry)} агрегатов/оценок. '
              'Числа собраны из сохранённых результатов; новые обучения не запускались.</p>')
    nav='<nav>'+''.join(f'<a href="#{anchor}">{label}</a>' for anchor,label in [
        ('definitions','Train / val / test / OOD'),('main','MVP'),('comparisons','Baseline'),
        ('forward','timur'),('diagnostics','Диагностики'),('telemetry','Журналы'),('registry','Все агрегаты'),('raw','Все исходные числа')])+'</nav>'
    document=head+overview+nav+''.join(sections)+'<footer>Самодостаточный отчёт: внешних зависимостей, запросов и серверной части нет. Один файл содержит таблицы и все использованные численные источники.</footer></main></body></html>'
    OUT.write_text(document)
    print(json.dumps({'output':str(OUT),'bytes':OUT.stat().st_size,'sources':len(records),
                      'journal_records':log_count,'quality_nodes':len(registry)},ensure_ascii=False))


if __name__ == '__main__':
    main()
