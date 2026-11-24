import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import save_json


def f(value):
    if value is None:
        return '—'
    return f'{value:.7g}' if isinstance(value,(float,int)) else str(value)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
        '| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(f(v) for v in row)+' |' for row in rows])


def build(directory):
    root=Path(directory)
    def read(name):
        return json.loads((root/(name+'.json')).read_text(encoding='utf-8'))
    machine=read('machine')
    smoke=machine.get('smoke',False)
    v=machine['variant']
    eq=read('equivalence')
    plain=read('plain-B')
    manual=read('manual-P2')
    strong=[read('strong-P1'),read('strong-P2')]
    ns=[read('strong-P1-ns'),read('strong-P2-ns')]
    weak=[read('weak-P1'),strong[1]]
    quality=[read('quality-small-B'),read('quality-ddp-B'),read('quality-linear')]
    qp=read('quality-plain-B')
    qr=read('recovery/quality-recovery')
    event=read('recovery/recovery')
    baseline={row['step']:row['loss'] for row in quality[1]['losses']}
    restored={row['step']:row['loss'] for row in qr['losses']}
    assert baseline.keys()==restored.keys()
    differences={s:abs(baseline[s]-restored[s]) for s in baseline}
    max_difference=max(differences.values())
    if max_difference>1e-6 or abs(qr['ACC']-quality[1]['ACC'])>1e-12:
        raise RuntimeError(f'Recovery mismatch: maximum loss difference {max_difference}')
    assert event['checkpoint_step']==20 and event['repeated_steps']==4
    for record in [plain,manual]+strong+ns+[weak[0]]:
        assert record['n_th']==machine['n_th']
        assert record['steps']==(3 if smoke else 50)
        assert record['warmup']==(1 if smoke else 10)
        assert len(record['step_seconds'])==record['steps']
        assert record['t_step']==median(record['step_seconds'])
        assert math.isclose(record['X'],record['steps']*record['global_batch']/record['elapsed'])
    if not smoke:
        assert qp['epochs']==3 and qp['optimizer_steps']==108
        for item,count in zip(quality,[216,108,108]):
            assert item['epochs']==3
            if not item.get('diverged'):
                assert item['optimizer_steps']==count
        assert qr['optimizer_steps']==108
    assert eq['steps']==50 and eq['max_relative_error']<=1e-4 and eq['weight_sum_spread']<=1e-9
    comm=[s['t_step']-n['t_step'] for s,n in zip(strong,ns)]
    fractions=[t/s['t_step'] for t,s in zip(comm,strong)]
    strong_rows=[]
    for i,item in enumerate(strong):
        P=i+1
        speed=item['X']/strong[0]['X']
        e=None if P==1 else (1/speed-1/P)/(1-1/P)
        strong_rows.append([P,item['local_batch'],1000*item['t_step'],1000*ns[i]['t_step'],
                            item['X'],speed,speed/P,e,fractions[i]])
    weak_rows=[]
    for i,item in enumerate(weak):
        P=i+1
        speed=item['X']/weak[0]['X']
        weak_rows.append([P,item['global_batch'],1000*item['t_step'],item['X'],speed,speed/P])
    figures=root/'figures'
    figures.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.8,4.2))
    ax.plot([1,2],[r['X'] for r in strong],'o-',label='Measured DDP')
    ax.plot([1,2],[strong[0]['X'],2*strong[0]['X']],'--',label='Ideal')
    ax.set(xlabel='Processes P',ylabel='Examples / second',xticks=[1,2],title='Strong scaling, global batch 192')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures/'strong-scaling.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,4.2))
    for rec,label in [(quality[1],'DDP, lr=0.02'),(quality[2],'DDP, lr=0.04'),(qr,'DDP after restart')]:
        ax.plot([r['step'] for r in rec['losses']],[r['loss'] for r in rec['losses']],label=label,alpha=.8)
    ax.axvline(25,color='gray',ls='--',lw=1)
    ax.set(xlabel='Optimizer step',ylabel='Mean training loss',title='Learning and recovery')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(figures/'training-loss.png',dpi=160)
    plt.close(fig)
    text=['# Лабораторна робота №6',
          '**Розподілене навчання нейромережі за даними на CPU**',
          'Дьяченко Валерія, ЗК-61мп. Варіант 7.',
          '## Мета',
          'Реалізувати синхронне навчання з ручним усередненням градієнтів та DDP, '
          'перевірити масштабування і відновлення після відмови.',
          '## 1. Дані, модель і стенд',
          'Використано DermaMNIST 28 × 28: перші 7000 навчальних зображень і всю тестову '
          'частину з 2005 зображень, 7 класів та 3 канали. Пікселі перетворено у діапазон '
          '[−1; 1]. Val не використовується. Дані перевірено за MD5 офіційного файлу. '
          'Мережа містить Conv(3→32), ReLU, MaxPool(2), Conv(32→64), ReLU, MaxPool(2), '
          'Flatten, Linear(3136→256), ReLU, Linear(256→7), без BatchNorm та Dropout.',
          'Зерно torch.manual_seed(0) задається перед створенням кожної моделі. '
          'Використано CPU, float32, gloo, SGD з momentum=0,9, η₀=0,02. '
          'Усі прогони мають однакове n_th=max(1,⌊C_phys/2⌋), включно з одним процесом. '
          'DataLoader працює без додаткових воркерів, з drop_last=True. '
          'DistributedSampler має seed=0; перед кожною епохою викликано set_epoch.',
          '### Таблиця 1. Стенд',
          table(['Процесор','C_phys','Логічних ядер','RAM, ГіБ','ОС','Python','PyTorch','n_th'],
                [[machine['processor'],machine['physical_cores'],machine['logical_cores'],
                  machine['memory_GiB'],machine['os'],machine['python'].split()[0],
                  machine['torch'],machine['n_th']]]),
          table(['Датасет','K','C','c1 / c2 / h','n_params','M, МіБ','N','B','P_max'],
                [['DermaMNIST',7,3,'32 / 64 / 256',machine['n_params'],
                  machine['gradient_bytes']/1024**2,7000,192,2]]),
          f'n_params = 9·3·32 + 32 + 9·32·64 + 64 + 49·64·256 + 256 + 256·7 + 7 = {machine["n_params"]}. '
          'M = 4·n_params байтів.',
          '## 2. Ручна синхронізація і DDP',
          'Ручний режим розсилає початкові параметри з rank 0 через broadcast. '
          'Після backward кожний тензор градієнта проходить all_reduce(SUM) і ділиться на P. '
          'Для перевірки взято перші 192 приклади. Однопроцесний градієнт порівняно '
          'з усередненими локальними градієнтами на індексах r, r+2, … . '
          'Окремо після рівно 50 синхронізованих кроків суми ваг у float64 зібрано через all_gather.',
          'DDP використовує типовий розмір кошиків без зміни bucket_cap_mb. '
          'Для кожного виміру швидкості виконано 10 кроків прогріву та 50 вимірюваних кроків. '
          't_step — медіана 50 часів rank 0; X = 50·B / t, де t — повний час цих 50 кроків. '
          'Час підготовки даних та запуску процесів не входить у замір. '
          'no_sync виміряно в окремому прогоні зі свіжою моделлю; контекст охоплює forward і backward. '
          'Ця модель не використовується для подальшого навчання.',
          '### Таблиця 2. Синхронізація градієнтів',
          table(['Режим','t_step, мс','Обмін на крок, мс','Частка обміну','X, прикл./с'],[
              ['1 процес, B=192',1000*plain['t_step'],0,0,plain['X']],
              ['2 процеси, вручну',1000*manual['t_step'],1000*manual['t_sync'],manual['t_sync']/manual['t_step'],manual['X']],
              ['2 процеси, DDP',1000*strong[1]['t_step'],1000*comm[1],fractions[1],strong[1]['X']],
              ['2 процеси, DDP no_sync',1000*ns[1]['t_step'],0,0,ns[1]['X']]]),
          table(['max відносна різниця градієнтів','Різниця сум ваг','ACC, 1 процес','ACC, DDP','AUC, 1 процес','AUC, DDP'],
                [[eq['max_relative_error'],eq['weight_sum_spread'],qp['ACC'],quality[1]['ACC'],qp['AUC'],quality[1]['AUC']]]),
          'Для ручного режиму t_sync виміряно від першого all-reduce до повернення з останнього. '
          'Для DDP t_comm,exp = t_step − t_step,ns; c = t_comm,exp / t_step. '
          'Оскільки це різниця двох окремих вимірів, невелике від’ємне значення може означати '
          'шум вимірювання; його не обрізано до нуля.',
          '## 3. Strong scaling',
          'Глобальний батч B=192; b=192 при P=1 і b=96 при P=2. Обидва рядки '
          'отримано через torchrun і DDP. Однопроцесний DDP є базою S та E.',
          '### Таблиця 3. Strong scaling',
          table(['P','b','t_step, мс','t_step,ns, мс','X, прикл./с','S','E','e','c'],strong_rows),
          'S=X(P)/X(1); E=S/P; e=(1/S−1/P)/(1−1/P). Для P=1 e не визначається.',
          '![Strong scaling](figures/strong-scaling.png)',
          '## 4. Weak scaling та швидкість навчання',
          'Локальний батч b=96. Глобальний батч зростає від 96 до 192. Рядок P=2 '
          'повторно використано зі strong scaling. S_w=X(P)/X(1), E_w=S_w/P.',
          '### Таблиця 4. Weak scaling',
          table(['P','B_P','t_step, мс','X, прикл./с','S_w','E_w'],weak_rows),
          'Моделі якості навчаються 3 епохи з однакового початкового стану. '
          + ('У прогоні лінійного правила зафіксовано розходження на кроці '
             + str(quality[2]['divergence_step']) + '; ACC/AUC після трьох епох відсутні. '
             if quality[2].get('diverged') else '') +
          'Лінійне правило: η_P=0,02·192/96=0,04, без розігріву η. '
          'Усі значення ACC та macro OVR AUC обчислено на повній тестовій частині. '
          'Loss — середня cross-entropy; у розподіленому режимі локальні втрати усереднено між процесами.',
          '### Таблиця 5. Якість за різних глобальних батчів',
          table(['Режим','B_P','η','Кроків за 3 епохи','ACC','AUC','max loss останньої епохи'],
                [[label,r['global_batch'],r['lr'],r['optimizer_steps'],r['ACC'],r['AUC'],r['last_epoch_max_loss']]
                 for label,r in zip(['P=1','P=2, η₀','P=2, лінійне правило'],quality)]),
          'За B=192 виконується 36 кроків за епоху і 108 за три епохи; '
          'за B=96 — 72 і 216 відповідно.',
          '## 5. Відмова та відновлення',
          'Контрольну точку кожні 10 кроків зберігає rank 0: ваги, стан SGD, епоха, '
          'наступний батч, глобальний крок, журнал втрат і стан генератора. '
          'Файл checkpoint.tmp атомарно замінює checkpoint.pt; після цього виконується barrier. '
          'Процес rank 1 на початку кроку 25 створює позначку і завершується через os._exit(1). '
          'torchrun використовує статичну адресу 127.0.0.1 і max-restarts=1. '
          'Після перезапуску всі процеси читають контрольну точку, відновлюють sampler.set_epoch '
          'і пропускають уже зафіксовані батчі.',
          '### Таблиця 6. Відмова і відновлення',
          table(['K_c','s_f','Крок checkpoint','Повторно виконано кроків','t_lost, с'],
                [[10,25,event['checkpoint_step'],event['repeated_steps'],event['t_lost']]]),
          table(['Крок','23','24','25','26','27','ACC після 3 епох'],[
              ['З відмовою']+[restored[s] for s in range(23,28)]+[qr['ACC']],
              ['Без відмови']+[baseline[s] for s in range(23,28)]+[quality[1]['ACC']]]),
          f'Відмова сталася до виконання кроку 25. Відновлено checkpoint кроку 20, '
          f'тому повторно виконано кроки 21–24: чотири завершені до відмови кроки. '
          f't_f−t_c={event["t_f"]-event["t_c"]:.6g} с; t_r={event["t_r"]:.6g} с; '
          f't_lost=(t_f−t_c)+t_r={event["t_lost"]:.6g} с. '
          f'Максимальна різниця loss між усіма відповідними кроками: {max_difference:.6g}.',
          '![Навчання та відновлення](figures/training-loss.png)',
          '## Висновок',
          f'При P=2 strong scaling дав S={strong_rows[1][5]:.5g}, E={strong_rows[1][6]:.5g}, '
          f'e={strong_rows[1][7]:.5g}. Частка неперекритого обміну c={fractions[1]:.5g}. '
          f'Weak scaling дав S_w={weak_rows[1][4]:.5g}, E_w={weak_rows[1][5]:.5g}. '
          'Ці результати характеризують баланс обчислень, синхронізації та ресурсів CPU цього вузла.',
          f'За η=0,02 і B=96 отримано ACC={quality[0]["ACC"]:.5g}; '
          f'за η=0,02 і B=192 — {quality[1]["ACC"]:.5g}; '
          f'за лінійним правилом η=0,04 і B=192 — {f(quality[2]["ACC"])}. '
          'Порівняння виконано за однакових трьох епох; велике B зменшує число оновлень. '
          'Один прогін кожного режиму не дає оцінки статистичної значущості різниці якості.',
          'Відновлення перевірено за журналом втрат і тестовою ACC; журнал відновленого '
          'прогону містить остаточну послідовність кроків без дублікатів.',
          '## Джерела',
          '1. Методичні вказівки до лабораторної роботи 6.\n'
          '2. MedMNIST v3.0: https://zenodo.org/records/10519652 . '
          'DermaMNIST: CC BY-NC 4.0.\n'
          '3. PyTorch Distributed: https://docs.pytorch.org/docs/stable/distributed.html']
    if smoke:
        text.insert(1,'**ТЕХНІЧНИЙ ТЕСТ: скорочені виміри та одна епоха; не для здачі.**')
    (root/'REPORT.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    with (root/'scaling.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['P','b','t_step_ms','t_step_ns_ms','X','S','E','e','c'])
        writer.writerows(strong_rows)
    save_json(root/'checks.json',dict(
        complete=True,smoke=smoke,gradient_equivalent=True,
        max_relative_gradient_error=eq['max_relative_error'],
        weight_sum_spread=eq['weight_sum_spread'],
        recovery_max_loss_difference=max_difference,
        recovery_ACC_difference=abs(qr['ACC']-quality[1]['ACC']),
        completed_steps=qr['optimizer_steps']))
    print('Six tables, figures and recovery comparison: OK',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    build(p.parse_args().directory)
