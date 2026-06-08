#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""账号2 并行任务: 用 PETCGDNN 当【快速裁判】, 在 RML22(spS=8) 上搜索 GPR 去噪超参数变体。

为什么 PET 当裁判: 已实测确认 PET 对 GPR 去噪敏感(+1.9%), 是"验证过刻度的尺子"; 用它给
不同去噪变体(L0/slope/σ_f²/负β)排序才有信噪比、才可信(ULCNN 太小可能对去噪不敏感)。

设计:
- 复用原版 main.py 的 train_selected_models / evaluate_selected_models -> 与 baseline 完全一致,
  且产出原版结果格式(overall_accuracy.txt + accuracy_by_snr.csv), 便于迁回本地汇总。
- 参数化去噪在本脚本里直接对 dict 做(调 apply_gpr_denoising_efficient_per_sample, 已加 L0/slope/sigma_f_mode),
  绕开 main.py 的去噪缓存(缓存名只含方法名不含超参 -> 会串味)。
- 每个变体: 去噪dict -> split(seed42, 与baseline同) -> train PET(SEARCH_EPOCHS, 早停) -> eval per-SNR -> 存 Drive。
- 断点续跑: 变体目录有 DONE.txt 则跳过。后台 subprocess 启动 -> MCP 断开不影响。

启动(在 notebook cell):
  os.environ['SEARCH_EPOCHS']='50'
  subprocess.Popen('python -u /content/radioML-v4b/script/method_search_pet.py',
                    shell=True, stdout=open('<drive>/methodsearch_stdout.log','a'),
                    stderr=subprocess.STDOUT)
"""
import os, sys, json, time
import numpy as np

REPO = '/content/radioML-v4b'
SRC = f'{REPO}/src'
sys.path.insert(0, SRC)
os.chdir(SRC)  # main.py 内部相对路径(config/training.yaml 等)需要在 src 下

DRIVE = '/content/drive/MyDrive'
DATA = f'{DRIVE}/RML22'                       # 与 baseline 同一份数据(pickle 文件)
OUT_ROOT = f'{DRIVE}/results_rml22_methodsearch'
os.makedirs(OUT_ROOT, exist_ok=True)

import tensorflow as tf
import main as M
from preprocess import split_data_raw
from efficient_gpr_per_sample import apply_gpr_denoising_efficient_per_sample

# 训练超参(取自 config/training.yaml, 与 baseline 一致; 搜索用较少 epoch + 早停加速)
BATCH = 128
LR = 1e-3
PAT_LR = 2
PAT_ES = 12        # 早停耐心(比 baseline 的 15 略小, 加速排序)
FACTOR = 0.7
SEARCH_EPOCHS = int(os.environ.get('SEARCH_EPOCHS', '50'))
JUDGE = 'pet'

# 变体集合: (name, denoise?, L0, slope, sigma_f_mode)
# 覆盖: L0 扫描 / slope(β) 扫描含负β / σ_f² 有无 / SNR非自适应(slope=0)
VARIANTS = [
    ('baseline_none',   False, None, None, None),          # 不去噪基准
    ('std_L5_s0.25',    True,  5.0,  0.25, 'unit'),         # 当前部署律(锚点, 预期≈+1.9%)
    ('L4_s0.25',        True,  4.0,  0.25, 'unit'),
    ('L6_s0.25',        True,  6.0,  0.25, 'unit'),
    ('L7_s0.25',        True,  7.0,  0.25, 'unit'),
    ('L5_s0_const',     True,  5.0,  0.0,  'unit'),         # L 不随 SNR 自适应
    ('L5_s0.5',         True,  5.0,  0.5,  'unit'),         # 低SNR时 L 增长更快
    ('L5_sNEG0.1',      True,  5.0, -0.1, 'unit'),          # 负β: 低SNR时 L 反而缩小
    ('L5_s0.25_sigf',   True,  5.0,  0.25, 'signal_var'),   # 带 σ_f²(eff_noise=1/SNR_lin)
    ('L6_s0.25_sigf',   True,  6.0,  0.25, 'signal_var'),
]

LOG = open(f'{OUT_ROOT}/search.log', 'a')
def log(m):
    LOG.write(m + '\n'); LOG.flush(); print(m, flush=True)


def run_variant(dataset, name, denoise, L0, slope, sigma_f_mode):
    vdir = f'{OUT_ROOT}/{name}'
    done = f'{vdir}/DONE.txt'
    if os.path.exists(done):
        log(f'skip {name} (done)'); return
    os.makedirs(vdir, exist_ok=True)
    models_dir = f'{vdir}/models'; results_dir = f'{vdir}/results'; plots_dir = f'{vdir}/plots'
    for d in (models_dir, results_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    # 去噪(dict版函数不修改输入 dict)
    if denoise:
        log(f'>>> {name}: denoise L0={L0} slope={slope} sigma_f={sigma_f_mode}')
        dn_dict, dt = apply_gpr_denoising_efficient_per_sample(
            dataset, L0=L0, slope=slope, sigma_f_mode=sigma_f_mode)
        log(f'    denoise done ({dt:.0f}s)')
    else:
        dn_dict = dataset

    # split(与 baseline 同一确定性切分 seed=42)
    Xtr, Xv, Xte, ytr, yv, yte, snrtr, snrv, snrte, mods = split_data_raw(dn_dict)
    ncls = len(mods); ishape = Xtr.shape[1:]
    ytr_oh = tf.keras.utils.to_categorical(ytr, ncls)
    yv_oh = tf.keras.utils.to_categorical(yv, ncls)
    yte_oh = tf.keras.utils.to_categorical(yte, ncls)

    suffix = '_search'
    # 训练 PET(原版 train_selected_models, 内部存 best/last + 逐epoch日志)
    M.train_selected_models([JUDGE], Xtr, ytr_oh, Xv, yv_oh, ishape, ncls,
                            models_dir, plots_dir, suffix, BATCH, SEARCH_EPOCHS,
                            LR, PAT_LR, PAT_ES, FACTOR)
    # 评估(原版 evaluate_selected_models -> overall_accuracy.txt + accuracy_by_snr.csv)
    M.evaluate_selected_models([JUDGE], Xte, yte_oh, snrte, mods,
                               models_dir, results_dir, suffix, results_suffix=suffix)

    json.dump({'name': name, 'denoise': denoise, 'L0': L0, 'slope': slope,
               'sigma_f_mode': sigma_f_mode, 'epochs': SEARCH_EPOCHS,
               'elapsed_s': time.time() - t0},
              open(f'{vdir}/variant.json', 'w'), indent=2)
    open(done, 'w').write(f'{time.time() - t0:.0f}s')
    log(f'DONE  {name} ({time.time() - t0:.0f}s)')


def main():
    log(f'=== PET method-search start === judge={JUDGE} epochs={SEARCH_EPOCHS} '
        f'variants={[v[0] for v in VARIANTS]}')
    M.set_random_seed(42)
    M.configure_gpu()
    dataset = M.load_radioml_data(DATA)
    log(f'dataset loaded: {len(dataset)} (mod,snr) keys, total '
        f'{sum(len(v) for v in dataset.values())} samples')
    for v in VARIANTS:
        try:
            run_variant(dataset, *v)
        except Exception as e:
            import traceback
            log(f'FAIL  {v[0]}: {e}\n{traceback.format_exc()[-1800:]}')
    log('=== ALL DONE ===')


if __name__ == '__main__':
    main()
