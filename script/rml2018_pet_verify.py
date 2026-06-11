#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RML2018 PET 探针复核(回应 -2.58pp 质疑): seed42 重跑 baseline + L-sweep{2,4,6,8} + seed123 换种子复跑 baseline/L8。

原探针(results_rml2018_pet/): seed42 baseline 0.5811 vs denoise_L8 0.5553 (-2.58pp)。
逐SNR签名: 低SNR(-12..-4dB)正向(+0.7~+2.7pp), 高SNR(>=4dB)一致 -4~-6pp -> 全部损失来自高SNR过平滑。
本脚本区分两个假设: 若小 L 时高SNR损伤收窄 -> "L0=8 锚定过大"; 若各 L 都显著负 -> 方法边界。
子采样与原探针完全一致(loader 内部 seed=42, per_group=300); 训练种子按变体显式重置。
"""
import os, sys, json, time
import numpy as np

REPO = '/content/radioML-v4b'; SRC = f'{REPO}/src'
sys.path.insert(0, SRC); os.chdir(SRC)
DRIVE = '/content/drive/MyDrive'
HDF5 = os.environ.get('HDF5_PATH', '/content/rml2018/GOLD_XYZ_OSC.0001_1024.hdf5')
OUT_ROOT = os.environ.get('OUT_ROOT', f'{DRIVE}/results_rml2018_pet_verify')
os.makedirs(OUT_ROOT, exist_ok=True)

import tensorflow as tf
import main as M
from preprocess import split_data_raw, load_rml2018_hdf5
from efficient_gpr_per_sample import apply_gpr_denoising_efficient_per_sample

BATCH = 128; LR = 1e-3; PAT_LR = 2; PAT_ES = 10; FACTOR = 0.7
EPOCHS = int(os.environ.get('SEARCH_EPOCHS', '30'))
MODEL = os.environ.get('JUDGE_MODEL', 'pet')
PER_GROUP = int(os.environ.get('PER_GROUP', '300'))

LOG = open(f'{OUT_ROOT}/verify.log', 'a')
def log(m): LOG.write(m + '\n'); LOG.flush(); print(m, flush=True)


def run_variant(dataset, name, denoise, L0, seed):
    vdir = f'{OUT_ROOT}/{name}'
    if os.path.exists(f'{vdir}/DONE.txt'):
        log(f'skip {name}'); return
    models_dir = f'{vdir}/models'; results_dir = f'{vdir}/results'; plots_dir = f'{vdir}/plots'
    for d in (models_dir, results_dir, plots_dir): os.makedirs(d, exist_ok=True)
    t0 = time.time()
    M.set_random_seed(seed)
    dn = dataset
    if denoise:
        dn, dt = apply_gpr_denoising_efficient_per_sample(dataset, L0=L0, slope=0.25,
                                                          sigma_f_mode='unit', noise_est='label')
        log(f'{name}: denoise L0={L0} ({dt:.0f}s)')
    Xtr, Xv, Xte, ytr, yv, yte, snrtr, snrv, snrte, mods = split_data_raw(dn)
    ncls = len(mods)
    ytr_oh = tf.keras.utils.to_categorical(ytr, ncls)
    yv_oh = tf.keras.utils.to_categorical(yv, ncls)
    yte_oh = tf.keras.utils.to_categorical(yte, ncls)
    M.train_selected_models([MODEL], Xtr, ytr_oh, Xv, yv_oh, Xtr.shape[1:], ncls,
                            models_dir, plots_dir, '_2018', BATCH, EPOCHS, LR, PAT_LR, PAT_ES, FACTOR)
    M.evaluate_selected_models([MODEL], Xte, yte_oh, snrte, mods,
                               models_dir, results_dir, '_2018', results_suffix='_2018')
    json.dump({'name': name, 'denoise': denoise, 'L0': L0, 'seed': seed,
               'model': MODEL, 'per_group': PER_GROUP, 'epochs': EPOCHS, 'elapsed_s': time.time()-t0},
              open(f'{vdir}/variant.json', 'w'), indent=2)
    open(f'{vdir}/DONE.txt', 'w').write(f'{time.time()-t0:.0f}s')
    log(f'DONE {name} ({time.time()-t0:.0f}s)')


def main():
    log(f'=== RML2018 PET verify === model={MODEL} epochs={EPOCHS} per_group={PER_GROUP}')
    M.set_random_seed(42); M.configure_gpu()
    t0 = time.time()
    dataset = load_rml2018_hdf5(HDF5, per_group=PER_GROUP)   # loader 内部 seed=42, 与原探针子采样一致
    log(f'loaded {len(dataset)} keys {sum(len(v) for v in dataset.values())} frames ({time.time()-t0:.0f}s)')
    # 子采样连同配置归档 Drive: 复核/复现不再依赖 21GB 原始文件
    arc = f'{DRIVE}/RML2018/subsample_pg300_seed42.npz'
    if not os.path.exists(arc):
        ta = time.time()
        np.savez(arc, **{f'{m}|{s}': v for (m, s), v in dataset.items()})
        json.dump({'per_group': PER_GROUP, 'loader_seed': 42, 'source': os.path.basename(HDF5)},
                  open(f'{DRIVE}/RML2018/subsample_pg300_seed42.json', 'w'))
        log(f'subsample archived to Drive ({time.time()-ta:.0f}s)')
    run_variant(dataset, 's42_baseline', False, 8.0, 42)
    for L in (2.0, 4.0, 6.0, 8.0):
        run_variant(dataset, f's42_L{L:g}', True, L, 42)
    run_variant(dataset, 's123_baseline', False, 8.0, 123)
    run_variant(dataset, 's123_L8', True, 8.0, 123)
    log('=== ALL DONE ===')


if __name__ == '__main__':
    main()
