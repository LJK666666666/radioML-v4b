#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""斜率修正变体(用户提出): 2016a 标签 SNR=实际/2 (SNR×2 bug), 在 2016 上调出的 slope=0.25 是
"每标签dB"; 换算到真实 dB 轴 = 0.125。2018 标签=实际 -> 同一物理律应取 slope=0.125。
试 L0∈{6,8} × slope=0.125 (L0 锚点只作用于 SNR>=0 段, 不受轴换算影响)。
接在 rml2018_pet_verify.py 的 7 个变体后跑, 共用 OUT_ROOT 与子采样。
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


def run_variant(dataset, name, L0, slope, seed):
    vdir = f'{OUT_ROOT}/{name}'
    if os.path.exists(f'{vdir}/DONE.txt'):
        log(f'skip {name}'); return
    models_dir = f'{vdir}/models'; results_dir = f'{vdir}/results'; plots_dir = f'{vdir}/plots'
    for d in (models_dir, results_dir, plots_dir): os.makedirs(d, exist_ok=True)
    t0 = time.time()
    M.set_random_seed(seed)
    dn, dt = apply_gpr_denoising_efficient_per_sample(dataset, L0=L0, slope=slope,
                                                      sigma_f_mode='unit', noise_est='label')
    log(f'{name}: denoise L0={L0} slope={slope} ({dt:.0f}s)')
    Xtr, Xv, Xte, ytr, yv, yte, snrtr, snrv, snrte, mods = split_data_raw(dn)
    ncls = len(mods)
    ytr_oh = tf.keras.utils.to_categorical(ytr, ncls)
    yv_oh = tf.keras.utils.to_categorical(yv, ncls)
    yte_oh = tf.keras.utils.to_categorical(yte, ncls)
    M.train_selected_models([MODEL], Xtr, ytr_oh, Xv, yv_oh, Xtr.shape[1:], ncls,
                            models_dir, plots_dir, '_2018', BATCH, EPOCHS, LR, PAT_LR, PAT_ES, FACTOR)
    M.evaluate_selected_models([MODEL], Xte, yte_oh, snrte, mods,
                               models_dir, results_dir, '_2018', results_suffix='_2018')
    json.dump({'name': name, 'denoise': True, 'L0': L0, 'slope': slope, 'seed': seed,
               'model': MODEL, 'per_group': PER_GROUP, 'epochs': EPOCHS, 'elapsed_s': time.time()-t0},
              open(f'{vdir}/variant.json', 'w'), indent=2)
    open(f'{vdir}/DONE.txt', 'w').write(f'{time.time()-t0:.0f}s')
    log(f'DONE {name} ({time.time()-t0:.0f}s)')


def main():
    log(f'=== RML2018 PET slope-corrected variants === epochs={EPOCHS} per_group={PER_GROUP}')
    M.set_random_seed(42); M.configure_gpu()
    t0 = time.time()
    dataset = load_rml2018_hdf5(HDF5, per_group=PER_GROUP)   # loader 内部 seed=42, 同一子采样
    log(f'loaded {len(dataset)} keys {sum(len(v) for v in dataset.values())} frames ({time.time()-t0:.0f}s)')
    run_variant(dataset, 's42_L6_slp125', 6.0, 0.125, 42)
    run_variant(dataset, 's42_L8_slp125', 8.0, 0.125, 42)
    log('=== SLOPE EXT DONE ===')


if __name__ == '__main__':
    main()
