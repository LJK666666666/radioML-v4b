#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""差分特征(Gemini最高EV零参数法)在 RML22 上测试: 能否处理相位/CFO -> 提升分类。

动机(数据分析): RML22 用ETU70信道但spS=8下最大时延≈0.6采样->近似平坦衰落(每帧≈常数复增益=幅度+相位)
+ CFO(渐进相位)。相对2016a的主导新增损坏是【相位/CFO】,非重ISI。GPR去噪去不掉相位/CFO->~0。
差分特征 z[n]=y[n]·conj(y[n-1]): 常数相位完全抵消、CFO变成CNN易学的静态偏置。
输入 2×N -> 4×N=[Re(y),Im(y),Re(z),Im(z)]。用相位无关的CNN当探针(phase-naive模型最受益)。

变体: raw(2×N基线) vs diff(4×N差分增强)。复用 main.py train/eval。模型/数据集可配。
"""
import os, sys, json, time
import numpy as np

REPO = '/content/radioML-v4b'; SRC = f'{REPO}/src'
sys.path.insert(0, SRC); os.chdir(SRC)
DRIVE = '/content/drive/MyDrive'
DATA = os.environ.get('DATA', f'{DRIVE}/RML22')
OUT_ROOT = f'{DRIVE}/results_diff_feature'
os.makedirs(OUT_ROOT, exist_ok=True)

import tensorflow as tf
import main as M
from preprocess import split_data_raw

BATCH = 128; LR = 1e-3; PAT_LR = 2; PAT_ES = 12; FACTOR = 0.7
EPOCHS = int(os.environ.get('SEARCH_EPOCHS', '50'))
MODEL = os.environ.get('JUDGE_MODEL', 'cnn2d')   # phase-naive CNN 当探针
LOG = open(f'{OUT_ROOT}/diff.log', 'a')
def log(m): LOG.write(m + '\n'); LOG.flush(); print(m, flush=True)


def add_diff_channels(dataset):
    """dict {(mod,snr):(N,2,128)} -> (N,4,128): [Re(y),Im(y),Re(z),Im(z)], z[n]=y[n]·conj(y[n-1])."""
    out = {}
    for k, X in dataset.items():
        y = X[:, 0, :] + 1j * X[:, 1, :]            # (N,128)
        z = np.zeros_like(y)
        z[:, 1:] = y[:, 1:] * np.conj(y[:, :-1])
        out[k] = np.stack([X[:, 0, :], X[:, 1, :], z.real, z.imag], axis=1).astype(np.float32)
    return out


def run_variant(dataset, name):
    vdir = f'{OUT_ROOT}/{name}'
    if os.path.exists(f'{vdir}/DONE.txt'):
        log(f'skip {name}'); return
    models_dir = f'{vdir}/models'; results_dir = f'{vdir}/results'; plots_dir = f'{vdir}/plots'
    for d in (models_dir, results_dir, plots_dir): os.makedirs(d, exist_ok=True)
    t0 = time.time()
    Xtr, Xv, Xte, ytr, yv, yte, snrtr, snrv, snrte, mods = split_data_raw(dataset)
    ncls = len(mods)
    log(f'{name}: input {Xtr.shape[1:]} ncls={ncls}')
    ytr_oh = tf.keras.utils.to_categorical(ytr, ncls)
    yv_oh = tf.keras.utils.to_categorical(yv, ncls)
    yte_oh = tf.keras.utils.to_categorical(yte, ncls)
    M.train_selected_models([MODEL], Xtr, ytr_oh, Xv, yv_oh, Xtr.shape[1:], ncls,
                            models_dir, plots_dir, '_diff', BATCH, EPOCHS, LR, PAT_LR, PAT_ES, FACTOR)
    M.evaluate_selected_models([MODEL], Xte, yte_oh, snrte, mods,
                               models_dir, results_dir, '_diff', results_suffix='_diff')
    json.dump({'name': name, 'model': MODEL, 'epochs': EPOCHS, 'elapsed_s': time.time()-t0},
              open(f'{vdir}/variant.json', 'w'), indent=2)
    open(f'{vdir}/DONE.txt', 'w').write(f'{time.time()-t0:.0f}s')
    log(f'DONE {name} ({time.time()-t0:.0f}s)')


def main():
    log(f'=== diff-feature validation === model={MODEL} epochs={EPOCHS} data={DATA}')
    M.set_random_seed(42); M.configure_gpu()
    dataset = M.load_radioml_data(DATA)
    log(f'loaded {len(dataset)} keys')
    run_variant(dataset, 'raw_2ch')                       # 基线 2×N
    run_variant(add_diff_channels(dataset), 'diff_4ch')   # 差分增强 4×N
    log('=== ALL DONE ===')


if __name__ == '__main__':
    main()
