#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RML2018.01A 全量数据 PET 对比实验 (baseline vs GPR 去噪), 供 A100/高内存 runtime 手动跑。

变体: baseline + L0=6/slope=0.25 (探针 sweep 最优) + L0=6/slope=0.125 (sps 规则锚定 + SNR 轴修正)。
内存预算: 全量 X≈21GB float32, 去噪副本+划分副本叠加峰值≈63GB -> 需要 A100 等高内存 runtime (~83GB)。
环境变量: HDF5_PATH / OUT_ROOT / SEARCH_EPOCHS / PER_GROUP(默认4096=全量) / BATCH / LR。
batch 提示: PET 仅~71k 参数, 显存不是约束; batch 增大需同步放大学习率(默认 512 配 2e-3)。
续训: RESUME_EPOCHS=<新的总epoch数> 重跑本脚本, 已有 last 权重的变体从 last 接着训到该总数
(日志接续、best 仅在超过历史最佳时覆盖、优化器/LR 状态随 .keras 一并恢复), 无 last 的变体
从头训到该总数; 此模式下忽略 DONE.txt 跳过逻辑。
"""
import os, sys, json, time
import numpy as np

REPO = '/content/radioML-v4b'; SRC = f'{REPO}/src'
sys.path.insert(0, SRC); os.chdir(SRC)
DRIVE = '/content/drive/MyDrive'
HDF5 = os.environ.get('HDF5_PATH', '/content/rml2018/GOLD_XYZ_OSC.0001_1024.hdf5')
OUT_ROOT = os.environ.get('OUT_ROOT', f'{DRIVE}/results_rml2018_pet_full')
os.makedirs(OUT_ROOT, exist_ok=True)

import tensorflow as tf
import main as M
from preprocess import split_data_raw, load_rml2018_hdf5
from efficient_gpr_per_sample import apply_gpr_denoising_efficient_per_sample
from train import train_model_resume
from model.custom_objects import get_custom_objects_for_model

BATCH = int(os.environ.get('BATCH', '512'))
LR = float(os.environ.get('LR', '2e-3'))
PAT_LR = 2; PAT_ES = 10; FACTOR = 0.7
EPOCHS = int(os.environ.get('SEARCH_EPOCHS', '30'))
RESUME_EPOCHS = int(os.environ.get('RESUME_EPOCHS', '0'))    # >0 = 续训到该总 epoch 数
MODEL = os.environ.get('JUDGE_MODEL', 'pet')
PER_GROUP = int(os.environ.get('PER_GROUP', '4096'))         # 4096 = 每组全量

LOG = open(f'{OUT_ROOT}/full.log', 'a')
def log(m): LOG.write(m + '\n'); LOG.flush(); print(m, flush=True)


def run_variant(dataset, name, denoise, L0, slope, seed):
    vdir = f'{OUT_ROOT}/{name}'
    if os.path.exists(f'{vdir}/DONE.txt') and not RESUME_EPOCHS:
        log(f'skip {name}'); return
    models_dir = f'{vdir}/models'; results_dir = f'{vdir}/results'; plots_dir = f'{vdir}/plots'
    for d in (models_dir, results_dir, plots_dir): os.makedirs(d, exist_ok=True)
    t0 = time.time()
    M.set_random_seed(seed)
    dn = dataset
    if denoise:
        dn, dt = apply_gpr_denoising_efficient_per_sample(dataset, L0=L0, slope=slope,
                                                          sigma_f_mode='unit', noise_est='label')
        log(f'{name}: denoise L0={L0} slope={slope} ({dt:.0f}s)')
    Xtr, Xv, Xte, ytr, yv, yte, snrtr, snrv, snrte, mods = split_data_raw(dn)
    if denoise:
        del dn                                                # 划分副本已生成, 释放去噪副本压内存峰值
    ncls = len(mods)
    ytr_oh = tf.keras.utils.to_categorical(ytr, ncls)
    yv_oh = tf.keras.utils.to_categorical(yv, ncls)
    yte_oh = tf.keras.utils.to_categorical(yte, ncls)
    last_path = f'{models_dir}/{MODEL}_model_2018_last.keras'
    if RESUME_EPOCHS and os.path.exists(last_path):
        log_json = f'{models_dir}/logs/{MODEL}_model_2018_detailed_log.json'
        log_csv = f'{models_dir}/logs/{MODEL}_model_2018_detailed_log.csv'
        prev = json.load(open(log_json))
        initial_epoch = len(prev['epochs'])
        if initial_epoch >= RESUME_EPOCHS:
            log(f'skip {name}: already {initial_epoch} epochs >= RESUME_EPOCHS={RESUME_EPOCHS}'); return
        log(f'{name}: resume from epoch {initial_epoch} -> {RESUME_EPOCHS}')
        model = tf.keras.models.load_model(last_path,
                                           custom_objects=get_custom_objects_for_model(MODEL))
        train_model_resume(model, Xtr, ytr_oh, Xv, yv_oh,
                           f'{models_dir}/{MODEL}_model_2018.keras', last_path,
                           batch_size=BATCH, epochs=RESUME_EPOCHS, initial_epoch=initial_epoch,
                           log_json_path=log_json, log_csv_path=log_csv, existing_log_data=prev)
    else:
        n_epochs = RESUME_EPOCHS or EPOCHS
        M.train_selected_models([MODEL], Xtr, ytr_oh, Xv, yv_oh, Xtr.shape[1:], ncls,
                                models_dir, plots_dir, '_2018', BATCH, n_epochs, LR, PAT_LR, PAT_ES, FACTOR)
    M.evaluate_selected_models([MODEL], Xte, yte_oh, snrte, mods,
                               models_dir, results_dir, '_2018', results_suffix='_2018')
    json.dump({'name': name, 'denoise': denoise, 'L0': L0, 'slope': slope, 'seed': seed,
               'model': MODEL, 'per_group': PER_GROUP, 'epochs': RESUME_EPOCHS or EPOCHS,
               'batch': BATCH, 'lr': LR, 'elapsed_s': time.time()-t0},
              open(f'{vdir}/variant.json', 'w'), indent=2)
    open(f'{vdir}/DONE.txt', 'w').write(f'{time.time()-t0:.0f}s')
    log(f'DONE {name} ({time.time()-t0:.0f}s)')


def main():
    log(f'=== RML2018 PET full-data === epochs={EPOCHS} per_group={PER_GROUP} batch={BATCH} lr={LR}')
    M.set_random_seed(42); M.configure_gpu()
    t0 = time.time()
    dataset = load_rml2018_hdf5(HDF5, per_group=PER_GROUP)
    log(f'loaded {len(dataset)} keys {sum(len(v) for v in dataset.values())} frames ({time.time()-t0:.0f}s)')
    run_variant(dataset, 'full_baseline', False, 6.0, 0.25, 42)
    run_variant(dataset, 'full_L6', True, 6.0, 0.25, 42)
    run_variant(dataset, 'full_L6_slp125', True, 6.0, 0.125, 42)
    log('=== ALL DONE ===')


if __name__ == '__main__':
    main()
