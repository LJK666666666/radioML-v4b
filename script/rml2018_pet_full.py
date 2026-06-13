#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RML2018.01A 全量数据 PET 对比实验 (baseline vs GPR 去噪), 供 A100/高内存 runtime 手动跑。

变体: baseline + L0=6/slope=0.25 (探针 sweep 最优) + L0=6/slope=0.125 (sps 规则锚定 + SNR 轴修正)。
内存(系统 RAM, 非显存): 全量 X≈21GB float32。每变体内部独立加载, 单变量名重绑定使去噪后原始随即释放,
划分后副本随即释放 -> 峰值≈42GB (去噪过程瞬时 原始+去噪副本 共 42GB; split 瞬时 去噪副本+划分 共 42GB)。
普通 A100(~51GB RAM)即可跑; 代价是每变体重读一次本地 21GB(SSD 顺序读, 1-2min)。去噪是纯 CPU, 与 GPU 无关。
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


def run_variant(name, denoise, L0, slope, seed):
    vdir = f'{OUT_ROOT}/{name}'
    if os.path.exists(f'{vdir}/DONE.txt') and not RESUME_EPOCHS:
        log(f'skip {name}'); return
    models_dir = f'{vdir}/models'; results_dir = f'{vdir}/results'; plots_dir = f'{vdir}/plots'
    for d in (models_dir, results_dir, plots_dir): os.makedirs(d, exist_ok=True)
    last_path = f'{models_dir}/{MODEL}_model_2018_last.keras'
    log_json = f'{models_dir}/logs/{MODEL}_model_2018_detailed_log.json'
    log_csv = f'{models_dir}/logs/{MODEL}_model_2018_detailed_log.csv'
    # 续训: 先看已训 epoch 数, 训够了直接跳过 (不必白加载 21GB 数据)
    resume = bool(RESUME_EPOCHS and os.path.exists(last_path))
    if resume:
        prev = json.load(open(log_json))
        initial_epoch = len(prev['epochs'])
        if initial_epoch >= RESUME_EPOCHS:
            log(f'skip {name}: already {initial_epoch} epochs >= RESUME_EPOCHS={RESUME_EPOCHS}'); return

    t0 = time.time()
    M.set_random_seed(seed)
    # 每变体独立加载, 用完即释放: 单一变量名重绑定, 去噪后原始随即被回收, 划分后副本随即被回收
    dataset = load_rml2018_hdf5(HDF5, per_group=PER_GROUP)     # 本地 SSD 顺序读, ~1-2min
    log(f'{name}: loaded {sum(len(v) for v in dataset.values())} frames')
    if denoise:
        dataset, dt = apply_gpr_denoising_efficient_per_sample(dataset, L0=L0, slope=slope,
                                                               sigma_f_mode='unit', noise_est='label')
        log(f'{name}: denoise L0={L0} slope={slope} ({dt:.0f}s)')   # 重绑定后原始 21GB 已被释放
    Xtr, Xv, Xte, ytr, yv, yte, snrtr, snrv, snrte, mods = split_data_raw(dataset)
    del dataset                                               # 划分副本已生成, 释放 -> 训练期峰值仅 splits
    ncls = len(mods)
    ytr_oh = tf.keras.utils.to_categorical(ytr, ncls)
    yv_oh = tf.keras.utils.to_categorical(yv, ncls)
    yte_oh = tf.keras.utils.to_categorical(yte, ncls)
    if resume:
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
    run_variant('full_baseline', False, 6.0, 0.25, 42)        # 每变体内部自行加载 -> 释放, 峰值~42GB
    run_variant('full_L6', True, 6.0, 0.25, 42)
    run_variant('full_L6_slp125', True, 6.0, 0.125, 42)
    log('=== ALL DONE ===')


if __name__ == '__main__':
    main()
