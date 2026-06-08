#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RML2018.01A 上测 GPR 去噪是否有效(扩展到第二个经典数据集)。

注意: 2018 是 1024 样本/帧、24类、且比2016更真实(O'Shea 2018带衰落/时钟/硬件损伤)。
按已得根因(GPR只去AWGN), 预判2018去噪增益可能也小; 但2018是标准基准, 实测才有完整结论。

流程: 子采样加载(h5py, 省内存) -> 测自相关定L0(=0.66×spS规律) -> PET base vs 去噪(标准/spectral) -> 逐SNR对比。
复用 main.py 的 train/eval(原版结果格式)。模型/子采样/SNR 可用环境变量配。
HDF5 路径: HDF5_PATH (默认 /content/rml2018/GOLD_XYZ_OSC.0001_1024.hdf5)。
"""
import os, sys, json, time
import numpy as np

REPO = '/content/radioML-v4b'; SRC = f'{REPO}/src'
sys.path.insert(0, SRC); os.chdir(SRC)
DRIVE = '/content/drive/MyDrive'
HDF5 = os.environ.get('HDF5_PATH', '/content/rml2018/GOLD_XYZ_OSC.0001_1024.hdf5')
OUT_ROOT = os.environ.get('OUT_ROOT', f'{DRIVE}/results_rml2018')   # 可配->同数据换模型(如pet)不与cnn2d结果冲突
os.makedirs(OUT_ROOT, exist_ok=True)

import tensorflow as tf
import main as M
from preprocess import split_data_raw, load_rml2018_hdf5
from efficient_gpr_per_sample import apply_gpr_denoising_efficient_per_sample

BATCH = 128; LR = 1e-3; PAT_LR = 2; PAT_ES = 10; FACTOR = 0.7
EPOCHS = int(os.environ.get('SEARCH_EPOCHS', '40'))
MODEL = os.environ.get('JUDGE_MODEL', 'pet')
PER_GROUP = int(os.environ.get('PER_GROUP', '500'))          # 每(mod,snr)组取多少帧(全量4096)
SNR_LIST = os.environ.get('SNRS', '')                         # 逗号分隔; 空=全部
SNRS = [int(s) for s in SNR_LIST.split(',')] if SNR_LIST else None

LOG = open(f'{OUT_ROOT}/rml2018.log', 'a')
def log(m): LOG.write(m + '\n'); LOG.flush(); print(m, flush=True)


def measure_autocorr_1e(dataset, n_show=3):
    """测纯净(高SNR)信号自相关1/e长度 -> 建议 L0 (=0.66×spS 规律的实测版)。"""
    keys = [k for k in dataset if k[1] >= 18][:50] or list(dataset)[:50]
    x = np.concatenate([dataset[k] for k in keys], axis=0)   # (M,2,1024)
    xc = x[:, 0, :] + 1j * x[:, 1, :]
    # 平均归一化自相关
    M_, n = xc.shape
    ac = np.zeros(n)
    for lag in range(n):
        ac[lag] = np.abs(np.mean(np.conj(xc[:, :n - lag]) * xc[:, lag:]))
    ac = ac / ac[0]
    one_e = np.argmax(ac < np.exp(-1)) if np.any(ac < np.exp(-1)) else n
    log(f'autocorr 1/e length ~ {one_e} (ac[1..{n_show}]={np.round(ac[1:n_show+1],3)}) -> suggest L0~{one_e}')
    return one_e


def run_variant(dataset, name, denoise, L0, noise_est='label'):
    vdir = f'{OUT_ROOT}/{name}'
    if os.path.exists(f'{vdir}/DONE.txt'):
        log(f'skip {name}'); return
    models_dir = f'{vdir}/models'; results_dir = f'{vdir}/results'; plots_dir = f'{vdir}/plots'
    for d in (models_dir, results_dir, plots_dir): os.makedirs(d, exist_ok=True)
    t0 = time.time()
    dn = dataset
    if denoise:
        dn, dt = apply_gpr_denoising_efficient_per_sample(dataset, L0=L0, slope=0.25,
                                                          sigma_f_mode='unit', noise_est=noise_est)
        log(f'{name}: denoise L0={L0} noise_est={noise_est} ({dt:.0f}s)')
    Xtr, Xv, Xte, ytr, yv, yte, snrtr, snrv, snrte, mods = split_data_raw(dn)
    ncls = len(mods)
    ytr_oh = tf.keras.utils.to_categorical(ytr, ncls)
    yv_oh = tf.keras.utils.to_categorical(yv, ncls)
    yte_oh = tf.keras.utils.to_categorical(yte, ncls)
    M.train_selected_models([MODEL], Xtr, ytr_oh, Xv, yv_oh, Xtr.shape[1:], ncls,
                            models_dir, plots_dir, '_2018', BATCH, EPOCHS, LR, PAT_LR, PAT_ES, FACTOR)
    M.evaluate_selected_models([MODEL], Xte, yte_oh, snrte, mods,
                               models_dir, results_dir, '_2018', results_suffix='_2018')
    json.dump({'name': name, 'denoise': denoise, 'L0': L0, 'noise_est': noise_est,
               'model': MODEL, 'per_group': PER_GROUP, 'epochs': EPOCHS, 'elapsed_s': time.time()-t0},
              open(f'{vdir}/variant.json', 'w'), indent=2)
    open(f'{vdir}/DONE.txt', 'w').write(f'{time.time()-t0:.0f}s')
    log(f'DONE {name} ({time.time()-t0:.0f}s)')


def main():
    log(f'=== RML2018 validation === model={MODEL} epochs={EPOCHS} per_group={PER_GROUP} snrs={SNRS}')
    M.set_random_seed(42); M.configure_gpu()
    t0 = time.time()
    dataset = load_rml2018_hdf5(HDF5, snrs=SNRS, per_group=PER_GROUP)
    log(f'loaded {len(dataset)} (mod,snr) keys, {sum(len(v) for v in dataset.values())} frames, '
        f'shape {next(iter(dataset.values())).shape} ({time.time()-t0:.0f}s)')
    L0 = float(os.environ.get('L0', '0')) or float(measure_autocorr_1e(dataset))
    L0 = max(L0, 1.0)
    log(f'using L0={L0}')
    run_variant(dataset, 'baseline_none', False, L0)
    run_variant(dataset, f'denoise_L{L0:g}', True, L0, 'label')
    if os.environ.get('RUN_SPECTRAL', '1') == '1':                  # PET确认run可关掉省一次训练
        run_variant(dataset, f'denoise_L{L0:g}_spectral', True, L0, 'spectral')
    log('=== ALL DONE ===')


if __name__ == '__main__':
    main()
