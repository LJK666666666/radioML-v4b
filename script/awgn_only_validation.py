#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AWGN-only 假设验证: 生成【纯净数字调制 + 仅AWGN(无衰落/时钟)】数据集, 测 GPR 去噪是否有效。

动机(RML22论文+实测): GPR时域平滑只能去AWGN。2016a低SNR是纯AWGN(且因SNR bug极端)->去噪大涨;
RML22叠加了fading+clock+CFO(GPR去不掉)->去噪~0。本脚本造一个"只有AWGN"的数据集(GNU-Radio-free,
纯numpy干净调制+RRC成形+加AWGN), 若GPR去噪在它上面有效 -> 证实"方法是AWGN专用, RML22失效因其真实artifact"。

数据格式与RML22一致: dict {(mod,snr):(N,2,128)}, 直接喂给 main.py 的 train/eval。
SNR采用正确约定 σz=10^(-SNR/20)(=RML22约定), 范围 -20..18。spS=8(与2016a/RML22同), 16符号/帧。
"""
import os, sys, json, time
import numpy as np

# ---------------- 纯净数字调制生成 (numpy, 无GNU Radio) ----------------
def rrc(beta, sps, span):
    N = span * sps
    t = (np.arange(N + 1) - N / 2.0) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4 * beta)) < 1e-8:
            h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                                          (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def constellation(mod):
    if mod == 'BPSK':
        c = np.array([1, -1], dtype=complex)
    elif mod == 'QPSK':
        c = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    elif mod == '8PSK':
        c = np.exp(1j * 2 * np.pi * np.arange(8) / 8)
    elif mod == 'PAM4':
        c = np.array([-3, -1, 1, 3], dtype=complex)
    elif mod == 'QAM16':
        lv = np.array([-3, -1, 1, 3])
        c = np.array([a + 1j * b for a in lv for b in lv], dtype=complex)
    elif mod == 'QAM64':
        lv = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        c = np.array([a + 1j * b for a in lv for b in lv], dtype=complex)
    else:
        raise ValueError(mod)
    return c / np.sqrt(np.mean(np.abs(c) ** 2))   # 单位平均功率


def gen_clean_frames(mod, n_frames, n=128, sps=8, beta=0.35, span=8, rng=None):
    rng = rng or np.random.default_rng(0)
    c = constellation(mod)
    h = rrc(beta, sps, span)
    n_sym = n // sps + span + 4                      # 多生成几个符号填满128
    out = np.empty((n_frames, n), dtype=complex)
    for i in range(n_frames):
        syms = c[rng.integers(0, len(c), n_sym)]
        up = np.zeros(n_sym * sps, dtype=complex)
        up[::sps] = syms
        sig = np.convolve(up, h, mode='full')
        start = len(h)                               # 跳过滤波器暂态
        frame = sig[start:start + n]
        frame = frame / np.sqrt(np.mean(np.abs(frame) ** 2) + 1e-12)  # 单位功率
        out[i] = frame
    return out


def gen_awgn_dataset(snrs, n_per=2000, seed=42, doubled_snr=False):
    """返回 dict {(mod,snr):(N,2,128)}。doubled_snr=True 模拟RML16的2×bug(σz=10^(-SNR/10))。"""
    mods = ['BPSK', 'QPSK', '8PSK', 'PAM4', 'QAM16', 'QAM64']
    rng = np.random.default_rng(seed)
    ds = {}
    for mod in mods:
        clean = gen_clean_frames(mod, n_per * len(snrs), rng=rng)  # 复用
        idx = 0
        for snr in snrs:
            sig = clean[idx:idx + n_per]; idx += n_per
            exp = snr / 10.0 if doubled_snr else snr / 20.0        # 正确=/20, bug=/10
            sigma = 10 ** (-exp)                                   # 噪声std(复) 使 E|z|²=σ², per-comp σ/√2
            noise = (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape)) * (sigma / np.sqrt(2))
            y = sig + noise
            arr = np.stack([y.real, y.imag], axis=1).astype(np.float32)  # (N,2,128)
            ds[(mod, int(snr))] = arr
    return ds


# ---------------- 验证主流程 (复用 main.py train/eval) ----------------
def main():
    REPO = '/content/radioML-v4b'; SRC = f'{REPO}/src'
    sys.path.insert(0, SRC); os.chdir(SRC)
    DRIVE = '/content/drive/MyDrive'
    OUT_ROOT = f'{DRIVE}/results_awgn_only'
    os.makedirs(OUT_ROOT, exist_ok=True)
    import tensorflow as tf
    import main as M
    from preprocess import split_data_raw
    from efficient_gpr_per_sample import apply_gpr_denoising_efficient_per_sample

    BATCH = 128; LR = 1e-3; PAT_LR = 2; PAT_ES = 12; FACTOR = 0.7
    EPOCHS = int(os.environ.get('SEARCH_EPOCHS', '50'))
    JUDGE = 'pet'
    SNRS = list(range(-20, 19, 2))
    LOG = open(f'{OUT_ROOT}/awgn.log', 'a')
    def log(m): LOG.write(m + '\n'); LOG.flush(); print(m, flush=True)

    M.set_random_seed(42); M.configure_gpu()
    log(f'=== AWGN-only validation === epochs={EPOCHS} snrs={SNRS[0]}..{SNRS[-1]}')
    dataset = gen_awgn_dataset(SNRS, n_per=2000, seed=42, doubled_snr=False)
    log(f'synth dataset: {len(dataset)} (mod,snr) keys, {sum(len(v) for v in dataset.values())} samples')

    for name, denoise in [('baseline_none', False), ('std_L5_s0.25', True)]:
        vdir = f'{OUT_ROOT}/{name}'
        if os.path.exists(f'{vdir}/DONE.txt'):
            log(f'skip {name}'); continue
        models_dir = f'{vdir}/models'; results_dir = f'{vdir}/results'; plots_dir = f'{vdir}/plots'
        for d in (models_dir, results_dir, plots_dir): os.makedirs(d, exist_ok=True)
        t0 = time.time()
        dn = dataset
        if denoise:
            dn, dt = apply_gpr_denoising_efficient_per_sample(dataset, L0=5.0, slope=0.25, sigma_f_mode='unit')
            log(f'{name}: denoise {dt:.0f}s')
        Xtr, Xv, Xte, ytr, yv, yte, snrtr, snrv, snrte, mods = split_data_raw(dn)
        ncls = len(mods)
        ytr_oh = tf.keras.utils.to_categorical(ytr, ncls)
        yv_oh = tf.keras.utils.to_categorical(yv, ncls)
        yte_oh = tf.keras.utils.to_categorical(yte, ncls)
        M.train_selected_models([JUDGE], Xtr, ytr_oh, Xv, yv_oh, Xtr.shape[1:], ncls,
                                models_dir, plots_dir, '_awgn', BATCH, EPOCHS, LR, PAT_LR, PAT_ES, FACTOR)
        M.evaluate_selected_models([JUDGE], Xte, yte_oh, snrte, mods,
                                   models_dir, results_dir, '_awgn', results_suffix='_awgn')
        open(f'{vdir}/DONE.txt', 'w').write(f'{time.time()-t0:.0f}s')
        log(f'DONE {name} ({time.time()-t0:.0f}s)')
    log('=== ALL DONE ===')


if __name__ == '__main__':
    main()
