#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""合成信道损坏生成器: 干净数字调制 → 平坦衰落+CFO+相位+AWGN → (corrupted, target) 对。

用于训练学习式均衡器(用户要求"据多径衰落升级方法"的主升级)。
设计(避开盲均衡绝对相位歧义):
  base_faded = g·clean·e^{jφ}   # 常数复衰落增益g + 常数相位φ, 干净, 无CFO无噪
  corrupted  = base_faded·e^{j2π·fcfo·n} + AWGN   # 加渐进CFO相位 + AWGN
  target     = base_faded        # 均衡器学:去CFO+去AWGN, 保留常数g/φ(由分类器/差分处理)
依据数据分析: RML22的ETU70在spS=8下≈平坦衰落(每帧常数复增益),主导新增损坏是相位/CFO非重ISI。

复用 awgn_only_validation.gen_clean_frames 造干净信号。
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from awgn_only_validation import gen_clean_frames


def apply_channel(clean, snr_db, rng, cfo_std=0.01, k_factor=4.0):
    """clean: (N,L) complex 单位功率 → (corrupted (N,2,L), target (N,2,L)) float32。

    cfo_std: CFO归一化频率标准差(clipped±3σ); k_factor: Rician K因子(dB能量比, 越大越接近LOS)。
    """
    N, L = clean.shape
    phi = rng.uniform(0, 2 * np.pi, (N, 1))                       # 常数相位偏移
    los = np.sqrt(k_factor / (k_factor + 1.0))
    nlos = np.sqrt(1.0 / (k_factor + 1.0))
    g = los + nlos * (rng.standard_normal((N, 1)) + 1j * rng.standard_normal((N, 1))) / np.sqrt(2)
    base_faded = g * clean * np.exp(1j * phi)                     # 常数衰落+相位, 干净
    fcfo = np.clip(rng.normal(0, cfo_std, (N, 1)), -3 * cfo_std, 3 * cfo_std)
    n = np.arange(L)[None, :]
    rx = base_faded * np.exp(1j * 2 * np.pi * fcfo * n)           # 加渐进CFO
    sig_pwr = np.mean(np.abs(base_faded) ** 2, axis=1, keepdims=True)
    sigma = np.sqrt(sig_pwr) * 10 ** (-snr_db / 20.0)            # AWGN相对每帧信号功率
    rx = rx + (rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape)) * (sigma / np.sqrt(2))

    def to2(x):
        return np.stack([x.real, x.imag], axis=1).astype(np.float32)
    return to2(rx), to2(base_faded)


def gen_pairs(snrs=range(-20, 19, 2), n_per=2000, seed=42, **ch):
    """造 (corrupted, target) 训练对。返回 X_corrupt (M,2,128), Y_target (M,2,128)。"""
    mods = ['BPSK', 'QPSK', '8PSK', 'PAM4', 'QAM16', 'QAM64']
    rng = np.random.default_rng(seed)
    Xc, Yt = [], []
    snr_list = list(snrs)
    for mod in mods:
        clean_c = gen_clean_frames(mod, n_per * len(snr_list), rng=rng)   # (M,128) complex
        idx = 0
        for snr in snr_list:
            c = clean_c[idx:idx + n_per]; idx += n_per
            corr, tgt = apply_channel(c, snr, rng, **ch)
            Xc.append(corr); Yt.append(tgt)
    return np.concatenate(Xc), np.concatenate(Yt)


if __name__ == '__main__':
    # 本地 sanity: 造一批, 检查 corrupted 是否类RML22(自相关/功率), target是否干净
    rng = np.random.default_rng(0)
    cc = gen_clean_frames('QPSK', 500, rng=rng)   # (500,128) complex
    for snr in (-10, 0, 10):
        corr, tgt = apply_channel(cc, snr, rng)
        cpwr = np.mean(corr[:, 0, :] ** 2 + corr[:, 1, :] ** 2)
        tpwr = np.mean(tgt[:, 0, :] ** 2 + tgt[:, 1, :] ** 2)
        mse = np.mean((corr - tgt) ** 2)
        print(f'snr={snr}: corrupt_pwr={cpwr:.3f} target_pwr={tpwr:.3f} MSE(corr,tgt)={mse:.4f}')
