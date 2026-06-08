#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU 版 GPR 去噪 + 超参数选择 / 全量验证统一管线（PyTorch + cu128, 适配 RTX 5070 Ti Blackwell）。

自包含：不依赖 TF/Keras（现有 radioml 环境的 TF 2.13 不支持 sm_120）。数据收集 + 确定性划分 +
GPR 谱分解去噪均在本文件内实现；分类器用 src/model/pet_torch_model.py 的 PyTorch PETCGDNN。

两个模式：
  --mode select   : domain-alignment 廉价选参。在最高 SNR 纯净数据训练参考分类器，
                    扫 (kernel, L) 用 mean log p(真类) 选 L*，拟合 L=L0(1+β|SNR|)。无需重训。
  --mode validate : 昂贵确认。按给定 L(snr) 律 + 核 + σ_f² 模式对全量数据去噪(或 baseline 不去噪)，
                    重训下游分类器，输出逐 SNR 测试准确率，对比 baseline。

数据特征自适应：GPR 长度尺度 L0 ≈ 信号自相关 1/e 长度 ≈ 0.66×samples_per_symbol。
  --autocorr_l0 时由数据自动测定 L0（免人工设定）。

结果保存在 results/{exp_name}_{num}/。图片英文、无标题。
"""

import os
import sys
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:                       # ljk 环境无 tqdm 时的轻量回退
    def tqdm(it, **kw):
        return it

import torch
import torch.nn as nn
from model.pet_torch_model import build_pet_torch


# ============================================================================
# 数据加载 / 确定性划分（自包含，复刻 preprocess.split_data_raw 的逻辑, random_state=42）
# ============================================================================

def load_data(file_path):
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def collect_arrays(dataset):
    mods = sorted(list(set(k[0] for k in dataset.keys())))
    snrs = sorted(list(set(k[1] for k in dataset.keys())))
    mod_to_idx = {m: i for i, m in enumerate(mods)}
    X, y, snr_v, comp = [], [], [], []
    for m in mods:
        for s in snrs:
            key = (m, s)
            if key in dataset:
                samp = dataset[key]
                X.append(samp)
                y.append(np.ones(len(samp)) * mod_to_idx[m])
                snr_v.append(np.ones(len(samp)) * s)
                comp.extend([f"{m}_{s}"] * len(samp))
    X = np.vstack(X).astype(np.float32)
    y = np.hstack(y).astype(int)
    snr_v = np.hstack(snr_v)
    comp = np.array(comp)
    return X, y, snr_v, comp, mods


def split_raw(dataset, test_size=0.2, val_split=0.1, seed=42):
    """确定性分层划分(numpy, 不依赖 sklearn)。按 (mod,snr) 复合组分层, 每组内固定种子打乱后
    按比例切 train/val/test(70/10/20)。对新数据集而言只需确定性+分层, 无需与 sklearn 逐索引一致。"""
    X, y, snr_v, comp, mods = collect_arrays(dataset)
    rng = np.random.default_rng(seed)
    tr_idx, va_idx, te_idx = [], [], []
    for c in np.unique(comp):
        idx = np.where(comp == c)[0]
        rng.shuffle(idx)
        ng = len(idx)
        n_te = int(round(ng * test_size))
        n_va = int(round(ng * val_split))
        te_idx.append(idx[:n_te])
        va_idx.append(idx[n_te:n_te + n_va])
        tr_idx.append(idx[n_te + n_va:])
    tr = np.concatenate(tr_idx); va = np.concatenate(va_idx); te = np.concatenate(te_idx)
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return (X[tr], X[va], X[te], y[tr], y[va], y[te],
            snr_v[tr], snr_v[va], snr_v[te], mods)


# ============================================================================
# 数据特征：纯净信号自相关 1/e 长度（普适 L0 预测量）
# ============================================================================

def autocorr_length(X_clean, max_lag=30):
    """X_clean: (N,2,L) 高SNR纯净样本。返回归一化自相关跨 1/e 的 lag（线性插值）。"""
    sig = X_clean.reshape(-1, X_clean.shape[-1]).astype(np.float64)   # (N*2, L)
    sig = sig - sig.mean(axis=1, keepdims=True)
    var = np.mean(sig ** 2, axis=1, keepdims=True) + 1e-12
    ac = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            ac[lag] = 1.0
        else:
            ac[lag] = np.mean(np.mean(sig[:, :-lag] * sig[:, lag:], axis=1, keepdims=True) / var)
    thr = np.exp(-1.0)
    for lag in range(1, len(ac)):
        if ac[lag] < thr:
            return (lag - 1) + (ac[lag - 1] - thr) / (ac[lag - 1] - ac[lag] + 1e-12), ac
    return float(len(ac) - 1), ac


# ============================================================================
# GPR 谱分解去噪（纯 numpy, 与 efficient_gpr_per_sample 一致）
# ============================================================================

def build_kernel_matrix(n, kernel, length_scale, matern_nu=1.5, rq_alpha=1.0):
    idx = np.arange(n, dtype=np.float64)
    d = np.abs(idx[:, None] - idx[None, :])
    d2 = d ** 2
    L = max(float(length_scale), 1e-12)
    k = kernel.lower()
    if k == 'rbf':
        K = np.exp(-0.5 * d2 / (L ** 2))
    elif k == 'matern':
        if abs(matern_nu - 1.5) < 1e-9:
            r = np.sqrt(3.0) * d / L
            K = (1.0 + r) * np.exp(-r)
        elif abs(matern_nu - 2.5) < 1e-9:
            r = np.sqrt(5.0) * d / L
            K = (1.0 + r + (r ** 2) / 3.0) * np.exp(-r)
        elif abs(matern_nu - 0.5) < 1e-9:
            K = np.exp(-d / L)
        else:
            raise ValueError(f"matern nu={matern_nu} unsupported")
    elif k == 'rational_quadratic':
        K = (1.0 + d2 / (2.0 * rq_alpha * L ** 2)) ** (-rq_alpha)
    else:
        raise ValueError(f"unknown kernel {kernel}")
    return K.astype(np.float32)


def denoise_batch(X, snr_values, power_values, K, sigma_f_mode='signal_var'):
    """谱域批去噪。X:(M,2,n)。sigma_f_mode: unit / signal_var / obs_var。"""
    M, _, n = X.shape
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 0.0).astype(np.float32)
    snr_lin = 10.0 ** (snr_values.astype(np.float64) / 10.0)
    noise_vars = power_values / (2.0 * (snr_lin + 1.0))
    Y = np.empty((n, M * 2), dtype=np.float32)
    Y[:, 0::2] = X[:, 0, :].T
    Y[:, 1::2] = X[:, 1, :].T
    if sigma_f_mode == 'unit':
        eff = noise_vars
    elif sigma_f_mode == 'signal_var':
        eff = 1.0 / snr_lin
    elif sigma_f_mode == 'obs_var':
        eff = noise_vars / np.maximum(0.5 * power_values, 1e-12)
    else:
        raise ValueError(sigma_f_mode)
    nv = np.empty(M * 2, dtype=np.float32)
    nv[0::2] = eff
    nv[1::2] = eff
    V = eigvecs.T @ Y
    S = eigvals[:, None] / (eigvals[:, None] + nv[None, :])
    dn = eigvecs @ (S * V)
    out = np.empty_like(X)
    out[:, 0, :] = dn[:, 0::2].T
    out[:, 1, :] = dn[:, 1::2].T
    return out


def length_scale_law(snr_db, L0, beta, cap=None):
    L = L0 if snr_db >= 0 else L0 * (1.0 + beta * (-snr_db))
    if cap is not None:
        L = min(L, cap)
    return L


def denoise_dataset_law(X, snr_values, kernel, L0, beta, sigma_f_mode,
                        cap=None, matern_nu=1.5, rq_alpha=1.0, desc="denoise"):
    """按 L(snr) 律对整批数据逐 SNR 组去噪。"""
    out = np.empty_like(X)
    power = np.mean(X[:, 0, :] ** 2 + X[:, 1, :] ** 2, axis=1)
    n = X.shape[2]
    uniq = np.unique(snr_values)
    for s in tqdm(uniq, desc=desc, leave=False):
        m = np.where(snr_values == s)[0]
        L = length_scale_law(float(s), L0, beta, cap)
        K = build_kernel_matrix(n, kernel, L, matern_nu, rq_alpha)
        out[m] = denoise_batch(X[m], snr_values[m], power[m], K, sigma_f_mode)
    return out


# ============================================================================
# PyTorch 训练 / 评估
# ============================================================================

def train_classifier(model, Xtr, ytr, Xva, yva, device, epochs=80, batch_size=512,
                     patience=12, lr=1e-3, log_prefix=""):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5,
                                                       patience=4, min_lr=1e-6)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).long()
    Xva_t = torch.from_numpy(Xva).float().to(device)
    yva_t = torch.from_numpy(yva).long().to(device)
    ntr = len(Xtr_t)

    best_acc, best_state, wait = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(ntr)
        tot_loss = 0.0
        for i in range(0, ntr, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr_t[idx].to(device, non_blocking=True)
            yb = ytr_t[idx].to(device, non_blocking=True)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
        # val
        model.eval()
        with torch.no_grad():
            vacc = batched_accuracy(model, Xva_t, yva_t, batch_size)
        sched.step(vacc)
        if vacc > best_acc:
            best_acc = vacc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        print(f"{log_prefix}epoch {ep:3d}/{epochs}  loss={tot_loss/ntr:.4f}  val_acc={vacc:.4f}  best={best_acc:.4f}  wait={wait}", flush=True)
        if wait >= patience:
            print(f"{log_prefix}early stop @ epoch {ep} (best val_acc={best_acc:.4f})", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


def batched_accuracy(model, X_t, y_t, batch_size=512):
    n = len(X_t)
    correct = 0
    for i in range(0, n, batch_size):
        out = model(X_t[i:i + batch_size])
        pred = out.argmax(1)
        correct += (pred == y_t[i:i + batch_size]).sum().item()
    return correct / n


@torch.no_grad()
def predict_probs(model, X, device, batch_size=512):
    model.eval()
    X_t = torch.from_numpy(X).float()
    outs = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i + batch_size].to(device)
        outs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0)


# ============================================================================
# 输出目录
# ============================================================================

def build_model(name, input_shape, num_classes):
    """统一模型注册分发(论文6架构的PyTorch版)。forward均接收(B,2,128)->logits。"""
    if name == 'pet':
        from model.pet_torch_model import build_pet_torch
        return build_pet_torch(input_shape, num_classes)
    if name == 'ulcnn':
        from model.ulcnn_torch_model import build_ulcnn_torch_model
        return build_ulcnn_torch_model(input_shape, num_classes)
    if name == 'fea_t':
        from model.fea_t_torch_model import build_fea_t_model
        return build_fea_t_model(input_shape, num_classes)
    if name == 'iqformer':
        from model.iqformer_torch_model import build_iqformer_model
        return build_iqformer_model(input_shape, num_classes)
    if name == 'mcldnn':
        from model.mcldnn_torch_model import build_mcldnn_torch_model
        return build_mcldnn_torch_model(input_shape, num_classes)
    if name == 'amcnet':
        from model.amcnet_torch_model import build_amcnet_torch_model
        return build_amcnet_torch_model(input_shape, num_classes)
    raise ValueError(f"unknown model: {name}")


PAPER_MODELS = ['ulcnn', 'mcldnn', 'pet', 'amcnet', 'fea_t', 'iqformer']


def next_run_dir(results_root, base):
    os.makedirs(results_root, exist_ok=True)
    num = 1
    while os.path.exists(os.path.join(results_root, f"{base}_{num}")):
        num += 1
    d = os.path.join(results_root, f"{base}_{num}")
    os.makedirs(d)
    return d


# ============================================================================
# 模式：select（domain-alignment 选参）
# ============================================================================

def run_select(args, device, run_dir, data):
    Xtr, Xva, Xte, ytr, yva, yte, str_, sva, ste, mods = data
    num_classes = len(mods)
    n = Xtr.shape[2]

    # autocorr 诊断
    hi = max(np.unique(str_))
    al, _ = autocorr_length(Xtr[str_ == hi])
    print(f"[autocorr] 最高SNR={hi}dB 纯净信号自相关1/e长度 = {al:.2f}  (建议 L0≈{al:.2f})")

    # 训练参考分类器（ref_snr 纯净数据）
    trm = str_ == args.ref_snr
    vam = sva == args.ref_snr
    print(f"参考分类器训练集 {Xtr[trm].shape}  验证集 {Xva[vam].shape}")
    model = build_pet_torch((2, n), num_classes)
    model, racc = train_classifier(model, Xtr[trm], ytr[trm], Xva[vam], yva[vam],
                                   device, epochs=args.epochs, batch_size=args.batch_size,
                                   patience=args.patience, lr=args.lr, log_prefix="[ref] ")
    print(f"参考分类器 {args.ref_snr}dB 验证准确率 = {racc:.4f}")
    torch.save(model.state_dict(), os.path.join(run_dir, 'reference_pet.pt'))

    # 评估池（验证集各扫描 SNR 抽样）
    rng = np.random.default_rng(args.seed)
    snrs_scan = np.array(args.snrs) if args.snrs else np.array(sorted(np.unique(sva)))
    Xe, ye, se = [], [], []
    for s in snrs_scan:
        idx = np.where(sva == s)[0]
        if len(idx) == 0:
            continue
        if args.samples_per_snr > 0 and len(idx) > args.samples_per_snr:
            idx = rng.choice(idx, args.samples_per_snr, replace=False)
        Xe.append(Xva[idx]); ye.append(yva[idx]); se.append(sva[idx])
    Xe = np.concatenate(Xe); ye = np.concatenate(ye).astype(int); se = np.concatenate(se)
    pe = np.mean(Xe[:, 0, :] ** 2 + Xe[:, 1, :] ** 2, axis=1)

    Ls = np.round(np.arange(args.ls_min, args.ls_max + 1e-9, args.ls_step), 4)
    print(f"扫描: kernels={args.kernels}  L={Ls.tolist()}  σ_f²={args.sigma_f_mode}")

    records = []
    for kern in args.kernels:
        for L in tqdm(Ls, desc=f"sweep[{kern}]"):
            K = build_kernel_matrix(n, kern, L, args.matern_nu, args.rq_alpha)
            Xdn = denoise_batch(Xe, se, pe, K, args.sigma_f_mode)
            probs = predict_probs(model, Xdn, device, args.batch_size)
            lp = np.log(probs[np.arange(len(ye)), ye] + 1e-12)
            pred = probs.argmax(1)
            for s in snrs_scan:
                m = se == s
                if not np.any(m):
                    continue
                records.append(dict(kernel=kern, length_scale=float(L), snr=float(s),
                                    mean_logprob=float(lp[m].mean()),
                                    accuracy=float((pred[m] == ye[m]).mean())))
    # 保存 sweep
    import csv
    with open(os.path.join(run_dir, 'sweep_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)

    # 选 L* / 拟合
    neg = sorted([s for s in snrs_scan if s < 0])
    # 选最优核：负SNR平均最优 logprob
    def best_lp(kern, s):
        vals = [r['mean_logprob'] for r in records if r['kernel'] == kern and r['snr'] == s]
        return max(vals) if vals else -1e9
    kscore = {k: float(np.mean([best_lp(k, s) for s in neg])) for k in args.kernels}
    sel_kernel = max(kscore, key=kscore.get)

    fit_pts = []
    for s in neg:
        cand = [r for r in records if r['kernel'] == sel_kernel and r['snr'] == s]
        best = max(cand, key=lambda r: r['mean_logprob'])
        fit_pts.append((abs(s), best['length_scale']))
    asnr = np.array([p[0] for p in fit_pts]); Lstar = np.array([p[1] for p in fit_pts])
    slope, intercept = np.polyfit(asnr, Lstar, 1)
    L0 = float(intercept); beta = float(slope / intercept) if intercept != 0 else float('nan')
    predL = intercept + slope * asnr
    r2 = 1.0 - np.sum((Lstar - predL) ** 2) / (np.sum((Lstar - Lstar.mean()) ** 2) + 1e-12)

    summary = {
        'dataset': os.path.basename(args.dataset_path),
        'autocorr_1e_length': float(al),
        'ref_val_acc': float(racc),
        'selected_kernel': sel_kernel,
        'kernel_scores_neg_snr': kscore,
        'linear_fit': {'L0': L0, 'beta': beta, 'slope': float(slope), 'r2': float(r2)},
        'fit_points': [{'abs_snr': float(a), 'L_star': float(l)} for a, l in zip(asnr, Lstar)],
        'optimal_per_snr': [
            {'snr': float(s),
             'best_L': max([r for r in records if r['kernel'] == sel_kernel and r['snr'] == s],
                           key=lambda r: r['mean_logprob'])['length_scale'],
             'best_acc': max([r for r in records if r['kernel'] == sel_kernel and r['snr'] == s],
                             key=lambda r: r['mean_logprob'])['accuracy']}
            for s in sorted(snrs_scan)],
    }
    with open(os.path.join(run_dir, 'select_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 图：L* vs |SNR|
    plt.figure(figsize=(8, 6))
    plt.plot(asnr, Lstar, 'o', ms=9, mec='k', label="Optimal $L^*$")
    xs = np.linspace(0, asnr.max() * 1.05, 100)
    plt.plot(xs, intercept + slope * xs, '-', lw=2,
             label=f"Fit: $L={L0:.2f}(1+{beta:.3f}|\\mathrm{{SNR}}|)$")
    plt.axhline(al, ls=':', color='green', label=f"autocorr $1/e$ length = {al:.2f}")
    plt.xlabel("$|\\mathrm{SNR}|$ (dB)", fontsize=13)
    plt.ylabel("Optimal length scale $L^*$", fontsize=13)
    plt.legend(fontsize=11); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'fig_optimal_L_vs_snr.png'), dpi=200)
    plt.close()
    print(f"\nselect 完成 -> {run_dir}")
    return summary


# ============================================================================
# 模式：validate（全量重训确认）
# ============================================================================

def run_validate(args, device, run_dir, data):
    Xtr, Xva, Xte, ytr, yva, yte, str_, sva, ste, mods = data
    num_classes = len(mods)
    n = Xtr.shape[2]

    L0, beta = args.L0, args.beta
    if args.autocorr_l0:
        hi = max(np.unique(str_))
        al, _ = autocorr_length(Xtr[str_ == hi])
        L0 = float(round(al, 2))
        print(f"[autocorr] 自动设定 L0 = {L0} (自相关1/e长度)")

    if args.denoise:
        print(f"去噪: kernel={args.kernel}  L0={L0}  beta={beta}  σ_f²={args.sigma_f_mode}  cap={args.cap}")
        Xtr = denoise_dataset_law(Xtr, str_, args.kernel, L0, beta, args.sigma_f_mode,
                                  args.cap, args.matern_nu, args.rq_alpha, "denoise-train")
        Xva = denoise_dataset_law(Xva, sva, args.kernel, L0, beta, args.sigma_f_mode,
                                  args.cap, args.matern_nu, args.rq_alpha, "denoise-val")
        Xte = denoise_dataset_law(Xte, ste, args.kernel, L0, beta, args.sigma_f_mode,
                                  args.cap, args.matern_nu, args.rq_alpha, "denoise-test")
    else:
        print("baseline: 不去噪")

    model = build_pet_torch((2, n), num_classes)
    model, vacc = train_classifier(model, Xtr, ytr, Xva, yva, device,
                                   epochs=args.epochs, batch_size=args.batch_size,
                                   patience=args.patience, lr=args.lr, log_prefix="[dwn] ")
    torch.save(model.state_dict(), os.path.join(run_dir, 'downstream_pet.pt'))

    # 逐 SNR 测试准确率
    probs = predict_probs(model, Xte, device, args.batch_size)
    pred = probs.argmax(1)
    overall = float((pred == yte).mean())
    per_snr = []
    for s in sorted(np.unique(ste)):
        m = ste == s
        per_snr.append({'snr': float(s), 'accuracy': float((pred[m] == yte[m]).mean()),
                        'n': int(m.sum())})

    summary = {
        'dataset': os.path.basename(args.dataset_path),
        'denoise': bool(args.denoise),
        'kernel': args.kernel if args.denoise else None,
        'L0': L0 if args.denoise else None,
        'beta': beta if args.denoise else None,
        'sigma_f_mode': args.sigma_f_mode if args.denoise else None,
        'cap': args.cap if args.denoise else None,
        'overall_test_acc': overall,
        'best_val_acc': float(vacc),
        'per_snr': per_snr,
    }
    with open(os.path.join(run_dir, 'validate_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    import csv
    with open(os.path.join(run_dir, 'per_snr_accuracy.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['snr', 'accuracy', 'n'])
        w.writeheader(); w.writerows(per_snr)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 图：逐 SNR 准确率
    plt.figure(figsize=(8, 6))
    ss = [p['snr'] for p in per_snr]; aa = [p['accuracy'] for p in per_snr]
    plt.plot(ss, aa, 'o-', lw=1.8, ms=6)
    plt.xlabel("SNR (dB)", fontsize=13); plt.ylabel("Test accuracy", fontsize=13)
    plt.grid(alpha=0.3); plt.ylim(0, 1); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'fig_per_snr_accuracy.png'), dpi=200)
    plt.close()
    print(f"\nvalidate 完成  overall={overall:.4f} -> {run_dir}")
    return summary


# ============================================================================
# main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['select', 'validate'], required=True)
    p.add_argument('--dataset_path', required=True)
    p.add_argument('--results_root', default=os.path.join(PROJECT_ROOT, 'results'))
    p.add_argument('--exp_name', default='gpu_denoise')
    p.add_argument('--seed', type=int, default=42)
    # 训练
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--lr', type=float, default=1e-3)
    # select
    p.add_argument('--ref_snr', type=float, default=18.0)
    p.add_argument('--kernels', nargs='+', default=['rbf'])
    p.add_argument('--ls_min', type=float, default=0.5)
    p.add_argument('--ls_max', type=float, default=12.0)
    p.add_argument('--ls_step', type=float, default=0.5)
    p.add_argument('--snrs', type=float, nargs='+', default=None)
    p.add_argument('--samples_per_snr', type=int, default=500)
    p.add_argument('--sigma_f_mode', default='signal_var',
                   choices=['signal_var', 'unit', 'obs_var'])
    # validate
    p.add_argument('--denoise', action='store_true', help='去噪(否则 baseline)')
    p.add_argument('--kernel', default='rbf')
    p.add_argument('--L0', type=float, default=5.0)
    p.add_argument('--beta', type=float, default=0.05)
    p.add_argument('--cap', type=float, default=None)
    p.add_argument('--autocorr_l0', action='store_true', help='用自相关长度自动定 L0')
    # 公共核参数
    p.add_argument('--matern_nu', type=float, default=1.5)
    p.add_argument('--rq_alpha', type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device = {device}  ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    if device != 'cuda':
        raise RuntimeError("GPU 不可用 —— 本管线要求 GPU 训练")

    run_dir = next_run_dir(args.results_root, args.exp_name)
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"结果目录: {run_dir}")

    print("加载/划分数据...")
    data = split_raw(load_data(args.dataset_path))
    print(f"调制数={len(data[-1])}  序列长n={data[0].shape[2]}  "
          f"train={len(data[0])} val={len(data[1])} test={len(data[2])}")

    if args.mode == 'select':
        run_select(args, device, run_dir, data)
    else:
        run_validate(args, device, run_dir, data)


if __name__ == "__main__":
    main()
