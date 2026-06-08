#!/usr/bin/env python3
"""
复现论文 paper/CL/Letter2/letter1_manuscript.tex 提出的核函数超参数选择方法
(Domain-Alignment Hyperparameter Optimization)。

方法流程：
  1. 用最高 SNR (18 dB) 的纯净数据训练一个参考分类器(默认 PETCGDNN)，
     该分类器捕获"目标"纯净特征空间。
  2. 对每个低 SNR 档位，用不同核函数(RBF / Matern / RationalQuadratic)与
     不同长度尺度 L 对样本进行 GPR 去噪(谱分解加速、单位方差核 σ_f^2=1，
     与部署的 efficient_gpr_per_sample 一致)。
  3. 用参考分类器衡量去噪样本的"可识别度"：
        metric = (1/N) Σ log p(y_i = c_i | x_i)
     即真实类别的平均对数似然(论文中的优化目标)。该指标直接度量下游分类
     效用，而非 MSE 等重建保真度。
  4. 对每个 SNR 选出使指标最大的核与长度尺度 L*。
  5. 聚合各负 SNR 档位的最优 L*，对 |SNR| 做线性拟合：
        L = L0 (1 + β |SNR|)   (SNR < 0)，   L = L0  (SNR >= 0)
     回归得到 L0(截距)与 β(=斜率/截距)，与论文 L0=5.0, β=0.05 对比。

结果保存在 results/kernel_hyperparam_selection_{num}/ 下。

用法示例：
  source activate radioml
  python script/kernel_hyperparam_selection.py                 # 完整运行
  python script/kernel_hyperparam_selection.py --quick         # 快速自测
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from preprocess import load_data, split_data_raw
from efficient_gpr_per_sample import spectral_gp_denoise_same_inputs


# ============================================================================
# 核函数矩阵构建 (规则网格 x = 0..n-1, 单位方差 σ_f^2 = 1)
# ============================================================================

def build_kernel_matrix(n, kernel, length_scale, matern_nu=1.5, rq_alpha=1.0):
    """构建 (n, n) 核矩阵，与 sklearn 同名核保持一致(单位方差)。"""
    idx = np.arange(n, dtype=np.float64)
    d = np.abs(idx[:, None] - idx[None, :])      # |i-j|
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
            raise ValueError(f"Unsupported matern nu={matern_nu}")
    elif k == 'rational_quadratic':
        K = (1.0 + d2 / (2.0 * rq_alpha * L ** 2)) ** (-rq_alpha)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    return K.astype(np.float32)


def denoise_batch_spectral(X, snr_values, power_values, K, sigma_f_mode='signal_var'):
    """对一批样本用单一核矩阵 K 做谱分解去噪。

    K 对全批样本相同(只依赖核与长度尺度)，每个样本的噪声方差 σ_n^2 不同。

    谱滤波器: S_i = (σ_f^2 λ_i) / (σ_f^2 λ_i + σ_n^2) = λ_i / (λ_i + σ_n^2/σ_f^2)
      - sigma_f_mode='unit'      : σ_f^2 = 1 (单位方差核，与部署的 efficient_gpr 一致)
                                   谱域有效噪声 = σ_n^2 (绝对尺度，含功率)
      - sigma_f_mode='signal_var': σ_f^2 = 纯净信号方差 P_signal/2 (正确的 MMSE 信号先验)
                                   谱域有效噪声 = σ_n^2/σ_f^2 = 1/SNR_lin (只依赖 SNR)
      - sigma_f_mode='obs_var'   : σ_f^2 = 每分量观测(含噪)信号方差 (低 SNR 被噪声主导，会欠平滑)

    Args:
        X: (M, 2, n) 原始 IQ
        snr_values: (M,) 每样本 SNR(dB)，用于算噪声方差
        power_values: (M,) 每样本平均功率 mean(I^2+Q^2)
        K: (n, n) 核矩阵
    Returns:
        (M, 2, n) 去噪结果
    """
    M, _, n = X.shape
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 0.0).astype(np.float32)   # 保证半正定

    snr_lin = 10.0 ** (snr_values.astype(np.float64) / 10.0)
    noise_vars = (power_values / (2.0 * (snr_lin + 1.0)))  # σ_n^2 per sample (绝对尺度)

    # 堆叠 I/Q 为列：偶列 I, 奇列 Q
    Y = np.empty((n, M * 2), dtype=np.float32)
    Y[:, 0::2] = X[:, 0, :].T
    Y[:, 1::2] = X[:, 1, :].T

    if sigma_f_mode == 'unit':
        eff_noise = noise_vars                       # σ_f^2 = 1
    elif sigma_f_mode == 'signal_var':
        eff_noise = 1.0 / snr_lin                    # σ_f^2 = P_signal/2 -> σ_n^2/σ_f^2 = 1/SNR_lin
    elif sigma_f_mode == 'obs_var':
        var_comp = 0.5 * power_values                # ≈ 每分量观测方差(零均值)
        eff_noise = noise_vars / np.maximum(var_comp, 1e-12)
    else:
        raise ValueError(f"Unknown sigma_f_mode: {sigma_f_mode}")

    nv_cols = np.empty(M * 2, dtype=np.float64)
    nv_cols[0::2] = eff_noise
    nv_cols[1::2] = eff_noise

    dn = spectral_gp_denoise_same_inputs(eigvecs, eigvals, Y, nv_cols.astype(np.float32))

    out = np.empty_like(X)
    out[:, 0, :] = dn[:, 0::2].T
    out[:, 1, :] = dn[:, 1::2].T
    return out


# ============================================================================
# 参考分类器 (在 18 dB 纯净数据上训练)
# ============================================================================

def build_reference_classifier(ref_model_name, input_shape, num_classes):
    if ref_model_name == 'pet':
        from model.pet_model import build_pet_model_main
        return build_pet_model_main(input_shape, num_classes)
    elif ref_model_name == 'ulcnn':
        from model.ulcnn_model import build_ulcnn_model
        return build_ulcnn_model(input_shape, num_classes)
    elif ref_model_name == 'cnn1d':
        from models import build_cnn1d_model
        return build_cnn1d_model(input_shape, num_classes)
    else:
        raise ValueError(f"Unsupported reference model: {ref_model_name}")


def train_reference_classifier(ref_model_name, X_tr, y_tr, X_va, y_va,
                               num_classes, model_path, plot_path,
                               epochs, batch_size, patience_es):
    """训练参考分类器并保存权重(best)。返回训练好的模型。"""
    from keras.utils import to_categorical
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from train import plot_training_history

    input_shape = X_tr.shape[1:]
    model = build_reference_classifier(ref_model_name, input_shape, num_classes)
    print(f"参考分类器 ({ref_model_name}) 参数量: {model.count_params():,}")

    y_tr_oh = to_categorical(y_tr, num_classes)
    y_va_oh = to_categorical(y_va, num_classes)

    callbacks = [
        EarlyStopping(monitor='val_accuracy', mode='max', patience=patience_es,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ]
    history = model.fit(
        X_tr, y_tr_oh, validation_data=(X_va, y_va_oh),
        batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=2,
    )
    # 直接保存(避免 ModelCheckpoint 在 .keras 原生格式下的 options 不兼容问题)
    model.save(model_path)
    print(f"参考分类器权重已保存: {model_path}")
    plot_training_history(history, plot_path)
    return model


# ============================================================================
# 评估指标
# ============================================================================

def evaluate_recognizability(model, X, y_true):
    """返回 (mean_logprob, accuracy)。X: (M,2,n)；y_true: (M,) int。"""
    probs = model.predict(X, batch_size=512, verbose=0)
    eps = 1e-12
    idx = np.arange(len(y_true))
    true_p = probs[idx, y_true]
    mean_logprob = float(np.mean(np.log(true_p + eps)))
    accuracy = float(np.mean(np.argmax(probs, axis=1) == y_true))
    return mean_logprob, accuracy


# ============================================================================
# 输出目录
# ============================================================================

def get_next_run_dir(results_root, base_name):
    os.makedirs(results_root, exist_ok=True)
    num = 1
    while os.path.exists(os.path.join(results_root, f"{base_name}_{num}")):
        num += 1
    run_dir = os.path.join(results_root, f"{base_name}_{num}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ============================================================================
# 绘图 (英文标注，无标题)
# ============================================================================

def plot_metric_vs_lengthscale(df, kernel, neg_snrs, length_scales, out_path):
    """选定核下：各负 SNR 的 mean log-likelihood 随长度尺度变化曲线，标出最优点。"""
    sub = df[df['kernel'] == kernel]
    plt.figure(figsize=(8, 6))
    colors = cm.viridis(np.linspace(0, 0.9, len(neg_snrs)))
    for c, snr in zip(colors, neg_snrs):
        s = sub[sub['snr'] == snr].sort_values('length_scale')
        if len(s) == 0:
            continue
        plt.plot(s['length_scale'], s['mean_logprob'], '-', color=c,
                 label=f"{int(snr)} dB", linewidth=1.6)
        best = s.loc[s['mean_logprob'].idxmax()]
        plt.plot(best['length_scale'], best['mean_logprob'], 'o',
                 color=c, markersize=7, markeredgecolor='k', markeredgewidth=0.6)
    plt.xlabel("Length scale $L$", fontsize=13)
    plt.ylabel("Mean log-likelihood of true class", fontsize=13)
    plt.legend(title="SNR", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_kernel_comparison(df, kernels, neg_snrs, out_path):
    """各核在每个负 SNR 上可达到的最优 mean log-likelihood 对比。"""
    plt.figure(figsize=(8, 6))
    markers = {'rbf': 'o-', 'matern': 's--', 'rational_quadratic': '^:'}
    names = {'rbf': 'RBF', 'matern': 'Matern (ν=1.5)',
             'rational_quadratic': 'Rational Quadratic'}
    for k in kernels:
        sub = df[df['kernel'] == k]
        best_per_snr = [sub[sub['snr'] == snr]['mean_logprob'].max() for snr in neg_snrs]
        plt.plot(neg_snrs, best_per_snr, markers.get(k, 'o-'),
                 label=names.get(k, k), linewidth=1.6, markersize=6)
    plt.xlabel("SNR (dB)", fontsize=13)
    plt.ylabel("Best mean log-likelihood", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_optimal_L_fit(abs_snr, L_star, L0, beta, slope, intercept, kernel, out_path):
    """关键图：最优 L* vs |SNR| 散点 + 回归直线 + 论文参考线。"""
    plt.figure(figsize=(8, 6))
    plt.plot(abs_snr, L_star, 'o', color='#2c7fb8', markersize=9,
             markeredgecolor='k', markeredgewidth=0.6, label="Optimal $L^*$ (data-driven)")
    xs = np.linspace(0, max(abs_snr) * 1.05, 100)
    plt.plot(xs, intercept + slope * xs, '-', color='#d95f0e', linewidth=2.0,
             label=f"Recovered fit: $L={L0:.2f}(1+{beta:.3f}|\\mathrm{{SNR}}|)$")
    plt.plot(xs, 5.0 * (1.0 + 0.05 * xs), '--', color='gray', linewidth=1.8,
             label="Paper: $L=5.0(1+0.05|\\mathrm{SNR}|)$")
    plt.xlabel("$|\\mathrm{SNR}|$ (dB)", fontsize=13)
    plt.ylabel("Optimal length scale $L^*$", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="复现核函数超参数选择 (domain-alignment)")
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'RML2016.10a_dict.pkl'))
    parser.add_argument('--results_root', type=str,
                        default=os.path.join(PROJECT_ROOT, 'results'))
    parser.add_argument('--exp_name', type=str, default='kernel_hyperparam_selection')
    parser.add_argument('--ref_model', type=str, default='pet',
                        choices=['pet', 'ulcnn', 'cnn1d'],
                        help='18dB 参考分类器架构 (默认 PETCGDNN)')
    parser.add_argument('--ref_snr', type=float, default=18.0,
                        help='训练参考分类器所用的纯净 SNR 档位')
    parser.add_argument('--epochs', type=int, default=100, help='参考分类器训练轮数')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience_es', type=int, default=15)
    parser.add_argument('--kernels', type=str, nargs='+',
                        default=['rbf', 'matern', 'rational_quadratic'])
    parser.add_argument('--matern_nu', type=float, default=1.5)
    parser.add_argument('--rq_alpha', type=float, default=1.0)
    parser.add_argument('--sigma_f_mode', type=str, default='signal_var',
                        choices=['signal_var', 'unit', 'obs_var'],
                        help="超参数选择阶段核函数 σ_f^2 设置: "
                             "signal_var(默认,σ_f^2=纯净信号方差->有效噪声1/SNR_lin,忠实复现) / "
                             "unit(σ_f^2=1,对齐部署) / obs_var(含噪观测方差)")
    parser.add_argument('--ls_min', type=float, default=1.0)
    parser.add_argument('--ls_max', type=float, default=20.0)
    parser.add_argument('--ls_step', type=float, default=0.5)
    parser.add_argument('--samples_per_snr', type=int, default=500,
                        help='每个 SNR 用于评估指标的样本数(从验证集随机抽取)，<=0 表示全部')
    parser.add_argument('--snrs', type=float, nargs='+', default=None,
                        help='参与扫描的 SNR 档位，默认数据集中全部')
    parser.add_argument('--ref_weights', type=str, default=None,
                        help='已训练参考分类器权重路径(.keras)，提供则跳过训练')
    parser.add_argument('--quick', action='store_true', help='快速自测(小规模)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.quick:
        args.epochs = 1
        args.kernels = ['rbf', 'matern']
        args.ls_min, args.ls_max, args.ls_step = 3.0, 11.0, 2.0
        args.samples_per_snr = 150
        args.snrs = [-12.0, -8.0, -4.0, -2.0, 0.0]

    # --- 输出目录 ---
    run_dir = get_next_run_dir(args.results_root, args.exp_name)
    print(f"结果输出目录: {run_dir}")
    with open(os.path.join(run_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(vars(args), f, allow_unicode=True)

    # --- 加载数据并划分 (确定性) ---
    print("加载数据集...")
    dataset = load_data(args.dataset_path)
    X_train, X_val, X_test, y_train, y_val, y_test, \
        snr_train, snr_val, snr_test, mods = split_data_raw(dataset)
    num_classes = len(mods)
    n = X_train.shape[2]
    print(f"调制类型数: {num_classes}, 序列长度: {n}")

    all_snrs = np.array(sorted(set(snr_train.tolist())))
    snrs_to_scan = np.array(args.snrs) if args.snrs is not None else all_snrs
    print(f"参与扫描的 SNR: {snrs_to_scan.tolist()}")

    length_scales = np.round(np.arange(args.ls_min, args.ls_max + 1e-9, args.ls_step), 4)
    print(f"长度尺度网格 ({len(length_scales)}): {length_scales.tolist()}")

    # --- 1. 训练/加载 18dB 参考分类器 ---
    ref_path = os.path.join(run_dir, f"reference_{args.ref_model}_{int(args.ref_snr)}dB.keras")
    if args.ref_weights and os.path.exists(args.ref_weights):
        import tensorflow as tf
        from model.custom_objects import get_custom_objects_for_model
        print(f"加载已训练参考分类器: {args.ref_weights}")
        ref_model = tf.keras.models.load_model(
            args.ref_weights,
            custom_objects=get_custom_objects_for_model(args.ref_model),
            safe_mode=False, compile=False)
    else:
        print(f"\n=== 在 {int(args.ref_snr)} dB 纯净数据上训练参考分类器 ({args.ref_model}) ===")
        tr_mask = snr_train == args.ref_snr
        va_mask = snr_val == args.ref_snr
        X_ref_tr, y_ref_tr = X_train[tr_mask], y_train[tr_mask]
        X_ref_va, y_ref_va = X_val[va_mask], y_val[va_mask]
        print(f"参考训练集: {X_ref_tr.shape}, 验证集: {X_ref_va.shape}")
        ref_model = train_reference_classifier(
            args.ref_model, X_ref_tr, y_ref_tr, X_ref_va, y_ref_va, num_classes,
            ref_path, os.path.join(run_dir, 'reference_training_history.png'),
            epochs=args.epochs, batch_size=args.batch_size, patience_es=args.patience_es,
        )

    # --- 2. 构建评估池 (各 SNR 从验证集抽样) ---
    print("\n构建评估池(验证集低SNR样本)...")
    X_eval_list, y_eval_list, snr_eval_list = [], [], []
    for snr in snrs_to_scan:
        mask = np.where(snr_val == snr)[0]
        if len(mask) == 0:
            continue
        if args.samples_per_snr > 0 and len(mask) > args.samples_per_snr:
            mask = rng.choice(mask, size=args.samples_per_snr, replace=False)
        X_eval_list.append(X_val[mask])
        y_eval_list.append(y_val[mask])
        snr_eval_list.append(snr_val[mask])
    X_eval = np.concatenate(X_eval_list, axis=0)
    y_eval = np.concatenate(y_eval_list, axis=0).astype(int)
    snr_eval = np.concatenate(snr_eval_list, axis=0)
    power_eval = np.mean(X_eval[:, 0, :] ** 2 + X_eval[:, 1, :] ** 2, axis=1)
    print(f"评估池: {X_eval.shape}, 每档样本数 ~{args.samples_per_snr}")

    # --- 3. 核 x 长度尺度 扫描 ---
    print(f"\n=== 开始扫描: {len(args.kernels)} 核 x {len(length_scales)} 长度尺度 ===")
    records = []
    total = len(args.kernels) * len(length_scales)
    pbar = tqdm(total=total, desc="kernel x L sweep")
    for kernel in args.kernels:
        for L in length_scales:
            K = build_kernel_matrix(n, kernel, L, args.matern_nu, args.rq_alpha)
            X_dn = denoise_batch_spectral(X_eval, snr_eval, power_eval, K,
                                          sigma_f_mode=args.sigma_f_mode)
            probs = ref_model.predict(X_dn, batch_size=512, verbose=0)
            eps = 1e-12
            idx_all = np.arange(len(y_eval))
            logp_all = np.log(probs[idx_all, y_eval] + eps)
            pred_all = np.argmax(probs, axis=1)
            for snr in snrs_to_scan:
                m = snr_eval == snr
                if not np.any(m):
                    continue
                records.append({
                    'kernel': kernel,
                    'length_scale': float(L),
                    'snr': float(snr),
                    'mean_logprob': float(np.mean(logp_all[m])),
                    'accuracy': float(np.mean(pred_all[m] == y_eval[m])),
                    'n_samples': int(np.sum(m)),
                })
            pbar.update(1)
    pbar.close()

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(run_dir, 'sweep_results.csv'), index=False)
    print(f"扫描结果已保存: sweep_results.csv ({len(df)} 行)")

    # --- 4. 每个 SNR 选最优 (核, L) ---
    opt_rows = []
    for snr in snrs_to_scan:
        s = df[df['snr'] == snr]
        best = s.loc[s['mean_logprob'].idxmax()]
        opt_rows.append({
            'snr': float(snr),
            'abs_snr': float(abs(snr)),
            'best_kernel': best['kernel'],
            'best_L': float(best['length_scale']),
            'best_logprob': float(best['mean_logprob']),
            'best_acc': float(best['accuracy']),
        })
    opt_df = pd.DataFrame(opt_rows).sort_values('snr')
    opt_df.to_csv(os.path.join(run_dir, 'optimal_per_snr.csv'), index=False)

    # 选定最优核：在负 SNR 上平均最优指标最高的核
    neg_snrs = sorted([s for s in snrs_to_scan if s < 0])
    kernel_scores = {}
    for k in args.kernels:
        sub = df[(df['kernel'] == k) & (df['snr'].isin(neg_snrs))]
        best_per_snr = [sub[sub['snr'] == snr]['mean_logprob'].max() for snr in neg_snrs]
        kernel_scores[k] = float(np.mean(best_per_snr))
    selected_kernel = max(kernel_scores, key=kernel_scores.get)
    win_counts = opt_df[opt_df['snr'] < 0]['best_kernel'].value_counts().to_dict()
    print(f"\n各核负SNR平均最优指标: {kernel_scores}")
    print(f"各核在负SNR档位胜出次数: {win_counts}")
    print(f"选定最优核: {selected_kernel}")

    # --- 5. 对选定核的负 SNR 最优 L* 做线性拟合 L* = a + b|SNR| ---
    # 用选定核每个负SNR的最优 L (而非全局最优核混合)，保证拟合一致性
    fit_rows = []
    for snr in neg_snrs:
        s = df[(df['kernel'] == selected_kernel) & (df['snr'] == snr)]
        best = s.loc[s['mean_logprob'].idxmax()]
        fit_rows.append((abs(snr), float(best['length_scale'])))
    abs_snr_arr = np.array([r[0] for r in fit_rows])
    L_star_arr = np.array([r[1] for r in fit_rows])

    slope, intercept = np.polyfit(abs_snr_arr, L_star_arr, 1)
    L0 = float(intercept)
    beta = float(slope / intercept) if intercept != 0 else float('nan')
    pred_L = intercept + slope * abs_snr_arr
    ss_res = float(np.sum((L_star_arr - pred_L) ** 2))
    ss_tot = float(np.sum((L_star_arr - np.mean(L_star_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    print(f"\n=== 线性拟合结果 (kernel={selected_kernel}) ===")
    print(f"  斜率 slope = {slope:.4f}, 截距 intercept(L0) = {L0:.4f}")
    print(f"  => L0 = {L0:.4f}, beta = {beta:.4f}, R^2 = {r2:.4f}")
    print(f"  论文参考: L0 = 5.0, beta = 0.05")

    # --- 6. 保存拟合与摘要 ---
    fit_result = {
        'selected_kernel': selected_kernel,
        'kernel_scores_neg_snr': kernel_scores,
        'kernel_win_counts_neg_snr': {k: int(v) for k, v in win_counts.items()},
        'linear_fit': {
            'slope': float(slope), 'intercept_L0': L0,
            'beta': beta, 'r_squared': float(r2),
        },
        'paper_reference': {'L0': 5.0, 'beta': 0.05, 'kernel': 'rbf'},
        'fit_points': [{'abs_snr': float(a), 'L_star': float(l)}
                       for a, l in zip(abs_snr_arr, L_star_arr)],
    }
    with open(os.path.join(run_dir, 'linear_fit.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(fit_result, f, allow_unicode=True, sort_keys=False)

    summary = [
        "Domain-Alignment 核函数超参数选择 - 复现摘要",
        "=" * 60,
        f"参考分类器: {args.ref_model} (在 {int(args.ref_snr)} dB 纯净数据训练)",
        f"评估指标: mean log p(true class) under reference classifier",
        f"扫描核: {args.kernels}",
        f"长度尺度网格: [{args.ls_min}, {args.ls_max}] step {args.ls_step}",
        "",
        f"选定最优核: {selected_kernel}",
        f"各核负SNR平均最优 mean-logprob: {kernel_scores}",
        f"各核负SNR胜出次数: {win_counts}",
        "",
        "线性拟合 L* = L0 (1 + beta|SNR|):",
        f"  L0 (intercept) = {L0:.4f}",
        f"  beta           = {beta:.4f}",
        f"  slope          = {slope:.4f}",
        f"  R^2            = {r2:.4f}",
        f"  论文参考       : L0=5.0, beta=0.05, kernel=rbf",
        "",
        "每个 SNR 的最优 (核, L*):",
    ]
    for _, row in opt_df.iterrows():
        summary.append(
            f"  SNR {int(row['snr']):>4d} dB: kernel={row['best_kernel']:<20s} "
            f"L*={row['best_L']:>5.1f}  logp={row['best_logprob']:.3f}  acc={row['best_acc']:.3f}"
        )
    summary_text = "\n".join(summary)
    with open(os.path.join(run_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text + "\n")
    print("\n" + summary_text)

    # --- 7. 绘图 ---
    print("\n生成图表...")
    plot_metric_vs_lengthscale(
        df, selected_kernel, neg_snrs, length_scales,
        os.path.join(run_dir, 'fig_metric_vs_lengthscale.png'))
    if len(args.kernels) > 1:
        plot_kernel_comparison(
            df, args.kernels, neg_snrs,
            os.path.join(run_dir, 'fig_kernel_comparison.png'))
    plot_optimal_L_fit(
        abs_snr_arr, L_star_arr, L0, beta, slope, intercept, selected_kernel,
        os.path.join(run_dir, 'fig_optimal_L_vs_snr.png'))

    print(f"\n完成。所有结果保存在: {run_dir}")


if __name__ == "__main__":
    main()
