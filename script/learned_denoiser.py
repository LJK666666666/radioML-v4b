#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学习式残差去噪器:自监督训练 + domain-alignment 代理验证(免重训快速筛选)。

流程:
  1. 加载数据集(确定性划分,复用 gpu_denoise_pipeline)。
  2. 自监督训练去噪器:取高SNR(>=hi_snr)近纯净样本作 target,每个 batch 加随机强度合成 AWGN
     作输入,UNet 预测残差,denoised=x-N̂,loss=MSE(clean,denoised)+λ_tv·TV(denoised)。
  3. 代理验证:在高SNR纯净数据训一个参考分类器(PETCGDNN),对低SNR验证样本分别用
     {raw / GPR(RBF L0=5) / 学习式} 处理,比较参考分类器的 mean log p(真类) 与 accuracy。
     谁让低SNR样本最"像纯净"(分数最高)谁更优 —— 免去为每个去噪器重训分类器。

用法:
  conda run -n ljk python script/learned_denoiser.py --dataset_path data/RML22 --exp_name ld_rml22
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'script'))

import gpu_denoise_pipeline as G
from model.residual_denoiser import build_residual_denoiser


def add_awgn(x, snr_db):
    """对 (B,2,L) 按每样本功率加 AWGN 到目标 SNR(dB)。"""
    psig = (x[:, 0, :] ** 2 + x[:, 1, :] ** 2).mean(dim=1, keepdim=True)  # (B,1) 每样本功率
    snr = 10.0 ** (snr_db / 10.0)
    sigma = torch.sqrt(psig / (2.0 * snr)).unsqueeze(1)  # (B,1,1) 每分量噪声std
    return x + torch.randn_like(x) * sigma


def tv_loss(x):
    return (x[:, :, 1:] - x[:, :, :-1]).abs().mean()


def train_denoiser(Xclean, device, epochs, bs, lr, snr_lo, snr_hi, tv_w, log_prefix=""):
    """Xclean: (N,2,L) 近纯净高SNR样本。返回训练好的去噪器。"""
    model = build_residual_denoiser(ch=2, base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xc = torch.from_numpy(Xclean).float()
    n = len(Xc)
    rng = np.random.default_rng(0)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            clean = Xc[idx].to(device)
            snr_db = float(rng.uniform(snr_lo, snr_hi))     # 随机噪声强度(blind)
            noisy = add_awgn(clean, snr_db)
            opt.zero_grad()
            den = model(noisy)
            loss = nn.functional.mse_loss(den, clean) + tv_w * tv_loss(den)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"{log_prefix}denoiser ep{ep:2d}/{epochs} loss={tot/n:.5f}", flush=True)
    return model


@torch.no_grad()
def apply_denoiser(model, X, device, bs=2048):
    model.eval()
    Xt = torch.from_numpy(X).float()
    out = np.empty_like(X)
    for i in range(0, len(Xt), bs):
        out[i:i + bs] = model(Xt[i:i + bs].to(device)).cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', required=True)
    p.add_argument('--exp_name', default='learned_denoiser')
    p.add_argument('--results_root', default=os.path.join(PROJECT_ROOT, 'results'))
    p.add_argument('--hi_snr', type=float, default=16.0, help='>=此SNR的样本当近纯净target')
    p.add_argument('--ref_snr', type=float, default=18.0, help='参考分类器训练SNR')
    p.add_argument('--snr_lo', type=float, default=-12.0)
    p.add_argument('--snr_hi', type=float, default=12.0)
    p.add_argument('--dn_epochs', type=int, default=20)
    p.add_argument('--dn_bs', type=int, default=256)
    p.add_argument('--dn_lr', type=float, default=1e-3)
    p.add_argument('--tv_w', type=float, default=0.01)
    p.add_argument('--ref_epochs', type=int, default=40)
    p.add_argument('--eval_snrs', type=float, nargs='+', default=[-10, -8, -6, -4, -2, 0])
    p.add_argument('--samples_per_snr', type=int, default=1000)
    p.add_argument('--gpr_L0', type=float, default=5.0)
    p.add_argument('--gpr_beta', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise RuntimeError("需要 GPU")
    run_dir = G.next_run_dir(args.results_root, args.exp_name)
    print(f"device={device} run_dir={run_dir}")

    data = G.split_raw(G.load_data(args.dataset_path))
    Xtr, Xva, Xte, ytr, yva, yte, str_, sva, ste, mods = data
    n = Xtr.shape[2]; nc = len(mods)
    print(f"mods={nc} n={n} train={len(Xtr)}")

    # --- 1. 训练去噪器(高SNR近纯净样本) ---
    clean_mask = str_ >= args.hi_snr
    Xclean = Xtr[clean_mask]
    print(f"近纯净训练样本(SNR>={args.hi_snr}): {len(Xclean)}")
    t0 = time.time()
    denoiser = train_denoiser(Xclean, device, args.dn_epochs, args.dn_bs, args.dn_lr,
                              args.snr_lo, args.snr_hi, args.tv_w)
    print(f"去噪器训练完成 {time.time()-t0:.0f}s")
    torch.save(denoiser.state_dict(), os.path.join(run_dir, 'denoiser.pt'))

    # --- 2. 参考分类器(纯净 ref_snr 数据) ---
    trm = str_ == args.ref_snr; vam = sva == args.ref_snr
    refmodel = G.build_pet_torch((2, n), nc)
    refmodel, racc = G.train_classifier(refmodel, Xtr[trm], ytr[trm], Xva[vam], yva[vam],
                                        device, epochs=args.ref_epochs, batch_size=256,
                                        patience=10, log_prefix="[ref] ")
    print(f"参考分类器 {args.ref_snr}dB val_acc={racc:.4f}")

    # --- 3. 构建低SNR评估池 ---
    rng = np.random.default_rng(args.seed)
    Xe, ye, se = [], [], []
    for s in args.eval_snrs:
        idx = np.where(sva == s)[0]
        if len(idx) == 0:
            continue
        if len(idx) > args.samples_per_snr:
            idx = rng.choice(idx, args.samples_per_snr, replace=False)
        Xe.append(Xva[idx]); ye.append(yva[idx]); se.append(sva[idx])
    Xe = np.concatenate(Xe); ye = np.concatenate(ye).astype(int); se = np.concatenate(se)
    pe = np.mean(Xe[:, 0, :] ** 2 + Xe[:, 1, :] ** 2, axis=1)

    # --- 4. 三种处理 ---
    X_raw = Xe
    X_gpr = G.denoise_dataset_law(Xe, se, 'rbf', args.gpr_L0, args.gpr_beta, 'unit')
    X_learned = apply_denoiser(denoiser, Xe, device)

    def score(Xp, tag):
        probs = G.predict_probs(refmodel, Xp, device, 512)
        lp = np.log(probs[np.arange(len(ye)), ye] + 1e-12)
        pred = probs.argmax(1)
        rows = {}
        for s in args.eval_snrs:
            m = se == s
            if not np.any(m):
                continue
            rows[int(s)] = dict(logp=float(lp[m].mean()), acc=float((pred[m] == ye[m]).mean()))
        overall = dict(logp=float(lp.mean()), acc=float((pred == ye).mean()))
        per_snr_str = ', '.join('%d:%.3f' % (k, v['acc']) for k, v in rows.items())
        print("  [%-8s] overall logp=%.3f acc=%.3f  per-snr acc=[%s]"
              % (tag, overall['logp'], overall['acc'], per_snr_str))
        return dict(per_snr=rows, overall=overall)

    print("\n=== domain-alignment 代理对比(参考分类器对低SNR处理样本的可识别度) ===")
    res = {'raw': score(X_raw, 'raw'), 'gpr': score(X_gpr, 'gpr'), 'learned': score(X_learned, 'learned')}
    res['meta'] = dict(dataset=os.path.basename(args.dataset_path), ref_val_acc=float(racc),
                       hi_snr=args.hi_snr, dn_epochs=args.dn_epochs, tv_w=args.tv_w)
    json.dump(res, open(os.path.join(run_dir, 'proxy_compare.json'), 'w'), indent=2)

    # 结论
    lo = lambda r: np.mean([r['per_snr'][s]['acc'] for s in r['per_snr']])
    print(f"\n低SNR平均代理acc:  raw={lo(res['raw']):.4f}  gpr={lo(res['gpr']):.4f}  learned={lo(res['learned']):.4f}")
    print(f"结果保存 -> {run_dir}")


if __name__ == "__main__":
    main()
