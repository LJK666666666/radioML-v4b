#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多模型去噪验证驱动(Colab GPU 后台跑)。论文 6 架构 × {baseline, 去噪} × {RML22(spS8), RML22.01A(spS2)}。
每个结果增量存 Drive(可断点续跑: 已存在则跳过), 进度写 multimodel_progress.log。

去噪配置(按数据特征自适应):
  RML22 (spS8): RBF L0=5, β=0.05, σ_f²=unit (论文部署律, 已验证有效)
  RML22.01A (spS2): 经验自相关 Wiener 核 (signal_var), RBF对spS=2会过平滑反而有害, Wiener自动学到几乎不平滑

默认 Colab 路径; 训练配置 epochs=40 patience=8 bs=1024 (公平且不过长)。
"""
import sys, os, time, json, contextlib, io
import numpy as np, torch

REPO = '/content/radioML-v4b'
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'script'))
import gpu_denoise_pipeline as G

device = 'cuda'
RES = '/content/drive/MyDrive/radioml_results'
os.makedirs(RES, exist_ok=True)
DATA = {'RML22': '/content/drive/MyDrive/RML22', 'RML22.01A': '/content/drive/MyDrive/RML22.01A'}
MODELS = ['ulcnn', 'mcldnn', 'pet', 'amcnet', 'fea_t', 'iqformer']
EPOCHS, BS, PAT = 40, 1024, 8

_logf = open(os.path.join(RES, 'multimodel_progress.log'), 'a')
def log(m):
    _logf.write(m + '\n'); _logf.flush(); print(m, flush=True)


def empirical_wiener_kernel(Xtr, str_, max_use=20000):
    hi = max(np.unique(str_)); Xc = Xtr[str_ == hi]
    sig = Xc.reshape(-1, Xc.shape[-1]).astype(np.float64)
    if len(sig) > max_use:
        sig = sig[np.random.default_rng(0).choice(len(sig), max_use, replace=False)]
    sig = sig - sig.mean(1, keepdims=True)
    var = np.mean(sig ** 2, 1, keepdims=True) + 1e-12
    n = sig.shape[1]; rho = np.ones(n)
    for lag in range(1, n):
        rho[lag] = np.mean(np.mean(sig[:, :-lag] * sig[:, lag:], 1, keepdims=True) / var)
    idx = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    K = rho[idx]
    w, V = np.linalg.eigh(K); w = np.maximum(w, 0.0)
    return ((V * w) @ V.T).astype(np.float32)


def main():
    log(f'=== multimodel driver start === models={MODELS} epochs={EPOCHS}')
    pw = lambda X: np.mean(X[:, 0, :] ** 2 + X[:, 1, :] ** 2, 1)
    for dsname, path in DATA.items():
        t0 = time.time()
        Xtr, Xva, Xte, ytr, yva, yte, str_, sva, ste, mods = G.split_raw(G.load_data(path))
        n, nc = Xtr.shape[2], len(mods)
        log(f'[{dsname}] loaded train={len(Xtr)} test={len(Xte)} ({time.time()-t0:.0f}s); denoising...')
        if dsname == 'RML22':
            dn_tag = 'dn_rbfL5'
            Dtr = G.denoise_dataset_law(Xtr, str_, 'rbf', 5.0, 0.05, 'unit')
            Dva = G.denoise_dataset_law(Xva, sva, 'rbf', 5.0, 0.05, 'unit')
            Dte = G.denoise_dataset_law(Xte, ste, 'rbf', 5.0, 0.05, 'unit')
        else:
            dn_tag = 'dn_wiener'
            K = empirical_wiener_kernel(Xtr, str_)
            Dtr = G.denoise_batch(Xtr, str_, pw(Xtr), K, 'signal_var')
            Dva = G.denoise_batch(Xva, sva, pw(Xva), K, 'signal_var')
            Dte = G.denoise_batch(Xte, ste, pw(Xte), K, 'signal_var')
        log(f'[{dsname}] denoise done; training {len(MODELS)} models x 2 conditions')
        for mname in MODELS:
            for cond, (A, B, C) in [('base', (Xtr, Xva, Xte)), (dn_tag, (Dtr, Dva, Dte))]:
                fn = os.path.join(RES, f'mm_{dsname}_{mname}_{cond}.json')
                if os.path.exists(fn):
                    log(f'skip {dsname} {mname} {cond} (exists)'); continue
                t1 = time.time(); torch.manual_seed(42); np.random.seed(42)
                try:
                    model = G.build_model(mname, (2, n), nc)
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        model, vacc = G.train_classifier(model, A, ytr, B, yva, device,
                                                         epochs=EPOCHS, batch_size=BS, patience=PAT)
                    probs = G.predict_probs(model, C, device, BS); pred = probs.argmax(1)
                    overall = float((pred == yte).mean())
                    per = {int(s): round(float((pred[ste == s] == yte[ste == s]).mean()), 4)
                           for s in sorted(np.unique(ste))}
                    lo = float(np.mean([per[s] for s in per if s <= 0]))
                    json.dump(dict(dataset=dsname, model=mname, cond=cond, overall=overall,
                                   low_snr_le0=lo, val=float(vacc), per_snr=per,
                                   params=int(sum(p.numel() for p in model.parameters())),
                                   sec=round(time.time() - t1, 1)), open(fn, 'w'), indent=2)
                    log(f'DONE {dsname} {mname} {cond}: overall={overall:.4f} low<=0={lo:.4f} ({time.time()-t1:.0f}s)')
                except Exception as e:
                    log(f'FAIL {dsname} {mname} {cond}: {repr(e)[:240]}')
    log('ALL_MULTIMODEL_DONE')


if __name__ == '__main__':
    main()
