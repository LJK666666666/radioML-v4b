#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 Colab 上用【原版 Keras+torch pipeline】(src/main.py) 跑 RML22(spS=8) 多模型 base vs 去噪。

聚焦 spS=8(用户指示, spS=2 波形太尖锐无去噪头绪)。复用原版结果保存格式(便于迁回本地)。
- 每个 (model, condition) 调一次 main.py(子进程),结果按原版格式存到 Drive 的 output_dir。
- 可断点续跑:已有该模型的结果目录则跳过。
- 后台 subprocess 启动 → MCP 断开也不影响。

前置(在 notebook cell 里先做):
  pip install "tensorflow==2.15.*"   # Keras 2, 兼容 repo 的 TF2.13 风格代码; torch 已自带
  git clone/pull repo 到 /content/radioML-v4b
启动:
  nohup python -u /content/radioML-v4b/script/colab_run_original.py > <drive>/orig_stdout.log 2>&1 &
"""
import os, sys, time, subprocess

REPO = '/content/radioML-v4b'
DRIVE = '/content/drive/MyDrive'
DATA = f'{DRIVE}/RML22'
OUT_ROOT = f'{DRIVE}/results_rml22_original'
EPOCHS = int(os.environ.get('EPOCHS', '80'))
# 先放 cnn1d 作快速 sanity(轻量Keras),再 6 个论文模型
MODELS = ['cnn1d', 'ulcnn', 'mcldnn', 'pet', 'amcnet', 'fea_t', 'iqformer']

os.makedirs(OUT_ROOT, exist_ok=True)
LOG = open(f'{OUT_ROOT}/run.log', 'a')
def log(m):
    LOG.write(m + '\n'); LOG.flush(); print(m, flush=True)


def run_one(model, denoise):
    cond = 'denoise' if denoise else 'base'
    method = 'efficient_gpr_per_sample' if denoise else 'none'
    out_dir = f'{OUT_ROOT}/{cond}'
    os.makedirs(out_dir, exist_ok=True)
    # 原版输出: out_dir/{model}_model.keras 等; 用存在性做断点续跑标记
    done_marker = f'{out_dir}/{model}_DONE.txt'
    if os.path.exists(done_marker):
        log(f'skip {model} {cond} (done)'); return
    cmd = [sys.executable, 'main.py',
           '--dataset', 'rml22',
           '--dataset_path', DATA,
           '--output_dir', out_dir,
           '--models', model,
           '--mode', 'all',
           '--denoising_method', method,
           '--epochs', str(EPOCHS)]
    log(f'>>> START {model} {cond}: {" ".join(cmd)}')
    t0 = time.time()
    p = subprocess.run(cmd, cwd=f'{REPO}/src', capture_output=True, text=True)
    dt = time.time() - t0
    # 保存该次 stdout/stderr 末尾,便于排错
    with open(f'{out_dir}/{model}_log.txt', 'w') as f:
        f.write(p.stdout[-8000:] + '\n===STDERR===\n' + p.stderr[-8000:])
    if p.returncode == 0:
        open(done_marker, 'w').write(f'{dt:.0f}s')
        log(f'DONE  {model} {cond} ({dt:.0f}s, rc=0)')
    else:
        log(f'FAIL  {model} {cond} (rc={p.returncode}, {dt:.0f}s) -- 末尾stderr: {p.stderr[-400:]}')


def main():
    log(f'=== original-pipeline RML22 run start === models={MODELS} epochs={EPOCHS}')
    for model in MODELS:
        for denoise in (False, True):
            run_one(model, denoise)
    log('=== ALL DONE ===')


if __name__ == '__main__':
    main()
