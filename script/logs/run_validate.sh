#!/bin/bash
set -e
PY=/d/020_Software/M/miniconda/Miniconda3/envs/ljk/python.exe
cd /d/010_CodePrograms/R/radioML/radioML-v4b

echo "########## [1/4] RML22 (spS8) BASELINE ##########"
$PY -u script/gpu_denoise_pipeline.py --mode validate --dataset_path data/RML22 \
  --exp_name gpu_val_rml22_base --epochs 60 --patience 10 --batch_size 512

echo "########## [2/4] RML22 (spS8) DENOISE L0=5 beta=0.05 unit ##########"
$PY -u script/gpu_denoise_pipeline.py --mode validate --dataset_path data/RML22 \
  --exp_name gpu_val_rml22_dn --epochs 60 --patience 10 --batch_size 512 \
  --denoise --kernel rbf --L0 5 --beta 0.05 --sigma_f_mode unit

echo "########## [3/4] RML22.01A (spS2) BASELINE ##########"
$PY -u script/gpu_denoise_pipeline.py --mode validate --dataset_path data/RML22.01A \
  --exp_name gpu_val_rml2201a_base --epochs 60 --patience 10 --batch_size 512

echo "########## [4/4] RML22.01A (spS2) DENOISE L0=1.5 beta=0.05 unit (autocorr-scaled) ##########"
$PY -u script/gpu_denoise_pipeline.py --mode validate --dataset_path data/RML22.01A \
  --exp_name gpu_val_rml2201a_dn15 --epochs 60 --patience 10 --batch_size 512 \
  --denoise --kernel rbf --L0 1.5 --beta 0.05 --sigma_f_mode unit

echo "ALL_VALIDATE_DONE"
