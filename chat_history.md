# Chat History

## 2026-04-09 会话 1

**用户提示词**:添加 SNR 出现偏差下的鲁棒性实验(模拟 SNR 随机误差下的预测准确率);将 SNR 预测神经网络的损失函数改成距离感知损失(参考 guide/update1.md)。

**工作内容**:
- `src/snr_predict.py`:
  - 新增 `make_gaussian_soft_label_matrix(C, sigma)` 构建 (C,C) 高斯软标签矩阵。
  - 新增 `GaussianSoftLabelLoss`:log_softmax + 软标签交叉熵(update1.md 方法1)。相邻类别误判轻罚,远距误判重罚。
  - `train_snr_predictor` 增加 `soft_label_sigma=1.0` 参数,默认启用软标签 CE;传 0 回退到硬 CE。
- `src/main.py`:
  - 新增 `--mode snr_robust`,新增 CLI `--snr_noise_stds`(默认 `0 1 2 4 6 10` dB)。
  - 流程:raw split → 对 `snr_test` 加高斯噪声并吸附到最近合法 SNR 档位 → 用扰动后 SNR 对 `X_test` 重新去噪 → 调用 `evaluate_selected_models` 评估(per-SNR 分组仍用真实 SNR)。
  - 每个噪声 std 的结果保存至 `results/snr_robust{suffix}/noisestd_{σ}/`,并存 `perturbed_snr.npz`。
  - `--models` 必填校验扩展到 `snr_robust`。
  - `denoise_split` 从 `denoise` 模块导入。

## 2026-04-09 会话 2

**用户提示词**:运行 `--mode evaluate --use_predicted_snr` 时,模型加载路径错误。训练时始终用真实 SNR,不应存在 `iqformer_model_..._predsnr_snr_cnn_stratified.pt` 这样的权重名。

**工作内容**:解耦模型权重后缀与结果目录后缀。
- `evaluate_model_variants` / `evaluate_selected_models` 新增 `results_suffix` 参数(默认等于 `suffix`)。评估结果子目录改用 `results_suffix`,模型权重仍用 `suffix`。
- train 模式:`suffix` 不再拼接 `_predsnr_{snr_model}`(训练一律真实 SNR)。
- evaluate 模式:`suffix` 用于定位权重(无 predsnr);`results_suffix` 在 `--use_predicted_snr` 时附加 `_predsnr_{snr_model}`,避免覆盖真实 SNR 评估结果。
- `snr_robust` 模式已使用无 predsnr 的 `train_suffix`,无需改动。
