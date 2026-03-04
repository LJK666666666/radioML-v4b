  修改内容                                                                                                                                                                  
                                                                                                                                                                            
  1. 新建 src/model/custom_objects.py

  将 main.py、main_2016b.py 和 model_benchmark.py 中重复的 ~140 行 custom objects 注册代码集中到一个模块，提供两个公共接口：
  - get_custom_objects_for_model(model_name) — 按模型名称返回所需的 custom objects
  - get_all_custom_objects() — 返回所有 custom objects（用于通用模型加载）

  2. 合并 main.py（吸收 main_2016b.py）

  - 新增 --dataset 参数，支持 2016a（默认）和 2016b
  - 通过 DATASET_CONFIGS 字典管理数据集特定的默认路径（数据路径、输出目录、缓存目录、文件后缀）
  - --dataset_path、--output_dir、--denoised_cache_dir 可手动覆盖默认值
  - 将 2016b 的 denoised cache 文件名后缀逻辑泛化为 prepare_data_with_cache_suffix() 函数
  - 删除了内联的 get_custom_objects_for_model 定义，改为从 model.custom_objects 导入

  3. 更新 model_benchmark.py

  - 删除了 ~70 行重复的 custom objects 导入和定义代码
  - 改为 from model.custom_objects import get_all_custom_objects as get_custom_objects_dict

  4. 删除 main_2016b.py

  5. 修复环境

  - 将 NumPy 从 2.2.6 降级到 1.24.3，使其与 TensorFlow 2.13.1 兼容

  使用方式

# 2016a 数据集（默认，与原来的 main.py 一致）
python main.py --models resnet --mode train

# 2016b 数据集（与原来的 main_2016b.py 一致）
python main.py --dataset 2016b --models resnet --mode train



  实施完成总结                                                                                                                                                                                                                                                                                                                                              
  新建文件                                                                                                                                                                                                                                                                                                                                                  
  1. src/model/snr_predictor_model.py — SNR预测CNN模型（PyTorch），20类分类（-20~18dB），注册表模式便于扩展                                                                     2. src/snr_predict.py — SNR预测训练、推理、保存/加载预测结果、评估（准确率/混淆矩阵/MAE）                                                                                     3. src/denoise.py — 独立去噪模块，支持按split去噪+增强+缓存                                                                                                                                                                                                                                                                                               
  修改文件

  4. src/preprocess.py — 提取 _collect_dataset_arrays、_perform_stratified_split、split_data_raw 三个共享函数，重构 prepare_data_by_snr_stratified 调用它们（向后兼容）       
  5. src/main.py — 主要改动：
    - --mode 新增 snr、denoise
    - --models 改为非必需（snr/denoise/explore 不需要）
    - 新增 --snr_model、--use_predicted_snr、--snr_predictions_path
    - explore 从 --mode all 中移除
    - 新增 get_cache_tag() 函数
  6. src/models.py — 导出 build_snr_predictor、get_available_snr_models
  7. src/config/training.yaml — 新增 snr_training 配置段

  典型使用流程

# 1. 训练SNR预测器
python main.py --mode snr --epochs 100

# 2. 用预测SNR去噪
python main.py --mode denoise --use_predicted_snr --augment_data

# 3. 训练分类模型（从缓存加载去噪数据）
python main.py --mode train --use_predicted_snr --models resnet --augment_data

# 4. 评估
python main.py --mode evaluate --use_predicted_snr --models resnet --augment_data