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