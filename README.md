# 使用说明

* code/file_process 用于文件预处理，首先运行 file_SMRT_processing.py，之后才能运行 file_target_processing.py

* code/target_process 在SMRT上训练基模型，然后迁移到小数据集上：首先运行 train_base_model.py，之后才能运行 train_target.py，最后运行 zget_dynamic_result.py 得到最终结果

* code/machine_learning 运行得到机器学习算法的结果

* code/target_process_no_baseDNN 运行得到不使用迁移学习的结果

* code/target_ratio: 训练数据为25, 50, 75, 100的结果对比(有迁移学习 vs 无迁移学习), 需要先运行split_data.py划分数据，然后运行train_target.py, 最后运行draw.py得到图形对比结果