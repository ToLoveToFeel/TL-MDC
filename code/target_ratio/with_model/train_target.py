# coding=utf-8
import os
import time
import keras
import numpy as np
import matplotlib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import load_model
import warnings

import utils
from utils import pro_path

matplotlib.use('agg')
warnings.filterwarnings("ignore")


def search(search_list):
    file_path = pro_path + "data/gen_data/target/"
    file_list = os.listdir(file_path)
    print(file_list)
    print(len(file_list))

    # (0) 'Eawag_XBridgeC18', (1) 'FEM_lipids', (2) 'FEM_long',
    # (3) 'IPB_Halle', (4) 'LIFE_new', (5) 'LIFE_old', (6) 'lipids', (7) 'MassBank1',
    # (8) 'MassBank2', (9) 'MetaboBASE', (10) 'Natural products', (11) 'pesticide',
    # (12) 'RIKEN_PlaSMA', (13) 'UniToyama_Atlantis'

    train_num_list = [25, 50, 75, 100]  # 训练样本数目

    for file in file_list:
        path = file_path + file + "/"
        print(path)
        # 读取数据
        Maccs = utils.read_csv(path + file + "_maccs.csv", header=None)
        Graph = utils.read_csv(path + file + "_graph_std.csv", header=None)
        Mordred = utils.read_csv(path + file + "_mordred_std.csv", header=None)

        X, y, com_name = utils.get_input(search_list, Maccs, Graph, Mordred)

        output_path = pro_path + "result/train_target/ratio/with_model/" + file + "/"
        model_path = pro_path + "result/train_base_model/" + com_name + "/"

        X_num = len(X)
        result_all = []
        for train_num in train_num_list:
            if train_num <= int(X_num * 0.75):
                # 分为训练集 和 测试集
                save_path = path + "split_index/"
                X_non_test_index = utils.read_csv(save_path + "_" + str(train_num) + "_X_non_test_index" + ".csv", header=None)[0]
                X_test_index = utils.read_csv(save_path + "_" + str(train_num) + "_X_test_index" + ".csv", header=None)[0]

                X_non_test, X_test = X[X_non_test_index], X[X_test_index]
                y_non_test, y_test = y[X_non_test_index], y[X_test_index]

                result = search_one(X_non_test, X_test, y_non_test, y_test, output_path + "train_num_" + str(train_num) + "/", model_path)
                result_all.append([train_num] + result)
                utils.write_csv(output_path, "result_all.csv", result_all)

                del X_non_test_index, X_test_index, X_non_test, X_test, y_non_test, y_test

        del Maccs, Graph, Mordred


# search_one 处理某一个小数据集
def search_one(X_non_test, X_test, y_non_test, y_test, path, model_path):

    batch_size_list = utils.train_target_params[0]  # 每次送入的数据量
    lr_list = utils.train_target_params[1]  # 学习率
    epochs_list = utils.train_target_params[2]  # 不同训练epochs
    outer_CV_times, inner_CV_times = utils.train_target_params[3]

    combine_list = []
    for batch_size in batch_size_list:
        for lr in lr_list:
            combine_list.append([batch_size, lr, epochs_list])

    del epochs_list, batch_size_list, lr_list

    start = time.perf_counter()  # 开始计时
    if not os.path.exists(path): os.makedirs(path)

    # result是一个列表
    result = cross_validate(X_non_test, X_test, y_non_test, y_test, model_path, path, combine_list, inner_CV_times)

    # 结果输出并写入文件
    print(["MAE", "MedAE", "MRE", "MedRE", "R_square", "RMSE"])
    print(result[3:])

    print("训练用时: " + utils.cost_time(start, time.perf_counter()))

    return result


def cross_validate(X_non_test, X_test, y_non_test, y_test, model_path, path, combine_list, inner_CV_times):

    # 使用 X_non_test、y_non_test 寻找最优参数，之后在 X_test、y_test 上测试结果
    result_all = []  # 存储 所有参数组合 combine_list 测试 inner_CV_times 的平均结果，[batch_size, lr, epochs, ...]
    for i in range(len(combine_list)):
        batch_size, lr, epochs_list = combine_list[i]

        # print("结果保存路径为: ", result_path)

        result = {}  # 是个map, (key: epochs, val: []), val是个二维列表，存储 inner_CV_times 次结果
        for j in range(inner_CV_times):
            X_train, X_val, y_train, y_val = train_test_split(X_non_test, y_non_test, test_size=0.1, random_state=j)
            print("当前搜索第" + str(i) + "大组参数，第" + str(j) + "次内部训练...")

            model = build_DNN(lr, model_path)
            history = utils.LossHistory()

            last_epochs = 0  # 类似于累积训练
            for k in range(len(epochs_list)):
                cur_epochs = epochs_list[k]
                print("当前搜索batch_size = " + str(batch_size) + ", lr = " + str(lr) + ", epochs = " + str(cur_epochs))
                model.fit(X_train, y_train, epochs=cur_epochs - last_epochs, batch_size=batch_size, verbose=0,
                          callbacks=[history])
                predict_val = np.array(model(X_val))[:, 0]
                last_epochs = cur_epochs

                index = utils.cal_index(y_val, predict_val)
                if cur_epochs not in result:
                    result[cur_epochs] = [index]
                else:
                    result[cur_epochs].append(index)

                # val_path = path + "model_" + str(batch_size) + "_" + str(lr) + "_" + str(
                #     cur_epochs) + "/validate" + str(j) + "/"
                # if not os.path.exists(val_path): os.makedirs(val_path)
                # history.loss_plot('epoch', val_path)  # 保存损失函数值，以及对应的曲线
                # Fun.write_csv(val_path + "res.csv", np.array(index, dtype=object).reshape(1, -1))  # 保存该次交叉验证的结果
                # Fun.write_csv(val_path + "true_predict.csv", [y_val, predict_val])  # 保存真实值，预测值

            # 删除中间变量
            del model, history, X_train, X_val, y_train, y_val
        for epochs in epochs_list:
            epochs_result = result[epochs]
            result_avg = []
            for u in range(len(epochs_result[0])):  # 遍历所有列
                s = 0
                for v in range(inner_CV_times):  # 遍历所有行
                    s += epochs_result[v][u]
                result_avg.append(round(s / inner_CV_times, 4))
            result_all.append([batch_size, lr, epochs] + result_avg)

    utils.write_csv(path, "result_all_params.csv", result_all)
    # 选择最好的一组参数在 X_test、y_test 上测试结果
    result_all.sort(key=lambda x: x[4])  # x[4]: median, 选择 median 最小的这组参数
    best_bs, best_lr, best_epochs = result_all[0][0], result_all[0][1], result_all[0][2]

    model = build_DNN(best_lr, model_path)
    history = utils.LossHistory()
    model.fit(X_non_test, y_non_test, epochs=best_epochs, batch_size=best_bs, verbose=0, callbacks=[history])
    predicts = np.array(model(X_test))[:, 0]

    index = utils.cal_index(y_test, predicts)
    utils.write_csv(path, "result_best.csv",
                  np.array([best_bs, best_lr, best_epochs] + index, dtype=object).reshape(1, -1))

    del model, history, predicts

    return [best_bs, best_lr, best_epochs] + index


# 负责创建模型
def build_DNN(num_lr, model_path):
    np.random.seed(1)
    tf.random.set_seed(1)

    model = load_model(model_path + "base_model.h5")

    adam = keras.optimizers.Adam(lr=num_lr)
    model.compile(optimizer=adam, loss='mae')
    return model


if __name__ == "__main__":
    search_list = ["maccs", "graph", "mordred"] # 需要根据最好的组合，改动这里
    search(search_list)
