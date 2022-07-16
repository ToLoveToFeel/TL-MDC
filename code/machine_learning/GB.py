# coding=utf-8
import os
import time
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from numpy.random import seed
import tensorflow as tf
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

    for file in file_list:
        path = file_path + file + "/"
        print(path)
        # 读取数据
        Maccs = utils.read_csv(path + file + "_maccs.csv", header=None)
        Graph = utils.read_csv(path + file + "_graph_std.csv", header=None)
        Mordred = utils.read_csv(path + file + "_mordred_std.csv", header=None)

        X, y, com_name = utils.get_input(search_list, Maccs, Graph, Mordred)

        output_path = pro_path + "result/train_target/machine_learning/GB/"

        search_one(X, y, output_path + file + "/")

        del Maccs, Graph, Mordred


# search_one 处理某一个小数据集
def search_one(X, y, path):
    outer_CV_times, inner_CV_times = utils.train_target_params[3]

    begin_time = time.perf_counter()
    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)

    # 结果汇总表
    result_all = []
    cnt = 0
    for i in range(outer_CV_times):
        start = time.perf_counter()  # 开始计时

        cnt += 1
        print("-" * 60)
        print("当前训练进度：", cnt, "/", outer_CV_times)

        result_path = path + "run_" + str(i) + "/"
        if not os.path.exists(result_path): os.makedirs(result_path)
        print("结果保存路径为: ", result_path)

        # result是一个列表
        result = cross_validate(X, y, i, result_path)

        # 结果输出并写入文件
        # print(["best_alpha", "MAE", "MedAE", "MRE", "MedRE", "R_square", "RMSE"])
        # print(result)
        print(["MAE", "MedAE", "MRE", "MedRE", "R_square", "RMSE"])
        print(result[1:])

        result_all.append(result)
        utils.write_csv(path, "result_all.csv", result_all)

        del result

        print("本次训练用时: " + utils.cost_time(start, time.perf_counter()))

    # 计算 result_all 平均值
    result_avg = []
    for u in range(len(result_all[0])):  # 遍历所有列
        s = 0
        for v in range(outer_CV_times):  # 遍历所有行
            s += result_all[v][u]
        result_avg.append(round(s / outer_CV_times, 4))
    utils.write_csv(path, "/result_avg.csv", np.array(result_avg, dtype=object).reshape(1, -1))

    del result_all
    print("总用时: " + utils.cost_time(begin_time, time.perf_counter()))


def cross_validate(X, y, seed, result_path):
    X_non_test, X_test, y_non_test, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    model = build_model()
    model.fit(X_non_test, y_non_test)
    predicts = model.predict(X_test)

    index = utils.cal_index(y_test, predicts)
    utils.write_csv(result_path, "result.csv", np.array(index, dtype=object).reshape(1, -1))

    del model, predicts

    return index


def build_model():
    seed(1)
    tf.random.set_seed(1)
    model = ensemble.GradientBoostingRegressor()
    return model


if __name__ == "__main__":
    for search_list in utils.search_lists:
        print("===============", search_list, "===============")
        search(search_list)
