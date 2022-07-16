# coding=utf-8
import os
import time
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
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

        output_path = pro_path + "result/train_target/machine_learning/lasso/"

        search_one(X, y, output_path + file + "/")

        del Maccs, Graph, Mordred


# search_one 处理某一个小数据集
def search_one(X, y, path):

    alpha_list = utils.alpha_list
    outer_CV_times, inner_CV_times = utils.train_target_params[3]

    combine_list = []
    for alpha in alpha_list:
        combine_list.append(alpha)

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
        result = cross_validate(X, y, combine_list, inner_CV_times, i, result_path)

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
    for u in range(1, len(result_all[0])):  # 遍历所有列
        s = 0
        for v in range(outer_CV_times):  # 遍历所有行
            s += result_all[v][u]
        result_avg.append(round(s / outer_CV_times, 4))
    utils.write_csv(path, "/result_avg.csv", np.array(result_avg, dtype=object).reshape(1, -1))

    del result_all, combine_list
    print("总用时: " + utils.cost_time(begin_time, time.perf_counter()))


def cross_validate(X, y, combine_list, inner_CV_times, seed, result_path):
    X_non_test, X_test, y_non_test, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    # 使用 X_non_test、y_non_test 寻找最优参数，之后在 X_test、y_test 上测试结果
    result_all = []  # 存储 所有参数组合 combine_list 测试 inner_CV_times 的平均结果，[batch_size, lr, epochs, ...]
    for i in range(len(combine_list)):
        alpha = combine_list[i]

        print("当前搜索第" + str(i) + "组参数...")
        result = []
        for j in range(inner_CV_times):
            X_train, X_val, y_train, y_val = train_test_split(X_non_test, y_non_test, test_size=0.1, random_state=j)

            model = Lasso(alpha=alpha)
            model.fit(X_train, y_train)
            predict_val = model.predict(X_val)

            index = utils.cal_index(y_val, predict_val)
            result.append(index)

        result_avg = []
        for u in range(len(result[0])):  # 遍历所有列
            s = 0
            for v in range(inner_CV_times):  # 遍历所有行
                s += result[v][u]
            result_avg.append(round(s / inner_CV_times, 4))
        result_all.append([alpha] + result_avg)

    utils.write_csv(result_path, "result_all_params.csv", result_all)
    # 选择最好的一组参数在 X_test、y_test 上测试结果
    result_all.sort(key=lambda x: x[1])  # x[1]: mae, 选择 mae 最小的这组参数
    best_alpha = result_all[0][0]

    model = Lasso(alpha=best_alpha)
    model.fit(X_non_test, y_non_test)
    predicts = model.predict(X_test)

    index = utils.cal_index(y_test, predicts)
    utils.write_csv(result_path, "result_best.csv", np.array([best_alpha] + index, dtype=object).reshape(1, -1))

    del model, predicts

    return [best_alpha] + index


# 根据各路结果，得到最终结果
def get_final_result():
    file_path = pro_path + "data/gen_data/target/"
    file_list = os.listdir(file_path)
    print(file_list)
    print(len(file_list))

    # (0) 'Eawag_XBridgeC18', (1) 'FEM_lipids', (2) 'FEM_long',
    # (3) 'IPB_Halle', (4) 'LIFE_new', (5) 'LIFE_old', (6) 'lipids', (7) 'MassBank1',
    # (8) 'MassBank2', (9) 'MetaboBASE', (10) 'Natural products', (11) 'pesticide',
    # (12) 'RIKEN_PlaSMA', (13) 'UniToyama_Atlantis'
    print(file_list)
    print(len(file_list))

    result_final = []
    com_name_list = []
    for search_list in utils.search_lists:
        com_name_list.append(utils.get_combine_name(search_list))

    for file in file_list:
        result_cro = []  # 验证集结果
        result_test = []  # 测试集结果
        for com_name in com_name_list:
            result_path = pro_path + "result/train_target/machine_learning/lasso/"
            res = search_best(result_path + file + "/")
            result_cro.append(res)
            result_test.append(utils.read_csv(result_path + file + "/result_avg.csv", header=None)[0])

        # 找到 result_cro 中 MAE 最小的对应的下标，使用其在验证集的结果
        index = 0
        for i in range(len(com_name_list)):
            if result_cro[i][1] < result_cro[index][1]:
                index = i

        result_final.append([file, com_name_list[index]] + result_test[index].tolist())

    utils.write_csv(pro_path + "result/train_target/machine_learning/lasso/", "result_final.csv", result_final)


def search_best(path):
    alpha_list = utils.alpha_list
    outer_CV_times, inner_CV_times = utils.train_target_params[3]

    combine_list = []
    for alpha in alpha_list:
        combine_list.append(alpha)

    # 结果汇总表
    result_all = []
    cnt = 0
    for i in range(outer_CV_times):
        cnt += 1
        print("-" * 60)
        print("当前训练进度：", cnt, "/", outer_CV_times)

        result_path = path + "run_" + str(i) + "/"
        if i == 0:
            result_all = utils.read_csv(result_path + "result_all_params.csv", header=None)
        else:
            result_tmp = utils.read_csv(result_path + "result_all_params.csv", header=None)
            for u in range(0, len(result_tmp)):
                for v in range(0, len(result_tmp[0])):
                    result_all[u][v] += result_tmp[u][v]

    result_all = result_all.tolist()
    result_all.sort(key=lambda x: x[1])
    return result_all[0]


if __name__ == "__main__":
    for search_list in utils.search_lists:
        print("===============", search_list, "===============")
        search(search_list)
    get_final_result()
