# coding=utf-8
import os
import csv
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 项目所在根目录
pro_path = "C:/Users/WXX/Desktop/vocation/method2/project/"

# 输入的组合方式
search_lists = [
    ["maccs"],
    ["maccs", "graph"],
    ["maccs", "mordred"],
    ["graph", "mordred"],
    ["maccs", "graph", "mordred"]
]

# train_base_model.py 训练基模型参数
# train_base_model_params = [
#     [16, 32, 64, 128],  # 每次送入的数据量
#     [0.001, 0.005, 0.01],  # 学习率
#     ["glorot_normal", "glorot_uniform", "random_normal"],
#     [10, 20, 40, 80],  # 不同训练epochs
# ]
train_base_model_params = [  # for test
    [64, 128],  # 每次送入的数据量
    [0.001, 0.005],  # 学习率
    ["glorot_normal", "glorot_uniform"],
    [10, 20],  # 不同训练epochs
]

# train_target.py 迁移学习寻优参数
# train_target_params = [
#     [16, 32],  # 每次送入的数据量
#     [0.0001, 0.0005, 0.001, 0.0025],  # 学习率
#     [50, 100, 150, 200, 250, 300, 350, 400],  # 不同训练epochs
#     [10, 10],  # outer_CV_times, inner_CV_times
# ]
train_target_params = [  # for test
    [16, 32],  # 每次送入的数据量
    [0.0001, 0.0005],  # 学习率
    [50, 100],  # 不同训练epochs
    [2, 2],  # outer_CV_times, inner_CV_times
]

# machine learning
# lasso
alpha_list = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# RF
tree_num_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def get_input(search_list, Maccs, Graph, Mordred):
    res = []
    com_name = ""
    if "maccs" in search_list:
        res.append(Maccs[:, 2:])
        com_name += "maccs_"
    if "graph" in search_list:
        res.append(Graph[:, 2:])
        com_name += "graph_"
    if "mordred" in search_list:
        res.append(Mordred[:, 2:])
        com_name += "mordred_"
    X = res[0]
    for i in range(1, len(res)):
        X = np.concatenate([X, res[i]], axis=1)
    y = Maccs[:, 1]
    X, y = X.astype("float64"), y.astype("float64")
    return X, y, com_name[:-1]


def get_combine_name(search_list):
    com_name = ""
    if "maccs" in search_list:
        com_name += "maccs_"
    if "graph" in search_list:
        com_name += "graph_"
    if "mordred" in search_list:
        com_name += "mordred_"
    return com_name[:-1]


# csv文件的读取
def read_csv(filename, header=0):
    df = pd.read_csv(filename, header=header)  # 默认第一行为表头，自动生成索引, dataFrame类型
    a = df.values  # a是numpy类型
    return a


# csv文件的写入
def write_csv(path, filename, rows):
    # 写入csv文件: 将list写入csv文件
    # headers = ['class', 'name', 'sex', 'height', 'year']
    # rows = [
    #     [1, 'xiaoming', 'male', 168, 23],
    #     [1, 'xiaohong', 'female', 162, 22],
    #     [2, 'xiaozhang', 'female', 163, 21],
    #     [2, 'xiaoli', 'male', 158, 21]
    # ]
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + filename, 'w', newline="") as f:  # newline="" 是为了去掉行与行之间的空格
        writer = csv.writer(f)
        writer.writerows(rows)


# 写入txt文件
def write_txt(filename, con):
    # 写入文件
    # con = ["hello\n", "12.0\n"]
    with open(filename, 'w') as f:
        f.writelines(con)


# 文本文件的处理: 每行数据是纯数字
def read_txt(filename):
    # 读取文件
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(line[:-1])
    # print(data)  # data是list类型
    return data


# 文本文件的处理: 每行数据是纯数字
def read_txt_smiles_rts(filename):
    # 读取文件
    smiles, rts = [], []
    with open(filename, 'r') as f:
        data = f.readlines()[1:]  # 过滤掉表头
        for line in data:
            a, b = line[:-1].split('\t')
            smiles.append(a)
            rts.append(float(b))
    # print(data)  # data是list类型
    return smiles, rts


def cal_relative_median_error(val, pre):
    res = []
    for i in range(val.shape[0]):
        res.append(np.abs((val[i] - pre[i]) / val[i]))
    return res


def cal_r_square(y_origin, y_predict):
    return r2_score(y_origin, y_predict)


def cal_relative_error(y_origin, y_predict):
    res = []
    for i in range(len(y_origin)):
        res.append(np.abs((y_origin[i] - y_predict[i]) / y_origin[i]))
    return res


def cal_index(y_origin, y_predict):
    error_val = list(map(lambda o, p: o - p, y_origin, y_predict))
    error_val_abs = list(map(abs, error_val))

    # ------ 计算绝对误差 ------
    # 计算平均误差
    error_mean = np.mean(error_val_abs)  # MAE
    # 计算中位数误差
    error_median = np.median(error_val_abs)  # MedAE

    # ------ 计算相对误差 ------
    relative_error_abs = cal_relative_error(y_origin, y_predict)
    # 计算相对中位数平均误差
    relative_error_mean = np.mean(relative_error_abs)  # MRE
    # 计算相对中位数误差
    relative_error_median = np.median(relative_error_abs)  # MedRE

    # ------ 计算R^2 ------
    R_square = cal_r_square(y_origin, y_predict)

    # ------ 计算RMSE(均方根误差) ------
    RMSE = np.sqrt(np.mean(np.square(y_origin - y_predict)))

    res = [
        error_mean,  # MAE
        error_median,  # MedAE
        relative_error_mean,  # MRE
        relative_error_median,  # MedRE
        R_square,
        RMSE,
    ]
    result = []
    for i in range(len(res)):
        result.append(round(res[i], 4))

    return result


def cost_time(start, end):
    return str(round(end - start, 1) // 60) + " min " + str(round(end - start, 0) % 60) + "s"


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = {'batch': [], 'epoch': []}

    # def on_train_begin(self, logs={}):
    #     self.losses = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))

    def loss_plot(self, loss_type, path):  # path为保存路径

        path_image = path + "loss.png"
        iters = range(len(self.losses[loss_type]))
        # 保存loss值
        write_csv(path, "loss.csv", np.array(self.losses[loss_type]).reshape(1, -1))
        # 保存loss曲线
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.title(path.split("/")[-2])
        plt.savefig(path_image)
        plt.close("all")
