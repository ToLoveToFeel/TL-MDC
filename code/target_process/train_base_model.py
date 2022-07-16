# coding=utf-8
import os
import keras
import time
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras import initializers
from keras.models import Model
from sklearn.model_selection import train_test_split

import utils
from utils import pro_path


def search(search_list, Maccs, Graph, Mordred):
    # 寻优参数
    batch_size_list = utils.train_base_model_params[0] # 每次送入的数据量
    lr_list = utils.train_base_model_params[1]  # 学习率
    init_list = utils.train_base_model_params[2]
    epochs_list = utils.train_base_model_params[3]  # 不同训练epochs
    combine_list = []
    for batch_size in batch_size_list:
        for lr in lr_list:
            for init in init_list:
                combine_list.append([batch_size, lr, init, epochs_list])

    begin_time = time.perf_counter()

    # 读取数据
    X, y, com_name = utils.get_input(search_list, Maccs, Graph, Mordred)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    output_path = pro_path + "result/train_base_model/" + com_name + "/"
    # 结果汇总表
    result_all = []
    cnt = 0
    for i in range(len(combine_list)):
        batch_size, lr, init, epochs_list_e = combine_list[i]
        start = time.perf_counter()  # 开始计时
        cnt += 1
        print("-" * 60)
        print("当前训练进度：", cnt, "/", len(combine_list))
        # 换成DNN训练
        result = cross_validate(X_train, X_test, y_train, y_test,batch_size, lr, init, epochs_list_e, output_path)
        # 结果输出并写入文件
        for e in result: result_all.append(e)
        utils.write_csv(output_path + "result/", "res_all.csv", result_all)
        print("本次训练用时: " + utils.cost_time(start, time.perf_counter()))

    print("总用时: " + utils.cost_time(begin_time, time.perf_counter()))


def cross_validate(X_train, X_test, y_train, y_test, batch_size, lr, init, epochs_list, output_path):
    model = build_DNN(X_train.shape[1], lr, init)
    history = utils.LossHistory()

    # print(model.summary())

    res = []
    last_epochs = 0  # 类似于累积训练
    best_mae = 0
    is_first = True
    for k in range(len(epochs_list)):
        cur_epochs = epochs_list[k]
        item = [batch_size, lr, init, cur_epochs]
        model.fit(X_train, y_train, epochs=cur_epochs - last_epochs, batch_size=batch_size, verbose=2,
                  callbacks=[history])
        predict_val = np.array(model(X_test))[:, 0]
        print("predict_val.shape =", predict_val.shape)
        last_epochs = cur_epochs

        index = utils.cal_index(y_test, predict_val)

        for e in index: item.append(e)
        res.append(item)

        result_path = output_path + "result/model_" + init + "_" + str(batch_size) + "_" + str(lr) + "_" + str(
            cur_epochs) + "/"
        if not os.path.exists(result_path): os.makedirs(result_path)
        print("结果保存路径为: ", result_path)
        print(["batch_size", "lr", "init", "cur_epochs", "MAE", "MedAE", "MRE", "MedRE", "R_square", "RMSE"])
        print(item)
        utils.write_csv(result_path, "res.csv", np.array(index, dtype=object).reshape(1, -1))
        history.loss_plot('epoch', result_path)
        if is_first:
            is_first = False
            best_mae = index[0]
            model.save(output_path + "base_model.h5")
        elif best_mae > index[0]:
            best_mae = index[0]
            model.save(output_path + "base_model.h5")

    return res


# 负责创建模型
def build_DNN(input_shape, num_lr, init_val):
    np.random.seed(1)
    tf.random.set_seed(1)

    layer_init_size = pow(10, len(str(input_shape)) - 1)

    init = None
    if init_val == "random_normal":
        init = initializers.random_normal(mean=0.0, stddev=0.05, seed=None)
    elif init_val == "glorot_normal":
        init = initializers.glorot_normal(seed=1)
    elif init_val == "glorot_uniform":
        init = initializers.glorot_uniform(seed=1)

    input_md = Input(shape=(input_shape,), name='input_md')
    x_md1 = Dense(int(layer_init_size * 1), kernel_initializer=init, bias_initializer='zeros', activation='relu',
                  name='AE1')(input_md)
    x_md2 = Dense(int(layer_init_size * 0.8), kernel_initializer=init, bias_initializer='zeros', activation='relu',
                  name='AE2')(x_md1)
    x_md3 = Dense(int(layer_init_size * 0.4), kernel_initializer=init, bias_initializer='zeros', activation='relu',
                  name='AE3')(x_md2)
    x_md4 = Dense(int(layer_init_size * 0.2), kernel_initializer=init, bias_initializer='zeros', activation='relu',
                  name='AE4')(x_md3)
    x_md5 = Dense(int(layer_init_size * 0.1), kernel_initializer=init, bias_initializer='zeros', activation='relu',
                  name='AE5')(x_md4)
    output = Dense(1, kernel_initializer=init, bias_initializer='zeros', name='output')(x_md5)

    model = Model([input_md], [output])
    adam = keras.optimizers.Adam(lr=num_lr)
    model.compile(optimizer=adam, loss='mae')
    return model


if __name__ == "__main__":

    Maccs = utils.read_csv(pro_path + "data/gen_data/fingerprint_MACCS.csv", header=None)
    Graph = utils.read_csv(pro_path + "data/gen_data/graph_norm.csv", header=None)
    Mordred = utils.read_csv(pro_path + "data/gen_data/mordred_fill_norm.csv", header=None)

    for search_list in utils.search_lists:
        print("===============", search_list, "===============")
        search(search_list, Maccs, Graph, Mordred)
