import numpy as np
import os

import utils
from utils import pro_path


def my_split(X_num, train_num, y_index):
    index1 = []  # y_index中的下标，还需要提取出 y_index 中实际的下标
    while len(index1) < train_num:
        x = int(np.random.uniform(0, X_num, 1)[0])
        if x not in index1:
            index1.append(x)
    index2 = []
    for i in range(X_num):
        if i not in index1:
            index2.append(i)

    X_non_test_index, X_test_index = [], []
    for i in index1:
        X_non_test_index.append(y_index[i][1])
    for i in index2:
        X_test_index.append(y_index[i][1])

    return X_non_test_index, X_test_index


if __name__ == "__main__":

    file_list = os.listdir(pro_path + "data/gen_data/target/")

    input_dirs = []
    for file in file_list:
        if len(file.split('.')) == 2:  # 说明是文件
            continue
        input_dirs.append(file)

    # (0) 'Eawag_XBridgeC18', (1) 'FEM_lipids', (2) 'FEM_long',
    # (3) 'IPB_Halle', (4) 'LIFE_new', (5) 'LIFE_old', (6) 'lipids', (7) 'MassBank1',
    # (8) 'MassBank2', (9) 'MetaboBASE', (10) 'Natural products', (11) 'pesticide',
    # (12) 'RIKEN_PlaSMA', (13) 'UniToyama_Atlantis'
    print(input_dirs)
    print(len(input_dirs))

    train_num_list = [25, 50, 75, 100]  # 训练样本数目
    np.random.seed(1)

    for file in input_dirs:
        path = pro_path + "data/gen_data/target/" + file + "/"
        print(path)
        # 读取数据
        X1 = utils.read_csv(path + file + "_maccs.csv", header=None)
        y = X1[:, 1]
        X = np.concatenate([X1[:, 2:]], axis=1)

        X, y = X.astype("float64"), y.astype("float64")

        # 将数据按照保留时间排序
        y_index = []
        for i in range(len(y)):
            y_index.append([y[i], i])
        y_index.sort()

        X_num = len(X)
        print(X_num)
        for train_num in train_num_list:
            if train_num <= int(X_num * 0.75):
                # 分为训练集 和 测试集
                X_non_test_index, X_test_index = my_split(X_num, train_num, y_index)
                save_path = path + "split_index/"
                if not os.path.exists(save_path): os.makedirs(save_path)
                utils.write_csv(save_path, "_" + str(train_num) + "_X_non_test_index" + ".csv", [X_non_test_index])
                utils.write_csv(save_path, "_" + str(train_num) + "_X_test_index" + ".csv", [X_test_index])
