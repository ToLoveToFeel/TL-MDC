# coding=utf-8
# Date: 2022/1/24 11:29
# 画出有无基模型的折线图结果
import os
import matplotlib.pyplot as plt

import utils
from utils import pro_path


def my_plot(filename, x, y1, y2, x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    a, = plt.plot(x, y1, color='r')
    b, = plt.plot(x, y2, color='g')
    plt.legend(handles=[a, b], labels=['with_model','without_model'], loc='best')
    plt.savefig(filename, dpi=300)
    plt.close()

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

    for file in input_dirs:

        print('-' * 40)
        print("当前处理的数据集: ", file)

        file_path = pro_path + "result/train_target/ratio/"

        res_with_model = utils.read_csv(file_path + "with_model/" + file + "/result_all.csv", header=None)
        res_without_model = utils.read_csv(file_path + "without_model/" + file +  "/result_all.csv", header=None)

        # train_num、batch_size、lr、epochs、MAE、MedAE、MRE、MedRE、R2、RMSE
        result_path = pro_path + "result/draw_result/" + file + '/'
        if not os.path.exists(result_path): os.makedirs(result_path)
        X = res_with_model[:, 0]  # train_num_list

        MAE_list1, MAE_list2 = res_with_model[:, 4], res_without_model[:, 4]
        my_plot(result_path + "MAE.png", X, MAE_list1, MAE_list2, "train_num", "MAE")

        MedAE_list1, MedAE_list2 = res_with_model[:, 5], res_without_model[:, 5]
        my_plot(result_path + "MedAE.png", X, MedAE_list1, MedAE_list2, "train_num", "MedAE")

        MRE_list1, MRE_list2 = res_with_model[:, 6], res_without_model[:, 6]
        my_plot(result_path + "MRE.png", X, MRE_list1, MRE_list2, "train_num", "MRE")

        MedRE_list1, MedRE_list2 = res_with_model[:, 7], res_without_model[:, 7]
        my_plot(result_path + "MedRE.png", X, MedRE_list1, MedRE_list2, "train_num", "MedRE")

        R2_list1, R2_list2 = res_with_model[:, 8], res_without_model[:, 8]
        my_plot(result_path + "R2.png", X, R2_list1, R2_list2, "train_num", "R2")

        RMSE_list1, RMSE_list2 = res_with_model[:, 9], res_without_model[:, 9]
        my_plot(result_path + "RMSE.png", X, RMSE_list1, RMSE_list2, "train_num", "RMSE")



