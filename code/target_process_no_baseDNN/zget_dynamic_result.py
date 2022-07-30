# coding=utf-8
import os
import utils
from utils import pro_path


def search(path):

    batch_size_list = utils.train_target_params[0]  # 每次送入的数据量
    lr_list = utils.train_target_params[1]  # 学习率
    epochs_list = utils.train_target_params[2]  # 不同训练epochs
    outer_CV_times, inner_CV_times = utils.train_target_params[3]
    combine_list = []
    for batch_size in batch_size_list:
        for lr in lr_list:
                combine_list.append([batch_size, lr, epochs_list])

    del epochs_list, batch_size_list, lr_list

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
    result_all.sort(key=lambda x: x[3])
    return result_all[0]


if __name__ == "__main__":

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

    result_dynamic = []
    com_name_list = []
    for search_list in utils.search_lists:
        com_name_list.append(utils.get_combine_name(search_list))

    for file in file_list:
        result_cro = []  # 验证集结果
        result_test = []  # 测试集结果
        for com_name in com_name_list:
            result_path = pro_path + "result/train_target_no_baseDNN/"
            res = search(result_path + file + "/")
            result_cro.append(res)
            result_test.append(utils.read_csv(result_path + file + "/result_avg.csv", header=None)[0])

        # 找到 result_cro 中 MAE 最小的对应的下标，使用其在验证集的结果
        index = 0
        for i in range(len(com_name_list)):
            if result_cro[i][3] < result_cro[index][3]:
                index = i

        result_dynamic.append([file, com_name_list[index]] + result_test[index].tolist())

    utils.write_csv(pro_path + "result/train_target_no_baseDNN/", "result_dynamic.csv", result_dynamic)

