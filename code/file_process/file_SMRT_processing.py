# coding=utf-8
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from mordred import Calculator, descriptors
import math
import utils
from utils import pro_path


def SMRT_processing():
    # Maccs 的计算
    print("start process maccs......")
    SMRT_processing_maccs()
    # mordred 的计算
    print("start process mordred......")
    SMRT_processing_mordred()
    print("start process graph......")
    # 图向量的处理
    SMRT_processing_graph()


def SMRT_processing_maccs():
    # 使用 MACCSkeys 计算 maccs
    SMRT_processing_maccs_step1()
    # 根据smiles排序
    SMRT_processing_maccs_step2()


def SMRT_processing_maccs_step1():
    smiles_rts = utils.read_csv(pro_path + "/data/input_data/graph_smiles_rt.csv", header=None)
    smiles, rts = smiles_rts[:, 0].tolist(), smiles_rts[:, 1].tolist()
    mols = [Chem.MolFromSmiles(e) for e in smiles]

    data = []
    cnt = 0  # 进度
    for mol in mols:
        cnt += 1
        if cnt % 100 == 0:
            print(cnt, "/", len(mols))

        # 计算指纹
        fps = MACCSkeys.GenMACCSKeys(mol)
        # 指纹转化为数组
        fp_arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fps, fp_arr)
        # 数组中每个元素转化为int, 因为只可能是0或者1
        fp_arr = [int(e) for e in fp_arr]  # list

        data.append(fp_arr[1:])  # 第0位是占位符，需要删除

    data = np.array(data).T
    new_data = [smiles, rts]
    for item in data:
        new_data.append(item)
    new_data = np.array(new_data).T
    print(new_data.shape)

    utils.write_csv(pro_path + "/data/gen_data/", "fingerprint_MACCS.csv", new_data)
    del data, new_data


def SMRT_processing_maccs_step2():
    data = utils.read_csv(pro_path + "/data/gen_data/fingerprint_MACCS.csv", header=None).tolist()
    data.sort(key=lambda x: x[0])
    utils.write_csv(pro_path + "/data/gen_data/", "fingerprint_MACCS.csv", data)
    del data


def SMRT_processing_mordred():
    # 使用 mordred 计算分子指纹
    SMRT_processing_mordred_step1()
    # 删除缺失值(或者0值)过多的特征：缺失值数量超过 40% 的特征、非缺失值中0值数量超过 60% 的特征也会被删除
    SMRT_processing_mordred_step2()
    # 缺失值填充
    SMRT_processing_mordred_step3()
    # 根据smiles排序
    SMRT_processing_mordred_step4()
    # 特征归一化
    SMRT_processing_mordred_step5()


def SMRT_processing_mordred_step1():
    calc = Calculator(descriptors, ignore_3D=True)
    # print(len(calc.descriptors))  # 1613
    # print(len(Calculator(descriptors, ignore_3D=True, version="1.0.0")))  # 1612

    smiles_rt = utils.read_csv(pro_path + "/data/input_data/graph_smiles_rt.csv", header=None)
    smiles, rt = smiles_rt[:, 0], smiles_rt[:, 1]
    mols = [Chem.MolFromSmiles(e) for e in smiles]

    data = []
    cnt = 0  # 用于查看进度
    for mol in mols:
        res = calc(mol)
        item = []
        cnt += 1
        if cnt % 100 == 0:
            print(cnt, "/", len(mols))
        for e in res.values():
            e = float(e)
            if math.isnan(e):  # 是NaN的话填充字符串：loss
                item.append("loss")
            else:
                item.append(e)
        data.append(item)
    utils.write_csv(pro_path + "data/tmp_data/mordred/", "mordred_descriptors.csv", data)
    del data


def SMRT_processing_mordred_step2():
    data = utils.read_csv(pro_path + "/data/tmp_data/mordred/mordred_descriptors.csv", header=None).T
    print(data.shape)  # (1613, 77980)

    new_data = []
    new_header = []
    sum = len(data[0])
    cnt = 0
    idx_mordred = []  # 记录筛选出的特征的位置
    for k in range(len(data)):
        item = data[k]
        loss_cnt = 0
        zero_cnt = 0
        cnt += 1
        if cnt % 50 == 0:
            print(cnt, "/", len(data))
        for i in range(len(item)):
            if item[i] == "loss":
                loss_cnt += 1
            elif item[i] == 0:
                zero_cnt += 1
        if loss_cnt / sum >= 0.4:
            continue
        if zero_cnt / (sum - loss_cnt) >= 0.6:
            continue
        idx_mordred.append(k)
        new_data.append(item)
        # print(loss_cnt, zero_cnt)
    new_data = np.array(new_data).T
    print(new_data.shape)
    print(len(new_header))  # 1156
    print(len(idx_mordred))

    utils.write_csv(pro_path + "data/gen_data/", "idx_mordred.csv", [idx_mordred])
    utils.write_csv(pro_path + "data/tmp_data/mordred/", "mordred_remove.csv", new_data)
    del data, new_data


def SMRT_processing_mordred_step3():
    data1 = utils.read_csv(pro_path + "data/tmp_data/mordred/mordred_remove.csv", header=None).T
    smiles_rts = utils.read_csv(pro_path + "/data/input_data/graph_smiles_rt.csv", header=None)
    smiles, rts = smiles_rts[:, 0].tolist(), smiles_rts[:, 1].tolist()

    mols = [Chem.MolFromSmiles(e) for e in smiles]
    fps = [Chem.RDKFingerprint(x) for x in mols]

    mp = {}
    topK = 5
    data = []
    cnt = 0
    for item1 in data1:  # 填充每个特征
        index_list = []
        loss_list = []

        cnt += 1
        print(cnt, "/", len(data1))
        print("len(mp):", len(mp))

        for i in range(len(item1)):
            if item1[i] == "loss":
                loss_list.append(i)
            else:
                index_list.append(i)
        if len(loss_list) == 0:
            data.append(item1)
            continue

        for i in loss_list:
            rank_list = []
            for j in index_list:
                tp = (i, j)
                if tp in mp:
                    rank_list.append(mp[tp])
                else:
                    sim = DataStructs.FingerprintSimilarity(fps[i], fps[j])
                    rank_list.append((sim, j))
                    if len(mp) < 2e7:
                        mp[(i, j)] = (sim, j)

            rank_list.sort(key=lambda x: -x[0])
            t = 0
            for j in range(min(len(rank_list), topK)):
                t += float(item1[rank_list[j][1]])
            item1[i] = t / min(len(rank_list), topK)
            del rank_list
        data.append(item1)

    new_data = [smiles, rts]
    for item in data:
        new_data.append(item)
    new_data = np.array(new_data).T
    utils.write_csv(pro_path + "/data/tmp_data/mordred/", "mordred_fill.csv", new_data)
    del data, new_data, mp


def SMRT_processing_mordred_step4():
    data = utils.read_csv(pro_path + "/data/tmp_data/mordred/mordred_fill.csv", header=None).tolist()
    data.sort(key=lambda x: x[0])
    utils.write_csv(pro_path + "data/tmp_data/mordred/", "mordred_fill.csv", data)
    del data


def SMRT_processing_mordred_step5():
    data = utils.read_csv(pro_path + "data/tmp_data/mordred/mordred_fill.csv", header=None).T
    new_data = [data[0], data[1]]

    for i in range(2, len(data)):
        item = data[i]
        mean = np.mean(item)
        std = np.std(item)
        new_item = (item - mean) / std
        new_data.append(new_item)

    new_data = np.array(new_data).T
    print(new_data.shape)
    utils.write_csv(pro_path + "data/gen_data/", "mordred_fill_norm.csv", new_data)
    del data, new_data


def SMRT_processing_graph():
    # 删除0值数量超过60%的特征
    SMRT_processing_graph_step1()
    # 根据smiles排序
    SMRT_processing_graph_step2()
    # 特征归一化
    SMRT_processing_graph_step3()


def SMRT_processing_graph_step1():
    train = utils.read_csv(pro_path + "data/input_data/SMRT_train_set_graph.csv", header=None).T
    test = utils.read_csv(pro_path + "data/input_data/SMRT_test_set_graph.csv", header=None).T

    graph_train = [train[0], train[1]]  # smiles、rt
    graph_test = [test[0], test[1]]

    sum = len(train[0]) + len(test[0])  # 77980

    idx_graph = []  # 记录筛选出的特征的位置
    for i in range(2, len(train)):
        a, b = train[i], test[i]
        cnt = 0  # 0值数量
        for e in a:
            if e == 0:
                cnt += 1
        for e in b:
            if e == 0:
                cnt += 1
        print(cnt)
        if cnt / sum > 0.60:
            continue
        idx_graph.append(i - 2)
        graph_train.append(a)
        graph_test.append(b)

    graph_train = np.array(graph_train).T  # 70182个化合物
    graph_test = np.array(graph_test).T  # 7798个化合物

    graph = []
    for e in graph_train:
        graph.append(e)
    for e in graph_test:
        graph.append(e)

    utils.write_csv(pro_path + "data/gen_data/", "idx_graph.csv", [idx_graph])
    utils.write_csv(pro_path + "data/tmp_data/graph/", "graph.csv", graph)
    del graph, graph_train, graph_test


def SMRT_processing_graph_step2():
    data = utils.read_csv(pro_path + "/data/tmp_data/graph/graph.csv", header=None).tolist()
    data.sort(key=lambda x: x[0])
    utils.write_csv(pro_path + "data/tmp_data/graph/", "graph.csv", data)
    del data


def SMRT_processing_graph_step3():
    data = utils.read_csv(pro_path + "data/tmp_data/graph/graph.csv", header=None).T
    new_data = [data[0], data[1]]

    for i in range(2, len(data)):
        item = data[i]
        mean = np.mean(item)
        std = np.std(item)
        new_item = (item - mean) / std
        new_data.append(new_item)

    new_data = np.array(new_data).T
    print(new_data.shape)
    utils.write_csv(pro_path + "data/gen_data/", "graph_norm.csv", new_data)
    del data, new_data


###########################################################

if __name__ == "__main__":
    # 处理 SMRT
    SMRT_processing()
