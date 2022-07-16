# coding=utf-8
import os
import utils
from utils import pro_path
from file_target_helper import *

DEBUG = True


# 保留idx对应位置的特征（注意：前两列为smiles、rt, 需要从第3列开始）
def handle_by_idx(data, idx):
    new_data = [data[:, 0], data[:, 1]]
    data = data[:, 2:].T
    for i in idx:
        new_data.append(data[i])
    return np.array(new_data).T


if __name__ == "__main__":

    # 1. 读取 pro_path + "data/input_data/target/" 文件夹下的所有文件
    file_list = os.listdir(pro_path + "data/input_data/target/")
    if DEBUG:
        print(file_list)

    # 2. 读取 idx 文件
    idx_graph = utils.read_csv(pro_path + "data/gen_data/idx_graph.csv", header=None)[0]
    idx_mordred = utils.read_csv(pro_path + "data/gen_data/idx_mordred.csv", header=None)[0]

    # 3. 遍历所有文件，并一次处理每个文件
    for file in file_list:
        if len(file.split('.')) == 1:  # 说明是文件夹
            continue
        # 前置处理：生成该文件对应的文件夹
        output_path = pro_path + "data/gen_data/target/" + file.split('.')[0] + "/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('-' * 20, "开始处理", file, '-' * 20)
        print("结果保存路径为: ", output_path)

        # 4. 获取文件内容
        print(pro_path + "data/input_data/target/" + file)
        smiles, rts = utils.read_txt_smiles_rts(pro_path + "data/input_data/target/" + file)
        new_smiles, new_rts = [], []
        # 5. 根据 smiles 计算分子 mol
        mols = []
        for i in range(len(smiles)):
            e = smiles[i]
            try:
                mol = Chem.MolFromSmiles(e)
                mol_gnn = Chem.AddHs(Chem.MolFromSmiles(e))

                mols.append(mol)
                new_smiles.append(smiles[i])
                new_rts.append(rts[i])
            except:
                if DEBUG:
                    print(e, "无法计算出 mol 或者 mol_gnn")
                pass
        # 6. 计算 图向量, 并根据 idx_graph 的下标，只保留 idx_graph 中对应的特征
        if DEBUG: print("开始计算图向量...")
        graph_vec = cal_gnn_vec(pro_path, new_smiles, new_rts)  # 返回numpy数据，前两列是smiles, rt
        graph_vec = handle_by_idx(graph_vec, idx_graph)
        utils.write_csv(output_path, file.split('.')[0] + "_graph.csv", graph_vec)
        if DEBUG:
            print(graph_vec.shape)
            print("图向量计算完成...")
        # 7. 计算 MACCS
        if DEBUG: print("开始计算MACCS...")
        maccs = cal_MACCS(new_smiles, new_rts, mols)  # 返回numpy数据，前两列是smiles, rt
        utils.write_csv(output_path, file.split('.')[0] + "_maccs.csv", maccs)
        if DEBUG:
            print(maccs.shape)
            print("MACCS计算完成...")
        # 8. 计算 mordred
        if DEBUG: print("开始计算mordred...")
        mordred = cal_mordred(new_smiles, new_rts, mols)  # 返回numpy数据，前两列是smiles, rt
        mordred = handle_by_idx(mordred, idx_mordred)
        if DEBUG: print("开始填充缺失值...")
        mordred = fill_mordred(mordred, mols)
        if DEBUG: print("缺失值填充完成")
        utils.write_csv(output_path, file.split('.')[0] + "_mordred.csv", mordred)
        if DEBUG:
            print(mordred.shape)
            print("mordred计算完成...")
        # 9. 对 图向量、mordred 中的每个特征进行标准化
        if DEBUG: print("开始标准化...")
        graph_std = standred(graph_vec)
        utils.write_csv(output_path, file.split('.')[0] + "_graph_std.csv", graph_std)
        mordred_std = standred(mordred)
        utils.write_csv(output_path, file.split('.')[0] + "_mordred_std.csv", mordred_std)
        if DEBUG: print("标准化完成....")
    pass
