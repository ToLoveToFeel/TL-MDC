# coding=utf-8
import numpy as np
from mordred import Calculator, descriptors
import math
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
import torch
import torch.nn as nn
import torch.nn.functional as F
import preprocess as pp

radius = 1
dim = 48
layer_hidden = 6
layer_output = 6
batch_train = 32
batch_test = 32
lr = 1e-4
lr_decay = 0.85
decay_interval = 10
iteration = 200
N = 5000

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i + m, j:j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        Smiles, fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)

        return Smiles, molecular_vectors

    def mlp(self, vectors):
        """ regressor based on multilayer perceptron."""
        for l in range(layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs

    def forward_regressor(self, data_batch, train):

        # inputs = data_batch[:-1]
        # correct_values = torch.cat(data_batch[-1])
        #
        # if train:
        #     Smiles, molecular_vectors = self.gnn(inputs)
        #     predicted_values = self.mlp(molecular_vectors)
        #     loss = F.mse_loss(predicted_values, correct_values)
        #     return loss
        # else:
        #     with torch.no_grad():
        #         Smiles, molecular_vectors = self.gnn(inputs)
        #         predicted_values = self.mlp(molecular_vectors)
        #     predicted_values = predicted_values.to('cpu').data.numpy()
        #     correct_values = correct_values.to('cpu').data.numpy()
        #     predicted_values = np.concatenate(predicted_values)
        #     correct_values = np.concatenate(correct_values)
        #     return Smiles, predicted_values, correct_values, molecular_vectors  # 更改

        inputs = data_batch

        if train:
            pass
        else:
            with torch.no_grad():
                Smiles, molecular_vectors = self.gnn(inputs)
            return Smiles, molecular_vectors  # 更改


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_regressor(self, dataset, rts, gen=False):  # 提取训练得到的向量
        N = len(dataset)
        SMILES, Ts, Ys = '', [], []
        data = []
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i + batch_test]))
            (Smiles, molecular_vectors) = self.model.forward_regressor(data_batch, train=False)
            SMILES += ' '.join(Smiles) + ' '
            correct_values = rts[i:i + batch_test]

            if gen:  # 添加
                for j in range(len(Smiles)):
                    item = [Smiles[j], correct_values[j]]
                    for k in range(len(molecular_vectors[j])):
                        item.append(molecular_vectors[j][k].item())
                    data.append(item)
        return np.array(data)


def cal_gnn_vec(pro_path, smiles, rts):
    # 加载数据
    dataset = pp.create_dataset(smiles)

    model = MolecularGraphNeuralNetwork(N, dim, layer_hidden, layer_output).to(device)
    # SMRT_model.h5 是卢红梅文章提供的模型
    model.load_state_dict(torch.load(pro_path + "data/input_data/SMRT_model.h5", map_location=torch.device('cpu')))
    tester = Tester(model)

    return tester.test_regressor(dataset, rts, True)


####################################################################################

def cal_MACCS(new_smiles, new_rts, mols):
    data = []
    cnt = 0  # 进度
    for i in range(len(mols)):
        mol = mols[i]
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

        data.append([new_smiles[i], new_rts[i]] + fp_arr[1:])  # 第0位是占位符，需要删除
    return np.array(data)


####################################################################################

def cal_mordred(new_smiles, new_rts, mols):
    calc = Calculator(descriptors, ignore_3D=True)
    data = []
    cnt = 0  # 用于查看进度
    for i in range(len(mols)):
        mol = mols[i]
        res = calc(mol)
        item = []

        cnt += 1
        if cnt % 20 == 0:
            print(cnt, "/", len(mols))

        for e in res.values():
            e = float(e)
            if math.isnan(e):  # 是NaN的话填充字符串：loss
                item.append("loss")
            else:
                item.append(e)
        data.append([new_smiles[i], new_rts[i]] + item)
    return np.array(data)


def fill_mordred(data, mols):
    fps = [Chem.RDKFingerprint(x) for x in mols]
    mp = {}
    topK = 5
    new_data = [data[:, 0], data[:, 1]]
    data = data[:, 2:].T
    cnt = 0
    for item in data:  # 填充每个特征
        index_list = []
        loss_list = []

        cnt += 1
        if cnt % 100 == 0:
            print(cnt, "/", len(data))
            print("len(mp):", len(mp))

        for i in range(len(item)):
            if item[i] == "loss":
                loss_list.append(i)
            else:
                index_list.append(i)
        if len(loss_list) == 0:
            new_data.append(item)
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
                t += float(item[rank_list[j][1]])
            item[i] = t / max(min(len(rank_list), topK), 1)  # rank_list可能为空，说明全部是缺失值
            del rank_list
        new_data.append(item)

    new_data = np.array(new_data).T
    return new_data


####################################################################################

def standred(data):
    data = data.T
    new_data = [data[0], data[1]]

    for i in range(2, len(data)):
        item = data[i]
        item = np.array([float(e) for e in item])
        mean = np.mean(item)
        std = np.std(item)
        if std != 0:
            new_item = (item - mean) / std
            new_data.append(new_item)
        else:
            new_data.append(item)

    new_data = np.array(new_data).T
    return new_data
