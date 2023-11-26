
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from collections import defaultdict
from time import time

ALL_TRAIN = False

class GetData(Dataset):

    def __init__(self, path="../data/RNADisease"):
        # train or test
        print(f'using {path} dataset')
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")

        print(f'loading [{path}]')
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        self.trainSize = 0
        self.testSize = 0

        df = pd.read_csv(train_file)
        self.trainUniqueSnoRNAs = pd.unique(df["snoRNA_id"])
        self.trainSnoRNA = df["snoRNA_id"].to_numpy()
        self.trainDisease = df["disease_id"].to_numpy()
        self.trainSize = len(df)

        self.n_snoRNA = 471
        self.m_disease = 84

        df = pd.read_csv(test_file)
        self.testUniqueSnoRNAs = pd.unique(df["snoRNA_id"])
        self.testSnoRNA = df["snoRNA_id"].to_numpy()
        self.testDisease = df["disease_id"].to_numpy()

        # print(f'self.')

        self.testSize = len(df)

        if ALL_TRAIN:
            self.trainSnoRNA = np.concatenate((self.trainSnoRNA, self.testSnoRNA), axis=0)
            self.trainDisease = np.concatenate((self.trainDisease, self.testDisease), axis=0)
            self.trainSize = self.trainSize + self.testSize

        self.Graph = None
        print(f"{self.trainSize} interactions for training")
        print(f"{self.testSize} interactions for testing")
        print(f"Sparsity : {(self.trainSize + self.testSize) / self.n_snoRNA / self.m_disease}")

        # (snoRNAs,diseases), bipartite graph
        self.SnoRNADiseaseNet = csr_matrix((np.ones(len(self.trainSnoRNA)),
                                        (self.trainSnoRNA, self.trainDisease)),
                                        shape=(self.n_snoRNA, self.m_disease))


        # pre-calculate
        self.allPos = self.getSnoRNAPosDiseases(list(range(self.n_snoRNA)))
        self.testDict = self.__build_test()
        print(f"Ready to go")

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def saveRatingMatrix(self):
        test_ratings = csr_matrix((np.ones(len(self.testSnoRNA)),
                                        (self.testSnoRNA, self.testDisease)),
                                        shape=(self.n_snoRNA, self.m_disease))
        sp.save_npz(self.path + '/train_mat.npz', self.SnoRNADiseaseNet)
        sp.save_npz(self.path + '/test_mat.npz', test_ratings)


    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_snoRNA + self.m_disease, self.n_snoRNA + self.m_disease), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.SnoRNADiseaseNet.tolil()
                adj_mat[:self.n_snoRNA, self.n_snoRNA:] = R
                adj_mat[self.n_snoRNA:, :self.n_snoRNA] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)


            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)

        return self.Graph


    def __build_test(self):
        test_data = defaultdict(set)
        for snoRNA, disease in zip(self.testSnoRNA, self.testDisease):
            test_data[snoRNA].add(disease)

        return test_data

    def getSnoRNADiseaseFeedback(self, snoRNAs, diseases):
        return np.array(self.SnoRNADiseaseNet[snoRNAs, diseases]).astype('uint8').reshape((-1,))

    def getSnoRNAPosDiseases(self, snoRNAs):
        posDiseases = []
        for snoRNA in snoRNAs:
            posDiseases.append(self.SnoRNADiseaseNet[snoRNA].nonzero()[1])
        return posDiseases