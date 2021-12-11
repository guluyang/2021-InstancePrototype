# coding: utf-8
"""
Author: Inki
Email: inki.yinji@qq.com
Create: 2021 1206
Last modify: 2021 1206
"""

import numpy as np
import torch
import torch.optim as opt
import torch.utils.data as data_utils
from sklearn.metrics import accuracy_score, f1_score
from Model import ShapeletGenerator
from MIL import MIL
from utils import get_k_cv_idx, pairwise_dist


class BagLoader(data_utils.Dataset):

    def __init__(self, bags, bags_label, idx=None):
        """"""
        self.bags = bags
        self.idx = idx
        if self.idx is None:
            self.idx = list(range(len(bags)))
        self.num_idx = len(self.idx)
        self.bags_label = bags_label[idx]

    def __getitem__(self, idx):
        bag = [self.bags[self.idx[idx], 0][:, :-1].tolist()]
        bag = torch.from_numpy(np.array(bag))

        return bag.double(), torch.tensor([self.bags_label[idx].tolist()]).double()

    def __len__(self):
        """"""
        return self.num_idx


class InstancePrototype(MIL):

    def __init__(self, file_name,
                 epoch=10,
                 lr_pro=0.001,
                 lr_model=0.001,
                 D=1,
                 k=5,
                 bag_space=None):
        """
        :param file_name:   文件名
        :param epoch:       批次
        :param lr_pro:      原型学习率
        :param lr_model:    模型学习率
        :param D:           原型数量
        :param k:           k-CV
        :param bag_space:   包空间
        """
        super(InstancePrototype, self).__init__(file_name, bag_space=bag_space)
        self.epoch = epoch
        self.lr_pro = lr_pro
        self.lr_model = lr_model
        self.D = D
        self.k = k
        self.loss = torch.nn.CrossEntropyLoss()

    def get_mapping(self):
        tr_idxes, te_idxes = get_k_cv_idx(self.N, k=self.k)
        for i, (tr_idx, te_idx) in enumerate(zip(tr_idxes, te_idxes)):
            # print("The %d-th CV" % i)
            # 载入数据集
            tr_loader = BagLoader(self.bag_space, self.bag_lab, tr_idx)
            # 载入网络和优化器
            net = ShapeletGenerator(self.D, self.d, self.C)
            optim1 = torch.optim.Adam([net.prototypes], lr=self.lr_pro)
            optim2 = torch.optim.Adam(list(net.linear_layer.parameters()), lr=self.lr_model)

            batch_count = 0
            for epoch in range(self.epoch):
                tr_loss = 0
                for data, label in tr_loader:
                    # 标签预测
                    logits, dist = net(data.float())
                    # 当前损失
                    cur_loss = self.loss(logits.float(), label.long())
                    optim1.zero_grad()
                    optim2.zero_grad()
                    # 计算原型的距离
                    ins_prototypes_dist = pairwise_dist(net.prototypes, net.prototypes)
                    reg_ins_prototypes = ins_prototypes_dist.sum()

                    # 计算总损失
                    weight_reg = 0
                    for param in net.linear_layer.parameters():
                        weight_reg += param.norm(p=1).sum()
                    reg_loss = 0.05 * weight_reg + 0.0005 * dist.sum() + reg_ins_prototypes * 0.0005
                    min_loss = cur_loss + reg_loss
                    tr_loss += min_loss
                    min_loss.backward()

                    # 优化
                    optim1.step()
                    optim2.step()

                    batch_count += 1
                # print("Epoch %d, loss %.4f" % (epoch + 1, tr_loss / batch_count))

            """映射"""
            ret_mat = np.zeros((self.N, self.D * 3))
            for j in range(self.N):
                _, dist = net(torch.from_numpy(self.get_bag(j)).reshape(1, self.bag_size[j], -1).float())
                ret_mat[j] = dist.detach().numpy()

            yield ret_mat[tr_idx], self.bag_lab[tr_idx], ret_mat[te_idx], self.bag_lab[te_idx], None
