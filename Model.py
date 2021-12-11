# coding: utf-8
"""
作者: 因吉
邮箱: inki.yinji@gmail.com
创建日期：2021 1206
近一次修改：2021 1206
"""


import torch
import torch.nn as nn
import numpy as np


class ShapeletGenerator(nn.Module):

    def __init__(self, D, L, C):
        """
        :param D:  实例原型的数量
        :param L:  实例维度
        :param C:  数据集类别
        """
        super(ShapeletGenerator, self).__init__()
        self.prototypes = (torch.randn((1, D, L))).requires_grad_()
        self.linear_layer = nn.Sequential(torch.nn.Linear(3 * D, C, bias=False),
                                          nn.Softmax())
        self.C = C

    def forward(self, X):
        """
        :param X:   包
        """
        X_norm = (X.norm(dim=2)[:, :, None])
        # 将维度进行换位,contiguous则使得内存中的表示与换位后匹配
        Y = self.prototypes.permute(0, 2, 1).contiguous()
        Y_norm = (self.prototypes.norm(dim=2)[:, None])
        Y = torch.cat([Y] * X.shape[0], dim=0)
        dist = X_norm + Y_norm - 2.0 * torch.bmm(X, Y)
        # 将距离限定在指定范围
        dist = torch.clamp(dist, 0.0, np.inf)
        min_dist = dist.min(dim=1)[0]
        max_dist = dist.max(dim=1)[0]
        mean_dist = dist.mean(dim=1)
        all_features = torch.cat([min_dist, max_dist, mean_dist], dim=1)
        logits = self.linear_layer(all_features)

        return logits, all_features


if __name__ == '__main__':
    _B = torch.rand(1, 5, 8)
    sg = ShapeletGenerator(10, 8, 2)
    sg.forward(_B)
