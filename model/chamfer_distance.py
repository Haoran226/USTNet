# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def dist_chamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = torch.sqrt(rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def f_score(pc1, pc2, threshold = 0.01):
    batch_size, num_point1, num_features = pc1.shape
    batch_size, num_point2, num_features = pc1.shape
    dist1 = torch.zeros([batch_size, num_point1])
    dist2 = torch.zeros([batch_size, num_point2])
    for i in range(batch_size):
        dist1[i], dist1[i] = dist_chamfer(pc1[i], pc2[i])
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore
