# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 09:52:11 2021

@author: WHR
"""
import torch
import random

def instance_center(point_cloud):
    if len(point_cloud.shape) == 3:
        pc = torch.zeros_like(point_cloud)
        for p in range(point_cloud.shape[0]):
            temp = point_cloud[p]
            mean = torch.mean(temp, axis=0)
            pc[p] = (temp - mean) 
    else:
        mean = torch.mean(point_cloud, axis=0)
        pc = (point_cloud - mean) 
    return pc

def instance_normalize(point_cloud):
    if len(point_cloud.shape) == 3:
        pc = torch.zeros_like(point_cloud)
        for p in range(point_cloud.shape[0]):
            temp = point_cloud[p]
            mean = torch.mean(temp, axis=0)
            std = torch.std(temp)
            pc[p] = (temp - mean) / std
    else:
        mean = torch.mean(point_cloud, axis=0)
        std = torch.std(point_cloud)
        pc = (point_cloud - mean) / std
    return pc


def pointcloud_split(pointcloud):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    z = random.uniform(0, 1)
    v = torch.tensor([x, z, y]).cuda(0)
    f = torch.matmul(pointcloud, v.T)
    sorted_pc = torch.zeros_like(pointcloud).detach_()
    for i in range(len(pointcloud)):
        _, indices = torch.sort(f)
        sorted_pc[i] = torch.gather(pointcloud[i], 0, indices[i].expand(pointcloud[i].T.shape).T)
    s = random.randint(int(pointcloud.shape[1] * 0.6), int(pointcloud.shape[1] * 0.9))
    return sorted_pc[:, 0: s], sorted_pc[:, s:]


def pointcloud_trans(pointcloud):
    pointcloud = pointcloud.detach()
    for i in range(len(pointcloud)):
        dim = random.randint(0, 2)
        times = random.uniform(0.8, 1.2)
        pointcloud[i, :, dim] *= times
    return pointcloud
    
def pointcloud_edge(pointcloud):
    pc = torch.zeros_like(pointcloud)
    for i in range(len(pointcloud)):
        # print(pointcloud[i][1:].shape, pointcloud[i][0:1].shape)
        pc[i] = torch.cat([pointcloud[i][1:], pointcloud[i][0:1]])
    return pc - pointcloud