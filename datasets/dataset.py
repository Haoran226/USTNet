# -*- coding: utf-8 -*-

import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import numpy as np
import torch

class dataset_unpair(data.Dataset):
    def __init__(self, opts):
        self.dataroot = opts.dataroot
        self.opts = opts
        self.pcd_a = os.listdir(os.path.join(self.dataroot, opts.phase, 'A'))
        self.pcd_b = os.listdir(os.path.join(self.dataroot, opts.phase, 'B'))
        self.pcd_ply = opts.pcdply
        self.dataset_size = len(self.pcd_a)
        self.input_point = opts.input_point
        print('%d shapes, select %d points' % (len(self.pcd_a) + len(self.pcd_b), self.input_point))
        return

    def __getitem__(self, index):
        data_A = self.load_pointcloud(index, self.input_point, 0)
        random_index = random.randint(0, len(self.pcd_b) - 1)
        data_B = self.load_pointcloud(random_index, self.input_point, 1)
        return data_A, data_B

    def load_pointcloud(self, index, input_point, n):
        if n == 0:
            file = os.path.join(self.dataroot, self.opts.phase, 'A', self.pcd_a[index])
        else:
            file = os.path.join(self.dataroot, self.opts.phase, 'B', self.pcd_b[index])
        if self.pcd_ply == 0:
            header = 11
        else:
            header = 7

        with open(file, ) as file:
            points = []
            row = 0

            while 1:
                line = file.readline()
                if not line:
                    break

                if row >= header:
                    strs = line.split(" ")
                    points.append([float(strs[0]), float(strs[1]), float(strs[2])])
                row += 1
        assert input_point <= len(points)
        random.shuffle(points)
        return torch.tensor(points[:input_point])

    def __len__(self):
        return self.dataset_size


class DatasetUnpair4x4(data.Dataset):
    def __init__(self, opts):
        self.data_root = opts.dataroot
        self.opts = opts
        # 文件夹
        self.pcd_a = os.listdir(os.path.join(self.data_root, opts.phase, 'A'))
        self.pcd_b = os.listdir(os.path.join(self.data_root, opts.phase, 'B'))
        self.pcd_c = os.listdir(os.path.join(self.data_root, opts.phase, 'C'))
        self.pcd_d = os.listdir(os.path.join(self.data_root, opts.phase, 'D'))
        random.shuffle(self.pcd_a)
        random.shuffle(self.pcd_b)
        random.shuffle(self.pcd_c)
        random.shuffle(self.pcd_d)
        self.pcd_ply = opts.pcdply
        self.lenlist = [len(self.pcd_a), len(self.pcd_b), len(self.pcd_c), len(self.pcd_d)]
        self.dataset_size = sum(self.lenlist)
        self.input_point = opts.input_point
        print('%d shapes, select %d points' % (self.dataset_size, self.input_point))
        return

    def __getitem__(self, index):
        c = 0
        rindex = index
        while index // sum(self.lenlist[: c+1]) > 0:
            rindex -= self.lenlist[c]
            c += 1
        data_A1 = self.load_pointcloud(rindex, self.input_point, c)
        random_cls = random.randint(0, 2)
        random_cls = (c + random_cls) % 4
        random_index1 = random.randint(0, self.lenlist[random_cls] - 1)
        data_B1 = self.load_pointcloud(random_index1, self.input_point, random_cls)
        return data_A1, data_B1, c, random_cls

    def load_pointcloud(self, index, input_point, n):
        if n == 0:
            file = os.path.join(self.data_root, self.opts.phase, 'A', self.pcd_a[index])
        elif n == 1:
            file = os.path.join(self.data_root, self.opts.phase, 'B', self.pcd_b[index])
        elif n == 2:
            file = os.path.join(self.data_root, self.opts.phase, 'C', self.pcd_c[index])
        else:
            file = os.path.join(self.data_root, self.opts.phase, 'D', self.pcd_d[index])
        # print(file, index)
        if self.pcd_ply == 0:
            header = 11
        else:
            header = 7
        with open(file) as file:
            points = []
            row = 0
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if row >= header:
                    points.append([float(strs[0]), float(strs[1]), float(strs[2])])
                row += 1
        assert input_point <= len(points)
        random.shuffle(points)
        return torch.tensor(points[:input_point])

    def __len__(self):
        return self.dataset_size

    def normalization(self, pc):
        mx = torch.max(pc, dim=0)[0]
        mn = torch.min(pc, dim=0)[0]
        return pc / torch.max(mx-mn)
