# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn import functional as F
from template import get_template
from model_blocks import Mapping2Dto3D, Identity
import random
from copy import deepcopy as dp
import numpy as np
from chamfer_distance import chamfer_distance_torch
from torch.autograd.function import InplaceFunction
from saver import Saver


class PointNet(nn.Module):
    def __init__(self, output, nlatent=1024, dim_input=3):
        """
        PointNet Encoder
        See : PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
                Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
        """

        super(PointNet, self).__init__()
        self.dim_input = dim_input
        self.conv1 = torch.nn.Conv1d(dim_input, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, output)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(output)

        self.nlatent = nlatent

    def forward(self, x, y=None):
        b = x.shape[0]
        if y is not None:
            x = torch.cat([x, y], dim=0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1))
        if y is not None:
            return x[0:b].squeeze(2), x[b:].squeeze(2)
        else:
            return x.squeeze(2)


class Atlasnet(nn.Module):
    def __init__(self, opt):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        Author : Thibault Groueix 01.11.2019
        """
        super(Atlasnet, self).__init__()
        self.opt = opt

        # Define number of points per primitives
        self.nb_pts_in_primitive = opt.input_point // opt.nb_primitives
        self.nb_pts_in_primitive_eval = opt.input_point // opt.nb_primitives

        if opt.remove_all_batchNorms:
            torch.nn.BatchNorm1d = Identity

            print("Replacing all batchnorms by identities.")

        # Initialize deformation networks
        self.decoder = nn.ModuleList([Mapping2Dto3D(opt) for i in range(0, opt.nb_primitives)])

    def forward(self, latent_vector, points=None, train=True):
        if self.opt.decoder_structure == 'mesh':
            return self.generate_mesh(latent_vector, train=train)
        else:
            return self.generate_pc(latent_vector, points=points, train=train)

    def generate_pc(self, latent_vector, points=None, train=True):
        """
        Deform points from self.template using the embedding latent_vector
        :param latent_vector: an opt.bottleneck size vector encoding a 3D shape or an image. size : batch, bottleneck
        :return: A deformed pointcloud os size : batch, nb_prim, num_point, 3
        """
        B = latent_vector.shape[0]

        self.template = [get_template(self.opt.template_type, device=latent_vector.device) for i in range(0, self.opt.nb_primitives)]
        if train:
            if points == None:
                input_points = [self.template[i].get_random_points(
                    torch.Size((1, self.template[i].dim, self.nb_pts_in_primitive)),
                    latent_vector.device) for i in range(self.opt.nb_primitives)]
            else:
                input_points = points.permute(0, 2, 1).unsqueeze(1)
        else:
            input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive_eval,
                                                                device=latent_vector.device)
                            for i in range(self.opt.nb_primitives)]

        # Deform each patch
        output_points = torch.cat([self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).unsqueeze(1) for i in
                                   range(0, self.opt.nb_primitives)], dim=1)

        s = output_points.shape
        return output_points.permute(0, 1, 3, 2).contiguous().view(s[0], s[1] * s[3], s[2])  # batch, nb_prim, num_point, 3

    def generate_mesh(self, latent_vector, train=False):
        assert latent_vector.size(0) == 1, "input should have batch size 1!"
        import trimesh
        B = latent_vector.size(0)
        self.template = [get_template(self.opt.template_type, device=latent_vector.device) for i in
                         range(0, self.opt.nb_primitives)]
        input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive_eval,
                                                                device=latent_vector.device)
                            for i in range(self.opt.nb_primitives)]

        input_points = [input_points[i] for i in range(self.opt.nb_primitives)]

        # Deform each patch
        output_patches = [self.decoder[i](input_points[i],
                                          latent_vector.unsqueeze(2)).unsqueeze(1)
                          for i in range(0, self.opt.nb_primitives)]
        output_points = torch.cat(output_patches, dim=1).squeeze(0)

        output_meshes = [trimesh.Trimesh(vertices=output_points[i].transpose(1, 0).contiguous().cpu().detach().numpy(),
                                          faces=self.template[i].mesh.faces)
                         for i in range(self.opt.nb_primitives)]

        # Deform return the deformed pointcloud
        mesh = trimesh.util.concatenate([output_meshes])

        return mesh


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.sq = nn.Sequential(nn.Linear(input_dim, 256),
                                nn.Dropout(0.5),
                                nn.ReLU(),
                                nn.Linear(256, output_dim))

    def forward(self, x):
        x = self.sq(x)
        return x


class Fuse(nn.Module):
    def __init__(self, stl_len, cont_len):
        super(Fuse, self).__init__()
        self.linear = nn.Sequential(nn.Linear(stl_len//2, cont_len), nn.ReLU(),
                                    nn.Linear(cont_len, cont_len))
        self.linear2 = nn.Sequential(nn.Linear(stl_len//2, cont_len), nn.ReLU(),
                                    nn.Linear(cont_len, cont_len))

    def forward(self, stlA, contA, stlB, contB):
        b = stlA.shape[0]
        stl = torch.cat([stlA, stlB], dim=0)
        self.log_var = self.linear(stl[:, 1::2])
        self.std = torch.exp(0.5 * self.log_var)
        self.log_var = torch.log(self.std[:b] ** 2 + self.std[b:] ** 2)
        self.mu = self.linear2(stl[:, ::2])
        self.mu_sum = self.mu[:b] + self.mu[b:]
        return contA * self.std[:b] + self.mu[:b], contB * self.std[b:] + self.mu[b:]

    def get_loss(self):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu_sum ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        return kld_loss

    def gen(self, stl, cont):
        log_var = self.linear(stl[:, 1::2])
        std = torch.exp(0.5 * log_var)
        mu = self.linear2(stl[:, ::2])
        return cont * std + mu


class FuseM(nn.Module):
    def __init__(self, stl_len, cont_len):
        super(FuseM, self).__init__()
        self.linear = nn.Sequential(nn.Linear(stl_len//2, cont_len), nn.ReLU(),
                                    nn.Linear(cont_len, cont_len))
        self.linear2 = nn.Sequential(nn.Linear(stl_len//2, cont_len), nn.ReLU(),
                                    nn.Linear(cont_len, cont_len))

    def forward(self, stlA, contA, stlB, contB, clsA, clsB):
        self.log_varA = self.linear(stlA[:, 1::2])
        self.log_varB = self.linear(stlB[:, 1::2])
        self.stdA = torch.exp(0.5 * self.log_varA)
        self.stdB = torch.exp(0.5 * self.log_varB)
        self.muA = self.linear2(stlA[:, ::2])
        self.muB = self.linear2(stlB[:, ::2])
        cls = set(torch.cat([clsA, clsB], dim=0).numpy())
        self.mu = torch.ones(0, contA.shape[1], device=contA.device)
        self.std = torch.ones(0, contA.shape[1], device=contA.device)
        for c in cls:
            maskA = clsA == c
            maskB = clsB == c
            m_t = torch.mean(torch.cat([self.muA[maskA], self.muB[maskB]], dim=0), dim=0, keepdim=True)
            s_t = torch.mean(torch.cat([self.stdA[maskA], self.stdB[maskB]], dim=0), dim=0, keepdim=True)
            self.mu = torch.cat([self.mu, m_t], dim=0)
            self.std = torch.cat([self.std, s_t], dim=0)
        self.mu = torch.sum(self.muA + self.muB, dim=0, keepdim=True)
        self.log_var = torch.log(torch.sum(self.std ** 2, dim=0, keepdim=True))
        return contA * self.stdA + self.muA, contB * self.stdB + self.muB

    def get_loss(self):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var -
                                               self.mu ** 2 - self.log_var.exp(), dim=1), dim=0)
        return kld_loss

    def gen(self, stlA, contA, stlB, contB):
        log_varA = self.linear(stlA[:, 1::2])
        log_varB = self.linear(stlB[:, 1::2])
        stdA = torch.exp(0.5 * log_varA)
        stdB = torch.exp(0.5 * log_varB)
        muA = self.linear2(stlA[:, ::2])
        muB = self.linear2(stlB[:, ::2])
        return contA * stdA + muA, contB * stdB + muB


class DisInfo(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.sq = nn.Sequential(nn.Linear(input_dim, 512),
                                nn.ReLU(),
                                nn.Linear(512, 128),
                                nn.ReLU(),
                                nn.Linear(128, 1))

    def forward(self, x):
        x = self.sq(x)
        return torch.sigmoid(x)


class DisInfoM(nn.Module):
    def __init__(self, input_dim, num_domain=2):
        super().__init__()
        self.sq = nn.Sequential(nn.Linear(input_dim, 512),
                                nn.ReLU(),
                                nn.Linear(512, 128),
                                nn.ReLU(),
                                nn.Linear(128, num_domain))

    def forward(self, x):
        x = self.sq(x)
        return torch.softmax(x, dim=1)
