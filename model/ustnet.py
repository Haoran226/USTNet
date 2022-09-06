# -*- coding: utf-8 -*-

import networks
import torch
import torch.nn as nn
import torch.nn.functional as F
from chamfer_distance import dist_chamfer
from torch.nn.utils import clip_grad_norm_
from chamfer_distance import f_score
from pointcloud import instance_normalize, pointcloud_split, pointcloud_trans


class Ustnet(nn.Module):
    def __init__(self, opts):
        super().__init__()
        lr = opts.initial_lr
        self.lr = lr
        self.opts = opts
        self.enc_c = networks.PointNet(1024)
        self.enc_s = networks.PointNet(512)
        self.fuse = networks.Fuse(512, 1024)
        self.dis_cont = networks.DisInfo(1024)
        self.decoder = networks.Atlasnet(opts)

        self.c_opt = torch.optim.Adam([{'params': self.enc_c.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.s_opt = torch.optim.Adam([{'params': self.enc_s.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.decoder_opt = torch.optim.Adam([{'params': self.decoder.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.dis_cont_opt = torch.optim.Adam([{'params': self.dis_cont.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.fuse_opt = torch.optim.Adam([{'params': self.fuse.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        self.distChamfer = dist_chamfer
        self.f_score = f_score

    def setgpu(self, gpu):
        self.gpu = gpu
        self.enc_c.cuda(self.gpu)
        self.enc_s.cuda(self.gpu)
        self.dis_cont.cuda(self.gpu)
        self.decoder.cuda(self.gpu)
        self.fuse.cuda(self.gpu)

    def forward(self, pointA, pointB):
        self.pointA = pointA
        self.pointB = pointB

    def forward_all(self, train=True):
        self.z_c_A, self.z_c_B = self.enc_c(self.pointA.permute(0, 2, 1),
                                self.pointB.permute(0, 2, 1))
        self.z_s_A, self.z_s_B = self.enc_s(self.pointA.permute(0, 2, 1),
                                self.pointB.permute(0, 2, 1))
        self.z_A2B, self.z_B2A = self.fuse(self.z_s_B, self.z_c_A, self.z_s_A, self.z_c_B)
        self.loss_vae = self.fuse.get_loss()

        self.recon_A2B = self.decoder(self.z_A2B, train=train)
        self.recon_B2A = self.decoder(self.z_B2A, train=train)
        self.z_A, self.z_B = self.fuse(self.z_s_A, self.z_c_A, self.z_s_B, self.z_c_B)
        self.loss_vae += self.fuse.get_loss()

        self.recon_A = self.decoder(self.z_A, train=train)
        self.recon_B = self.decoder(self.z_B, train=train)
        self.loss_vae += self.fuse.get_loss()

        self.recon_c_A, self.recon_c_B = self.enc_c(self.recon_A2B.permute(0, 2, 1),
                                    self.recon_B2A.permute(0, 2, 1))
        self.recon_s_A, self.recon_s_B = self.enc_s(self.recon_B2A.permute(0, 2, 1),
                                    self.recon_A2B.permute(0, 2, 1))

        self.loss_re2_eval = torch.mean(self.fscore(self.recon_A, self.pointA) / 2 + self.fscore(self.recon_B, self.pointB) / 2).item()

    def generate_from_z(self, s, c):
        z = self.fuse.gen(s, c)
        gen = self.decoder(z)
        return gen

    def update_all(self):
        # D
        self.forward_all()
        self.dis_cont_opt.zero_grad()
        self.backward_D()
        clip_grad_norm_(self.dis_cont.parameters(), 10)
        self.dis_cont_opt.step()
        # E&G
        self.c_opt.zero_grad()
        self.s_opt.zero_grad()
        self.decoder_opt.zero_grad()
        self.fuse_opt.zero_grad()
        self.backward_EG()
        clip_grad_norm_(self.enc_c.parameters(), 10)
        clip_grad_norm_(self.enc_s.parameters(), 10)
        clip_grad_norm_(self.fuse.parameters(), 10)
        clip_grad_norm_(self.decoder.parameters(), 10)
        self.c_opt.step()
        self.s_opt.step()
        self.fuse_opt.step()
        self.decoder_opt.step()

    def backward_EG(self):
        # latent consistency loss
        mse = torch.nn.MSELoss()
        loss_cont = mse(self.z_c_A, self.recon_c_A) + \
                    mse(self.z_c_B, self.recon_c_B)
        loss_stl = mse(self.z_s_A, self.recon_s_A) + \
                   mse(self.z_s_B, self.recon_s_B)
        loss_latent = loss_stl + loss_cont

        # content adversarial Loss
        outs_cA = self.dis_cont(self.z_c_A)
        outs_cB = self.dis_cont(self.z_c_B)
        all_ones = torch.ones_like(outs_cA).cuda(self.gpu)
        loss_dis_c = F.binary_cross_entropy(outs_cA, all_ones * 0.5) / 2 + \
                     F.binary_cross_entropy(outs_cB, all_ones * 0.5) / 2

        # fuse network KL loss
        loss_vae = self.loss_vae

        # reconstruction loss
        loss_pointA1, loss_pointA2 = self.distChamfer(self.recon_A, self.pointA)
        loss_pointB1, loss_pointB2 = self.distChamfer(self.recon_B, self.pointB)
        loss_dist = torch.mean((loss_pointA1 + loss_pointB1) * 0.5 + (loss_pointA2 + loss_pointB2) * 0.5)

        loss_eg = loss_latent * 0.1 + loss_vae * 0.01 + loss_dis_c + loss_dist * 5
        loss_eg.backward(retain_graph=True)

        self.loss_dist0 = loss_dist.item()
        self.loss_dis_c = loss_dis_c.item()
        self.loss_vae = loss_vae.item()
        self.loss_latent = loss_latent.item()

    def backward_D(self):
        dis_c_A = self.dis_cont(self.z_c_A)
        dis_c_B = self.dis_cont(self.z_c_B)
        ones = torch.ones_like(dis_c_A)
        loss_dis_cont = F.binary_cross_entropy(dis_c_A, ones * 0.99) + F.binary_cross_entropy(dis_c_B, ones * 0.01)
        loss_dis_cont = torch.mean(loss_dis_cont) * 0.25
        loss_dis_cont.backward(retain_graph=True)
        self.loss_dis_cont = loss_dis_cont.item()

    def update_lr(self, ep):
        if ep in [120, 140, 145]:
            self.lr /= 10
            lr = self.lr
            self.c_opt = torch.optim.Adam([{'params': self.enc_c.parameters(), 'initial_lr': lr}], lr=lr,
                                          betas=(0.5, 0.999), weight_decay=0.0001)
            self.s_opt = torch.optim.Adam([{'params': self.enc_s.parameters(), 'initial_lr': lr}], lr=lr,
                                          betas=(0.5, 0.999), weight_decay=0.0001)
            self.decoder_opt = torch.optim.Adam([{'params': self.decoder.parameters(), 'initial_lr': lr}], lr=lr,
                                                weight_decay=0.0001)
            self.dis_cont_opt = torch.optim.Adam([{'params': self.dis_cont.parameters(), 'initial_lr': lr}], lr=lr,
                                                 weight_decay=0.0001)
            self.fuse_opt = torch.optim.Adam([{'params': self.fuse.parameters(), 'initial_lr': lr}], lr=lr,
                                             betas=(0.5, 0.999), weight_decay=0.0001)

    def save(self, filename, ep, total_it):
        state = {
             'enc_c': self.enc_c.state_dict(),
             'enc_s': self.enc_s.state_dict(),
             'decoder': self.decoder.state_dict(),
             'dis_cont': self.dis_cont.state_dict(),
             'fuse': self.fuse.state_dict(),
             'c_opt': self.c_opt.state_dict(),
             's_opt': self.s_opt.state_dict(),
             'decoder_opt': self.decoder_opt.state_dict(),
             'dis_cont_opt': self.dis_cont_opt.state_dict(),
             'fuse_opt': self.fuse_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
        torch.save(state, filename)
        return

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        self.enc_c.load_state_dict(checkpoint['enc_c'])
        self.enc_s.load_state_dict(checkpoint['enc_s'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.fuse.load_state_dict(checkpoint['fuse'])

        if train:
            self.dis_cont.load_state_dict(checkpoint['dis_cont'])
            self.c_opt.load_state_dict(checkpoint['c_opt'])
            self.s_opt.load_state_dict(checkpoint['s_opt'])
            self.decoder_opt.load_state_dict(checkpoint['decoder_opt'])
            self.dis_cont_opt.load_state_dict(checkpoint['dis_cont_opt'])
            self.fuse_opt.load_state_dict(checkpoint['fuse_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def output_pc(self):
        return (self.pointA.detach().cpu(), self.pointB.detach().cpu(), self.recon_A2B.detach().cpu(),
                self.recon_B2A.detach().cpu(), self.recon_A.detach().cpu(), self.recon_B.detach().cpu())

    def output_pc_simple(self):
        return (self.recon_A.detach().cpu(),
                self.recon_B.detach().cpu())
