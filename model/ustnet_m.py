# -*- coding: utf-8 -*-

import networks
import torch
import torch.nn as nn
import torch.nn.functional as F
from chamfer_distance import distChamfer
from chamfer_distance import f_score
from pointcloud import instance_normalize, pointcloud_split, pointcloud_trans


class UstnetM(nn.Module):
    def __init__(self, opts):
        super().__init__()
        
        lr = opts.initial_lr
        self.opts = opts
        self.enc_c = networks.PointNet(1024)
        self.enc_s = networks.PointNet(512)
        
        self.fuse = networks.Fuse_M(512, 1024)
        self.decoder = networks.Atlasnet(opts)
        self.dis_cont = networks.Dis_info_m(1024, 4)
        self.c_opt = torch.optim.Adam([{'params': self.enc_c.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.s_opt = torch.optim.Adam([{'params': self.enc_s.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.decoder_opt = torch.optim.Adam([{'params': self.decoder.parameters(), 'initial_lr': lr}], lr=lr, weight_decay=0.0001)
        self.dis_cont_opt = torch.optim.Adam([{'params': self.dis_cont.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.fuse_opt = torch.optim.Adam([{'params': self.fuse.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        self.distChamfer = distChamfer
        self.f_score = f_score

    def set_gpu(self, gpu):
        self.gpu = gpu
        self.enc_c.cuda(self.gpu)
        self.enc_s.cuda(self.gpu)
        self.dis_cont.cuda(self.gpu)
        self.decoder.cuda(self.gpu)
        self.fuse.cuda(self.gpu)

    def forward(self, pointA, pointB, cls1, cls2):
        self.pointA = pointA
        self.pointB = pointB
        self.cls1 = cls1
        self.cls2 = cls2

    def forward_for_test(self, pointA, pointB):
        self.pointA = pointA
        self.pointB = pointB
        self.z_c_A, self.z_c_B = self.enc_c(self.pointA.permute(0, 2, 1),
                                            self.pointB.permute(0, 2, 1))
        self.z_s_A, self.z_s_B = self.enc_s(self.pointA.permute(0, 2, 1),
                                            self.pointB.permute(0, 2, 1))
        self.z_A2B, self.z_B2A = self.fuse.gen(self.z_s_B, self.z_c_A, self.z_s_A, self.z_c_B)
        # self.loss_vae = self.fuse.get_loss()
        self.recon_A2B = self.decoder(self.z_A2B, train=False)
        self.recon_B2A = self.decoder(self.z_B2A, train=False)

        self.z_A, self.z_B = self.fuse.gen(self.z_s_A, self.z_c_A, self.z_s_B, self.z_c_B)
        self.recon_A = self.decoder(self.z_A, train=False)
        self.recon_B = self.decoder(self.z_B, train=False)
        
    def forward_all(self):
        self.z_c_A, self.z_c_B = self.enc_c(self.pointA.permute(0, 2, 1),
                                self.pointB.permute(0, 2, 1))
        self.z_s_A, self.z_s_B = self.enc_s(self.pointA.permute(0, 2, 1),
                                self.pointB.permute(0, 2, 1))

        self.z_A2B, self.z_B2A = self.fuse(self.z_s_B, self.z_c_A, self.z_s_A, self.z_c_B, self.cls2, self.cls1)
        self.loss_vae = self.fuse.get_loss()
        self.recon_A2B = self.decoder(self.z_A2B)
        self.recon_B2A = self.decoder(self.z_B2A)

        self.z_A, self.z_B = self.fuse(self.z_s_A, self.z_c_A, self.z_s_B, self.z_c_B, self.cls1, self.cls2)
        self.loss_vae += self.fuse.get_loss()
        self.recon_A = self.decoder(self.z_A)
        self.recon_B = self.decoder(self.z_B)

        self.recon_c_A, self.recon_c_B = self.enc_c(self.recon_A2B.permute(0, 2, 1),
                                    self.recon_B2A.permute(0, 2, 1))
        self.recon_s_A, self.recon_s_B = self.enc_s(self.recon_B2A.permute(0, 2, 1),
                                    self.recon_A2B.permute(0, 2, 1))

        self.loss_re2_eval = torch.mean(self.fscore(self.recon_A, self.pointA) / 2 +  self.fscore(self.recon_B, self.pointB) / 2).item()

    def one_hot(self, label, class_num):
        m_zeros = torch.zeros(1, class_num)
        onehot = m_zeros.scatter_(1, label, 1)  # (dim,index,value)
        return onehot

    def backward_d(self, pointB, pointA, cls_r, cls_f):
        pred_fake = self.dis_realB(pointA.detach())
        pred_real = self.dis_realB(pointB.detach())
        zeros = torch.zeros_like(pred_real, device=pointA.device)
        real = zeros.scatter(1, torch.tensor(cls_r, device=pointA.device).unsqueeze(-1), 1)
        pred_real = torch.sum(real * pred_real, dim=1)
        pred_fake = torch.sum(real * pred_fake, dim=1)
        all_ones = torch.ones_like(pred_real, device=pointA.device)
        ad_true_loss = nn.functional.binary_cross_entropy(pred_real, all_ones)
        ad_fake_loss = nn.functional.binary_cross_entropy(pred_fake, all_ones * 0)
        loss_D = (ad_true_loss + ad_fake_loss) / 10
        loss_D.backward(retain_graph=True)
        return loss_D
    
    def update_all(self, it=0):
        # D
        self.forward_all()
        self.dis_cont_opt.zero_grad()
        self.backward_D()
        self.dis_cont_opt.step()

        # E
        self.c_opt.zero_grad()
        self.s_opt.zero_grad()
        self.decoder_opt.zero_grad()
        self.fuse_opt.zero_grad()
        self.backward_EG()
        self.backward_encoder()
        self.c_opt.step()
        self.s_opt.step()
        self.fuse_opt.step()
        self.decoder_opt.step()

    def backward_encoder(self):
        # latent consistency loss
        mse = torch.nn.MSELoss()
        loss_cont = mse(self.z_c_A, self.recon_c_A) + \
                    mse(self.z_c_B, self.recon_c_B)
        loss_stl = mse(self.z_s_A, self.recon_s_A) + \
                   mse(self.z_s_B, self.recon_s_B)
        loss_encoder = (loss_stl + loss_cont) / 10
        loss_encoder.backward(retain_graph=True)
        self.loss_encoder = loss_encoder.item()

    def backward_EG(self):
        # 两个对抗性损失
        outs_c = self.dis_cont(torch.cat([self.z_c_A, self.z_c_B], dim=0))
        all_zeros = torch.zeros_like(outs_c).cuda(self.gpu) + 1 / outs_c.shape[1]
        loss_dis_c = F.binary_cross_entropy(outs_c, all_zeros)
        # 融合vae
        loss_vae = self.loss_vae

        z_c = torch.cat([self.z_c_A, self.z_c_B], dim=0)
        std = torch.std(z_c, dim=1, keepdim=True)
        mu = torch.mean(z_c, dim=1, keepdim=True)
        log_var = torch.log(std**2)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - std ** 2 , dim=1), dim=0)

        loss_pointA1, loss_pointA2 = self.distChamfer(self.recon_A, self.pointA)
        loss_pointB1, loss_pointB2 = self.distChamfer(self.recon_B, self.pointB)
        loss_dist = torch.mean((loss_pointA1 + loss_pointB1) * 0.5 + (loss_pointA2 + loss_pointB2) * (1 - 0.5))

        loss_eg = loss_dist * 5 + kld_loss + loss_vae * 0.01 + loss_dis_c
        loss_eg.backward(retain_graph=True)

        self.loss_dist = loss_dist.item()
        self.loss_dis_c = loss_dis_c.item()
        self.loss_kld = kld_loss.item()
        self.loss_vae = loss_vae.item()

    
    
        
    def backward_D(self):
        outs_c_A = self.dis_cont(self.z_c_A, self.idx_randD)
        outs_c_B = self.dis_cont(self.z_c_B, self.idx_randD)
        all_zeros = torch.zeros_like(outs_c_A, device=self.pointA.device)
        realA = all_zeros.scatter(1, torch.tensor(self.cls1, device=self.pointA.device).unsqueeze(-1), 1)
        realB = all_zeros.scatter(1, torch.tensor(self.cls2, device=self.pointA.device).unsqueeze(-1), 1)

        loss_dis_cont = F.binary_cross_entropy(outs_c_A, realA) + F.binary_cross_entropy(outs_c_B, realB)

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
