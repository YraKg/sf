from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from trainer import Trainer

import os
import numpy as np
import time
import torch
from torch.nn import functional as F
from utils import AverageMeter, mixup_data, mixup_criterion
from spgd import SparsePGD


class DynamicTrainerEpochs(Trainer):
    """ Dynamic trainer to adapt to multiplte Linf/L0 ratios """

    def __init__(self, model, optimizer, cfg, epsilon_k_list = [((16/255), 512), ((32/255), 256), ((64/255), 128)], summary_writer=None,
                 print_freq=1, output_freq=1, is_cuda=True, base_lr=0.1,
                 max_epoch=100, steps=[], rate=1., loss='adv', trades_beta=1., scheduler=None, mode='rand'):
        self.model = model
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.iter = 0
        self.epsilon_k_list = epsilon_k_list
        self.cfg = cfg
        self.print_freq = print_freq
        self.output_freq = output_freq
        self.is_cuda = is_cuda
        self.base_lr = base_lr
        self.max_epoch = max_epoch
        self.steps = steps
        self.rate = rate
        assert loss in ['adv', 'trades'], 'loss should be either adv or trades'
        self.loss = loss
        self.trades_beta = trades_beta
        self.get_lr_mults()
        self.scheduler = scheduler
        self.mode = mode

    def train(self, epoch, data_loader, subset_loader):
        self.model.train()

        batch_time = AverageMeter()
        adv_time = AverageMeter()
        loss_meter = AverageMeter()
        adv_loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        adv_acc_meter = AverageMeter()

        if self.scheduler is None:
            self.decrease_lr(epoch)

        end = time.time()

        attack_names = [f"Linf={eps}, L0={k}, Alpha={eps/8}" for (eps,k) in self.epsilon_k_list]

        for l, (eps, k) in enumerate(self.epsilon_k_list):

            attack = SparsePGD(self.model, epsilon=(eps/255), k=k, t=self.cfg.n_iters, alpha=self.cfg.alpha, beta=self.cfg.beta, patience=self.cfg.patience,
                    unprojected_gradient=True if self.cfg.train_mode == 'unproj' else False)
            
            max_loss = float('-inf')
            max_index = 0

            for (x,y) in subset_loader:
                if self.is_cuda:
                    x = x.cuda()
                    y = y.cuda()
                x_adv, _, _ = attack.perturb(x, y)
                if self.loss == 'adv':
                    adv_loss, adv_pred = self.adv_loss(x_adv, y)
                else:
                    adv_loss, adv_pred = self.trades_loss(x, x_adv, y)

                if max_loss < adv_loss:
                    max_loss = adv_loss
                    max_index = l

        attack = SparsePGD(self.model, epsilon=(self.epsilon_k_list[l][0]/255), k=self.epsilon_k_list[l][1], t=self.cfg.n_iters, alpha=self.cfg.alpha, beta=self.cfg.beta, patience=self.cfg.patience,
                    unprojected_gradient=True if self.cfg.train_mode == 'unproj' else False)
        
        print(f"Best performing attack: {attack_names[l]}")

        if self.mode == 'rand':
            if np.random.rand() < 1/2:
                attack.change_masking()

        for i, data in enumerate(data_loader):
            if self.mode == 'rand':
                if np.random.rand() < 1/2:
                    attack.change_masking()

            x, y = data
            if self.is_cuda:
                x = x.cuda()
                y = y.cuda()

            # Compute Adversarial Perturbations
            t0 = time.time()
            x_adv, _, _ = attack.perturb(x, y)
            adv_time.update(time.time() - t0)

            if self.loss == 'adv':
                adv_loss, adv_pred = self.adv_loss(x_adv, y)
            else:
                adv_loss, adv_pred = self.trades_loss(x, x_adv, y)

            self.optimizer.zero_grad()
            adv_loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            adv_loss_meter.update(adv_loss.item())
            adv_acc = self.accuracy(adv_pred, y)
            adv_acc_meter.update(adv_acc[0].item())

            self.scheduler.step()

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('adv_loss_iter', adv_loss_meter.val, self.iter)
                self.summary_writer.add_scalar('adv_acc_iter', adv_acc_meter.val, self.iter)

            if (i + 1) % self.output_freq == 0:
                with torch.no_grad():
                    pred = self.model(x)
                    loss = F.cross_entropy(pred, y)
                    loss_meter.update(loss.item())
                acc = self.accuracy(pred, y)
                acc_meter.update(acc[0].item())
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar('loss_iter', loss_meter.val, self.iter)
                    self.summary_writer.add_scalar('acc_iter', acc_meter.val, self.iter)

            if (i + 1) % self.print_freq == 0:
                p_str = "Epoch:[{:>3d}][{:>3d}|{:>3d}] Time:[{:.3f}/{:.3f}] " \
                        "Loss:[{:.3f}/{:.3f}] AdvLoss:[{:.3f}/{:.3f}] " \
                        "Acc:[{:.3f}/{:.3f}] AdvAcc:[{:.3f}/{:.3f}] ".format(
                    epoch, i + 1, len(data_loader), batch_time.val,
                    adv_time.val, loss_meter.val, loss_meter.avg,
                    adv_loss_meter.val, adv_loss_meter.avg, acc_meter.val,
                    acc_meter.avg, adv_acc_meter.val, adv_acc_meter.avg)
                print(p_str)

            self.iter += 1
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('loss_epoch', loss_meter.avg, epoch)
            self.summary_writer.add_scalar('adv_loss_epoch', adv_loss_meter.avg, epoch)
            self.summary_writer.add_scalar('acc_epoch', acc_meter.avg, epoch)
            self.summary_writer.add_scalar('adv_acc_epoch', adv_acc_meter.avg, epoch)
