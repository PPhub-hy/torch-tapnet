# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 20:44:36 2020

@author: Hongyu Li
"""

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch_Tapnet import calculate_accuracy, test_model
import os
import math

# Training settings
no_cuda =False
#  seed = 0

cuda = not no_cuda and torch.cuda.is_available()

#  torch.manual_seed(seed)
#  if cuda:
#    torch.cuda.manual_seed(seed)


def train_dan(model, use_pseudo_label,
              s_data_generator, t_data_generator, target_dataset, test_subsets,
              Iters, print_frequency, learning_rate, mome, w_d, save_model):
    correct = 0
    alpha = 0
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': learning_rate},
        ], lr=learning_rate, momentum=mome, weight_decay=w_d)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, Iters)
    #  scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 500, T_mult = 2)
    for i in range(1, Iters+1):
        model.train()
        scheduler.step()
        
        src_data, src_label = next(s_data_generator)  
        tgt_data, tgt_label = next(t_data_generator)
        
        src_data = torch.from_numpy(src_data).float().cuda()
        tgt_data = torch.from_numpy(tgt_data).float().cuda()
        src_label = torch.Tensor(src_label).long().cuda()
        tgt_label = torch.Tensor(tgt_label).long().cuda()
        
        #  if cuda:
        #    src_data, src_label = src_data.cuda(), src_label.cuda()
        #    tgt_data = tgt_data.cuda()
        optimizer.zero_grad()
        src_pred, mmd_loss = model(src_data, tgt_data)
        tgt_pred, _ = model(tgt_data, tgt_data)
        cls_loss = criterion(src_pred, src_label)

        #  lambd = 2 / (1 + math.exp(-10 * i / Iters)) - 1
        lambd = 1
        loss = cls_loss + lambd * mmd_loss
        if use_pseudo_label:
            pseudo_labels = tgt_pred.argmax(dim=1)
            pseudo_cls_loss = criterion(tgt_pred, pseudo_labels)
            if int(Iters * 0.1) <= i <= int(Iters * 0.2):
                alpha = (i - int(Iters * 0.1)) / int(Iters * 0.1)
            loss = loss + alpha * pseudo_cls_loss
        #  mmd_loss.backward(retain_graph = True)
        loss.backward()
        optimizer.step()
        
        if i % print_frequency == 0:
            print('step {}:'.format(i))
            accuracy_source = calculate_accuracy(src_pred, src_label)
            accuracy_target = calculate_accuracy(tgt_pred, tgt_label)

            print(">>> lr: {:.4f}".format(optimizer.param_groups[1]['lr']))
            print(">>> lambd: {:.4f}".format(lambd))
            print(">>> alpha: {:.4f}".format(alpha))
            print('>>> loss_cls: {:.3f}'.format(cls_loss))
            print('>>> loss_mmd: {:.3f}'.format(mmd_loss))
            print('>>> accuracy: (s:{:.3f};t:{:.3f})'.format(accuracy_source, accuracy_target))
        
        if i % (print_frequency*5) == 0:
            t_correct = test_dan(model, target_dataset, test_subsets, 500)
            if t_correct > correct:
                if os.path.exists(save_model.format(round(correct, 4)) + '_model.pkl'):
                    os.remove(save_model.format(round(correct, 4)) + '_model.pkl')
                correct = t_correct
                torch.save(model, save_model.format(round(correct, 4)) + '_model.pkl')
                print("model saved as", save_model.format(round(correct, 4)) + '_model.pkl')

            print('current acc: {: .2f} max acc: {: .2f}'.format(t_correct, 100*correct))
    return correct


def test_dan(net, dataset, test_subsets, batchsize):
    net.eval()
    with torch.no_grad():
        print('testing.............')
        test_x_batches, test_y_batches = dataset.get_test_data_batches(test_subsets, batchsize)
        test_accuracys = []
        test_samples = []
        for idx in range(len(test_x_batches)):
            test_x = test_x_batches[idx]
            test_y = test_y_batches[idx]
            test_x = torch.from_numpy(test_x).float()
            test_x = test_x.cuda()
            test_output, _ = net(test_x)
        
            test_y = torch.Tensor(test_y).long()
            test_y = test_y.cuda()
            test_accuracys.append(calculate_accuracy(test_output, test_y))
            test_samples.append(len(test_y))
        
        #  print(test_samples)
        test_accuracy = np.sum([test_accuracys[i] * test_samples[i] for i in range(len(test_accuracys))]) / np.sum(test_samples)
    net.train()
    return test_accuracy


if __name__ == '__main__':
    model = models.DANNet(num_classes=31)
    print(model)
    if cuda:
        model.cuda()
    train_dan(model)
