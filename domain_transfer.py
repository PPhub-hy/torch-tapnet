# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:09:22 2020

@author: Hongyu Li
"""

import numpy as np
import torch
import random
from torch_Tapnet import train_model, test_model
from data.tapnet_dataset import Tapnet_Dataset
from lib.networks import load_net
from lib.networks import plainCNN, ResNet
from lib.networks import DANNet
from lib.densenet import DenseNet_vib as DenseNet
from transfer.DAN import train_dan, test_dan

Learning_Rate = 0.01
Weight_Decay = 1e-4
Momentum = 0.9
Iters = 50000
BatchSize = 64
BatchSize_test = 50
print_frequency = 200
# 准备数据
data_file = 'data/Coin_tap_data_{}.pkl'
materials = ['GFRP','Al','Acrylic']
split_times = 3
subsets = [subset for subset in range(split_times)]
classes = 7
target_labeled_rate = 0 # %

Used_encoders = [#['plainCNN', 64, 0.01],
                 #['ResNet', 64, 0.005],
                 ['DenseNet', 105, 0.01]]

def train_test_without_DT():

    if not target_labeled_rate == 0:
        split_times = 100 // target_labeled_rate
    else:
        split_times = 3

    test_accuracy = {}
    for target_domain in materials:

        source_domain = [domain for domain in materials if not domain == target_domain]
        target_domain = [target_domain]
        print('source domain = ', source_domain)
        print('target domain = ', target_domain)

        source_dataset = Tapnet_Dataset(source_domain,
                                        data_file=data_file,
                                        fine_label=fine_label,
                                        split_times=split_times)
        target_dataset = Tapnet_Dataset(target_domain,
                                        data_file=data_file,
                                        fine_label=fine_label,
                                        split_times=split_times)
        if not target_labeled_rate == 0:
            for _ in range(split_times * 10):
                source_dataset.subsets.append(target_dataset.subsets[0])
            source_dataset.split_subsets()
            target_dataset.subsets.pop(0)

        data_generator = source_dataset.batch_generator(subsets, batch_size = BatchSize)

        #net = plainCNN(classes = classes)
        net = DenseNet(classes)
        print(net)

        net.apply(net.weights_init)

        train_model(net, target_dataset, subsets, data_generator, BatchSize, Iters, print_frequency,
                    learning_rate = Learning_Rate, mome = Momentum, w_d = Weight_Decay)

        accuracy = test_model(net, target_dataset, subsets, BatchSize_test)

        test_accuracy['{}-->{}'.format(source_domain, target_domain)] = accuracy
        print('accuracy {}-->{} = '.format(source_domain, target_domain), accuracy)

        model_name = 'models/DAN/{}/'.format(Used_encoders[0][0]) + '{}_noTransfer_{}-->{}'.format(Used_encoders[0][0], source_domain, target_domain) + '_acc{}'

        torch.save(net, model_name.format(round(accuracy, 4)) + '_model.pkl')
        print("model saved as", model_name.format(round(accuracy, 4)) + '_model.pkl')
    
    print(test_accuracy)

def train_DAN():
    Use_pseudo_label = True  #False #

    print(materials)
    print('PLL:{}'.format(Use_pseudo_label))
    sigma = 1
    kernel_num = 5

    if not target_labeled_rate == 0:
        split_times = 100 // target_labeled_rate
    else:
        split_times = 3

    test_ACCURACYs = {}
    for Used_encoder in Used_encoders:

        test_accuracy = {}
        test_std = {}

        for target_domain in materials:

            source_domain = [domain for domain in materials if not domain==target_domain]
            target_domain = [target_domain]
            print('source domain = ', source_domain)
            print('target domain = ', target_domain)

            source_dataset = Tapnet_Dataset(source_domain,
                                            data_file=data_file,
                                            split_times=split_times)
            target_dataset = Tapnet_Dataset(target_domain,
                                            data_file=data_file,
                                            split_times=split_times)

            if not target_labeled_rate == 0:
                for _ in range(split_times * 10):
                    source_dataset.subsets.append(target_dataset.subsets[0])
                source_dataset.split_subsets()
                target_dataset.subsets.pop(0)

            best_acc = []
            target_acc = []
            source_acc = []
            for i in range(3):#split_times):
                if not target_labeled_rate == 0:
                    test_subsets = [n for n in range((split_times - 1)//3*3) if n % 3 == i]
                    train_subsets = [subset for subset in range(split_times - 1) if subset not in test_subsets]
                else:
                    test_subsets = [n for n in range(split_times // 3 * 3) if n % 3 == i]
                    train_subsets = [subset for subset in range(split_times) if subset not in test_subsets]
                print(test_subsets)

                if Used_encoder[0] == 'plainCNN':
                    net_encoder = plainCNN(classes)
                elif Used_encoder[0] == 'ResNet':
                    net_encoder = ResNet(classes)
                elif Used_encoder[0] == 'DenseNet':
                    net_encoder = DenseNet(classes)
                n_features = Used_encoder[1]
                #net_encoder.apply(net_encoder.weights_init)
                danNet = DANNet(net_encoder, n_features, num_classes=classes,
                                kernel_num=kernel_num, fix_sigma=sigma)
                danNet.cuda()

                s_data_generator = source_dataset.batch_generator(subsets, batch_size=BatchSize)
                t_data_generator = target_dataset.batch_generator(train_subsets, batch_size=BatchSize)

                save_path = 'models/DAN/{}/'.format(Used_encoder[0])
                filename = '{}_{}_to_{}_bestACC'.format(Used_encoder[0], source_domain,target_domain)
                save_model = save_path + filename + '{}'

                b_acc = train_dan(danNet, Use_pseudo_label,
                                  s_data_generator, t_data_generator, target_dataset, test_subsets,
                                  Iters=Iters, print_frequency=print_frequency,
                                  learning_rate=Used_encoder[2], mome=Momentum, w_d=Weight_Decay,
                                  save_model=save_path + filename + '{}')
                t_acc = test_dan(danNet, target_dataset, test_subsets, BatchSize_test)
                s_acc = test_dan(danNet, source_dataset, subsets, BatchSize_test)

                torch.save(danNet, save_model.format(round(t_acc, 4)) + '_Final_model.pkl')
                print("model saved as", save_model.format(round(t_acc, 4)) + '_Final_model.pkl')

                best_acc.append(b_acc)
                target_acc.append(t_acc)
                source_acc.append(s_acc)

            test_accuracy['{}-->{} [source, target, best]'.format(source_domain, target_domain)] = [
                np.mean(source_acc), np.mean(target_acc),
                np.mean(best_acc)]
            test_std['{}-->{} [source, target, best]'.format(source_domain, target_domain)] = [
                np.std(source_acc), np.std(target_acc),
                np.std(best_acc)]
            print(test_accuracy['{}-->{} [source, target, best]'.format(source_domain, target_domain)])
            print(test_std['{}-->{} [source, target, best]'.format(source_domain, target_domain)])

        print(test_accuracy)
        test_ACCURACYs[Used_encoder[0]] = test_accuracy
    print(test_ACCURACYs)


if __name__ == '__main__':

    # train_test_without_DT()
    # Iters = 15500
    train_DAN()
