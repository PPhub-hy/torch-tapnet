# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:04:50 2019

@author: Hongyu Li
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:07:06 2019

@author: Hongyu Li
"""

import torch
import numpy as np
from data.tapnet_dataset import Tapnet_Dataset
from lib.networks import TICNN
from lib.networks import plainCNN
from lib.networks import ResNet
from lib import networks
from lib.densenet import DenseNet_vib as DenseNet
import torch.nn as nn
import torch.optim as optim

BatchSize_test = 300

def calculate_accuracy(output, target):
    predictions = output.argmax(dim=1)
    accuracy=[]
    for i in range(len(target)):
        if predictions[i] == target[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
    accuracy = torch.mean(torch.tensor(accuracy).float())
    return accuracy
  
def train_model(net, train_dataset, test_subsets, data_generator, BatchSize, Iters, print_frequency, learning_rate, mome, w_d):
    print(net.get_parameter_number())    
    net.train()
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=mome, weight_decay=w_d)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, Iters)
    
    net = net.cuda()
    for i in range(Iters):
        scheduler.step()
        train_batch_x, train_batch_y = next(data_generator)
        
        #正向传递计算输出
        train_batch_x = np.array(train_batch_x)
        train_batch_x = torch.from_numpy(train_batch_x).float()
        train_batch_x = train_batch_x.cuda()
        batch_output = net(train_batch_x)
        #计算损失函数
        train_batch_y = torch.Tensor(train_batch_y).long()
        train_batch_y = train_batch_y.cuda()
        loss = criterion(batch_output, train_batch_y)
        #梯度归零（否则累加）
        optimizer.zero_grad()
        #反向传递计算梯度
        loss.backward()
        #按照当前的梯度更新一轮变量
        optimizer.step()
        
        if i%print_frequency==0:
            print('step {}:'.format(i))
            accuracy_train = calculate_accuracy(batch_output, train_batch_y)
            accuracy_test = test_model(net, train_dataset, test_subsets, BatchSize_test)
        
            print(">>> lr: {:.4f}".format(optimizer.param_groups[0]['lr']))
            print('>>> loss: {:.3f}'.format(loss))
            print('>>> accuacy_train: {:.3f}'.format(accuracy_train))
            print('>>> accuacy_test: {:.3f}'.format(accuracy_test))
        
def test_model(net, dataset, test_subsets, batchsize):
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
            test_output = net(test_x)
        
            test_y = torch.Tensor(test_y).long()
            test_y = test_y.cuda()
            test_accuracys.append(calculate_accuracy(test_output, test_y))
            test_samples.append(len(test_y))

        test_accuracy = np.sum([test_accuracys[i] * test_samples[i] for i in range(len(test_accuracys))]) / np.sum(test_samples)
        
        #print('test accuracy = {}'.format(test_accuracy))
    net.train()
    return test_accuracy
       
if __name__ == '__main__':
    Learning_Rate = 0.01
    Weight_Decay = 1e-4
    Momentum = 0.9
    Iters = 50000
    BatchSize = 32
    print_frequency = 1000
    #prepare data
    data_file = 'data/Coin_tap_data_{}.pkl'
    materials = ['GFRP', 'Acrylic', 'Al']
    split_times = 3
    classes = 7
    
    ACCURACYs = {}
    STDs = {}

    for material in materials:
        train_dataset = Tapnet_Dataset([material],
                                       data_file = data_file,
                                       split_times = split_times)
        test_accuracy = []
        
        for i in range(split_times):
            train_subsets = [subset for subset in range(split_times) if subset != i]
            test_subsets = [i]

            data_generator = train_dataset.batch_generator(train_subsets, batch_size = BatchSize)

            #net = TICNN(classes = classes)
            #net = plainCNN(classes = classes)
            #net = ResNet(classes)
            net = DenseNet(classes)
            print(net)
            print(net.get_parameter_number())

            net.apply(net.weights_init)

            train_model(net, train_dataset, test_subsets, data_generator, BatchSize, Iters, print_frequency, learning_rate = Learning_Rate, mome = Momentum, w_d = Weight_Decay)

            test_accuracy += [test_model(net, train_dataset, test_subsets, BatchSize_test)]

        save_path = 'models/'
        filename = '{}-test-{}-{}'.format(material, round(np.mean(test_accuracy),4), round(np.std(test_accuracy),4))
        net.save_model(save_path, filename)
        print('accuracy = ', np.mean(test_accuracy))
        print('std = ', np.std(test_accuracy))
        ACCURACYs[material] = round(np.mean(test_accuracy),4)
        STDs[material] = round(np.std(test_accuracy),4)
    
    print('ACCURACYs: ', ACCURACYs)
    print('STDs: ', STDs)
