# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:18:52 2019

@author: Hongyu Li
"""

"""
Load CIFAR-10 dataset and generate batch-wise data stream for training image classification task.
"""

import os
import numpy as np
import pickle
import random
import torch
import math

class Tapnet_Dataset:
    def __init__(self,
                 material=None,
                 data_file='tapdata.pkl',
                 fine_label=True,
                 mix_up=True,
                 split_times=3,
                 rescale=None,
                 reverse=None):
        
        self.material = material
        self.data_file = data_file
        self.split_times = split_times
        self.fine_label = fine_label
        self.num_labeled = [0, 0, 0, 0, 0, 0, 0]
        self.num_total = 0
        self.subsets = []
        self.max_negative = 3000
        self.rescale = rescale
        self.reverse = reverse

        self.load_all_data()
        if mix_up:
            self.split_subsets()
    
    def load_all_data(self):
        """
        Load all the data from CIFAR-10 dataset.
        """

        if self.material is None:
            self.material = 'GFRP'#, 'Acrylic', 'Al']

        for material in self.material:
            # Load meta information
            with open(self.data_file.format(material), 'rb') as f:
                tapdata = pickle.load(f)
            # Load data
            for i in range(len(tapdata[material])):
                self.subsets.append([])
            for sample_idx in tapdata[material].keys():
                random.shuffle(tapdata[material][sample_idx])
                negative_num = 0
                for sample in tapdata[material][sample_idx]:
                    if sample['label'] == 0: #负样本统计
                        negative_num += 1
                    elif not self.fine_label: #二分类处理
                        sample['label'] = 1

                    if self.rescale is not None:
                        sample['data'] = torch.from_numpy(sample['data']).unsqueeze(dim=0).unsqueeze(dim=0)
                        sample['data'] = torch.nn.functional.interpolate(sample['data'], scale_factor=self.rescale,
                                                                         mode='linear', align_corners=False)
                        if self.reverse:
                            sample['data'] = torch.nn.functional.interpolate(sample['data'], scale_factor=1 / self.rescale,
                                                                             mode='linear', align_corners=False)
                        sample['data'] = np.array(sample['data'].squeeze())

                    self.subsets[int(sample_idx) - 1].append(sample)
                    self.num_labeled[sample['label']] += 1
                    self.num_total += 1

        self.sample_num = [len(subset) for subset in self.subsets]

        print('=' * 10, 'Dataset Constructed', '=' * 10)
        print('- included materials: ', self.material)
        print('- all data: %d' % self.num_total)
        print('- number of labeled classes: ', self.num_labeled)
        print('- subset number: %d' % len(self.subsets))
        print('- samples in subsets: ', self.sample_num)
        print('- rescale = ', self.rescale, '; reverse = ', self.reverse)
        print(' ')

    def split_subsets(self):
        """
        Split part of training data into validation set.
        """
        self.data_list = np.concatenate(self.subsets)

        random.shuffle(self.data_list)
        num_subset = math.floor(len(self.data_list) / self.split_times)
        # Split
        self.subsets = []

        for i in range(self.split_times):
            if i + 1 == self.split_times:
                self.subsets.append(self.data_list[num_subset * i: max(num_subset * (i + 1), len(self.data_list))])
            else:
                self.subsets.append(self.data_list[num_subset * i: num_subset * (i + 1)])

        self.sample_num = [len(subset) for subset in self.subsets]
        print('- all data: %d' % (len(self.data_list)))
        print('- subset number: %d' % (self.split_times))
        print('- samples in subsets: ', self.sample_num)
        print(' ')
    
    def batch_generator(self, Subsets, 
                        batch_size = 16,
                        shake = False, 
                        cutout = False,
                        noise=False,
                        noise_SNR=None):
        """
        A generator to provide batch-wise data stream for training image classification task.
        """
        data = [self.subsets[subset] for subset in Subsets]
        data = np.concatenate(data, axis=0)
        
        print('data generater seted, used subset: ', Subsets)
        print ('shake = ', shake)
        print ('cutout = ', cutout)
        print ('noise = ', noise)
        print ('noise_SNR = ', noise_SNR)
        if self.fine_label:
            print('Using fine labels.......')
        else:
            print('Using binary labels.......')
        
        # Record the traverse of data 
        data_idx = -1
        while True:
            # Construct batch data containers
            batch_data = np.zeros((batch_size, 1, 2048), np.float32)
            if self.rescale is not None and self.reverse == False:
                batch_data = np.zeros((batch_size, 1, int(2048 * self.rescale)), np.float32)
            elif self.rescale is not None and self.reverse == True:
                batch_data = np.zeros((batch_size, 1, int(int(2048 * self.rescale) /self.rescale)), np.float32)
            batch_gt = np.zeros((batch_size), np.float32)
            
            i = 0
            negative_count = 0
            while i < batch_size:
                data_idx = (data_idx + 1) % len(data)
                # Shuffle training data at the start of a new traverse
                if data_idx == 0:
                    random.shuffle(data)

                while negative_count >= batch_size // 3 and data[data_idx]['label'] == 0:
                    data_idx = (data_idx + 1) % len(data)
                    if data_idx == 0:
                        random.shuffle(data)

                # Load data into batch
                this_data = data[data_idx]['data']

                if noise:
                    this_data, snr, arg = add_noise(this_data, noise_SNR)

                #if split == 'train':
                if shake:
                    # pad the signal
                    this_data = np.pad(this_data, ((512,0),(0,0)), mode='constant')
                    # shake the img
                    randx = random.randint(0,512)
                    #print(randx, ' ' ,randy)
                    this_data = this_data[randx:randx+2048, :]
                
                if cutout:
                    this_data = self.img_cutout(this_data)

                batch_data[i] = this_data
                batch_gt[i] = data[data_idx]['label']
                if self.fine_label and batch_gt[i] == 0:
                    negative_count += 1

                i += 1
            yield batch_data, batch_gt
    
    def get_test_data_batches(self, Subsets, batchsize, noise=False, noise_SNR=None):
        datas = [self.subsets[subset] for subset in Subsets]
        datas = np.concatenate(datas, axis=0)
        random.shuffle(datas)
        test_data_batches = []
        test_label_batches = []
        test_datas = []
        test_labels = []
        batch_counter = 0
        for idx in range(len(datas)):
            data = datas[idx]['data'][np.newaxis, :]
            if noise:
                data, snr, arg = add_noise(data, noise_SNR)
            # Normalize
            #data = (data - self.mean) / self.std
            #img = img.astype(np.float32)/128 - 1
            test_datas.append(data)
            # Background = 0, object = 1 ~ 10
            '''
            if fine_label:
                label = datas[idx]['fine_label']
            else:
                label = datas[idx]['label']
            '''
            label = datas[idx]['label']
            test_labels.append(label)
            batch_counter += 1
            if batch_counter == batchsize:
                test_data_batches.append(np.array(test_datas))
                test_label_batches.append(np.array(test_labels))
                test_datas = []
                test_labels = []
                batch_counter = 0
        if len(test_datas) > 0:
            test_data_batches.append(np.array(test_datas))
            test_label_batches.append(np.array(test_labels))
        
        print('='*10, 'test data prepared, used subset: ', Subsets, '='*10)
        print ('- noise = ', noise)
        print ('- noise_SNR = ', noise_SNR)
        print(' ')
        
        return test_data_batches, test_label_batches
  
def add_noise(signal, SNR = 0, return_noise = False):
    # 给数据加指定SNR的高斯噪声
    noise = np.random.randn(signal.shape[0],signal.shape[1]) 	#产生N(0,1)噪声数据
    noise = noise-np.mean(noise) 								#均值为0
    signal_power = np.linalg.norm( signal )**2 / signal.size	#此处是信号的std**2
    noise_variance = signal_power/np.power(10,(SNR/10))         #此处是噪声的std**2
    noise_arg = np.sqrt(noise_variance) / np.std(noise)
    noise = noise_arg*noise    ##此处是噪声的std**2
    signal_noise = noise + signal
    
    Ps = ( np.linalg.norm(signal - signal.mean()) )**2          #signal power
    Pn = ( np.linalg.norm(signal - signal_noise ) )**2          #noise power
    snr = 10*np.log10(Ps/Pn)
    if return_noise:
        return signal_noise, noise, snr, noise_arg
    else:
        return signal_noise, snr, noise_arg
