# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:08:50 2019

@author: Hongyu Li
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import transfer.MMD as mmd

def load_net(save_path):
    #net = Net()
    net = torch.load(save_path)
    return net

class Net(nn.Module):
    def __init__(self):
         super().__init__()

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    def save_model(self, save_path, filename = None):
        if filename == None:
            time_now = time.strftime('%y-%m-%d_%H-%M-%S',time.localtime(time.time()))
            print("model saved at", save_path + time_now + '_model.pkl')
            torch.save(self, save_path + time_now + '_model.pkl')
        else:
            torch.save(self, save_path + filename + '_model.pkl')
            print("model saved at", save_path + filename + '_model.pkl')
        
    def weights_init(m, leaky_a=0.1):
        if isinstance(m, nn.Conv1d):
            #nn.init.xavier_normal_(m.weight, gain = nn.init.calculate_gain('leaky_relu', leaky_a))
            nn.init.kaiming_normal_(m.weight, a=leaky_a)
            #nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            #nn.init.xavier_normal_(m.weight, gain = nn.init.calculate_gain('leaky_relu', leaky_a))
            nn.init.kaiming_normal_(m.weight, a=leaky_a)
            nn.init.constant_(m.bias, 0)
        
class TICNN(Net):
    def __init__(self, channels = 16, DW_layers = 5, classes = 2, iters = 5000):
        super().__init__()
        self.iters = iters
        self.channels = channels
        self.classes = classes
        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=64, stride=8, padding=31, bias=False),
                                        nn.MaxPool1d(kernel_size = 2, padding=0),
                                        nn.ReLU(channels),
                                        nn.BatchNorm1d(num_features=channels)))
        self.convs.append(self.act_conv(channels, channels*2))
        self.convs.append(self.act_conv(channels*2, channels*4))
        
        for i in range(DW_layers - 3):
            self.convs.append(self.act_conv(channels*4, channels*4))
        '''
        self.convs.append(nn.Sequential(nn.ReLU(channels*4),
                                        nn.BatchNorm1d(num_features=channels*4),
                                        nn.Conv1d(in_channels=channels*4, out_channels=self.classes, kernel_size=1, stride=1, padding=0, bias=False)
                                        )
                          )'''
        self.convs.append(self.act_conv(channels*4, channels*4, padding = 0))
        self.fc1 = nn.Linear(in_features=channels*4*3, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=self.classes)
     
    def act_conv(self, in_channels, out_channels, padding=1):
        return nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding, bias=False),
                             nn.MaxPool1d(kernel_size = 2, padding=0),
                             nn.ReLU(out_channels),
                             nn.BatchNorm1d(num_features=out_channels)
                             )
    
    def forward(self, x, all_tensors = False):
        #输入直接dropout，滤除噪声
        #x = F.dropout(x, p = 0.5, training=self.training)
        if all_tensors:
            out_tensors = []
            out_tensors.append(x)
        for conv in self.convs:
            x = conv(x)
            if all_tensors:
                out_tensors.append(x)
        x = x.view(-1, 64*3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(x, dim=1)
        
        if all_tensors:
            return out_tensors
        else:
            return x

class ResNet(Net):
    def __init__(self, classes = 2, channels = 8):
        super().__init__()
        self.classes = classes
        self.convs = nn.ModuleList()
        self.channels = [channels, channels*2, channels*4] + [channels*8 for i in range(4)]
        self.kernels = [15, 7, 7] + [3 for i in range(3)]
        self.paddings = [7, 3, 3] + [1 for i in range(3)]

        first_channels = channels
        
        self.firstconv = nn.Conv1d(in_channels=1, out_channels=first_channels, kernel_size=31, 
                                stride=1, padding=15, bias=False)
        
        for idx in range(len(self.channels)-1):
            in_channel = self.channels[idx]
            out_channel = self.channels[idx + 1]
            self.convs.append(self.act_conv(in_channels=in_channel, 
                                        out_channels=out_channel, 
                                        kernel=self.kernels[idx],
                                        padding=self.paddings[idx]))
        
        self.conv11 = nn.Sequential(nn.BatchNorm1d(num_features=self.channels[-1]),
                                     nn.ReLU(self.channels[-1]),
                                     nn.Conv1d(in_channels=self.channels[-1], out_channels=self.classes, kernel_size=1, 
                                                  stride=1, padding=0, bias=True)
                                     )
        self.fine_tune_layers = nn.ModuleList()
        self.fine_tune_layers += self.convs[2:]
        self.fine_tune_layers.append(self.conv11)
                                    
    def act_conv(self, in_channels, out_channels, kernel, padding, stride = 1):
        return nn.Sequential(nn.BatchNorm1d(in_channels),
                             nn.ReLU(in_channels),
                             nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, 
                                       stride=stride, padding=padding, bias=False),
                             )

    def forward(self, x, all_tensors = False, give_features = False):
        #输入直接dropout，滤除噪声
        #x = F.dropout(x, p = 0.5, training=self.training)
        if all_tensors:
            out_tensors = []
            out_tensors.append(x)
        
        x = self.firstconv(x)
        x = F.max_pool1d(x, kernel_size = 2)
        identity = x
        for idx in range(len(self.convs)):
            if all_tensors:# and not idx == len(self.convs) - 1:
                out_tensors.append(x)
            x = self.convs[idx](x)
            x = F.max_pool1d(x, kernel_size=2)
            if idx % 2 == 1:
                pad_dim = self.channels[idx+1] - self.channels[idx-1]
                if not pad_dim == 0:
                    identity = F.pad(identity, (0, 0, 0, pad_dim))
                mask = torch.arange(identity.shape[2]) % 4 == 0
                mask = mask.expand(*identity.shape)
                x = identity[mask].reshape(*x.shape) + x
                identity = x

            #print(x.shape)
                
        x = F.avg_pool1d(x, kernel_size = 16)
        #x = F.dropout(x, p=0.5, training=self.training)
        if all_tensors:
            out_tensors.append(x.squeeze(-1))
        if give_features:
            return x.squeeze(-1)
        x = self.conv11(x)
        x = x.squeeze(-1)
        if all_tensors:
            out_tensors.append(x)
            return out_tensors
        x = F.log_softmax(x, dim=1)
        return x
       
class plainCNN(Net):
    def __init__(self, classes = 2, channels = 8,
                 Use_wavelet = False, Multi_scales = True, Wavelet_degree = 4, Wave = ['sym4'], replace = False):
        super().__init__()
        self.classes = classes
        self.convs = nn.ModuleList()
        self.use_wavelet = Use_wavelet
        self.wavelet_degree = Wavelet_degree
        self.wave = Wave
        self.multi_scales = Multi_scales
        self.channels = [channels, channels*2, channels*4] + [channels*8 for i in range(4)]
        self.kernels = [15, 7, 7] + [3 for i in range(3)]
        self.paddings = [7, 3, 3] + [1 for i in range(3)]
        wl_layer_idx = 1 
        if Use_wavelet and wl_layer_idx <= Wavelet_degree and self.multi_scales:
            first_channels = channels #- 2 ** wl_layer_idx
            #wl_layer_idx += 1
            if replace:
                first_channels = channels - 2 ** wl_layer_idx * len(self.wave)
                wl_layer_idx += 1
        else:
            first_channels = channels
        self.convs.append(nn.Sequential(nn.Conv1d(in_channels=1, out_channels=first_channels, kernel_size=31, 
                                                  stride=1, padding=15, bias=False),
                                        nn.MaxPool1d(kernel_size = 2, padding=0),
                                        nn.BatchNorm1d(num_features=first_channels),
                                        nn.ReLU(first_channels)
                                        ))
        
        
        for idx in range(len(self.channels)-1):
            if self.multi_scales and Use_wavelet and wl_layer_idx <= Wavelet_degree\
                or Use_wavelet and wl_layer_idx == Wavelet_degree:
                in_channel = self.channels[idx] + 2 ** wl_layer_idx * len(self.wave)
                out_channel = self.channels[idx + 1] 
                if replace:
                    in_channel = self.channels[idx]
                    out_channel = self.channels[idx + 1] - 2 ** wl_layer_idx * len(self.wave)
            else:
                in_channel = self.channels[idx]
                out_channel = self.channels[idx + 1]
            wl_layer_idx += 1
            self.convs.append(self.act_conv(in_channels=in_channel, 
                                        out_channels=out_channel, 
                                        kernel=self.kernels[idx],
                                        padding=self.paddings[idx]))
        self.conv11 = nn.Conv1d(in_channels=self.channels[-1], out_channels=self.classes, kernel_size=1, 
                                                  stride=1, padding=0, bias=True)
        self.fine_tune_layers = nn.ModuleList()
        self.fine_tune_layers += self.convs[3:]
        self.fine_tune_layers.append(self.conv11)
     
    def act_conv(self, in_channels, out_channels, kernel, padding, stride = 1):
        return nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, 
                                       stride=stride, padding=padding, bias=False),
                             nn.MaxPool1d(kernel_size = 2, padding=0),
                             nn.BatchNorm1d(num_features=out_channels),
                             nn.ReLU(out_channels)
                             )
        
    def arrange_wl_features(self, WP_features):
        WP_features_tensor = []
        for sample_idx in range(len(WP_features)):
            WP_features_tensor_layers=[]
            for layer_idx in range(len(WP_features[sample_idx])):
                if self.multi_scales or layer_idx + 1 == self.wavelet_degree:
                    tensors_in_this_layer = torch.cat(tuple([torch.from_numpy(channel_features).unsqueeze(0) \
                                                         for channel_features in WP_features[sample_idx][layer_idx]]), 0)
                else:
                    tensors_in_this_layer = torch.Tensor([])  
                WP_features_tensor_layers.append(tensors_in_this_layer)
            WP_features_tensor.append(WP_features_tensor_layers)
        WP_features_tensor_rerange = []
        for layer_idx in range(len(WP_features_tensor[0])):
            if self.multi_scales or layer_idx + 1 == self.wavelet_degree:
                WP_features_tensor_rerange.append(torch.cat(tuple([WP_features_tensor[sample_idx][layer_idx].unsqueeze(0) \
                                                         for sample_idx in range(len(WP_features_tensor))]), 0))
            else:
                WP_features_tensor_rerange.append(None)
        return WP_features_tensor_rerange

    def forward(self, x, all_tensors = False, give_features = False):
        #输入直接dropout，滤除噪声
        #x = F.dropout(x, p = 0.5, training=self.training)
        if all_tensors:
            out_tensors = []
            out_tensors.append(x)
        
        if self.use_wavelet:
            WP_features = []
            for sample in x:
                sample = sample.squeeze().cpu().numpy()
                WP_feature = wavelets(sample, n = self.wavelet_degree, multi_layers = True, wavelet = self.wave)
                WP_features.append(WP_feature)
            wl_features = self.arrange_wl_features(WP_features)
        
        for idx in range(len(self.convs)):
            if all_tensors and not idx == 0: #not idx == len(self.convs) - 1 and 
                out_tensors.append(x)
            x = self.convs[idx](x)
            if self.multi_scales and self.use_wavelet and idx < self.wavelet_degree \
                or self.use_wavelet and idx + 1 == self.wavelet_degree:
                x = torch.cat((x, wl_features[idx].cuda()),1)
            #print(x.shape)
                
        x = F.avg_pool1d(x, kernel_size = 16)
        #x = F.dropout(x, p=0.5, training=self.training)
        if all_tensors:
            out_tensors.append(x.squeeze(-1))
        if give_features:
            return x.squeeze(-1)

        x = self.conv11(x)
        x = x.squeeze(-1)
        if all_tensors:
            out_tensors.append(x)
            return out_tensors
        #x = x.view(-1, self.channels[-1]*2)
        x = F.log_softmax(x, dim=1)
        return x

class DANNet(Net):
    def __init__(self, sharednet, n_features, num_classes=5, kernel_num = None, fix_sigma = None):
        super().__init__()
        self.sharedNet = sharednet
        self.cls_fc = nn.Linear(n_features, num_classes)
        self.fix_sigma = fix_sigma
        self.kernel_num = kernel_num

    def forward(self, source, target = None, multi_layer = False, joint_muiti_layer = False, all_tensors = False):
        loss = 0
        source = self.sharedNet(source, all_tensors = all_tensors or multi_layer, give_features = not all_tensors and not multi_layer)
        if self.training == True:
            target = self.sharedNet(target, all_tensors = all_tensors or multi_layer, give_features = not all_tensors and not multi_layer)
            if not multi_layer:
                #loss += mmd.mmd_rbf_accelerate(source, target)
                loss += mmd.mmd_rbf_noaccelerate(source, target, kernel_num = self.kernel_num, fix_sigma=self.fix_sigma)
            else:
                last_n_layers = 2
                if joint_muiti_layer:
                    loss += mmd.jmmd_rbf_noaccelerate(source[len(source)-last_n_layers:], target[len(source)-last_n_layers:], kernel_num = self.kernel_num, fix_sigma=self.fix_sigma)
                else:
                    loss_list = []
                    for i in range(last_n_layers):
                        loss_list.append(mmd.mmd_rbf_accelerate(source[-1-i].view(source[0].shape[0], -1), target[-1-i].view(target[0].shape[0], -1), kernel_num = self.kernel_num, fix_sigma=self.fix_sigma))
                    for i in range(len(loss_list)):
                        K = loss_list[i] / sum(loss_list)
                        loss += K * loss_list
        
        #if all_tensors:
            #source.append(self.cls_fc(source[-1]))
        if not all_tensors:
            if multi_layer:
                source = self.cls_fc(source[-2])
            else:
                source = self.cls_fc(source)
            source = F.log_softmax(source, dim=1)
        #target = self.cls_fc(target)

        return source, loss