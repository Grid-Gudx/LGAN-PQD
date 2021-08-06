# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:04:39 2021

@author: gdx
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    input_shape: batchsize * 1 * 640
    output_shape: batchsize * num_labels
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(32, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2)
        )

    def forward(self, x):
        x=x.float()
        x = self.stage1(x)
        return x
    
class Decoupling(nn.Module):
    '''
    input:
        feature map (batch*channel*location)
        label_feature (1*label_num*label_feature)    
    output:
        batch * label_num * d2 * location
    '''
    def __init__(self, num_classes, seq_feature_dim, label_feature_dim, d1=128, d2=128):
        super(Decoupling, self).__init__()
        self.num_classes = num_classes #PQD type == label num (8)
        self.seq_feature_dim = seq_feature_dim #conv channel num
        self.label_feature_dim = label_feature_dim #feature number of each label
        self.d1 = d1 #the dimensions of the joint embeddings
        self.d2 = d2 #the dimensions of the output features
        self.fc_1 = nn.Linear(self.seq_feature_dim, self.d1, bias=False)
        self.fc_2 = nn.Linear(self.label_feature_dim, self.d1, bias=False)
        self.fc_3 = nn.Linear(self.d1, self.d2)

    def forward(self,feature_map, word_features):
        # feature_map size: batch * channel * location
        convsize = feature_map.size()[2] #location
        batch_size = feature_map.size()[0]
        word_features = word_features[0] #label feature (8*8)
        
        feature_map = torch.transpose(feature_map, 1, 2) #output:  batch * location * channal
        f_wh_feature = feature_map.contiguous().view(batch_size*convsize, -1) #out: (batch * location) * channel
        f_wh_feature = self.fc_1(f_wh_feature).view(batch_size*convsize, 1, -1).repeat(1, self.num_classes, 1) #channel -> d1, out:(batch * location) * label_num * d1
        f_wd_feature = self.fc_2(word_features).view(1, self.num_classes, self.d1).repeat(batch_size*convsize,1,1) #label feature -> d1, out:(batch * location) * label_num * d1
        lb_feature = self.fc_3(torch.tanh(f_wh_feature*f_wd_feature).view(-1,self.d1)) #d1 -> d2, out: (batch * location * label_num) * d2  

        lb_feature = lb_feature.view(batch_size, convsize, self.num_classes, -1) #out: batch * location * label_num * d2
        lb_feature = torch.transpose(torch.transpose(lb_feature, 1, 2),2,3) #out: batch * label_num * d2 * location
        return lb_feature

#feature Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1   = nn.Conv1d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv1d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
#location Attention
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv1d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel, ratio):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class BRNN(nn.Module):
    def __init__(self,inputs=40,hidden=4):
        super(BRNN, self).__init__()
        self.rnn = nn.RNN(input_size=inputs,  # if use nn.RNN(), it hardly learns
                            hidden_size=hidden,     # hidden
                            num_layers=1,           # rnn layer
                            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
                            #dropout=0.2,
                            bidirectional = True
                          )
        self.lin = nn.Linear(2*hidden, 1) #, nn.ReLU(True))

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.lin(out)
        out = out.view(-1,8)
        return out

class cnn_atten_rnn(nn.Module):
    def __init__(self):
        super(cnn_atten_rnn, self).__init__()
        self.cnn = CNN()
        self.dec = Decoupling(num_classes=8, 
                              seq_feature_dim=128, 
                              label_feature_dim=8, 
                              d1=64, d2=64)
        
        self.cbam = CBAM(channel=64, ratio=8)
        self.cnn1d = nn.Sequential(nn.Conv1d(64,64,7,1,3),
                                   nn.ReLU(True),
                                   nn.BatchNorm1d(64),
                                   nn.MaxPool1d(2, 2),
                                   nn.Conv1d(64,64,7,1,3),
                                   nn.ReLU(True),
                                   nn.BatchNorm1d(64),
                                   nn.MaxPool1d(2, 2),
                                   nn.Conv1d(64, 64, 7, 1, 3),
                                   nn.ReLU(True),
                                   nn.BatchNorm1d(64),
                                   nn.AdaptiveAvgPool1d(1)                                  
                                   )
        self.brnn = BRNN(inputs=64, hidden=4)
        
    def forward(self, x, label_features):
        x = self.cnn(x)
        x = self.dec(x, label_features)        
        
        batch_size = x.size()[0] #batch
        class_num = x.size()[1] #class
        
        x = x.contiguous().view(batch_size*class_num,64,-1)
        x = self.cbam(x)
        x = self.cnn1d(x)
        x = x.view(batch_size,class_num,-1)
        x = self.brnn(x)
        x = torch.sigmoid(x)
        return x


