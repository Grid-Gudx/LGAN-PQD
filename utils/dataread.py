# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:07:58 2021

@author: gdx
"""

import torch
from torch.utils.data import Dataset
from scipy import io
import numpy as np

class Mc_dataread(Dataset):
    def __init__(self,file_path,data_name,label_num):
        self.x,self.y = self.data_read(file_path,data_name,label_num)
        self.len = len(self.x)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def data_read(self,file_path,data_name,label_num):
        data_path = file_path+data_name +'.mat'
        features_struct = io.loadmat(data_path)
        train_data = features_struct['data']
        #train_data = train_data.astype(np.float32)  #change the data type       
        lable_start = train_data.shape[1] - label_num
        x=train_data[:,0:lable_start]
        y=train_data[:,lable_start:]
        return self.shape_transform(x,y)
    
    def shape_transform(self,x,y):
        x=torch.from_numpy(x)
        y=torch.from_numpy(y.astype(np.int64))
        x=x.reshape((x.shape[0],1,-1)) #sample_num * 1 *sample_dim
        return x,y


class Ml_dataread(Dataset):
    def __init__(self,file_path,data_name,label_num):
        self.x,self.y = self.data_read(file_path,data_name,label_num)
        self.len = len(self.x)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def data_read(self,file_path,data_name,label_num):
        data_path = file_path+data_name +'.mat'
        features_struct = io.loadmat(data_path)
        train_data = features_struct['data']
        #train_data = train_data.astype(np.float32)  #change the data type       
        lable_start = train_data.shape[1] - label_num
        x=train_data[:,0:lable_start] 
        y=train_data[:,lable_start:]
        return self.shape_transform(x,y)
    
    def shape_transform(self,x,y):
        x=torch.from_numpy(x)
        y=torch.from_numpy(y.astype(np.int64))
        x=x.reshape((x.shape[0],1,-1)) #sample_num * 1 *sample_dim
        ll=torch.ones((x.shape[0],1))
        y=torch.cat((ll,y),dim=1)
        #y_r=torch.nn.functional.one_hot(y, num_classes=48)
        return x,y