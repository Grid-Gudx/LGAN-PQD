# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 22:11:39 2021

@author: gdx
"""
import numpy as np
import torch
from tensorflow.keras.models import load_model
from utils.dataread import Ml_dataread, Mc_dataread
from utils.metrics import Ml_metrics, Mc_metrics, Ml_metrics_np
from network.cnn_atten_rnn import cnn_atten_rnn
from network.mc_cnn import MC_CNN, deep_cnn, CNN_LSTM
from network.ml_cnn import ML_CNN

from thop import profile
def params_get():
    label_features = torch.sparse.torch.eye(8).reshape(-1,8,8)
    dummy_input = torch.randn(1, 1, 640)
    
    net1 = cnn_atten_rnn()
    net2 = ML_CNN()
    net3 = MC_CNN()
    net4 = deep_cnn()
    net5 = CNN_LSTM()
    
    flops, params = profile(net1, inputs=(dummy_input, label_features ))
    params_list = [params]
    flops_list = [flops]
    net_list = [net2, net3, net4, net5]
    for i in range(len(net_list)):
        flops, params = profile(net_list[i], inputs=(dummy_input, ))
        params_list.append(params)
        flops_list.append(flops)
    
    return params_list, flops_list

# print(params_get())


### cnn_atten_rnn test
def ml_test(model_name, data_name):
    model_path ='./model/'+ model_name
    data_path = './datasets/test_dataset/'

    net = cnn_atten_rnn()
    net.load_state_dict(torch.load(model_path, map_location= 'cpu'))
    net.eval()
    
    test_data = Ml_dataread(data_path,data_name,label_num=7)
    input_=test_data.x
    label_=test_data.y
    label_features = torch.sparse.torch.eye(8).reshape(-1,8,8)
    with torch.no_grad():
        output_ = net(input_, label_features)
    
    metric = Ml_metrics()
    acc=metric.multi_acc(output_,label_)
    ham=metric.hamming_loss(output_,label_)
    oe=metric.one_errer(output_,label_)
    rl=metric.rank_loss(output_,label_)
    ce=metric.coverage(output_,label_)
    ap=metric.average_precision(output_,label_)

    return acc,ham,oe,rl,ce,ap

# model_name = 'cnn_atten_rnn35.pth'
# data_name = 'test_ml_100_rand'
# print(ml_test(model_name, data_name))

### ML_CNN test
def ml_cnn_test(model_name, data_name):
    model_path ='./model/'+ model_name
    data_path = './datasets/test_dataset/'

    net = ML_CNN()
    net.load_state_dict(torch.load(model_path, map_location= 'cpu'))
    net.eval()
    
    test_data = Ml_dataread(data_path,data_name,label_num=7)
    input_=test_data.x
    label_=test_data.y
    with torch.no_grad():
        output_ = net(input_)
    
    metric = Ml_metrics()
    acc=metric.multi_acc(output_,label_)
    ham=metric.hamming_loss(output_,label_)
    oe=metric.one_errer(output_,label_)
    rl=metric.rank_loss(output_,label_)
    ce=metric.coverage(output_,label_)
    ap=metric.average_precision(output_,label_)

    return acc,ham,oe,rl,ce,ap

# model_name = 'ml_cnn35.pth'
# data_name = 'test_ml_100_rand'
# print(ml_cnn_test(model_name, data_name))


### MC_CNN deep_cnn and cnn_lstm test
def mc_test(model_name, data_name):
    model_path ='./model/'+ model_name
    data_path = './datasets/test_dataset/'
    
    # net = MC_CNN()
    # net = CNN_LSTM()
    net = deep_cnn()
    net.load_state_dict(torch.load(model_path, map_location= 'cpu'))
    net.eval()
    
    test_data = Mc_dataread(data_path, data_name, label_num=1)
    input_=test_data.x
    label_=test_data.y.squeeze()
    
    metric = Mc_metrics()
    with torch.no_grad():
        output_ = net(input_)
    
    acc = metric.acc(output_,label_)
    return acc  

# # model_name = 'mc_cnn31.pth'
# # model_name = 'cnn_lstm31.pth'
# model_name = 'deep_cnn35.pth'
# data_name = 'test_sl_100_rand'
# print(mc_test(model_name, data_name))


# bpmll and cnn_bpmll test
def bpmll_test(model_name, data_name):
    file_path='./datasets/test_dataset/'
    test_data = Ml_dataread(file_path, data_name, label_num=7)
    X_test = test_data.x.squeeze().numpy()
    ## if model is bpmll, Comment next line
    X_test = X_test.reshape(-1,X_test.shape[1],1)    
    Y_test = test_data.y.numpy()
    
    load_path ='./model/'+ model_name
    #load_path = './model/bpmll.h5'
    model = load_model(load_path, compile = False)
    output_ = model.predict(X_test)
    label_ = Y_test
    
    metric = Ml_metrics_np()
    acc=metric.multi_acc(output_,label_)
    ham=metric.hamming_loss(output_,label_)
    oe=metric.one_errer(output_,label_)
    rl=metric.rank_loss(output_,label_)
    ce=metric.coverage(output_,label_)
    ap=metric.average_precision(output_,label_)
   
    return acc,ham,oe,rl,ce,ap

# model_name = 'bpmll35.h5'
model_name = 'cnn_bpmll35.h5'
data_name = 'test_ml_100_rand'
print(bpmll_test(model_name, data_name))

