# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:47:12 2021

@author: gdx
"""

import numpy as np
from skmultilearn.adapt import MLkNN
from skmultilearn.adapt import MLARAM

from utils.dataread import Ml_dataread, Mc_dataread
from utils.metrics import Ml_metrics_np 

from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import accuracy_score,hamming_loss,\
coverage_error,average_precision_score,label_ranking_loss

import joblib #save model
import time
import sys

def model_get(model_type):
    if model_type=='MLKNN':# predict_proba + toarray
        model = MLkNN(k=10, s=1.0) #Laplace smoothing parameter is 1.
    elif model_type == 'MLHARAM':# predict_proba  Clustering Vigilance
        model = MLARAM(threshold=0.02, vigilance=0.975) #threshold is 0.02， Clustering Vigilance is 0.975       
    else:
        print('please input correct model_type!')
        sys.exit(1111)
    return model

def model_fit(save_path,model_name,model_type,train_x,train_y):
    print('training start...')
    start=time.process_time()   
    clf = model_get(model_type)
    clf.fit(train_x,train_y)
    joblib.dump(clf,save_path+model_name)
    print('running time is ',time.process_time()-start)

def model_test(save_path, model_type, model_suffix, test_x, test_y):
    model = joblib.load(save_path+model_type+model_suffix)
    if model_type=='mlknn':
        y_pred = model.predict(test_x).toarray()
        y_score = model.predict_proba(test_x).toarray()   
    else:
        y_pred = model.predict(test_x)
        y_score = model.predict_proba(test_x)
    
    y_true = test_y
    
    metric = Ml_metrics_np()
    acc = accuracy_score(y_true, y_pred)
    ham = hamming_loss(y_true, y_pred)
    oe = metric.one_errer(y_score, y_true)
    rl = label_ranking_loss(y_true, y_score)
    ce = coverage_error(y_true, y_score)-1.
    ap = average_precision_score(y_true, y_score)
    return acc,ham,oe,rl,ce,ap

if __name__ == '__main__':  
### train model
    # file_path='./datasets/train_dataset/'
    # data_name='train_ml_300_rand'
    # train_data = Ml_dataread(file_path, data_name, label_num=7)
    # xx = train_data.x.numpy()
    # yy = train_data.y.numpy()
    # train_x = xx.reshape(-1,640)
    # train_y = yy
    
    # save_path = './model/'
    # model_name = 'mlharam1'
    # model_type = 'MLHARAM'    
    # model_fit(save_path, model_name, model_type, train_x, train_y)
    
    
### model test
    file_path='./datasets/test_dataset/'
    data_name='test_ml_100_rand'
    train_data = Ml_dataread(file_path, data_name, label_num=7)
    xx = train_data.x.numpy()
    yy = train_data.y.numpy()
    test_x = xx.reshape(-1,640)
    test_y = yy
        
    save_path = './model/'
    model_type = 'mlknn'
    # model_type = 'mlharam'
    model_suffix = '1'
    print(model_test(save_path, model_type, model_suffix, test_x, test_y))

