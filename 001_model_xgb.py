# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
import warnings
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import math
import numpy
import random
import scipy.special as special
import tqdm
import gc
import pickle
import xgboost as xgb
warnings.filterwarnings("ignore")

import time

def feat_select(train, test):

    features = train.drop(['is_trade','time', 'context_timestamp'], axis=1).columns.tolist()
    feature = [] 
    target = ['is_trade']

    return features, target

def xgbCV(train, test):
    
    features, target = feat_select(train, test)
    
    X = train[features]
    y = train[target]
    X_tes = test[features]
    y_tes = test[target]

    print('Training XGB model...')
    X_train_set = xgb.DMatrix(X, label=y, missing=np.nan)
    X_validate_set = xgb.DMatrix(X_tes, label=y_tes, missing=np.nan)
    watchlist = [(X_train_set, 'train'), (X_validate_set, 'eval')]
    params = {'max_depth':7,
              'nthread': 25,
              'eta': 0.01,
              'eval_metric': 'logloss',
              'objective': 'binary:logistic',
              'subsample': 0.85,
              'colsample_bytree': 0.85,
              'silent': 1,
              'seed': 0,
              'min_child_weight': 6
              #'scale_pos_weight':0.5
              }
    gbm = xgb.train(params, X_train_set, num_boost_round=3000, evals=watchlist, early_stopping_rounds=50)

    best_iter_num = gbm.best_iteration
    return best_iter_num

def sub(train, test, best_iter_num):

    features, target = feat_select(train, test)

    X = train[features]
    y = train[target]
    X_train_set = xgb.DMatrix(X, label=y, missing=np.nan)
    X_test_set = xgb.DMatrix(test[features], missing=np.nan)
    print('Training XGB model...')
    params = {'max_depth':7,
              'nthread': 25,
              'eta': 0.01,
              'eval_metric': 'logloss',
              'objective': 'binary:logistic',
              'subsample': 0.85,
              'colsample_bytree': 0.85,
              'silent': 1,
              'seed': 0,
              'min_child_weight': 6
              #'scale_pos_weight':0.5
              }
    gbm = xgb.train(params, X_train_set, num_boost_round=best_iter_num)

    pred = gbm.predict(X_test_set)
    test['predicted_score'] = pred
    sub = test[['instance_id', 'predicted_score']]
    mean = sub['predicted_score'].mean()
    sub[['instance_id', 'predicted_score']].to_csv('xgb_mean_%s.txt' % mean ,sep=" ",index=False)


if __name__ == "__main__":  
    
    path = './data/' 
    
    data = pd.read_csv(path+'601_a_b_trick_clickT_have_list_sorted_423.csv')
    
    train_data, test_data = data[data.hour < 12], data[data.hour >= 12]
    
    del data
    gc.collect()
    
    "----------------------------------------------------线下----------------------------------------"
    train= train_data.loc[train_data.hour<10]
    test= train_data.loc[train_data.hour>=10]
    
    best_iter = xgbCV(train, test)
    "----------------------------------------------------线上----------------------------------------"
    train = train_data[train_data.is_trade.notnull()]
    
    test_b = pd.read_table(path+'round2_ijcai_18_test_b_20180515.txt',sep=" ")
    test_b = test_b[['instance_id']]
    test_b = pd.merge(test_b,test_data, how='left', on='instance_id')
    
    sub(train, test_b ,best_iter)

