#coding:utf-8
import pandas as pd
import numpy as np
import time
import datetime
import gc
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn

def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll

def lgb_model():

    print('LGBMClassifier...')
    lgb_clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=2000,
                               max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                               min_child_weight=5, min_child_samples=10, subsample=1, subsample_freq=1,
                               colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, nthread=-1, silent=True)
    return lgb_clf

def feat_select(train, test):
  
    features = train.drop(['is_trade', 'time', 'context_timestamp', 'day'], axis=1).columns.tolist()

    target = ['is_trade']

    return features, target

def lgbCV(train, test, All):
    features, target = feat_select(train, test)

    if All == False:
        train_x = train.loc[train.hour < 10]
        print(train_x.shape)
        test_x = train.loc[train.hour >= 10]
        print(test_x.shape)
    if All == True:
        train_x = train.loc[train.day < 7]
        train_x1 = train.loc[(train.day == 7) & (train.hour < 10)]
        train_x = pd.concat([train_x, train_x1])
        print(train_x.shape)
        test_x = train.loc[(train.day == 7) & (train.hour >= 10)]
        print(test_x.shape)
        del train_x1
        gc.collect()

    lgb_clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=2000,
                               max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                               min_child_weight=5, min_child_samples=10, subsample=1, subsample_freq=1,
                               colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, nthread=-1, silent=True)

    lgb_model = lgb_clf.fit(train_x[features], train_x[target], eval_set=[(test_x[features], test_x[target])], early_stopping_rounds=200)
    best_iter = lgb_model.best_iteration_ 
    
    # 特征重要性
    lgb_predictors = [i for i in train_x[features].columns]
    lgb_feat_imp = pd.Series(lgb_model.feature_importances_, lgb_predictors).sort_values(ascending=False)
    lgb_feat_imp.to_csv('lgb_feat_imp.csv')
    
    # 训练模型
    lgb_clf.fit(train_x[features], train_x[target])
    test_x['lgb_predict'] = lgb_clf.predict_proba(test_x[features])[:, 1]
    lgb_loss = log_loss(test_x[target], test_x['lgb_predict'])  
    lgb_auc = roc_auc_score(test_x[target], test_x['lgb_predict'])
    print('Training loss: %.6f, Training AUC: %.6f' % (lgb_loss, lgb_auc))
    
    return best_iter,lgb_loss

def sub(train, test, best_iter, lgb_loss):

    features, target = feat_select(train, test)
    
    lgb_clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=best_iter,
                               max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                               min_child_weight=5, min_child_samples=10, subsample=1, subsample_freq=1,
                               colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, nthread=-1, silent=True)

    lgb_clf = lgb_model()
    
    lgb_clf.fit(train[features], train[target])
    test['predicted_score'] = lgb_clf.predict_proba(test[features])[:, 1]

    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    mean = test['predicted_score'].mean()
    print('predicted_score mean:', mean)
    test[['instance_id', 'predicted_score']].to_csv("./submit/lgb_%s_%s.csv" % (round(lgb_loss,4)*10000, round(mean,5)*100000), index=False, sep=" ")

def main():
    path = './data/'
    
    All = False

    if All == False:

        data = pd.read_csv(path+"final_data_05-11-05-51.csv")
        train = data.loc[data.hour < 12]
        test = pd.read_csv(path+"test_day7.csv")
        test = test[['instance_id']]
        test = pd.merge(test, data, how='left', on='instance_id')

    if All == True:

        data = pd.read_csv(path+"all_final_data_05-11-06-46.csv")
        train = pd.read_csv(path+"train_all.csv")
        train = train[['instance_id']]
        train =pd.merge(train, data, how='left', on='instance_id')

        test = pd.read_csv(path+"test_all.csv")
        test = test[['instance_id']]
        test = pd.merge(test, data, how='left', on='instance_id')

    print('The end of load data!')
    print(train.shape)
    print(test.shape)
    del data
    gc.collect()
    
    print('Start predicting...')       

    print("----------------------------------------------------线下----------------------------------------")
    best_iter, lgb_loss = lgbCV(train, test, All)

    print("----------------------------------------------------线上----------------------------------------")
    test_b = pd.read_table(path+'round2_ijcai_18_test_b_20180510.txt', sep=" ")
    test_b = test_b[['instance_id']]
    test = pd.merge(test_b, test, how='left', on='instance_id')
    sub(train, test, best_iter, lgb_loss)

if __name__ == '__main__':
    main()
