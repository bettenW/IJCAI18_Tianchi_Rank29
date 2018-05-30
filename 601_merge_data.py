#coding:utf-8
import pandas as pd
import numpy as np
import time
import datetime
import gc
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
import lightgbm as lgb

import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn

path = './data/'

'''
101_wang_feat.py                active,expose mean特征                               全局数据构造
102_trick_feat.py               时间段时间差特征,点击标签                            全局数据构造
103_statistics_feat.py          滑窗                                                 day7数据直接构造                     
201_meng_feat.py                分段,组合,各类比率,刻画兴趣度                        全局数据构造                  
301_timediff_last_next_feat.py  依然时间间隔,user_id组合下天,maphour,hour点击次数    全部数据直接构造
401_list_till_feat.py           till_now_cnt特征,,item相关比率, 新特征               day7数据直接构造
501_clickTran_feat.py           转化率特征                  



分两个数据集构造
All为全部数据作为训练集
复赛放弃了转化率特征以及全量训练集
'''
All = False

if All == False:
    print('----------------101_wang_feat,all,7----------------------')
    data = pd.read_csv(path+"101_wang_feat.csv")
    print('data:\n', data.shape)

    print('----------------102_trick_feat,all,7---------------------')
    trick_feat = pd.read_csv(path+"102_trick_feat.csv")
    print('102_trick_feat:\n', trick_feat.shape)
    data = data.merge(trick_feat, how='left', on='instance_id')
    del trick_feat
    gc.collect()

    print('---------------103_statistics_feat,7 ---------------------')
    statistics_feat = pd.read_csv(path+"103_statistics_feat.csv")
    print('103_statistics_feat:\n', statistics_feat.shape)
    data = data.merge(statistics_feat, how='left', on='instance_id')
    del statistics_feat
    gc.collect()

    print('----------------201_meng_feat, all,7--------------------')
    meng_feat = pd.read_csv(path+"201_meng_feat.csv")
    print('201_meng_feat:\n', meng_feat.shape)
    data = data.merge(meng_feat, how='left', on='instance_id')
    del meng_feat
    gc.collect()

    print('------------301_timediff_last_next_feat,all,7---------------')
    timediff_last_next_feat = pd.read_csv(path+"301_timediff_last_next_feat.csv")
    print('301_timediff_last_next_feat:\n', timediff_last_next_feat.shape)
    data = data.merge(timediff_last_next_feat, how='left', on='instance_id')
    del timediff_last_next_feat
    gc.collect()

    print('--------------401_list_till_feat,all,7----------------------')
    list_till_feat = pd.read_csv(path+"401_list_till_feat.csv")
    print('401_list_till_feat:\n', list_till_feat.shape)
    data = data.merge(list_till_feat, how='left', on='instance_id')
    del list_till_feat
    gc.collect()
    
    #print('------------------501 转化率,7------------------')
    #clickTran_feat = pd.read_csv(path+"501_clickTran_feat.csv")
    #print('501_clickTran_feat:\n', clickTran_feat.shape)
    #data = data.merge(clickTran_feat, how='left', on='instance_id')
    #del clickTran_feat
    #gc.collect()

    # the last feature, wd_feat
    wide_deep_feat_online = pd.read_csv("wide_deep_feat_online.csv")
    data = data.merge(wide_deep_feat_online, how='left', on='instance_id')

    # qunge satcking feature
    stacking_lgbm = pd.read_csv("stacking_lgbm.csv")
    data = data.merge(stacking_lgbm, how='left', on='instance_id')

    print('最终维度:',data.shape)

    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    data.to_csv(path+"rate_final_data_%s.csv" % now, index=False)

if All == True:
    print('----------------101_wang_feat,all,7----------------------')
    data = pd.read_csv(path+"101_wang_feat_all.csv")
    print('data:\n', data.shape)

    print('----------------102_trick_feat,all,7---------------------')
    trick_feat = pd.read_csv(path+"102_trick_feat_all.csv")
    print('102_trick_feat:\n', trick_feat.shape)
    data = data.merge(trick_feat, how='left', on='instance_id')
    del trick_feat
    gc.collect()

    print('----------------201_meng_feat, all,7--------------------')
    meng_feat = pd.read_csv(path+"201_meng_feat_all.csv")
    print('201_meng_feat:\n', meng_feat.shape)
    data = data.merge(meng_feat, how='left', on='instance_id')
    del meng_feat
    gc.collect()

    print('------------301_timediff_last_next_feat,all,7---------------')
    timediff_last_next_feat = pd.read_csv(path+"301_timediff_last_next_feat_all.csv")
    print('301_timediff_last_next_feat:\n', timediff_last_next_feat.shape)
    data = data.merge(timediff_last_next_feat, how='left', on='instance_id')
    del timediff_last_next_feat
    gc.collect()

    print('--------------401_list_till_feat,all,7----------------------')
    list_till_feat = pd.read_csv(path+"401_list_till_feat_all.csv")
    print('401_list_till_feat:\n', list_till_feat.shape)
    data = data.merge(list_till_feat, how='left', on='instance_id')
    del list_till_feat
    gc.collect()

    print('最终维度:',data.shape)

    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    data.to_csv(path+"all_final_data_%s.csv" % now, index=False)