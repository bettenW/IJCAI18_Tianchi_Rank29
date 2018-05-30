import numpy as np
import pandas as pd
import scipy as sp
import gc
import datetime
import time
import random
import scipy.special as special
import pickle
##########################################

# 获取click_time之前所有天的点击数和转化数, 为转化率和进行贝叶斯平滑做准备#

##########################################

print("Preprocessing...")

path = './data/'
        
train = pd.read_csv(path+'train_all.csv')
test = pd.read_csv(path+'test_all.csv')

data = pd.concat([train, test])
print(data['instance_id'].head())

data['period'] = data['hour'] // 12 + 1
data['period'] = data['day'] % 31 * 2 + data['period']
max_p = data['period'].max()+1

cols = data.columns.tolist()
cols.remove('instance_id')

# print("-----------------转化率（贝叶斯平滑）--------------------------")
# f = open('./smooth_parameters.txt','rb')
# smooth_para = pickle.load(f)

print("单特征........")
temp=data[['user_age_level','user_gender_id','user_occupation_id','item_id','shop_id','item_brand_id','item_city_id',
           'context_page_id','period','hour','is_trade']]
# for feat_1 in ['user_age_level','user_gender_id','user_occupation_id','item_id','shop_id','item_brand_id','item_city_id',
#            'context_page_id']:
for feat_1 in ['item_id','shop_id','user_age_level']:
    gc.collect()
    res=pd.DataFrame()
    for period in range(2,max_p):
        print(feat_1,'The',period,'starting')
        
        count=temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['period']<period).values].count()).reset_index(name=feat_1+'_all')
        count1=temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['period']<period).values].sum()).reset_index(name=feat_1+'_1')
        
        count_shift = temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['period']==(period-1)).values].count()).reset_index(name=feat_1+'_shift_all')
        count_shift_1 = temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['period']==(period-1)).values].sum()).reset_index(name=feat_1+'_shift_1')
        count[feat_1+'_shift_all'] = count_shift[feat_1+'_shift_all']
        count[feat_1+'_shift_1'] = count_shift_1[feat_1+'_shift_1']

        count[feat_1+'_1']=count1[feat_1+'_1']
        count.fillna(value=0, inplace=True)
        ######hyper########
        # hyper = smooth_para['{}_{}'.format(feat_1, period)]
        # count[feat_1+'_rate_hyper'] = round((count[feat_1+'_1']+hyper[0]) / (count[feat_1+'_all'] + hyper[0] + hyper[1]), 5)
        # count[feat_1+'_rate_hyper'].fillna(round(hyper[0]/(hyper[0] + hyper[1]), 5), inplace = True)
        ###################
        count[feat_1+'_rate'] = round(count[feat_1+'_1'] / count[feat_1+'_all'], 5)

        count[feat_1+'_shift_rate'] = round(count[feat_1+'_shift_1'] / count[feat_1+'_shift_all'], 5)

        count['period']=period
        count.drop([feat_1+'_all', feat_1+'_1', feat_1+'_shift_all', feat_1+'_shift_1'],axis=1,inplace=True)
        count.fillna(value=0, inplace=True)
        res=res.append(count,ignore_index=True)
    data = pd.merge(data,res, how='left', on=[feat_1,'period'])
    print(feat_1,' over')
del temp
gc.collect()

print("双特征........")
temp=data[['user_age_level','user_gender_id','user_occupation_id','item_id','shop_id','item_brand_id','item_city_id',
           'context_page_id','period','hour','is_trade']]
# for feat_1,feat_2 in [('user_age_level','item_id'), ('user_occupation_id','item_brand_id'),
#                     ('item_id','item_city_id'),('item_city_id','shop_id'),('item_brand_id','shop_id')]:
for feat_1,feat_2 in [('user_age_level','item_id'),('item_id','item_city_id'),('item_brand_id','shop_id'), ('user_occupation_id','item_brand_id')]:
    gc.collect()
    res=pd.DataFrame()
    for period in range(2,17):
        print(feat_1,feat_2,'The',period,'starting')
        count=temp.groupby([feat_1, feat_2]).apply(lambda x: x['is_trade'][(x['period']<period).values].count()).reset_index(name=feat_1+'_'+feat_2+'_all')
        count1=temp.groupby([feat_1, feat_2]).apply(lambda x: x['is_trade'][(x['period']<period).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_1')
        count.fillna(value=0, inplace=True)
        count[feat_1+'_'+feat_2+'_1']=count1[feat_1+'_'+feat_2+'_1']
        count[feat_1+'_'+feat_2+'_rate'] = round(count[feat_1+'_'+feat_2+'_1'] / count[feat_1+'_'+feat_2+'_all'], 5)
        count['period'] = period
        count.drop([feat_1+'_'+feat_2+'_all', feat_1+'_'+feat_2+'_1'],axis=1,inplace=True)
        count.fillna(value=0, inplace=True)
        res=res.append(count,ignore_index=True)
    data = data.merge(res, how='left', on=[feat_1,feat_2,'period'])
    print(feat_1,feat_2,' over')
del temp
gc.collect()

print("仨特征........")
temp=data[['user_age_level','user_gender_id','user_occupation_id','item_id','shop_id','item_brand_id','item_city_id',
           'context_page_id','period','hour','is_trade']]
# for feat_1,feat_2,feat_3 in [('user_gender_id','user_age_level','item_brand_id'),('user_gender_id','user_age_level','shop_id'),
#                           ('user_gender_id','user_occupation_id','item_brand_id'),('item_id','item_brand_id','item_city_id'),
#                           ('item_id','item_brand_id','context_page_id'),('item_brand_id','shop_id','context_page_id'),
#                           ('user_age_level','user_occupation_id','item_brand_id')]:
for feat_1,feat_2,feat_3 in [('user_gender_id','user_age_level','shop_id'),('item_id','item_brand_id','item_city_id'),
                             ('item_id','item_brand_id','context_page_id'),('user_age_level','user_occupation_id','item_brand_id')]:
    gc.collect()
    res=pd.DataFrame()
    for period in range(2,17):
        print(feat_1,feat_2,feat_3,'The',period,'starting')
        count=temp.groupby([feat_1, feat_2, feat_3]).apply(lambda x: x['is_trade'][(x['period']<period).values].count()).reset_index(name=feat_1+'_'+feat_2+'_'+feat_3+'_all')
        count1=temp.groupby([feat_1, feat_2, feat_3]).apply(lambda x: x['is_trade'][(x['period']<period).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_'+feat_3+'_1')
        count[feat_1+'_'+feat_2+'_'+feat_3+'_1']=count1[feat_1+'_'+feat_2+'_'+feat_3+'_1']
        count.fillna(value=0, inplace=True)
        count[feat_1+'_'+feat_2+'_'+feat_3+'_rate'] = round(count[feat_1+'_'+feat_2+'_'+feat_3+'_1'] / count[feat_1+'_'+feat_2+'_'+feat_3+'_all'], 5)
        count['period'] = period
        count.drop([feat_1+'_'+feat_2+'_'+feat_3+'_all', feat_1+'_'+feat_2+'_'+feat_3+'_1'], axis=1, inplace=True)
        count.fillna(value=0, inplace=True)
        res=res.append(count,ignore_index=True)
    data = data.merge(res, how='left', on=[feat_1,feat_2,feat_3,'period'])
    print(feat_1,feat_2,feat_3,' over')
del temp
gc.collect()
print(data.columns.tolist())
print(cols)
data = data.drop(cols, axis=1)

print('经过处理后,最终维度:', data.shape)

data.to_csv(path+'501_clickTran_feat_all.csv', index=False)