#coding:utf-8
import pandas as pd
import numpy as np
import time
import datetime
from collections import defaultdict
import gc

def clickTran(data):
    print("单特征........")
    temp=data[['user_age_level','user_gender_id','user_occupation_id','user_id','item_id','shop_id','item_brand_id','item_city_id',
           'context_page_id','hour','is_trade']]
    ######
    temp['temp_hour'] = temp['hour']
    temp['temp_hour'].loc[temp.temp_hour>12] = 12

    data['temp_hour'] = data['hour']
    data['temp_hour'].loc[data.temp_hour>12] = 12
    ######
    for feat_1 in ['item_city_id', 'item_id', 'shop_id', 'item_brand_id', 'user_id']:
        gc.collect()
        res=pd.DataFrame()
        for temp_hour in range(1,13):
            print(feat_1,'The',temp_hour,'starting')        
            count=temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['temp_hour']<temp_hour).values].count()).reset_index(name=feat_1+'_all')
            count1=temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['temp_hour']<temp_hour).values].sum()).reset_index(name=feat_1+'_1')       
            count['temp_hour'] = temp_hour
            count[feat_1+'_1']=count1[feat_1+'_1']
            count[feat_1+'_rate'] = round(count[feat_1+'_1'] / count[feat_1+'_all'], 5)
            count.drop([feat_1+'_all', feat_1+'_1'],axis=1,inplace=True)
            count.fillna(value=0, inplace=True)
            res=res.append(count,ignore_index=True)
        data = pd.merge(data,res, how='left', on=[feat_1,'temp_hour'])
        print(feat_1,' over')
    del temp
    gc.collect()

    print("双特征........")
    temp=data[['user_age_level','user_gender_id','user_occupation_id','item_id','shop_id','item_brand_id','item_city_id',
               'context_page_id','hour','is_trade']]
    ######
    temp['temp_hour'] = temp['hour']
    temp['temp_hour'].loc[temp.temp_hour>12] = 12
    ######
    for feat_1,feat_2 in [('user_age_level','item_id'),('item_id','item_city_id'),('item_brand_id','shop_id'), ('user_occupation_id','item_brand_id')]:
        gc.collect()
        res=pd.DataFrame()
        for temp_hour in range(1,13):
            print(feat_1,feat_2,'The',temp_hour,'starting')
            count=temp.groupby([feat_1, feat_2]).apply(lambda x: x['is_trade'][(x['temp_hour']<temp_hour).values].count()).reset_index(name=feat_1+'_'+feat_2+'_all')
            count1=temp.groupby([feat_1, feat_2]).apply(lambda x: x['is_trade'][(x['temp_hour']<temp_hour).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_1')
            count['temp_hour'] = temp_hour
            count[feat_1+'_'+feat_2+'_1']=count1[feat_1+'_'+feat_2+'_1']
            count[feat_1+'_'+feat_2+'_rate'] = round(count[feat_1+'_'+feat_2+'_1'] / count[feat_1+'_'+feat_2+'_all'], 5)
            count.drop([feat_1+'_'+feat_2+'_all', feat_1+'_'+feat_2+'_1'],axis=1,inplace=True)
            count.fillna(value=0, inplace=True)
            res=res.append(count,ignore_index=True)
        data = data.merge(res, how='left', on=[feat_1,feat_2,'temp_hour'])
        print(feat_1,feat_2,' over')
    del temp
    gc.collect()

    print("仨特征........")
    temp=data[['user_age_level','user_gender_id','user_occupation_id','item_id','shop_id','item_brand_id','item_city_id',
           'context_page_id','hour','is_trade']]
    ######
    temp['temp_hour'] = temp['hour']
    temp['temp_hour'].loc[temp.temp_hour>12] = 12
    ######
    for feat_1,feat_2,feat_3 in [('user_gender_id','user_age_level','item_brand_id'),('user_gender_id','user_age_level','shop_id'),
                              ('user_gender_id','user_occupation_id','item_brand_id'),('user_age_level','user_occupation_id','item_brand_id')]:
        gc.collect()
        res=pd.DataFrame()
        for temp_hour in range(1,13):
            print(feat_1,feat_2,feat_3,'The',temp_hour,'starting')
            count=temp.groupby([feat_1, feat_2, feat_3]).apply(lambda x: x['is_trade'][(x['temp_hour']<temp_hour).values].count()).reset_index(name=feat_1+'_'+feat_2+'_'+feat_3+'_all')
            count1=temp.groupby([feat_1, feat_2, feat_3]).apply(lambda x: x['is_trade'][(x['temp_hour']<temp_hour).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_'+feat_3+'_1')
            count['temp_hour'] = temp_hour
            count[feat_1+'_'+feat_2+'_'+feat_3+'_1']=count1[feat_1+'_'+feat_2+'_'+feat_3+'_1']
            count[feat_1+'_'+feat_2+'_'+feat_3+'_rate'] = round(count[feat_1+'_'+feat_2+'_'+feat_3+'_1'] / count[feat_1+'_'+feat_2+'_'+feat_3+'_all'], 5)
            count.drop([feat_1+'_'+feat_2+'_'+feat_3+'_all', feat_1+'_'+feat_2+'_'+feat_3+'_1'], axis=1, inplace=True)
            count.fillna(value=0, inplace=True)
            res=res.append(count,ignore_index=True)
        data = data.merge(res, how='left', on=[feat_1,feat_2,feat_3,'temp_hour'])
        print(feat_1,feat_2,feat_3,' over')
    del temp

    del data['temp_hour']
    gc.collect()

    return data

def new_clickTran(data):
    print("单特征........")
    temp=data[['item_id', 'user_id', 'item_price_level', 'item_sales_level','day','is_trade']]
    for feat_1 in ['item_id', 'user_id', 'item_price_level', 'item_sales_level']:
        print(feat_1,'starting')        
        count=temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['day']!=7).values].count()).reset_index(name=feat_1+'_all')
        count1=temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['day']!=7).values].sum()).reset_index(name=feat_1+'_1')       
        count['day'] = 7
        count[feat_1+'_1']=count1[feat_1+'_1']
        count[feat_1+'_rate'] = round(count[feat_1+'_1'] / count[feat_1+'_all'], 5)
        count.drop([feat_1+'_all', feat_1+'_1'],axis=1,inplace=True)
        data = pd.merge(data,count, how='left', on=[feat_1,'day'])
        print(feat_1,' over')
    del temp
    gc.collect()

    print("双特征........")
    temp=data[['user_id','item_id','item_category_list','day','is_trade']]
    for feat_1,feat_2 in [('user_id','item_id'),('user_id','item_category_list')]:
        print(feat_1,feat_2,'starting')
        count=temp.groupby([feat_1, feat_2]).apply(lambda x: x['is_trade'][(x['day']!=7).values].count()).reset_index(name=feat_1+'_'+feat_2+'_all')
        count1=temp.groupby([feat_1, feat_2]).apply(lambda x: x['is_trade'][(x['day']!=7).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_1')
        count['day'] = 7
        count[feat_1+'_'+feat_2+'_1']=count1[feat_1+'_'+feat_2+'_1']
        count[feat_1+'_'+feat_2+'_rate'] = round(count[feat_1+'_'+feat_2+'_1'] / count[feat_1+'_'+feat_2+'_all'], 5)
        count.drop([feat_1+'_'+feat_2+'_all', feat_1+'_'+feat_2+'_1'],axis=1,inplace=True)
        data = pd.merge(data,count, how='left', on=[feat_1,feat_2,'day'])
        print(feat_1,feat_2,' over')
    del temp
    gc.collect()

def main():
    path = './data/'
    
    train = pd.read_csv(path+'train_all.csv')
    test = pd.read_csv(path+'test_all.csv')

    # train = pd.read_csv(path+'train_day7.csv')
    # test = pd.read_csv(path+'test_day7.csv')

    data = pd.concat([train, test])

    print('初始维度:', data.shape)

    cols = data.columns.tolist()
    cols.remove('instance_id')

    ##################################
    # data = clickTran(data)
    # print('clickTran:', data.shape)

    data = new_clickTran(data)
    print('new_clickTran:', data.shape)
    ##################################
    
    # 得到7号训练集
    data = data.drop(cols, axis=1)
    data = data.loc[data.day==7]
    print('经过处理后,7号训练集最终维度:', data.shape)
    print(data.columns.tolist())
    data.to_csv(path+'501_clickTran_feat.csv', index=False)
    
if __name__ == '__main__':
    main()

# 7号转化率
