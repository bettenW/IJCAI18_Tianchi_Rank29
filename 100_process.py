#coding:utf-8
import pandas as pd
import numpy as np
import time
import datetime
import gc
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn


def map_hour(s):
    if s < 6:
        return 1
    elif s < 12:
        return 2
    elif s < 18:
        return 3
    else:
        return 4


def pre_process(data):
    
    data['time'] = pd.to_datetime(data.context_timestamp, unit='s')
    data['time'] = data['time'].apply(lambda x: x + datetime.timedelta(hours=8))
    data['day'] = data['time'].apply(lambda x: int(str(x)[8:10]))
    data['hour'] = data['time'].apply(lambda x: int(str(x)[11:13]))   
    data['minute'] =  data['time'].apply(lambda x: int(str(x)[14:16]))


    data['maphour'] = data['hour'].map(map_hour)
    data['mapmin'] = data['minute'] % 15 + 1

    data_item_category = data.item_category_list.str.split(';', expand=True).add_prefix('item_category_')

    for i in range(3):
        data['item_category_'+str(i)] =  data_item_category['item_category_'+str(i)]     
    del data['item_category_0']
    data['item_category_1'] = data['item_category_1'].apply(int)
    data['item_category_2'].fillna(value=0, inplace=True)
    data['item_category_2'] = data['item_category_2'].apply(int)


    label = ['item_category_1',  'item_category_2', 'context_id',  'item_brand_id', 'item_city_id', 'item_id', 'user_id', 'shop_id']
    
    short_label = ['context_page_id', 'shop_star_level', 'user_age_level', 'user_occupation_id', 'user_star_level']

    score_label = ['shop_score_service', 'shop_review_positive_rate', 'shop_score_delivery', 'shop_score_description']
    
    for col in label:
       col_encoder = LabelEncoder()
       col_encoder.fit(data[col])
       data[col] = col_encoder.transform(data[col])

    for col in short_label:
       col_encoder = LabelEncoder()
       col_encoder.fit(data[col])
       data[col] = col_encoder.transform(data[col])

    for col in score_label:
        data[col] = round(data[col], 3)

    return data


def main():
    path = './data/'
    
    # 读取全部数据
    train = pd.read_table(path+'round2_train.txt', sep=" ")
    test_a = pd.read_table(path+'round2_ijcai_18_test_a_20180425.txt', sep=" ")
    test_b = pd.read_table(path+'round2_ijcai_18_test_b_20180510.txt', sep=" ")
    test = test_a.append(test_b)
    data = pd.concat([train, test])
    
    print('原始特征:', data.columns.tolist())
    print('初始维度:', data.shape)

    data = pre_process(data)
    print('pre_process:', data.shape)

    del data['predict_category_property']
    del data['item_property_list']

    # 全量数据集
    len = train.shape[0]
    train_all = data[:len]
    test_all = data[len:]
    train_all.to_csv(path+'train_all.csv', index=False)
    test_all.to_csv(path+'test_all.csv', index=False)
    
    # 7号数据集
    train_day7 = data.loc[(data.day==7) & (data.hour<12)]
    test_day7 = data.loc[(data.day==7) & (data.hour>=12)]
    train_day7.to_csv(path+'train_day7.csv', index=False)
    test_day7.to_csv(path+'test_day7.csv', index=False)
    print(data.columns.tolist())
    

if __name__ == '__main__':
    main()

# 构造全部和7号数据集