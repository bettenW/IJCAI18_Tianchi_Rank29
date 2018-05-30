#coding:utf-8
import pandas as pd
import numpy as np
import time
import datetime
import gc
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import lightgbm as lgb

'''
此处参考技术圈涵涵开源的代码,
'''

def pre_process(data):

    cols = data.columns.tolist()
    keys = ['instance_id', 'day']
    for k in keys:
        cols.remove(k)

    return data, cols

def zuhe(data):

    for col in ['user_gender_id','user_age_level','user_occupation_id','user_star_level']:
        data[col] = data[col].apply(lambda x: 0 if x == -1 else x)

    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',
                'user_gender_id','user_age_level','user_occupation_id','user_star_level',
                'shop_review_num_level', 'shop_star_level']:
        data[col] = data[col].astype(str)

    print('item两两组合')
    data['sale_price'] = data['item_sales_level'] + data['item_price_level']
    data['sale_collect'] = data['item_sales_level'] + data['item_collected_level']
    data['price_collect'] = data['item_price_level'] + data['item_collected_level']

    print('user两两组合')
    data['gender_star'] = data['user_gender_id'] + data['user_star_level']

    print('shop两两组合')
    data['review_star'] = data['shop_review_num_level'] + data['shop_star_level']

    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',  'sale_price','sale_collect', 'price_collect',
                'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']:
        data[col] = data[col].astype(int)

    del data['review_star']

    return data

def item(data):

    print('一个item有多少brand,price salse collected level……')

    itemcnt = data.groupby(['item_id'], as_index=False)['instance_id'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_id'], how='left')
    
    # 去除 'item_brand_id','item_city_id'
    for col in ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_id'], as_index=False)['instance_id'].agg({str(col) + '_item_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_id'], how='left')
        data[str(col) + '_item_prob']=data[str(col) + '_item_cnt']/data['item_cnt']
        del data[str(col) + '_item_cnt']
    del data['item_cnt']

    print('一个brand有多少price salse collected level……')

    itemcnt = data.groupby(['item_brand_id'], as_index=False)['instance_id'].agg({'item_brand_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_brand_id'], how='left')

    for col in ['item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_brand_id'], as_index=False)['instance_id'].agg({str(col) + '_brand_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_brand_id'], how='left')
        data[str(col) + '_brand_prob'] = data[str(col) + '_brand_cnt'] / data['item_brand_cnt']
        del data[str(col) + '_brand_cnt']
    del data['item_brand_cnt']

    print('一个city有多少item_price_level，item_sales_level，item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_city_id'], as_index=False)['instance_id'].agg({'item_city_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_city_id'], how='left')
    for col in ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg({str(col) + '_city_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_city_prob'] = data[str(col) + '_city_cnt'] / data['item_city_cnt']
        del data[str(col) + '_city_cnt']
    del data['item_city_cnt']

    print('一个price有多少item_sales_level，item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_price_level'], as_index=False)['instance_id'].agg({'item_price_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_price_level'], how='left')
    for col in ['item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg({str(col) + '_price_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_price_prob'] = data[str(col) + '_price_cnt'] / data['item_price_cnt']
        del data[str(col) + '_price_cnt']
    del data['item_price_cnt']

    print('一个item_sales_level有多少item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_sales_level'], as_index=False)['instance_id'].agg({'item_salse_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_sales_level'], how='left')
    for col in ['item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_sales_level'], as_index=False)['instance_id'].agg({str(col) + '_salse_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_sales_level'], how='left')
        data[str(col) + '_salse_prob'] = data[str(col) + '_salse_cnt'] / data['item_salse_cnt']
        del data[str(col) + '_salse_cnt']
    del data['item_salse_cnt']

    print('一个item_collected_level有多少item_pv_level')

    itemcnt = data.groupby(['item_collected_level'], as_index=False)['instance_id'].agg({'item_coll_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_collected_level'], how='left')
    for col in ['item_pv_level']:
        itemcnt = data.groupby([col, 'item_collected_level'], as_index=False)['instance_id'].agg({str(col) + '_coll_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_collected_level'], how='left')
        data[str(col) + '_coll_prob'] = data[str(col) + '_coll_cnt'] / data['item_coll_cnt']
        del data[str(col) + '_coll_cnt']
    del data['item_coll_cnt']
 
    return data

def user_item(data):

    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')

    print('一个user有多少item_id,item_brand_id……')
    for col in ['item_id','item_category_list',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg({str(col)+'_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']
        del data[str(col) + '_user_cnt']

    print('一个user_gender有多少item_id,item_brand_id……')
    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg({'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')
    for col in ['item_id', 'item_category_list',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg({str(col)+'_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data['user_gender_cnt']
        del data[str(col) + '_user_gender_cnt']

    print('一个user_age_level有多少item_id,item_brand_id……')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg({'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')
    for col in ['item_id', 'item_category_list',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg({str(col)+'_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']
        del data[str(col) + '_user_age_cnt']

    print('一个user_occupation_id有多少item_id,item_brand_id…')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg({'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['item_id', 'item_category_list',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg({str(col)+'_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']
        del data[str(col) + '_user_occ_cnt']

    return data

def user_shop(data):

    print('一个user有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']
        del data[str(col) + '_user_cnt']
    del data['user_cnt']

    print('一个user_gender有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data['user_gender_cnt']
        del data[str(col) + '_user_gender_cnt']
    del data['user_gender_cnt']

    print('一个user_age_level有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']
        del data[str(col) + '_user_age_cnt']
    del data['user_age_cnt']

    print('一个user_occupation_id有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']
        del data[str(col) + '_user_occ_cnt']
    del data['user_occ_cnt']

    return data

def shop_item(data):
    
    print('一个shop有多少item_id,item_brand_id,item_city_id,item_price_level……')
    itemcnt = data.groupby(['shop_id'], as_index=False)['instance_id'].agg({'shop_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['shop_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'shop_id'], as_index=False)['instance_id'].agg({str(col)+'_shop_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'shop_id'], how='left')
        data[str(col) + '_shop_prob'] = data[str(col) + '_shop_cnt'] / data['shop_cnt']
        del data[str(col) + '_shop_cnt']
    del data['shop_cnt']

    print('一个shop_review_num_level有多少item_id,item_brand_id,item_city_id,item_price_level……')
    itemcnt = data.groupby(['shop_review_num_level'], as_index=False)['instance_id'].agg({'shop_rev_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['shop_review_num_level'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'shop_review_num_level'], as_index=False)['instance_id'].agg({str(col)+'_shop_rev_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'shop_review_num_level'], how='left')
        data[str(col) + '_shop_rev_prob'] = data[str(col) + '_shop_rev_cnt'] / data['shop_rev_cnt']
        del data[str(col) + '_shop_rev_cnt']
    del data['shop_rev_cnt']

    return data

def main():
    path = './data/'
    
    train = pd.read_csv(path+'train_all.csv')
    test = pd.read_csv(path+'test_all.csv')

    data = pd.concat([train, test])
    print(data.columns.tolist())
    data = data.drop(['time', 'context_timestamp'], axis=1) 
    del train
    del test
    gc.collect()

    print('初始维度:', data.shape)

    data, cols = pre_process(data)
    print('pre_process:', data.shape)

    #############################
    data = zuhe(data)
    print('zuhe:', data.shape)
    
    # 均为比率
    data = item(data)
    print('item:', data.shape)

    data = user_item(data)
    print('user_item:', data.shape)

    data = user_shop(data)
    print('user_shop:', data.shape)

    data = shop_item(data)
    print('shop_item:', data.shape)
    ###############################

    data = data.drop(cols, axis=1)

    # 得到全部训练集
    print('经过处理后,全部训练集最终维度:', data.shape)
    data.to_csv(path+'201_meng_feat_all.csv', index=False)

    # 得到7号训练集
    data = data.loc[data.day == 7]
    data = data.drop('day', axis=1)
    print('经过处理后,7号训练集最终维度:', data.shape)
    print(data.columns.tolist())
    data.to_csv(path+'201_meng_feat.csv', index=False)

if __name__ == '__main__':
    main()
