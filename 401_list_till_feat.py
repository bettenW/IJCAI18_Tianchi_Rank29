#coding:utf-8
import pandas as pd
import numpy as np
import time
import datetime
from collections import defaultdict
import gc

def dolastCount(data):
    
    keys = ['shop_id', 'item_id', 'user_id', 'item_brand_id', 'item_city_id', 'item_sales_level']

    for colname in keys:
        print(colname,'starting.....')
        count = data.groupby([colname]).apply(lambda x: x['instance_id'][(x['day']!=7).values].count()).reset_index(name='cnt_'+colname)
        sums = data.groupby([colname]).apply(lambda x: x['item_sales_level'][(x['day']!=7).values].sum()).reset_index(name='sum_'+colname)
        data = pd.merge(data,count, how='left', on=[colname])
        data = pd.merge(data,sums, how='left', on=[colname])
    
    for colname in keys:
        if(colname != 'user_id'):
            print(colname,'starting.....')
            count = data.groupby([colname, 'user_id']).apply(lambda x: x['instance_id'][(x['day']!=7).values].count()).reset_index(name='cnt_'+'user_id_'+colname)
            sums = data.groupby([colname, 'user_id']).apply(lambda x: x['item_sales_level'][(x['day']!=7).values].sum()).reset_index(name='sum_'+'user_id_'+colname)
            data = pd.merge(data,count, how='left', on=[colname, 'user_id'])
            data = pd.merge(data,sums, how='left', on=[colname, 'user_id'])
    
    data['item_user_ratio'] = data['cnt_user_id_item_id']/data['cnt_user_id']
    data['shop_user_ratio'] = data['cnt_user_id_shop_id']/data['cnt_user_id']
    data['brand_user_ratio'] = data['cnt_user_id_item_brand_id']/data['cnt_user_id'] 

    return data

def doNew(data):

    # collect pv sales 之间的小时比率
    print('collect pv sales 之间的小时比率')
    coll_query = data.groupby(['day', 'hour'], as_index=False)['item_collected_level'].agg({'hour_query_collect': 'sum'})
    pv_query = data.groupby(['day', 'hour'], as_index=False)['item_pv_level'].agg({'hour_query_pv': 'sum'})
    sales_query = data.groupby(['day', 'hour'], as_index=False)['item_sales_level'].agg({'hour_query_sales': 'sum'})
    coll_query = coll_query.merge(pv_query, how='left', on=['day', 'hour'])
    coll_query = coll_query.merge(sales_query, how='left', on=['day', 'hour'])

    coll_query['coll_sales_hour_ratio'] = round(coll_query['hour_query_collect'] / coll_query['hour_query_sales'],5)
    coll_query['coll_sales_hour_ratio'] = coll_query['coll_sales_hour_ratio'] - coll_query['coll_sales_hour_ratio'].min()
    coll_query['pv_sales_hour_ratio'] = round(coll_query['hour_query_pv'] / coll_query['hour_query_sales'], 5)
    coll_query['pv_sales_hour_ratio'] = coll_query['pv_sales_hour_ratio'] - coll_query['pv_sales_hour_ratio'].min()
    
    del coll_query['hour_query_collect']
    del coll_query['hour_query_pv']
    del coll_query['hour_query_sales']

    data = pd.merge(data, coll_query, how='left', on=['day', 'hour'])
    
    #重复次数是否大于2
    print('重复次数是否大于2')
    subset = ['item_brand_id', 'item_id', 'shop_id', 'user_id']
    temp=data.groupby(subset)['is_trade'].count().reset_index()
    temp.columns=['item_brand_id', 'item_id', 'shop_id', 'user_id','large2']
    temp['large2']=1*(temp['large2']>2)
    data = pd.merge(data, temp, how='left', on=subset)

    shop_query = data.groupby(['shop_id', 'day']).size().reset_index().rename(columns={0: 'shop_id_query_day'})
    category_2_query = data.groupby(['item_category_2', 'day']).size().reset_index().rename(columns={0: 'category_2_query_day'})
    data = pd.merge(data, shop_query, how='left', on=['shop_id', 'day'])
    data = pd.merge(data, category_2_query, how='left', on=['item_category_2', 'day'])

    print('price diff......')
    print('item_price_level')
    temp = data[['item_id', 'item_price_level']].loc[data.day==4]
    item_price_4 = temp.groupby(['item_id'], as_index=False)['item_price_level'].agg({'price_4': 'mean'})
    data = pd.merge(data, item_price_4, how='left', on='item_id')
    del temp
    gc.collect()
    print('item_price_level')
    temp = data[['item_id', 'item_price_level']].loc[data.day==5]
    item_price_5 = temp.groupby(['item_id'], as_index=False)['item_price_level'].agg({'price_5': 'mean'})
    data = pd.merge(data, item_price_5, how='left', on='item_id')
    del temp
    gc.collect()
    print('item_price_level')
    temp = data[['item_id', 'item_price_level']].loc[data.day==6]
    item_price_6 = temp.groupby(['item_id'], as_index=False)['item_price_level'].agg({'price_6': 'mean'})
    data = pd.merge(data, item_price_6, how='left', on='item_id')
    del temp
    gc.collect()

    data['price_diff_7_6'] = data['item_price_level'] = data['price_6']
    data['price_diff_7_5'] = data['item_price_level'] = data['price_5']
    data['price_diff_7_4'] = data['item_price_level'] = data['price_4']

    del data['price_6']
    del data['price_5']
    del data['price_4']

    return data

def tillNow(data):
    
    for feat in ['user_id', 'shop_id', 'item_id', 'item_brand_id']:
        lists = data[feat].values
        dicts = defaultdict(lambda: 0)
        till_now_cnt = np.zeros(len(data))
        for i in range(len(data)):
            till_now_cnt[i] = dicts[lists[i]]
            dicts[lists[i]] += 1
        if(feat == 'item_brand_id'):
            data[feat.split('_')[1]+'_till_now_cnt'] = till_now_cnt
        else:
            data[feat.split('_')[0]+'_till_now_cnt'] = till_now_cnt

    return data

def main():
    path = './data/'
    
    train = pd.read_csv(path+'train_all.csv')
    test = pd.read_csv(path+'test_all.csv')

    data = pd.concat([train, test])

    print('初始维度:', data.shape)

    cols = data.columns.tolist()
    keys = ['instance_id', 'day']
    for k in keys:
        cols.remove(k)

    ##################################
    data = dolastCount(data)
    print('dolastCount:', data.shape)

    data = doNew(data)
    print('doNew:', data.shape)

    data = tillNow(data)
    print('tillNow:', data.shape)    
    ##################################

    data = data.drop(cols, axis=1)

    # 得到全部训练集
    # print('经过处理后,最终维度:', data.shape)
    # data.to_csv(path+'401_list_till_feat_all.csv', index=False)

    # 得到7号训练集
    data = data.loc[data.day==7]
    data = data.drop('day', axis=1)
    print('经过处理后,7号训练集最终维度:', data.shape)
    print(data.columns.tolist())
    data.to_csv(path+'401_list_till_feat.csv', index=False)
    
if __name__ == '__main__':
    main()
