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


def doTime(data):
    
    # item_id expose hour
    exp_hour = data.groupby(['item_id', 'hour']).instance_id.nunique().to_frame()
    exp_hour.columns = ['item_hour_cnt']
    exp_hour = exp_hour.reset_index()
    exp_hour['item_hour_cnt_ratio'] = round(exp_hour.item_hour_cnt / exp_hour.groupby('item_id').item_hour_cnt.transform(np.sum), 5)
    exp_hour = exp_hour[['item_id', 'hour', 'item_hour_cnt_ratio']]
    data = data.merge(exp_hour, how='left', on=['item_id', 'hour'])
    
    # item_id expose maphour
    exp_tz = data.groupby(['item_id', 'maphour']).instance_id.nunique().to_frame()
    exp_tz.columns = ['item_maphour_cnt']
    exp_tz = exp_tz.reset_index()
    exp_tz['item_maphour_cnt_ratio'] = round(exp_tz.item_maphour_cnt / exp_tz.groupby('item_id').item_maphour_cnt.transform(np.sum), 5)
    exp_tz = exp_tz[['item_id', 'maphour', 'item_maphour_cnt_ratio']]
    data = data.merge(exp_tz, how='left', on=['item_id', 'maphour'])
    
    # user_id expose hour
    exp_hour = data.groupby(['user_id', 'hour']).instance_id.nunique().to_frame()
    exp_hour.columns = ['user_hour_cnt']
    exp_hour = exp_hour.reset_index()
    exp_hour['user_hour_cnt_ratio'] = round(exp_hour.user_hour_cnt / exp_hour.groupby('user_id').user_hour_cnt.transform(np.sum), 5)
    exp_hour = exp_hour[['user_id', 'hour', 'user_hour_cnt_ratio']]
    data = data.merge(exp_hour, how='left', on=['user_id', 'hour'])

    # user_id expose maphour
    exp_tz = data.groupby(['user_id', 'maphour']).instance_id.nunique().to_frame()
    exp_tz.columns = ['user_maphour_cnt']
    exp_tz = exp_tz.reset_index()
    exp_tz['user_maphour_cnt_ratio'] = round(exp_tz.user_maphour_cnt / exp_tz.groupby('user_id').user_maphour_cnt.transform(np.sum), 5)
    exp_tz = exp_tz[['user_id', 'maphour', 'user_maphour_cnt_ratio']]
    data = data.merge(exp_tz, how='left', on=['user_id', 'maphour'])
    gc.collect()

    #重要：同一个context_timestamp下，有多次曝光的，转化率??
    add = pd.DataFrame(data.groupby(["user_id", "context_timestamp"]).instance_id.nunique()).reset_index()
    add.columns = ["user_id", "context_timestamp", "same_time_expo_cnt"]
    data = data.merge(add, on=["user_id", "context_timestamp"], how="left")

    return data

def doAvg(data):

    # 小时均值特征
    grouped = data.groupby('user_id')['hour'].mean().reset_index()
    grouped.columns = ['user_id', 'user_mean_hour']
    data = data.merge(grouped, how='left', on='user_id')
    grouped = data.groupby('item_id')['hour'].mean().reset_index()
    grouped.columns = ['item_id', 'item_mean_hour']
    data = data.merge(grouped, how='left', on='item_id')
    grouped = data.groupby('item_brand_id')['hour'].mean().reset_index()
    grouped.columns = ['item_brand_id', 'brand_mean_hour']
    data = data.merge(grouped, how='left', on='item_brand_id')
    grouped = data.groupby('shop_id')['hour'].mean().reset_index()
    grouped.columns = ['shop_id', 'shop_mean_hour']
    data = data.merge(grouped, how='left', on='shop_id')

    # 年龄均值特征
    grouped = data.groupby('user_id')['user_age_level'].mean().reset_index()
    grouped.columns = ['user_id', 'user_mean_age']
    data = data.merge(grouped, how='left', on='user_id')
    grouped = data.groupby('item_id')['user_age_level'].mean().reset_index()
    grouped.columns = ['item_id', 'item_mean_age']
    data = data.merge(grouped, how='left', on='item_id')
    grouped = data.groupby('item_brand_id')['user_age_level'].mean().reset_index()
    grouped.columns = ['item_brand_id', 'brand_mean_age']
    data = data.merge(grouped, how='left', on='item_brand_id')
    grouped = data.groupby('shop_id')['user_age_level'].mean().reset_index()
    grouped.columns = ['shop_id', 'shop_mean_age']
    data = data.merge(grouped, how='left', on='shop_id')

    return data

def doActive(data):
 
    #小时特征
    add = pd.DataFrame(data.groupby(["user_id"]).hour.nunique()).reset_index()
    add.columns = ["user_id", "user_active_hour"]
    data = data.merge(add, on=["user_id"], how="left")

    # 活跃item_id数特征
    add = pd.DataFrame(data.groupby(["item_category_list", "day"]).item_id.nunique()).reset_index()
    add.columns = ["item_category_list", "day", "category_day_active_item"]
    data = data.merge(add, on=["item_category_list", "day"], how="left")


    # 活跃city数特征
    add = pd.DataFrame(data.groupby(["user_id", "day"]).item_city_id.nunique()).reset_index()
    add.columns = ["user_id", "day", "user_day_active_city"]
    data = data.merge(add, on=["user_id", "day"], how="left")

    add = pd.DataFrame(data.groupby(["user_id", "day", "hour"]).item_city_id.nunique()).reset_index()
    add.columns = ["user_id", "day", "hour", "user_hour_active_city"]
    data = data.merge(add, on=["user_id", "day", "hour"], how="left")

    #活跃user数特征
    add = pd.DataFrame(data.groupby(["item_id", "day"]).user_id.nunique()).reset_index()
    add.columns = ["item_id", "day", "item_day_active_user"]
    data = data.merge(add, on=["item_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["shop_id", "day"]).user_id.nunique()).reset_index()
    add.columns = ["shop_id", "day", "shop_day_active_user"]
    data = data.merge(add, on=["shop_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_brand_id", "day"]).user_id.nunique()).reset_index()
    add.columns = ["item_brand_id", "day", "brand_day_active_user"]
    data = data.merge(add, on=["item_brand_id", "day"], how="left")
    add = pd.DataFrame(data.groupby(["item_category_list", "day"]).user_id.nunique()).reset_index()
    add.columns = ["item_category_list", "day", "category_day_active_user"]
    data = data.merge(add, on=["item_category_list", "day"], how="left")


    add = pd.DataFrame(data.groupby(["item_id", "day", "hour"]).user_id.nunique()).reset_index()
    add.columns = ["item_id", "day", "hour", "item_hour_active_user"]
    data = data.merge(add, on=["item_id", "day", "hour"], how="left")
    add = pd.DataFrame(data.groupby(["shop_id", "day", "hour"]).user_id.nunique()).reset_index()
    add.columns = ["shop_id", "day", "hour", "shop_hour_active_user"]
    data = data.merge(add, on=["shop_id", "day", "hour"], how="left")

    # 活跃shop数特征
    add = pd.DataFrame(data.groupby(["user_id", "day"]).shop_id.nunique()).reset_index()
    add.columns = ["user_id", "day", "user_day_active_shop"]
    data = data.merge(add, on=["user_id", "day"], how="left")

    # 活跃brand数特征 
    add = pd.DataFrame(data.groupby(["user_id", "day"]).item_brand_id.nunique()).reset_index()
    add.columns = ["user_id", "day", "user_day_active_brand"]
    data = data.merge(add, on=["user_id", "day"], how="left")

    add = pd.DataFrame(data.groupby(["user_id", "day", "hour"]).item_brand_id.nunique()).reset_index()
    add.columns = ["user_id", "day", "hour", "user_hour_active_brand"]
    data = data.merge(add, on=["user_id", "day", "hour"], how="left")

    return data

def item_mean_ratio(df):

    # shop_id仅求均值全局
    df_shop_item = df[['shop_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']]
    df_shop_item.columns = ['shop_id', 'mean_item_price_level', 'mean_item_sales_level', 'mean_item_collected_level', 'mean_item_pv_level']
    shop_item = df_shop_item.groupby(['shop_id']).mean().reset_index()
    df = pd.merge(df, shop_item, 'left', on='shop_id')
    del df_shop_item
    del shop_item

    for colname in ['item_category_list', 'item_brand_id', 'item_city_id']:
        gc.collect()
        grouped = df.groupby([colname])
        meancols = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']
        df_g = grouped[meancols].mean().reset_index()
        colnames = [i for i in df_g.columns]
        for i in range(len(colnames)):
            if colnames[i] != colname:
                colnames[i] += '_mean_by_'+colname.split('_')[1]
        df_g.columns = colnames
        df=pd.merge(df, df_g, how='left', on=colname)
        colnames = colnames[1:]
        for i in range(len(colnames)):
            df[colnames[i]+'_ratio'] = round((df[meancols[i]]/df[colnames[i]]), 5)

    return df

def main():
    path = './data/'
   
    train = pd.read_csv(path+'train_all.csv')
    test = pd.read_csv(path+'test_all.csv')
    print(train['day'].unique())
    data = pd.concat([train, test])
    print('原始特征:', data.columns.tolist())
    print('初始维度:', data.shape)


    ###########挖掘新的特征###########
    
    data = doTime(data)
    print('doTime:', data.shape)

    data = doAvg(data)
    print('doAvg:', data.shape)

    data = doActive(data)
    print('doActive:', data.shape)

    data = item_mean_ratio(data)
    print('item_mean_ratio:', data.shape)

    ############挖掘新的特征###########

    del data['item_category_list']
    gc.collect()

    # 得到全部训练集
    print('经过处理后,全部训练集最终维度:', data.shape)
    data.to_csv(path+'101_wang_feat_all.csv', index=False)

    # 得到7号训练集
    data = data.loc[data.day==7]
    print('经过处理后,最终维度:', data.shape)
    print(data.columns.tolist())
    data.to_csv(path+'101_wang_feat.csv', index=False)
    

if __name__ == '__main__':
    main()
