#coding:utf-8
import pandas as pd
import numpy as np
import time
import datetime
import gc

def pre_process(data):
    
    df = data[['shop_id', 'item_brand_id', 'item_id', 'item_category_1','item_pv_level','item_sales_level','item_collected_level',
              'item_price_level','context_page_id', 'day', 'hour', 'maphour', 'context_timestamp', 'user_id', 'instance_id']]
    del data
    gc.collect()

    cols = df.columns.tolist()
    keys = ['instance_id', 'day']
    for k in keys:
        cols.remove(k)

    return df, cols

def user_check(df, behaviour):

    df.sort_values(['user_id', 'context_timestamp'], inplace=True)

    user_day = df.groupby(['user_id', 'day', behaviour]).size().reset_index().rename(columns={0: 'user_id_query_day_{}'.format(behaviour)})
    df = pd.merge(df, user_day, how = 'left', on=['user_id', 'day',behaviour])
    user_day_hour = df.groupby(['user_id', 'day', 'hour', behaviour]).size().reset_index().rename(columns={0: 'user_id_query_day_hour_{}'.format(behaviour)})
    df = pd.merge(df, user_day_hour, how = 'left', on=['user_id', 'day', 'hour',behaviour])
    user_day_hour_map = df.groupby(['user_id', 'day', 'maphour', behaviour]).size().reset_index().rename(columns={0: 'user_id_query_day_hour_map_{}'.format(behaviour)})
    df = pd.merge(df, user_day_hour_map, how = 'left', on=['user_id', 'day', 'maphour',behaviour])

    n = 0
    check_time_day = np.ones((len(df),1))
    check_time_difference_last = np.ones((len(df),1))
    num = {}
    timeseries = {}
    bd = df.day.min()
    for u, i, d in zip(df.user_id, df[behaviour], df.day):
        n += 1
        try:
            num[(u,i)] += 1
            # timeseries[(u,i)] = df.min_series_full[n-1] - timeseries[(u,i)]
            check_time_difference_last[n-1] = df.context_timestamp[n-1] - timeseries[(u,i)]
            timeseries[(u,i)] = df.context_timestamp[n-1]
        except:
            num[(u,i)] = 0
            timeseries[(u,i)] = df.context_timestamp[n-1]
            check_time_difference_last[n-1] = -1

        check_time_day[n-1] = num[(u,i)]
        if d > bd:
            num = {}
        bd = d
    check_time_difference_next = np.ones((len(df),1))
    timeseries = {}
    for i in range(len(df)): #df.user_id[::-1]:
        u = df.user_id[len(df)-i-1]
        b = df[behaviour][len(df)-i-1]
        try:
            check_time_difference_next[len(df)-i-1] = timeseries[(u,b)]- df.context_timestamp[len(df)-i-1]
        except:
            check_time_difference_next[len(df)-i-1] = -1
        timeseries[(u,b)] = df.context_timestamp[len(df)-i-1]

    df['check_{}_min_diff_last'.format(behaviour)] = check_time_difference_last
    df['check_{}_min_diff_next'.format(behaviour)] = check_time_difference_next

    df['check_{}_time_day'.format(behaviour)] = check_time_day
    df['check_{}_ratio'.format(behaviour)] = df['check_{}_time_day'.format(behaviour)] / df['user_id_query_day_{}'.format(behaviour)]

    return df

def main():
    path = './data/'
    
    train = pd.read_csv(path+'train_all.csv')
    test = pd.read_csv(path+'test_all.csv')

    # train = pd.read_csv(path+'train_day7.csv')
    # test = pd.read_csv(path+'test_day7.csv')

    data = pd.concat([train, test])
    print('初始维度:', data.shape)
    
    #####################################
    data, cols = pre_process(data)
    print('pre_process:', data.shape)

    for f in ['shop_id', 'item_brand_id', 'item_id', 'item_category_1','item_pv_level','item_sales_level','item_collected_level',
              'item_price_level','context_page_id']:
        print(f,'starting...')
        data = user_check(data, f)
    #####################################

    data = data.drop(cols, axis=1)

    # 得到全部训练集
    print('经过处理后,全部训练集最终维度:', data.shape)
    data.to_csv(path+'301_timediff_last_next_feat_all.csv', index=False)

    # 得到7号训练集
    data = data.loc[data.day==7]
    data = data.drop('day', axis=1)
    print('经过处理后,7号训练集最终维度:', data.shape)
    print(data.columns.tolist())
    data.to_csv(path+'301_timediff_last_next_feat.csv', index=False)
    

if __name__ == '__main__':
    main()
