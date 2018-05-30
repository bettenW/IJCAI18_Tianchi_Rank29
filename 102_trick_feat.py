#coding:utf-8
import pandas as pd
import numpy as np
import time
import datetime
import gc
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

def pre_process(data):

    cols = data.columns.tolist()
    keys = ['instance_id', 'day']
    for k in keys:
        cols.remove(k)

    return data, cols
    

def doTrick1(data):

    data.sort_values(['user_id', 'context_timestamp'], inplace=True)
    
    subset = ['user_id', 'day']
    data['click_user_lab'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_lab'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_lab'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_lab'] = 3
    del pos
    gc.collect()

    subset = ['item_id', 'user_id', 'day']
    data['click_user_item_lab'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_item_lab'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_item_lab'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_item_lab'] = 3
    del pos
    gc.collect()

    subset = ['item_brand_id','user_id', 'day']
    data['click_user_brand_lab'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_brand_lab'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_brand_lab'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_brand_lab'] = 3
    del pos
    gc.collect()

    subset = ['shop_id','user_id', 'day']
    data['click_user_shop_lab'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_shop_lab'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_shop_lab'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_shop_lab'] = 3
    del pos
    gc.collect()

    subset = ['item_city_id','user_id', 'day']
    data['click_user_city_lab'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_city_lab'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_city_lab'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_city_lab'] = 3
    del pos
    gc.collect()


    return data

def doTrick2(data):

    data.sort_values(['user_id', 'context_timestamp'], inplace=True)

    #user_id
    subset = ['user_id', 'day']
    temp = data.loc[:,['context_timestamp', 'user_id', 'day']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'u_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['u_day_diffTime_first'] = data['context_timestamp'] - data['u_day_diffTime_first']
    del temp
    gc.collect()
    temp = data.loc[:,['context_timestamp', 'user_id', 'day']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'u_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['u_day_diffTime_last'] = data['u_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['u_day_diffTime_first', 'u_day_diffTime_last']] = -1

    #item_id
    subset = ['item_id', 'day']
    temp = data.loc[:,['context_timestamp', 'item_id', 'day']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'i_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['i_day_diffTime_first'] = data['context_timestamp'] - data['i_day_diffTime_first']
    del temp
    gc.collect()
    temp = data.loc[:,['context_timestamp', 'item_id', 'day']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'i_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['i_day_diffTime_last'] = data['i_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['i_day_diffTime_first', 'i_day_diffTime_last']] = -1

    #item_brand_id, user_id
    subset = ['item_brand_id', 'user_id', 'day']
    temp = data.loc[:,['context_timestamp', 'item_brand_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'b_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['b_day_diffTime_first'] = data['context_timestamp'] - data['b_day_diffTime_first']
    del temp
    gc.collect()
    temp = data.loc[:,['context_timestamp', 'item_brand_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'b_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['b_day_diffTime_last'] = data['b_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['b_day_diffTime_first', 'b_day_diffTime_last']] = -1
    
    #shop_id, user_id
    subset = ['shop_id', 'user_id', 'day']
    temp = data.loc[:,['context_timestamp', 'shop_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 's_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['s_day_diffTime_first'] = data['context_timestamp'] - data['s_day_diffTime_first']
    del temp
    gc.collect()
    temp = data.loc[:,['context_timestamp', 'shop_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 's_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['s_day_diffTime_last'] = data['s_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['s_day_diffTime_first', 's_day_diffTime_last']] = -1

    return data

def lasttimeDiff(data):
    for column in ['user_id', 'item_id']:
        gc.collect()
        data[column+'_lasttime_diff'] = 0
        train_data = data[['context_timestamp', column, column+'_lasttime_diff']].values
        lasttime_dict = {}
        for df_list in train_data:
            if df_list[1] not in lasttime_dict:
                df_list[2] = -1
                lasttime_dict[df_list[1]] = df_list[0]
            else:
                df_list[2] = df_list[0] - lasttime_dict[df_list[1]]
                lasttime_dict[df_list[1]] = df_list[0]
        data[['context_timestamp', column, column+'_lasttime_diff']] = train_data
    return data

def nexttimeDiff(data):
    for column in ['user_id', 'item_id']:
        gc.collect()
        data[column+'_nexttime_diff'] = 0
        train_data = data[['context_timestamp', column, column+'_nexttime_diff']].values
        nexttime_dict = {}
        for df_list in train_data:
            if df_list[1] not in nexttime_dict:
                df_list[2] = -1
                nexttime_dict[df_list[1]] = df_list[0]
            else:
                df_list[2] = nexttime_dict[df_list[1]] - df_list[0]
                nexttime_dict[df_list[1]] = df_list[0]
        data[['context_timestamp', column, column+'_nexttime_diff']] = train_data

    return data

def main():
    path = './data/'
    
    train = pd.read_csv(path+'train_all.csv')
    test = pd.read_csv(path+'test_all.csv')

    data = train.append(test, ignore_index=True)

    data, cols = pre_process(data)
    print('pre_process data:', data.shape)

    ###########挖掘新的特征###########

    # 对不同点击进行标记
    data = doTrick1(data)
    print('doTrick1 data:', data.shape)

    # 同一天点击时间差
    data = doTrick2(data)
    print('doTrick2 data:', data.shape)
    
    # 单特征距离上一次点击时间差
    data = lasttimeDiff(data)
    print('lasttimeDiff data:', data.shape)
    
    # 单特征距离下一次点击时间差
    data = nexttimeDiff(data)
    print('lasttimeDiff data:', data.shape)

    ############挖掘新的特征###########

    data = data.drop(cols, axis=1)

    # 得到全部训练集
    print('经过处理后,全部训练集最终维度:', data.shape)
    data.to_csv(path+'102_trick_feat_all.csv', index=False)

    # 得到7号训练集
    data = data.loc[data.day == 7]
    data = data.drop('day', axis=1)
    print('经过处理后,7号数据集最终维度:', data.shape)
    print(data.columns.tolist())
    data.to_csv(path+'102_trick_feat.csv', index=False)
    

if __name__ == '__main__':
    main()
