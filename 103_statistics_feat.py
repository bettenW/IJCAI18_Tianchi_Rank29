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

def dorollWin(data):

    data['context_timestamp_str'] = data['context_timestamp'].astype(str)
    user_time_join = data.groupby('user_id')['context_timestamp_str'].agg(lambda x:';'.join(x)).reset_index()
    user_time_join.rename(columns={'context_timestamp_str':'user_time_join'},inplace = True)
    data = pd.merge(data,user_time_join,on=['user_id'],how='left')
    user_shop_time_join = data.groupby(['user_id','shop_id'])['context_timestamp_str'].agg(lambda x:';'.join(x)).reset_index()
    user_shop_time_join.rename(columns={'context_timestamp_str':'user_shop_time_join'},inplace = True)
    data = pd.merge(data,user_shop_time_join,on=['user_id','shop_id'],how='left')
    user_item_time_join = data.groupby(['user_id','item_id'])['context_timestamp_str'].agg(lambda x:';'.join(x)).reset_index()
    user_item_time_join.rename(columns={'context_timestamp_str':'user_item_time_join'},inplace = True)
    data = pd.merge(data,user_item_time_join,on=['user_id','item_id'],how='left')
    data['index_']=data.index
    del user_time_join,user_shop_time_join,user_item_time_join
    
    nowtime=data.context_timestamp.values
    user_time=data.user_time_join.values
    user_shop_time=data.user_shop_time_join.values
    user_item_time=data.user_item_time_join.values
    
    data_len=data.shape[0]
    user_time_10_bf=np.zeros(data_len)
    user_time_10_af=np.zeros(data_len)
    user_shop_time_10_bf=np.zeros(data_len)
    user_shop_time_10_af=np.zeros(data_len)
    user_item_time_10_bf=np.zeros(data_len)
    user_item_time_10_af=np.zeros(data_len)
    a=time.time()
    for i in range(data_len):
        df1=nowtime[i]
        df2=user_time[i].split(';')
        df2_len=len(df2)
        for j in range(df2_len):
            if ((int(df2[j])-df1)<600) & ((int(df2[j])-df1)>0):
                user_time_10_bf[i]+=1
            if ((int(df2[j])-df1)>-600) & ((int(df2[j])-df1)<0):
                user_time_10_af[i]+=1
        
        df3=user_shop_time[i].split(';')
        df3_len=len(df3)
        for j in range(df3_len):
            if ((int(df3[j])-df1)<600) & ((int(df3[j])-df1)>0):
                user_shop_time_10_bf[i]+=1
            if ((int(df3[j])-df1)>-600) & ((int(df3[j])-df1)<0):
                user_shop_time_10_af[i]+=1
                
        df4=user_item_time[i].split(';')
        df4_len=len(df4)
        for j in range(df4_len):
            if ((int(df4[j])-df1)<600) & ((int(df4[j])-df1)>0):
                user_item_time_10_bf[i]+=1
            if ((int(df4[j])-df1)>-600) & ((int(df4[j])-df1)<0):
                user_item_time_10_af[i]+=1
                
    print(time.time()-a)
    
    data['user_count_10_bf']=user_time_10_bf
    data['user_count_10_af']=user_time_10_af
    data['user_shop_count_10_bf']=user_shop_time_10_bf
    data['user_shop_count_10_af']=user_shop_time_10_af
    data['user_item_count_10_bf']=user_item_time_10_bf
    data['user_item_count_10_af']=user_item_time_10_af

    drops = ['context_timestamp_str', 'user_time_join', 'user_shop_time_join',
       'user_item_time_join', 'index_']
    data = data.drop(drops, axis=1)
    
    return data

def doSize(data):

    add = pd.DataFrame(data.groupby(["shop_id", "day"]).item_id.nunique()).reset_index()
    add.columns = ["shop_id", "day", "shop_item_unique_day"]
    data = data.merge(add, on=["shop_id", "day"], how="left")

    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_id_query_day'})
    data = pd.merge(data, user_query_day, how='left', on=['user_id', 'day'])
    
    data['min_10'] = data['minute'] // 10
    data['min_15'] = data['minute'] // 15
    data['min_30'] = data['minute'] // 30
    data['min_45'] = data['minute'] // 45
    
    # user 不同时间段点击次数
    min10_user_click = data.groupby(['user_id', 'day', 'hour', 'min_10']).size().reset_index().rename(columns={0:'min10_user_click'})
    min15_user_click = data.groupby(['user_id', 'day', 'hour', 'min_15']).size().reset_index().rename(columns={0:'min15_user_click'})
    min30_user_click = data.groupby(['user_id', 'day', 'hour', 'min_30']).size().reset_index().rename(columns={0:'min30_user_click'})
    min45_user_click = data.groupby(['user_id', 'day', 'hour', 'min_45']).size().reset_index().rename(columns={0:'min45_user_click'})

    data = pd.merge(data, min10_user_click, 'left', on=['user_id', 'day', 'hour', 'min_10'])
    data = pd.merge(data, min15_user_click, 'left', on=['user_id', 'day', 'hour', 'min_15'])
    data = pd.merge(data, min30_user_click, 'left', on=['user_id', 'day', 'hour', 'min_30'])
    data = pd.merge(data, min45_user_click, 'left', on=['user_id', 'day', 'hour', 'min_45'])
    
    del data['min_10']
    del data['min_15']
    del data['min_30']
    del data['min_45']

    return data

def doElse(data):

    pass

def main():
    path = './data/'

    train = pd.read_csv(path+'train_day7.csv')
    test = pd.read_csv(path+'test_day7.csv')

    data = pd.concat([train, test])
    print('初始维度:', data.shape)
    
    data, cols = pre_process(data)
    print('pre_process:', data.shape)
    
    ##################################
    data = dorollWin(data)
    print('dorollWin:', data.shape)

    data = doSize(data)
    print('doSize:', data.shape)
    ##################################
    
    data = data.drop(cols, axis=1)
    
    # 得到7号训练集
    data = data.loc[data.day == 7]
    data = data.drop('day', axis=1)
    print('经过处理后,7号数据集最终维度::',data.shape)
    print(data.columns.tolist())
    data.to_csv(path+'103_statistics_feat.csv', index=False)

if __name__ == '__main__':
    main()
