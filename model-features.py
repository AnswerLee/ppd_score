# -*- coding: utf-8 -*-
"""
Created on Sun May 20 12:53:49 2018

@author: AnswerLee
"""

#%%
import pandas as pd
import numpy as np
import importlib
import datetime
import collections
import numbers
import random
from itertools import combinations
import statsmodels as sm
from pandas import DataFrame,Series
import os
os.chdir(r'D:\give_me_five\githome\ppd-competition')
from scorecard_fucntions import *

#%%
# 载入数据，主要信息表、登录信息表、用户信息更新表
data_master = pd.read_csv('./dataset/PPD_Training_Master_GBK_3_1_Training_Set.csv',encoding='gbk')
data_loginfo = pd.read_csv('./dataset/PPD_LogInfo_3_1_Training_Set.csv',encoding='gbk')
data_update = pd.read_csv('./dataset/PPD_Userupdate_Info_3_1_Training_Set.csv',encoding='gbk')
#%%
######################################################################################################################################################
# 衍生变量
######################################################################################################################################################
# 四个代表城市的列，是否都一样，构建一个特征。
data_master.UserInfo_2 = data_master.UserInfo_2.map(lambda x:str(x).strip().replace('市',''))
data_master.UserInfo_4 = data_master.UserInfo_4.map(lambda x:str(x).strip().replace('市',''))
data_master.UserInfo_8 = data_master.UserInfo_8.map(lambda x:str(x).strip().replace('市',''))
data_master.UserInfo_20 = data_master.UserInfo_20.map(lambda x:str(x).strip().replace('市',''))
data_master['city_match'] = data_master.apply(lambda x:int(x.UserInfo_2 == x.UserInfo_4 == x.UserInfo_8 == x.UserInfo_20),axis=1)
# 剔除掉这四个城市列
data_master.drop(['UserInfo_2','UserInfo_4','UserInfo_8','UserInfo_20'],axis=1,inplace=True)
data_master.shape
#%%
# 构建登录信息的衍生变量
# 构建出每次登录和申请时间的间隔
data_loginfo['LogInfo'] = data_loginfo.LogInfo3.map(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d'))
data_loginfo['ListingInfo'] = data_loginfo.Listinginfo1.map(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d'))
data_loginfo['LogGap'] = data_loginfo[['LogInfo','ListingInfo']].apply(lambda x:(x[1]-x[0]).days,axis=1)
#%%
# 时间窗口选择  根据每个时间窗口对所有gap的覆盖率，得出选择180天
# 使用180天作为data1中的特征的最大时间跨度，一般可以选用7天，30天，
# 60天，90天，120天，150天，180天等。通过计算时间跨度内样本的总数以及非
# 重复的样本数来确定。
timeWindows = TimeWindowSelection(data_loginfo,'LogGap',range(30,361,30))
timeWindows
#%%
# 构建出LogInfo1和LogInfo2在每个时间窗口中的总次数，总类别数，以及类别平均次数
time_window = [7,30,60,90,120,150,180]
var_list = ['LogInfo1','LogInfo2']
datal_Idx = DataFrame({'Idx':data_loginfo.Idx.drop_duplicates()})
for tw in time_window:
    data_loginfo['TruncatedLogInfo'] = data_loginfo['ListingInfo'].map(lambda x:x+datetime.timedelta(-tw))
    temp = data_loginfo.loc[data_loginfo['LogInfo'] >= data_loginfo['TruncatedLogInfo']]
    for var in var_list:
        # 统计LogInfo1和LogInfo2分别的总次数
        col_count = str(var)+'_'+str(tw)+'_count'
        col_unique = str(var)+'_'+str(tw)+'_unique'
        col_avg_count = str(var)+'_'+str(tw)+'_avg_count'
        count_stats = temp.groupby(['Idx'])[var].count().to_dict()
        datal_Idx[col_count] = datal_Idx['Idx'].map(lambda x:count_stats.get(x,0))
        
        # 统计LogInfo1和LogInfo2分别的类别数，
        Idx_UserupdateInfo1 = temp[['Idx',var]].drop_duplicates()
        uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])[var].count().to_dict()
        datal_Idx[col_unique] = datal_Idx['Idx'].map(lambda x:uniq_stats.get(x,0))
        
        # 统计LogInfo1 和LogInfo2 分别的平均次数
        datal_Idx[col_avg_count] = datal_Idx[[col_count,col_unique]].apply(lambda x:round(x[0]*1.0/x[1]),axis=1)
        # x[1] 为0 时，返回NaN，使用0填充
        datal_Idx[col_avg_count].fillna(0,inplace=True)
#%%
# 构建修改信息表的衍生变量
# 计算出用户修改信息到申请时间的间隔
data_update['UpdateInfo'] = data_update.UserupdateInfo2.map(lambda x:datetime.datetime.strptime(x,'%Y/%m/%d'))
data_update['ListingInfo'] = data_update.ListingInfo1.map(lambda x:datetime.datetime.strptime(x,'%Y/%m/%d'))
data_update['UpdateGap'] = data_update[['UpdateInfo','ListingInfo']].apply(lambda x:(x[1]-x[0]).days,axis=1)
# 处理更新信息表单中的数据，观察发现UserupdateInfo1中存在同一个意思，
# 但是大小写不统一的情况，所以先对大小写进行处理。且将MobilePhone和Phone统一称Phone。
data_update['UpdateInfo_upper'] = data_update.UserupdateInfo1.map(ChangeContent)
#%%
datau_Idx = DataFrame({'Idx':data_update.Idx.drop_duplicates()})
time_window = [7,30,60,90,120,150,180]
for tw in time_window:
    col_freq = 'UpdateInfo_'+str(tw)+'_freq'
    col_unique = 'UpdateInfo_'+str(tw)+'_unique'
    col_avg_count = 'UpdateInfo_'+str(tw)+'_avg_count'
    
    data_update['TruncatedLogInfo'] = data_update['ListingInfo'].map(lambda x:x+datetime.timedelta(-tw))
    temp = data_update.loc[data_update.UpdateInfo >= data_update['TruncatedLogInfo']]

    # 更新 的总次数
    freq_stats = temp.groupby(['Idx'])['UpdateInfo_upper'].count().to_dict()
    datau_Idx[col_freq] = datau_Idx['Idx'].map(lambda x:freq_stats.get(x,0))
    
    # 更新信息有多少类
    Idx_UserupdateInfo1 = temp[['Idx','UpdateInfo_upper']].drop_duplicates()
    uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])['UpdateInfo_upper'].count().to_dict()
    datau_Idx[col_unique] = datau_Idx['Idx'].map(lambda x:uniq_stats.get(x,0))
    
    # 更新平均值（每类）
    datau_Idx[col_avg_count] = datau_Idx[[col_freq, col_unique]].apply(lambda x:round(x[0]*1.0/x[1]),axis=1)
    # x[1] 为0 时，返回NaN，使用0填充
    datau_Idx.fillna(0,inplace=True)
    
    # 改变的类别是_IDNUMBER,_HASBUYCAR,_MARRIAGESTATUSID,_PHONE
    # 先将每一个item变成list，之后再求sum，而list的sum就是合并
    # 合并之后再分别使用每个item查询in
    Idx_UserupdateInfo1['UpdateInfo_upper'] = Idx_UserupdateInfo1['UpdateInfo_upper'].map(lambda x:[x])
    Idx_UserupdateInfo1_V2=Idx_UserupdateInfo1.groupby(['Idx'])['UpdateInfo_upper'].sum()
    for item in ['_IDNUMBER','_HASBUYCAR','_MARRIAGESTATUSID','_PHONE']:
        item_dict = Idx_UserupdateInfo1_V2.map(lambda x:int(item in x)).to_dict()
        datau_Idx['UpdateInfo_'+str(tw)+str(item)]=datau_Idx['Idx'].map(lambda x:item_dict.get(x,0))
#%%
# 存储构建变量后的表，all_data_0.csv
all_data_0 = pd.concat([data_master.set_index('Idx'),datal_Idx.set_index('Idx'),datau_Idx.set_index('Idx')],axis=1)
all_data_0.to_csv('./dataset/all_data_0.csv',encoding='gbk',index=True)
#%%
######################################################################################################################################################
# 缺失值处理
######################################################################################################################################################
# 读取all_data_0.csv数据
data_all = pd.read_csv('./dataset/all_data_0.csv',encoding='gbk')
allFeatures = list(data_all.columns)
allFeatures.remove('Idx')
allFeatures.remove('target')
allFeatures.remove('ListingInfo')
#%%
# UserInfo9特征处理，'中国移动 ' 和'中国移动' 是一个类别，去除空格
data_all.UserInfo_9 = data_all.UserInfo_9.map(lambda x:str(x).strip())
#%%
# 删除特征的值为常量的特征
for col in allFeatures:
    unique = set(data_all[col])
    if len(unique) == 1:
        print('{} is deleted'.format(col))
        data_all.drop([col],axis=1,inplace=True)
        allFeatures.remove(col)
#%%
# 把类别型的变量和数值型的变量分开
# 原则是小于10个类别的数值型视作类别型变量来处理
# 大于10个类别的数值型变量视作数值型变量
numerical_cols = []
for col in allFeatures:
    unique = list(set(data_all[col]))
    if np.nan in unique:
        unique.remove(np.nan)
    if len(unique) >= 10 and isinstance(unique[0],numbers.Real):
        numerical_cols.append(col)
categorical_cols = [i for i in allFeatures if i not in numerical_cols]
#%%
# 对于类别型变量，如果缺失值的比例占到50%以上，那么把它移除。
# 否则使用一个特殊的状态来填充缺失值。
missing_rate_threshold_c = 0.5
for col in categorical_cols:
    rate = MissingCategorial(data_all,col)
    print('{0} has missing rate as {1}'.format(col,rate))
    if rate > missing_rate_threshold_c:
        print('drop',col)
        data_all.drop([col],axis=1,inplace=True)
        categorical_cols.remove(col)
        allFeatures.remove(col)
    if 0 < rate <= missing_rate_threshold_c:
        data_all[col] = data_all[col].map(lambda x:str(x).upper())
#%%   
# 对于连续性变量，缺失值的比例超过0.3的直接剔除
# 未超过0.3的缺失值使用该列值的中位数
# 这个地方的缺失值填充能否使用多重插补？
# 是否可以使用该列值集合的随机值？
missing_rate_threshold_n = 0.3
for col in numerical_cols:
    rate = MissingContinuous(data_all,col)
    print('{0} has missing rate as {1}'.format(col,rate))
    if rate > missing_rate_threshold_n:
        data_all.drop(col,axis=1,inplace=True)
        numerical_cols.remove(col)
        allFeatures.remove(col)
        print('we drop variable {} because of its high missing rate'.format(col))
    elif rate > 0:
#使用随机值
#            not_missing = data_all[data_all[col] == data_all[col]][col]
#            makeuped = data_all[col].map(lambda x: MakeupRandom(x, list(not_missing)))
#            data_all.drop(col,axis=1,inplace=True)
            data_all[col].fillna(data_all[col].dropna().median(),inplace=True)
            missingRate2 = MissingContinuous(data_all, col)
            print('missing rate after making up is:{}'.format(str(missingRate2)))  

#%%
######################################################################################################################################################
# 变量分箱，计算WOE和IV值
######################################################################################################################################################
deleted_features = []
encoded_features = {}
merged_features = {}
var_IV = {}
var_WOE = {}
# 处理类别型变量
# 类别型变量的类别个数大于5，则将类别使用bad_rate来替代，变成数值型变量
# 类别形变量的类别个数小于等于5
# 查看其中有一个类别的占比是否超过90%，超过则剔除掉
# 不超过90%，查看其中最小的badrate值，如果为0，则和最低的badrate类别合并
# 对于所有类别都不存在badrate等于0的列，计算WOE和IV值，返回WOE字典和IV值
for col in categorical_cols:
    if len(set(data_all[col])) > 5:
        print('{} is encoded with bad rate'.format(col))
        col0 = str(col) + '_encoding'
        encoding_result = Func_BadRateEncoding(data_all,col,'target')
        data_all[col0] = encoding_result['encoding']
        br_encoding = encoding_result['br_rate']
        numerical_cols.append(col0)
        encoded_features[col] = [col0,br_encoding]
        deleted_features.append(col)
    else:
        maxPcnt = Func_MaximumBinPcnt(data_all,col)
        if maxPcnt > 0.9:
            print('{} is deleted because of large percentage of single bin'.format(col))
            deleted_features.append(col)
            categorical_cols.remove(col)
            continue
        else:
            bad_rate = data_all.groupby([col]).sum()
            if min(bad_rate) == 0:
                print('{} has 0 bad sample!'.format(col))
                col1 = col + '_mergeByBadRate'
                mergeBin = Func_MergeBad0(data_all,col,'target')
                data_all[col1] = data_all[col].map(mergeBin)
                maxPcnt = Func_MaximumBinPcnt(data_all,col1)
                if maxPcnt > 0.9:
                    print('{} is deleted because of large percentage of single bin'.format(col1))
                    deleted_features.append(col)
                    categorical_cols.remove(col)
                    continue
                merged_features[col] = [col1,mergeBin]
                WOE_IV = Func_CalcWOE(data_all,col1,'target')
                var_WOE[col] = WOE_IV['WOE']
                var_IV[col] = WOE_IV['IV']
            else:
                WOE_IV = Func_CalcWOE(data_all,col,'target')
                var_WOE[col] = WOE_IV['WOE']
                var_IV[col] = WOE_IV['IV']
#%%
var_cutoff = {}
for col in numerical_cols:
    print ("{} is in processing".format(col))
    col1 = str(col) + '_Bin'
    #(1), split the continuous variable and save the cutoff points. Particulary, -1 is a special case and we separate it into a group
    if -1 in set(data_all[col]):
        special_attribute = [-1]
    else:
        special_attribute = []
    cutOffPoints = ChiMerge_MaxInterval(data_all, col, 'target',special_attribute=special_attribute)
    var_cutoff[col] = cutOffPoints
    data_all[col1] = data_all[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))

    #(2), check whether the bad rate is monotone
    BRM = BadRateMonotone(data_all, col1, 'target',special_attribute=special_attribute)
    if not BRM:
        for bins in range(4,1,-1):
            cutOffPoints = ChiMerge_MaxInterval(data_all, col, 'target',max_interval = bins,special_attribute=special_attribute)
            data_all[col1] = data_all[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
            BRM = BadRateMonotone(data_all, col1, 'target',special_attribute=special_attribute)
            if BRM:
                break
        var_cutoff[col] = cutOffPoints

    #(3), check whether any single bin occupies more than 90% of the total
    maxPcnt = MaximumBinPcnt(data_all, col1)
    if maxPcnt > 0.9:
        deleted_features.append(col)
        numerical_cols.remove(col)
        print ('we delete {} because the maximum bin occupies more than 90%'.format(col))
        continue
    WOE_IV = CalcWOE(data_all, col1, 'target')
    var_IV[col] = WOE_IV['IV']
    var_WOE[col] = WOE_IV['WOE']

