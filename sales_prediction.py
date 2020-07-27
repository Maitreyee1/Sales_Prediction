# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time
import sys
import gc
import pickle
from itertools import product

from xgboost import XGBRegressor
from xgboost import plot_importance

train_data = pd.read_csv("sales_train.csv")
test_data = pd.read_csv("test.csv")
items_data = pd.read_csv("items.csv")
shops_data = pd.read_csv("shops.csv")
cate_data = pd.read_csv("item_categories.csv")

train_data.head()

train_data.shape

train_data.isnull().sum()

color = sns.color_palette("hls", 8)
sns.set(style="darkgrid")
plt.figure(figsize=(15, 5))
sns.countplot(x=train_data['shop_id'], data=train_data, palette=color)

plt.figure(figsize=(10, 5))
sns.distplot(train_data['item_id'], color="red");

plt.figure(figsize=(10, 5))
sns.distplot(train_data['item_price'], color="red");

plt.figure(figsize=(10, 5))
sns.distplot(np.log(train_data['item_price']), color="red");

plt.figure(figsize=(10, 5))
train_data['item_cnt_day'].plot(kind='hist', alpha=0.7, color='blue')

test_data.head()

test_data.shape

len(test_data['shop_id'].unique())

len(test_data['item_id'].unique())

plt.figure(figsize=(10, 5))
sns.countplot(x=test_data['shop_id'], data=test_data)

plt.figure(figsize=(10, 5))
sns.distplot(test_data['item_id'], color="green");

"""#Feature Engineering
##Outliers

Outliers in item_cnt_day
"""

train_data['item_cnt_day'].sort_values(ascending=False).head(5)

plt.figure(figsize=(10, 5))
color = sns.color_palette("hls", 8)
plt.xlim(-200, 3500)
sns.boxplot(x=train_data.item_cnt_day, color="red", palette="Set3")

train_data = train_data[train_data.item_cnt_day<=1000]

plt.figure(figsize=(10, 5))
color = sns.color_palette("hls", 8)
plt.xlim(-200, 3500)
sns.boxplot(x=train_data.item_cnt_day, color="red", palette="Set3")

"""item price outliers"""

train_data['item_price'].sort_values(ascending=False).head()

plt.figure(figsize=(10, 5))
color = sns.color_palette("hls", 8)
plt.xlim(train_data.item_price.min(), train_data.item_price.max()*1.1)
sns.boxplot(x=train_data.item_price, color="red", palette="Set3")

train_data = train_data[train_data['item_price'] < 100000]

train_data[train_data['item_price'] < 0]

predict_mean_price = train_data[(train_data['date_block_num'] == 4) & (train_data['shop_id'] == 32) & (train_data['item_id'] == 2973) & (train_data['item_price'] > 0)]['item_price'].mean()
predict_mean_price

train_data.loc[train_data['item_price'] < 0, 'item_price'] = predict_mean_price

train_data[(train_data['date_block_num'] == 4) & (train_data['shop_id'] == 32) & (train_data['item_id'] == 2973) & (train_data['item_price'] > 0)]

"""#Analysis

Shops
"""

unq_train_shops = train_data['shop_id'].unique()
unq_test_shops = test_data['shop_id'].unique()
print(len(unq_train_shops))
print(len(unq_test_shops))

set(unq_test_shops).issubset(set(unq_train_shops))

shops_data.head()

def listToString(s):  
    
    # initialize an empty string 
    str1 = " " 
    
    # return string
    return (str1.join(s))

import re
for i, shop in enumerate(shops_data['shop_name']):
    if i == 6:
        pass
    else:
        string = shop
        cleanString = re.sub(r'[?|$|.|!]',r'',string)
        splitStr = cleanString.split()[:-1]
        finalStr = listToString(splitStr)
        result = shops_data[shops_data['shop_name'].str.contains(finalStr)]
        if result.shape == (2, 2) or result.shape == (0, 2):
            print(result)
            print('-'*50)

shops_data[shops_data['shop_name'].str.contains('Якутск ТЦ')]

train_data.loc[train_data.shop_id == 0, 'shop_id'] = 57
test_data.loc[test_data.shop_id == 0, 'shop_id'] = 57

train_data.loc[train_data.shop_id == 1, 'shop_id'] = 58
test_data.loc[test_data.shop_id == 1, 'shop_id'] = 58

train_data.loc[train_data.shop_id == 10, 'shop_id'] = 11
test_data.loc[test_data.shop_id == 10, 'shop_id'] = 11

"""City from shops"""

shops_data.head()

shops_data.loc[shops_data['shop_name'] == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops_data['city'] = shops_data['shop_name'].str.split(' ').map(lambda x:x[0])
shops_data.loc[shops_data['city'] == '!Якутск', 'city'] = 'Якутск'
shops_data.head()

city_label_encoder = preprocessing.LabelEncoder()
shops_data['city_code'] = city_label_encoder.fit_transform(shops_data['city'])

shops_data = shops_data[['shop_id','city_code']]
shops_data.head()

"""Item Analysis"""

items_data.head()

unq_train_item =  train_data['item_id'].unique()
unq_test_item =  test_data['item_id'].unique()
print(len(unq_train_item))
print(len(unq_test_item))

set(unq_test_item).issubset(set(unq_train_item))

len(set(unq_test_item).difference(set(unq_train_item)))

cat_in_test_data = items_data.loc[items_data['item_id'].isin(sorted(test_data['item_id'].unique()))].item_category_id.unique()
cat_in_test_data

cat_not_in_test = cate_data[~cate_data['item_category_id'].isin(cat_in_test_data)].item_category_id.unique()
cat_not_in_test

"""Category Analysis"""

cate_data.head()

splt_cate = cate_data['item_category_name'].str.split('-')
cate_data['main_cate'] = splt_cate.map(lambda x: x[0].strip())
cate_data['main_cate_id'] = preprocessing.LabelEncoder().fit_transform(cate_data['main_cate'])

cate_data['sub_cate'] = splt_cate.map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cate_data['sub_cate_id'] = preprocessing.LabelEncoder().fit_transform(cate_data['sub_cate'])

cate_data = cate_data[['item_category_id', 'main_cate_id', 'sub_cate_id']]

cate_data.head()

"""Matrix shop item"""

print('Item not in train:', len(set(unq_test_item).difference(set(unq_train_item))))
print('Total item in test:', len(test_data['item_id'].unique()))
print('Total shop in test:', len(test_data['shop_id'].unique()))

ts = time.time()
matrix = []

months = train_data.date_block_num.unique()
for month in months:
    sales = train_data[train_data.date_block_num == month]
    unq_shop = sales['shop_id'].unique()
    unq_item = sales['item_id'].unique()
    append_arr = np.array(list(product(unq_shop, unq_item, [month])), dtype='int16')
    matrix.append(append_arr)

cols = ['shop_id','item_id', 'date_block_num']
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix.head()

matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix.sort_values(['date_block_num','shop_id','item_id'],inplace=True)
matrix.head()

matrix.shape

color = sns.color_palette("hls", 8)
sns.set(style="darkgrid")
plt.figure(figsize=(15, 5))
sns.countplot(x=matrix['shop_id'], data=matrix, palette=color)

"""Aggregate Sale"""

train_data['revenue'] = train_data['item_price'] * train_data['item_cnt_day']
train_data.head()

group_data = train_data.groupby(by=['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': 'sum'})
group_data.columns = ['item_cnt_month']
group_data.reset_index(inplace = True)
group_data.head()

matrix = pd.merge(matrix, group_data, on=cols, how='left')
matrix.head()

matrix['item_cnt_month'] = (matrix['item_cnt_month'].fillna(0).clip(0, 20).astype(np.float16))
matrix.head()

matrix.shape

test_data.head()

test_data['date_block_num'] = 34
test_data['date_block_num'] = test_data['date_block_num'].astype(np.int8)
test_data['shop_id'] = test_data['shop_id'].astype(np.int8)
test_data['item_id'] = test_data['item_id'].astype(np.int16)

matrix = pd.concat([matrix, test_data], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True)
matrix = matrix.drop('ID', axis=1)
matrix.head()

"""Merge features"""

items_data.drop(['item_name'], axis=1, inplace=True)
items_data.head()

cate_data.head()

matrix = pd.merge(matrix, shops_data, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items_data, on=['item_id'], how='left')
matrix = pd.merge(matrix, cate_data, on=['item_category_id'], how='left')

matrix.head()

matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['main_cate_id'] = matrix['main_cate_id'].astype(np.int8)
matrix['sub_cate_id'] = matrix['sub_cate_id'].astype(np.int8)

"""#Implement Lags"""

matrix.head()

def generate_lag(df, months, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
#     print(tmp)
    for month in months:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id',col+'_lag_'+str(month)]
        shifted['date_block_num'] += month
#         print(month)
#         print(shifted)
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
                      
    return df

matrix = generate_lag(matrix, [1, 2, 3, 6, 12], 'item_cnt_month')
matrix.head()

"""#Feature Mean Encoding

data_block_num
"""

group = matrix.groupby(['date_block_num']).agg({'item_cnt_month' : ['mean']})
group.columns = [ 'date_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
matrix = generate_lag(matrix, [1], 'date_avg_item_cnt')
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)

"""item"""

group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix = generate_lag(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)

"""Shop"""

group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_shop_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
matrix = generate_lag(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)

"""item_category_id"""

group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_cat_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_category_id'], how='left')
matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
matrix = generate_lag(matrix, [1], 'date_cat_avg_item_cnt')
matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)

"""shop and item_category_id"""

group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_cat_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)
matrix = generate_lag(matrix, [1], 'date_shop_cat_avg_item_cnt')
matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)

"""shop and main_cate_id"""

group = matrix.groupby(['date_block_num', 'shop_id', 'main_cate_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_type_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'main_cate_id'], how='left')
matrix['date_shop_type_avg_item_cnt'] = matrix['date_shop_type_avg_item_cnt'].astype(np.float16)
matrix = generate_lag(matrix, [1], 'date_shop_type_avg_item_cnt')
matrix.drop(['date_shop_type_avg_item_cnt'], axis=1, inplace=True)

"""shop and sub_cate_id"""

group = matrix.groupby(['date_block_num', 'shop_id', 'sub_cate_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_subtype_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'sub_cate_id'], how='left')
matrix['date_shop_subtype_avg_item_cnt'] = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)
matrix = generate_lag(matrix, [1], 'date_shop_subtype_avg_item_cnt')
matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)

"""city_code"""

group = matrix.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'city_code'], how='left')
matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)
matrix = generate_lag(matrix, [1], 'date_city_avg_item_cnt')
matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)

"""item and city_code"""

group = matrix.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)
matrix = generate_lag(matrix, [1], 'date_item_city_avg_item_cnt')
matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)

"""Main category id"""

group = matrix.groupby(['date_block_num', 'main_cate_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_type_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'main_cate_id'], how='left')
matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)
matrix = generate_lag(matrix, [1], 'date_type_avg_item_cnt')
matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)

"""sub_cate_id"""

group = matrix.groupby(['date_block_num', 'sub_cate_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_subtype_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'sub_cate_id'], how='left')
matrix['date_subtype_avg_item_cnt'] = matrix['date_subtype_avg_item_cnt'].astype(np.float16)
matrix = generate_lag(matrix, [1], 'date_subtype_avg_item_cnt')
matrix.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)

matrix.head()

"""##Trend Analysis"""

train_data.head()

group = train_data.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_avg_item_price']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['item_id'], how='left')
matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)
matrix.head()

group = train_data.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_avg_item_price']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)
matrix.head()

lags = [1,2,3,4,5,6]
matrix = generate_lag(matrix, lags, 'date_item_avg_item_price')
matrix.head()

for i in lags:
    matrix['delta_price_lag_'+str(i)] = (matrix['date_item_avg_item_price_lag_'+str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

def select_trend(row):
    for i in lags:
        if row['delta_price_lag_'+str(i)]:
            return row['delta_price_lag_'+str(i)]
    return 0

matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
matrix['delta_price_lag'].fillna(0, inplace=True)

fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
for i in lags:
    fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
    fetures_to_drop += ['delta_price_lag_'+str(i)]

matrix.drop(fetures_to_drop, axis=1, inplace=True)

"""shop revenue trend"""

group = train_data.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
group.columns = ['date_shop_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
group.columns = ['shop_avg_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)

matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)

matrix = generate_lag(matrix, [1], 'delta_revenue')

matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)

"""Adding more features"""

matrix['month'] = matrix['date_block_num'] % 12
matrix.head()

"""Number of days in a month. There are no leap years"""

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)
matrix.head()

cache = {}
matrix['item_shop_last_sale'] = -1
matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)

for idx, row in matrix.iterrows():    
    key = str(row.item_id)+' '+str(row.shop_id)
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        matrix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
        cache[key] = row.date_block_num

cache = {}
matrix['item_last_sale'] = -1
matrix['item_last_sale'] = matrix['item_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():    
    key = row.item_id
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        if row.date_block_num>last_date_block_num:
            matrix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num

matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')

matrix = matrix[matrix.date_block_num > 11]

def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df

matrix = fill_na(matrix)

matrix.to_pickle('data.pkl')

gc.collect();

data = pd.read_pickle('data.pkl')

data = data[[
    'date_block_num',
    'shop_id',
    'item_id',
    'item_cnt_month',
    'city_code',
    'item_category_id',
    'main_cate_id',
    'sub_cate_id',
    'item_cnt_month_lag_1',
    'item_cnt_month_lag_2',
    'item_cnt_month_lag_3',
    'item_cnt_month_lag_6',
    'item_cnt_month_lag_12',
    'date_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_2',
    'date_item_avg_item_cnt_lag_3',
    'date_item_avg_item_cnt_lag_6',
    'date_item_avg_item_cnt_lag_12',
    'date_shop_avg_item_cnt_lag_1',
    'date_shop_avg_item_cnt_lag_2',
    'date_shop_avg_item_cnt_lag_3',
    'date_shop_avg_item_cnt_lag_6',
    'date_shop_avg_item_cnt_lag_12',
    'date_cat_avg_item_cnt_lag_1',
    'date_shop_cat_avg_item_cnt_lag_1',
    'date_city_avg_item_cnt_lag_1',
    'date_item_city_avg_item_cnt_lag_1',
    'delta_price_lag',
    'month',
    'days',
    'item_shop_last_sale',
    'item_last_sale',
    'item_shop_first_sale',
    'item_first_sale',
]]

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)

Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test_data.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('boost_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))
pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plot_features(model, (10,14))