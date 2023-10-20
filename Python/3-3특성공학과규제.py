#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:42:31 2023

@author: sl
"""

import pandas as pd

df = pd.read_csv("https://bit.ly/perch_csv_data")
#%%

perch_full = df.to_numpy() 
#길이 높이 두께 가 있음 

print(perch_full)
#%%
# 넘파이 데이터로 싸이킷 런 으로 사용 할 예정임. 
# 농어의 무게 데이터 셋 준비 
import numpy as np

perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )

#%%
#데이터 셋 나누기  기존 나누는 방식과 동일 

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

#%%
from sklearn.preprocessing import PolynomialFeatures 

#degree = 2 (제곱을 의미함)
poly = PolynomialFeatures(include_bias=False)

poly.fit ([[1,2,3]])

#%%
array = np.array([1,2,3])
print(array)
#%%
value = poly.transform([array])

#%%
print(value)

#%%
print(poly.get_feature_names_out())

#%%

#Train_input에 대한 PolynominalFeature(다항특성) 만들기 
po = PolynomialFeatures(include_bias= False)

#%%
po.fit(train_input) 

transform_result = po.transform(train_input)
#%%
print(transform_result)

#%%
print(po.get_feature_names_out())

#%%
print(transform_result.shape)


#%% 






































