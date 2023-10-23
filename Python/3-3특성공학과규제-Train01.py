#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:04:15 2023

전체 프로세스 (암기용)

1.농어의 무게 데이터 셋 준비 
2.농어의 길이, 높이, 두깨 : 세가지 데이터를 가진 데이터 셋을 다운(pd.read_csv)받아 numpy 2D array perch_full 로 저장
3.훈련셋트(train) , 테스트셋트(test)로 나눔 : sklearn.model_selection trin_test_split()  함수사용 
4.새로운 특성만들기  : sklearn PloynominalFeature transforming 사용 다항특성 만들기 
5.새로운 특성으로 LinearRegrssion 학습 하고 스코어링 하기 
6.고차원 차수(degree)에 대한 특성으로 과대 접합 상황 만들기  
7. 
@author: sl
"""

#%%
# 넘파이 데이터로 싸이킷 런 으로 사용 할 예정임. 
# 1.농어의 무게 데이터 셋 준비 
import numpy as np
import pandas as pd 


#농어의 무계 데이터 준비 
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0])

#%%
# 2.농어의 길이, 높이, 두깨 : 세가지 데이터를 가진 데이터 셋을 다운(pd.read_csv)받아 numpy 2D array perch_full 로 저장 함. 
df = pd.read_csv('http://bit.ly/perch_csv_data')
perch_full = df.to_numpy() 

print(perch_full)
#%%
# 3.훈련셋트(train) , 테스트셋트(test)로 나눔 : sklearn.model_selection trin_test_split()  함수사용 

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_full,perch_weight, random_state=42) 

print(train_input.shape)


#%%
# 4.새로운 특성만들기  : sklearn PloynominalFeature transforming 사용 다항특성 만들기 

from sklearn.preprocessing import PolynomialFeatures

#%% 
poly = PolynomialFeatures(include_bias= False)
poly.fit([[2,3,4]])
print(poly.transform([[2,3,4]]))
#특성끼리 곱하고, 제곱한 항 을 추가함. 자신, 곱셈, 각자제곱 , 1은 기본 : 1, 2,3,(2^2),(2*3),(3^2)  

#%%
print(poly.get_feature_names_out())


#%%
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)

print(train_poly.shape)

test_poly = poly.transform(test_input)

#%%
# 5. 새로운 특성으로 LinearRegrssion 학습 하고 스코어링 하기 
from sklearn.linear_model import LinearRegression 

lr = LinearRegression()

lr.fit(train_poly,train_target)
#%%

lr.score(test_poly,test_target)
# 0.9714559911594159 

#%%# 6. 고차원 차수(degree)에 대한 특성으로 과대 접합 상황 만들기  
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)

test_poly = poly.transform(test_input)
#%%

lr.fit(train_poly, train_target)

lr.score(train_poly, train_target)
#%%

lr.score(test_poly, test_target)3
#%%





































