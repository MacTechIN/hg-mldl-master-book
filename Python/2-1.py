#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:43:33 2023

@author: sl
"""
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 
               475.0, 500.0,500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 
               575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 
               920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 
               9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


#%%
# 위에 두 데이터를 2차원 배열 fish_data 로 만듥 

fish_input = [[w,l] for w,l in zip(fish_length,fish_weight)]

print(fish_input)

#%%

fish_target = [1]*35 + [0]*14 

print(fish_target)

#%%
#KNNeighborsClassifier 이용 객체 만들기

from sklearn.neighbors import KNeighborsClassifier

Kn = KNeighborsClassifier()
#%%
 
print(fish_input[4])

#%%
#편향적이지 않는 학습 과 테스트 데이터 셋을 만들기위해 Numpy 라이브러리 사용 셔플을 이용해서 샘플을 섞어준다.

import numpy as np 

input_arr = np.array(fish_input)

target_arr =np.array(fish_target)

#%%
print(input_arr.shape)


#%%
# np random test 

np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)

print(index)

#%%
#Train set 셔플 
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

print(train_input[13], train_input[0])

#%%
# 테스트용 셔플 

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

print(test_input.shape)

#%% 
#도미와 빙어가 잘썩여있는지 확인하기 

import matplotlib.pyplot as plt 

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:,1])
plt.xlabel("length")
plt.ylabel('weight')
#%%
#학습시키기 fit() 

Kn.fit(train_input, train_target)
#%%
Kn.score(test_input, test_target)

#%%
# 검증하기 
Kn.predict(test_input)

print(test_target)


