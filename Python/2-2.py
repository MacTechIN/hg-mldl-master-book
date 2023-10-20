#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:45:27 2023

@author: sl
"""

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

print(len(fish_length))
print(len(fish_weight))
#%%
#위에 두 데이터를 합쳐서 하나의 fish_data 로 만든다.

fish_data = [[l,w] for l,w in zip(fish_length, fish_weight)]
#%%
# 도미 35마리, 빙어 49-35 = 14 
fish_target = [1] * 35 + [0] * 14 

print(fish_target)
#%%
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()


#%%
# 훈련 데이터 셋과 트레이닝 데이터셋을 나누는 방법 

train_input = fish_data[:35] 
train_target = fish_target[:35]

test_input = fish_data[35:]
test_target = fish_target[35:]

#%%

kn.fit(train_input, train_target)
kn.score(test_input, test_target)#정확도가 0.0으로 나옴 
#%%
#트레인 데이터셋에 도미와 빙어가 함께 썩여 있어야 의미가 있음
#편향적(bais) 데이터 셋트임. 
#골고루 썩어야함. 

import numpy as np 

fish_data_arr = np.array(fish_data)
target_data_arr = np.array(fish_target)

#%%
print(fish_data_arr)
print(target_data_arr)

#%%
print(fish_data_arr.shape)




