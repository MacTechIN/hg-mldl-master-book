#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04-1 로지스틱 휘귀 

Created on Mon Oct 23 16:09:15 2023

@author: sl

#로지스틱 회귀 
이진 부류 문제 클래스 확율을 예측 합니다.

"""

#%% 
import pandas as pd 
import numpy as np 

fish = pd.read_csv('./fish.csv')
fish.head() 

#%%
print(pd.unique(fish['Species']))

# ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']

#%%
fish_input = fish[['Weight', 'Length' , 'Diagonal' ,  'Height' , 'Width']].to_numpy()
fish_target = fish[['Species']].to_numpy()

print(fish_input[:5])
#%%
# train_test_split() 데이터셋 테스트셋 으로 데이터셋 분류 하기 

from sklearn.model_selection import train_test_split

train_input,test_input, train_target,  test_target = train_test_split(fish_input, )

