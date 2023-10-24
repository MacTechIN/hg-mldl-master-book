#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04-1 로지스틱 휘귀 

Created on Mon Oct 23 16:09:15 2023

@author: sl

<전체프로세스>
01. fish.csv 읽어 오기 (판다스)
02.데이터 셋에서 물고기의 종류 7가지 확인
03. 데이터 셋에서 fish_input과  fish_target 을 numpy array로 나눠서 보관 
04. rain_test_split() 데이터셋 테스트셋 으로 데이터셋 분류 하기 
05. input data size 확인
06. 표준화 처리 : standardScaler를 사용 훈련세트의 통계값으로 테스트 세트  변환 (중요!!)
07. K-nearest neighbors classifier 클래스 이용 객체를 만들고 , train_input셋트로 훈련 하고 훈련셋트와 테스트세트의 점수 확인 하기 
08. train set 과 test set scorea 비교하기
09. test_scaled 로 예측 한 결과 확인 하기 

 
#로지스틱 회귀 
이진 부류 문제 클래스 확율을 예측 합니다.

"""

#%% 

# 01. fish.csv 읽어 오기 (판다스)
import pandas as pd 
import numpy as np 

fish = pd.read_csv('./fish.csv')
fish.head() 

#%%
# 02.데이터 셋에서 물고기의 종류 7가지 확인 

print(pd.unique(fish['Species']))

# ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']

#%%
# 03. 데이터 셋에서 fish_input과  fish_target 을 numpy array로 나눠서 보관 
fish_input = fish[['Weight', 'Length' , 'Diagonal' ,  'Height' , 'Width']].to_numpy()
fish_target = fish[['Species']].to_numpy()

print(fish_input[:5])
print(fish_target[:5])
#%%
# 04. rain_test_split() 데이터셋 테스트셋 으로 데이터셋 분류 하기 

from sklearn.model_selection import train_test_split

train_input,test_input, train_target, test_target = train_test_split(fish_input, fish_target)
#%%
# 05 input data size 확인 
print("train_input shape:", train_input.shape)
print("train_target shape:", test_input.shape)

#%% 06. 표준화 처리 : standardScaler를 사용 훈련세트의 통계값으로 테스트 세트  변환 (중요!!)

from sklearn.preprocessing  import StandardScaler

ss = StandardScaler()

ss.fit(train_input) # train_input -> scaling 

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 데이터 준비 완료 

#%%
# 07. K-nearest neighbors classifier 클래스 이용 객체를 만들고 , train_input셋트로 훈련 하고 훈련셋트와 테스트세트의 점수 확인 하기 
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3)

kn.fit (train_scaled, train_target) 
#%%
# 08 train set 과 test set scorea 비교하기 
#Score 
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target)) 

#%%

print(kn.classes_) 

#%%
# 09. test_scaled 로 예측 한 결과 확인 하기 
print(kn.predict(test_scaled[:5]))
#%%

pb 




