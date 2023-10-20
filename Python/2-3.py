#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:07:40 2023
2-2 데이터 전처리 
@author: sl
"""
#%%

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 
               475.0, 500.0,500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 
               575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 
               920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 
               9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


#%% #1 Numpy 로 fish_data 만들기 
import numpy as np 

fish_data = np.column_stack((fish_length, fish_weight)) 
print(fish_data.shape)
#%% #2 fish_target 만들기 

fish_target = np.concatenate((np.ones(35),np.zeros(14)))
print(fish_target)
#%% #3 사이킷으로 train set , test set 나누기 

from sklearn.model_selection import train_test_split
 
train_input, test_input, train_target, test_target = train_test_split(fish_data,fish_target, stratify= fish_target, random_state= 42 )
#%%
from sklearn.neighbors import KNeighborsClassifier

Kn = KNeighborsClassifier()
#%%
#훈련시키기 
Kn.fit(train_input, train_target)

#%%
#확인하기 
Kn.score(test_input, test_target)

#%% 검증하기 25Cm, `150Gram 물고기는 어느 과에 속하는가 ?
print(Kn.predict([[25,150]]))

#빙어로 나옴 아래 그래프로 확인하기 

#%% #4 Matplot 그리기


import matplotlib.pyplot as plt 

plt.scatter(train_input[:,0],train_input[:,1])
plt.scatter(25,150, marker='^')
plt.xlabel("Length")
plt.ylabel("Weight")
plt.show()

#그래프로 그려본결과 도미가 가까운데 0으로 나옴 

#%%
#Kn.kneighbors 로 거리 및 인덱스 확인 

distance, indexes = Kn.kneighbors([[25,150]])

plt.scatter(train_input[:,0],train_input[:,1])
plt.scatter(25,150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1],marker='D')
plt.xlabel("Length")
plt.ylabel("Weight")
plt.show()
#확인결과 빙어가 4개로 나옴 (0)
#%%
print(train_input[indexes])
print(train_target[indexes])

#%% 
print(distance)
#거리에 문제 발견 (90과 130 차이가 너무 크게 발생 X축과 Y축에 단위가 다름 )
#%%
distance, indexes = Kn.kneighbors([[25,150]])

plt.scatter(train_input[:,0],train_input[:,1])
plt.scatter(25,150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1],marker='D')
plt.xlim((0,1000))
plt.xlabel("Length")
plt.ylabel("Weight")
plt.show()
#%%
#데이터 전처리 
#표준점수 = (입력값- 평균) / 표준편차(std) 

mean = np.mean(train_input, axis=0)
print(mean)
#%%

std = np.std(train_input,axis = 0)
print(std)

#%%

print(train_input[:4])

#%%
train_scaled = (train_input - mean) / std
#%%
print(train_scale_ratio)  
plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(25,150, marker='^')

plt.xlabel("Length")
plt.ylabel("Weight")
plt.show()

#%%

new_fish =([25,150] - mean) / std

plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new_fish[0],new_fish[1], marker='^')
plt.xlabel("Length")
plt.ylabel("Weight")
plt.show()

#%%
#test set도 스케일 해줘야함.

test_scaled = (test_input - mean) / std

#%%

Kn.fit(train_scaled, train_target)

Kn.score(train_scaled, train_target)

#%%


plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new_fish[0],new_fish[1], marker='^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes, 1], marker="D")

plt.xlabel("Length")
plt.ylabel("Weight")
plt.show()



































