#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:29:33 2023

@author: sl
"""
# 생선 분류 문제 
# 도미 와 빙어 데이터 

#%%
#도미 데이터 길이(Cm), 무게(g)  데이터 

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
#데이터 갯수 35 
print(len(bream_length))
#%%
# scatter 그래프 준비 

import matplotlib.pyplot as plt 

plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylable('weight')
plt.show() 
#%%
#빙어 데이터 준비하기 

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
#데이터 갯수 14 
print(len(smelt_length))

#%%
plt.scatter(bream_length, bream_weight) #도미
plt.scatter(smelt_length, smelt_weight) #빙어 
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#%%
plt.scatter(smelt_length, smelt_weight) #빙어 
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#%%

#도미와 빙어 데이터를 하나로 합침 

length = bream_length+smelt_length
weight = bream_weight+smelt_weight

#%%
#데이터를 2차원 배열화 시킨다. 

fish_data = [[l,w] for l , w in zip(length,weight)]

print(fish_data)
# len : 49 
print(len(fish_data))
#%%

test = [1]*35
test2 = [0]*14 

print(test)
print(test2)
#%%
# 정답 만들기 

# list[1] * 35 : 도미, 
# list[0] * 14 : 빙어 
# 데이터를 표시하기 위해 인코딩 방식으로 배열을 만든다. 

fish_target = [1]*35 + [0]*14
print(fish_target)

#%%

from sklearn.neighbors import KNeighborsClassifier

#%%
# 두 데이터를 넣고 지도 학습 시킨다. 

kn = KNeighborsClassifier() #기본 값은 5이다. 

#fish_data : 도미 + 빙어 의 2차원 배열 
#finish_target : 1,0 으로 도미와 빙어를 의미함.(답) 

kn.fit(fish_data, fish_target)
#
#%%
# 평가 

kn.score(fish_data, fish_target)

#%% 
# 새로운 ㅇ생선 예측 

# 2차원 데이터 로 입력 해야함. 
predict = kn.predict([[30,600]])
#사이킷 이 넘파이 어레이임. 
#%%

print(predict) # 1(도미) 을 리턴 한다. 



