#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:01:19 2023


1. fish_input 를 두 데이터 셋을 합쳐서 2차원 배열로 바꾼다. 
2. fish_target 도 1이 35 어레이 + 0이 14개 array 로 만든다. 
3. KNeighborClassifier 객채 생성 
4. numpy 로 input_arr, target_arr 를 만든다 
5. random.seed(42) 사용 1~49 개의 랜덤 index를 만들어 셔플 시킨다. 
6. index가 셔플로 잘 섞였는지 확인한다. 
7. train_input, target,(35개) test_input, target(나머지 전부) 이렇게 인덱스를 이용 나눈다. 
8. 그래프로 확인하기 matplotlib scatter() 잘쎡였는지 확인 
9. fit(x,y) 
10.  score(x,y)
11. predict(test_input) 과 test_target 이 동일한지 확인한다. 
12. 끝 

@author: sl
"""
#%% 도미, 빙어 ROW data sets 
# 도미 35와 빙어 14 데이터 셋트 
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 
               475.0, 500.0,500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 
               575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 
               920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 
               9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

#%% #1. fish_input에 두데이터를 2차원 배열로 만들기 

fish_data = [[l,w] for l,w in zip(fish_length,fish_weight)]
print(fish_data[:5])
#%% #2. fish_target 도 도미 1 35개 , 빙어 0 14개 배열을 만든다. 

fish_target = [1]*35 + [0]*14
print(fish_target)
#%% #3. KneighborClassifier 객체 생성

from sklearn.neighbors import KNeighborsClassifier
Kn = KNeighborsClassifier()
#%% #4. numpy fh input_arr, target_arr Row array만든다. 
import numpy as np
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
#%% #5,6 random.seed(42)fh index 생성 후 셔플로 섞는다.
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
print(index)
#%% #7 train_input, train_target , test_input, test_target 셔플로 생성 

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)
#%% #8 graph 확인 
import matplotlib.pyplot as plt 

plt.scatter(train_input[:,0],train_input[:,1])
plt.scatter(test_input[:,0],test_input[:,1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()
#%% #9 fit , score 

Kn.fit(train_input, train_target)
Kn.score(test_input, test_target)

#%% #10 검증하기 

p_result = Kn.predict(test_input)
print("predict Result : ", p_result)   
print("real Target : ", test_target) 

#%%

# 25Cm , 150g  도미를 검증하기 
r = Kn.predict([[25,150]])
print(r)













