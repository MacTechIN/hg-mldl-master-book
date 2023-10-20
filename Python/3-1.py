#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:02:59 2023
이웃 회귀 
@author: sl
"""
#%%

import numpy as np 
# 샘플 데이터 56 
# 길이만 


perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )

# 무게 예측 
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

import matplotlib.pyplot as plt 

plt.scatter(perch_length, perch_weight)
plt.xlabel("Weight")
plt.ylabel('Wegith')

plt.show ()

#%%
# 훈련세트 준비 

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state = 42)


#%% 

train_input = train_input.reshape(-1,1)
test_input =test_input.reshape(-1,1)

print(train_input)
print(test_input)
print(train_target )
print(test_target)

#%%
#KNeighborsRegressor 

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
#%%
knr.fit (train_input, train_target )

#%%
knr.score(test_input, test_target)
# 0.992809406101064
#%%

from sklearn.metrics import mean_absolute_error

test_predicition = knr.predict(test_input)

#%%

mae = mean_absolute_error(test_target , test_predicition)
print(mae)

# 19.157142857142862 


#%%

print(knr.score(train_input, train_target ))

# train 용 Score : 0.9698823289099254
# 훈련(train) 셋이 더 낮음 즉 과소적합임.

#%%
# 이웃의 숫자를 줄여서 모델을 더 복잡하게 만들기 
# 이웃을 줄이면 : 국지적 패턴에 민감(복잡해짐) 
# 이웃을 늘리면 : 일반적인 패턴을 따라감 (단순해짐 ) 
 

knr.n_neighbors = 3 
#%%
knr.fit(train_input, train_target)

#%%

knr_score = knr.score(test_input, test_target)

# 0.9746459963987609
print(knr_score)



#%%

# 농어무게 : 8.4 로 길이 예측하기 

predict_1 = knr.predict([[13.7]])
print(predict_1)


#%%
#연습문제 
import matplotlib.pyplot as plt 

x = np.arange(5,45).reshape(-1,1)

#%%
for n in [1,5,10]:
    knr.n_neighbors = n
    knr.fit(train_input , train_target)
    
    prediction = knr.predict(x)
    plt.scatter(train_input, train_target)
    plt.plot(x,prediction)
    plt.title('n_neighbors = {}'.format(n))
    plt.show() 


#%%

new_length = np.arange(5,50).reshape(-1,1)

print(new_length)

prediction2 = knr.predict(new_length)
plt.plot(new_length,prediction2)
plt.show()
