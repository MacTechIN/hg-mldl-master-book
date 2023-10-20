#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:02:59 2023
이웃 회귀 
@author: sl
"""
#%%
import matplotlib.pyplot as plt 
import numpy as np 
# 샘플 데이터 56 
# 길이 만으로 무게를 예측 해야한다. 

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

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state= 42)

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

#%%
#최근접 이웃 을 3으로 하는 모델학습하기 

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)

knr.fit (train_input, train_target)

knr.score (test_input,test_target)

#%%
#길이가 50Cm 인 농어 예측하기 

predict_50 = knr.predict([[50.0]])

print(predict_50)
# [1033.33333333]

#%%

# 이웃이 셋이므로 
dist, indexes = knr.kneighbors([[50.0]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes],train_target[indexes], marker='D' )

plt.scatter(50,1033, marker='^')

plt.xlabel("length")
plt.ylabel("weight")

plt.shouw() 

#%%  

print(np.mean(train_target[indexes]))

#%%
dist, indexes = knr.kneighbors([[100.0]])

plt.scatter(train_input, train_target)

plt.scatter(train_input[indexes],train_target[indexes], marker='D' )

plt.scatter(1000,1033, marker='^')

plt.xlabel("length")
plt.ylabel("weight")

plt.shouw() 
#%%
#선형회귀 
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(train_input, train_target)

#%%

print(lr.predict([[50]]))
#%%

print(lr.coef_, lr.intercept_)
# 절편 과 기울기에 대응하는 요소(모델 파라미터) :   계수 coefficient , 가중치 intercept 
# 머신러닝에서는 이 
# [39.01714496] -709.0186449535474
#%%

plt.scatter(train_input, train_target)

plt.plot([15,50], [15 *lr.coef_+lr.intercept_ , 50*lr.coef_ + lr.intercept_])

plt.scatter(50,1241.8, marker="^")
plt.xlabel("length")
plt.ylebel("weight")

plt.show() 

#%%
#스코어 확인하기 
print("Train score : {}".format(lr.score(train_input,train_target)))

print("Test score : {}".format(lr.score(test_input, test_target)))

#%%
#다항회귀 
# y = ax^2 + bx + c  
# train_input , test_input 항을 앞에 1열 추가 하여 제곱 값을 넣어 보도록 하자 

train_poly = np.column_stack((train_input**2, train_input)) 
test_poly = np.column_stack((test_input**2, test_input)) 

#%%
print(train_poly.shape)
print(train_poly[:5])

#%%

lr = LinearRegression()
lr.fit(train_poly, train_target)

#%%
print(lr.predict([[50**2, 50]]))

#%%
print(lr.coef_, lr.intercept_)

#%%
point = np.arange(15, 50)

plt.scatter(train_input, train_target)

plt.plot(point, 1.01*point**2 - 21.6*point + 116.05) 

plt.scatter(50,1573.98)

plt.show()


#%%
#검증 

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

















