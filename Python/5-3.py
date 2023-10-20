#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:29:28 2023

@author: sl
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
#%%
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)

# test_score : 검증용 샘플 스코어 
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
#0.9973541965122431 0.8905151032797809


#%%
rf.fit(train_input, train_target)

print(rf.feature_importances_)
#0.8934000384837406
#%%
# n_jobs : cpu 코어 설정 
# oob : 남은 샘플을 가지고 평가를 하기 위해서는 default 는 False 이다.
 
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)

rf.fit(train_input, train_target)
print(rf.oob_score_)
# 0.8934000384837406
#%%
# 액스트라트리 
# randomforest 와 흡사함. 
# 100개의 트리 사용 
# 랜덤하게 노드를 분활할 
# 부트스트랩 셈플은 사용하지 않음 , 전체 샘플 사용함 
# 과대 적합 방지 하는 방법 : 노드 분활시 A B C 렌덤하게 분할 에서 최적에 값을 찾아서 노드 분할 (트리성능이 좋지않다) 
# 랜덤하게 생성되기 때문에 속도가 빠르다. 대신 트리 겟수를 늘려야 성능이 올라간다. 
 
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)

#교차검증 
#return_train_score = True : 검증 점수, 훈련 세트에 대한 점수도 리턴 
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
#과대 적합 
#훈련 데이터에 대해서는 높은 점수 , 테스트 데이터에 대해서는 낮은 점수 
#0.9974503966084433 0.8887848893166506
#%%
et.fit(train_input, train_target)
print(et.feature_importances_)

#[알코올,      당도,       산도       ] -> 당도가 높음 (레드와인 과 화이트와인 분류시사용됨.) 
#[0.20183568 0.52242907 0.27573525]

#이전 decisiontree(5-1) 
#[0.15210271 0.70481604 0.14308125]

#%%
# 그레이디언트 부스팅


#%%

































