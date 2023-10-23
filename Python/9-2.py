#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:26:33 2023

@author: sl
"""

#%%
# 01. 
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism() 


#%%
#02. 데이터 셋 로드 (num_words = 300) 이전 판에서 300, 현재는 500으로 사용  

from tensorflow.keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data( num_words= 300)
#%%

print(train_input.shape, test_input.shape)

#%%

print(len(train_input[0]))  # 218 , 영화평 첫 데이터 출력 
print(len(train_input[1]))  # 189 




#%%
#영화 첫 번째 데이터 출력 
print(train_input[0])

#%%
# 정답 20개 만 출력 
print(train_target[:20])
# [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]

#%%
# 훈련데이터 25,000건을 훈련 과 검증 셋트로 분류하기  
# Train Data Set으로 검증세트(Val_train, target )만들기 
from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

#%%
# 실제 훈련용 셋트(80%) : 20,000개 
# 검증용 셋트(20%) : 5000개 
print(len(train_input))
print(len(val_input))

#%%

import numpy as np

# 20,000개의 행을 하나씩 읽어서 각 행의 토큰의 갯수를 구해서 numpy array 타입에 length  
print(len(train_input[0]))  # 259
#%%

lengths = np.array([len(x) for x in train_input])

#%%
print(lengths.max)
                                                                                    
#%%

# 토큰의 평군 갯수, 중간값 
print(np.mean(lengths), np.median(lengths))

# 239.00925 178.0

#%%
# 데이터가 한쪽으로 치우쳐져 있음 
# 평균이 239개 이고 중간값이 178으로 보아 오른쪽 끝에 아주 큰 데이터가 있기 때문, 
# 1000단어 이상 리뷰도 있다. 

import matplotlib.pyplot as plt

plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()
#%%

# 축소 : 100단어로 축소
# 패딩 : 100단어 안되는 데이터는 0으로 체움  
# maxlen = 100 
# 100단어가 안되는 데이터는 0으로 체움 
# 2000 * 100
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 뒤에서 부터 100개를 선택.  
#turncate :'pre' ,'post
# 'pre' : 기본값, 앞쪽에서 짜름 
# 'post' : 뒤쪽에서 짜름 
# 유용한 정보가 딧쪽에 있을 것이라고 기대 

train_seq = pad_sequences(train_input, maxlen=100 )
# 
#%%
print(train_input.shape)

#%%
print(train_seq.shape)
# (20000, 100)

#%%
import matplotlib.pyplot as plt

plt.hist(train_seq)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()
#%%

print(train_seq[0])

"""
[ 10   4  20   9   2   2   2   5  45   6   2   2  33 269   8   2 142   2
   5   2  17  73  17 204   5   2  19  55   2   2  92  66 104  14  20  93
  76   2 151  33   4  58  12 188   2 151  12 215  69 224 142  73 237   6
   2   7   2   2 188   2 103  14  31  10  10   2   7   2   5   2  80  91
   2  30   2  34  14  20 151  50  26 131  49   2  84  46  50  37  80  79
   6   2  46   7  14  20  10  10   2 158]
"""

#%%

print(train_input[0][-10:])

"""
[6, 2, 46, 7, 14, 20, 10, 10, 2, 158]
"""
#%%
val_seq = pad_sequences(val_input, maxlen=100)

#%%

# 순환신경 만들기 
from tensorflow import keras

model = keras.Sequential()

# input_shape() = (100,300)
# 100: 샘플의 길이 
# 300 : 어휘는 300단어 
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 300)))
# activate = 'sigmoid' : 이진분류 
model.add(keras.layers.Dense(1, activation='sigmoid'))

#%%

# 원-핫 인코딩 : 
# 300단어 중 하나만 1이고 나머지는 0으로 만들든 정수 
train_oh = keras.utils.to_categorical(train_seq)

#%%
print(train_oh.shape)

#%%

print(train_oh[0][0][:12])

# 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]

#%%
print(np.sum(train_oh[0][0]))

# 1.0 

#%%
# 원-핫 인코딩 : 
# 300단어 중 하나만 1이고 나머지는 0으로 만들든 정수 
val_oh = keras.utils.to_categorical(val_seq)
model.summary()

# total param : 2472  + *+ 1 = 

"""
뉴런의 갯수 : 8 
원핫인코딩 : 된 갯수 : 300 

    3000* 8 = 24000
       8*8  = 64 
 +)      8  = 8 (절편 )
=========================
              2472 
"""

#%%
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy',
              metrics=['accuracy'])

# 20000/64 = 312.5 => 313
## 1에포크 : 313번 훈련 / 검증 반복 
# 100 에포크 : 3번 연속 변화가 없으면 중단 

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

# 베치 사이즈 가 64인 이유 ? 하나의 셋트가 2만개 
history = model.fit(train_oh, train_target, epochs= 100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])


#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

#%%
## 단어 임벧딩(Word Embedding) 
# 각 단어를 고정ㅇ된 크기의 실수 벡터로 변경 
# 자연어 처리에서 더 좋은 성능을 낸다 
# 케라스에서 Embedding() 클래스 제공 
# 원-핫 인코딩을 할 필요가 없음. 


model2 = keras.Sequential()

# 300 -> 16  
# 원핫 인코딩인 경우  train_oh 를 사용하였으나
# 워드 임배딩시 에는 그냥 16으로 

model2.add(keras.layers.Embedding(300, 16, input_length=100))

# 뉴런 겟수 : 8 로 지정 
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

model2.summary()

#%%
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy',
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.h5',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])


#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()


#%%

#예측 하기  (소스코드에 없는 부분임)

preds = model2.predict(val_seq[0:12])

print(preds)
      
#%%

test_seq = pad_sequences(test_input, maxlen = 100)
preds_test = model2.predict(test_seq[:10])

print(preds_test)



# [0,1,0,0,0,1,0,1,1,1,0,0,1] 


              
