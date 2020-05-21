#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


## 필요 module load
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime


# In[ ]:


## 데이터 load
a_train_origin = pd.read_csv("/content/drive/My Drive/DataForColab/A_train.csv")
a_test_origin = pd.read_csv("/content/drive/My Drive/DataForColab/A_test.csv")
a_train_set = a_train_origin.copy()
a_test_set =  a_test_origin.copy()


# In[ ]:


## 42개 null값 처리 후 필요 set 추출
a_train_set = a_train_set.fillna(0)

a_train_1 = a_train_set.loc[a_train_set["op_end"]>0,:]
a_test_1 = a_test_set.loc[a_train_set["op_end"]>0,:]

y_train_data = a_train_1["op_result"]
x_train_data = a_train_1.iloc[:,24:47]


# In[ ]:


## train data set 전처리 (공장 가동중에 대한 데이터 추출)
sp_df = a_train_set[((a_train_set["op_start"]==1) | (a_train_set["op_end"]==1)) ==True]
sp_df
sp_x_df = sp_df.iloc[:,24:47]
sp_y_df = sp_df.loc[:,["op_start","op_end","op_result"]]


# In[ ]:


## predict용 test data set 전처리
pp_df = a_test_set[((a_test_set["op_start"]==1) | (a_test_set["op_end"]==1)) ==True]
pp_x_df=pp_df.loc[:,["seq","d15","d16","d17","d18","d19","d20","d21","d22","d23","d24","d25","d26","d27","d28","d29","d30","d31","d32","d33","d34","d35","d36","d37"]]


# In[116]:


## 양품 / 불량품에 대한 그룹핑 (가동 중 result값을 불량품일때 0으로 처리 / 양품일때 1로 처리)
a_train_set.iloc[:,24:47].sum()
n = 0
for i in range(int(len(sp_y_df))):
    
    if (sp_y_df.op_end[i:i+1].item() == 1) and (sp_y_df.op_result[i:i+1].item() == 0):
        sp_y_df.op_result[n:i] = 0    
        n = i
    elif (sp_y_df.op_end[i:i+1].item() == 1) and (sp_y_df.op_result[i:i+1].item() == 1):
        sp_y_df.op_result[n:i] = 1
        n = i
    else:
        pass
print(sp_y_df)    


# In[ ]:


## train data 정규화 및 train set / test set 분할

sp_x_df.shape
train_num = int(sp_x_df.shape[0] * 0.7)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(sp_x_df[:train_num].values)
x_test = scaler.fit_transform(sp_x_df[train_num:].values)
y_train = sp_y_df["op_result"][:train_num].values.reshape([-1,1])
y_test = sp_y_df["op_result"][train_num:].values.reshape([-1,1])
y_train.shape
x_pre = scaler.fit_transform(pp_x_df.values)


# In[ ]:


# placeholer / reset
tf.reset_default_graph()
X = tf.placeholder(shape=[None,23], dtype=tf.float32)
Y = tf.placeholder(shape=[None,1], dtype=tf.float32)
drop_rate = tf.placeholder(dtype=tf.float32)

W1 = tf.get_variable("weight1", shape=[23,100],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([100]), name="bias1")
L1 = tf.sigmoid(tf.matmul(X, W1)+b1)

W2 = tf.get_variable("weight2", shape=[100,100],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([100]), name="bias2")
L2 = tf.sigmoid(tf.matmul(L1, W2)+b2)

W3 = tf.get_variable("weight3", shape=[100,1],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([1]), name="bias3")

# L4 = tf.nn.relu(tf.matmul(L3, W4)+b4)
logit = tf.matmul(L2, W3)+b3

H = tf.sigmoid(logit)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logit,
                                                             labels = Y))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[119]:


for step in range(3000):
    _, cost_val = sess.run([train, cost], feed_dict={ X : x_train,
                                                        Y : y_train})
    if step % 300 == 0:
        print(f"Cost 값은 : {cost_val}")
        
        
predict = tf.cast(H > 0.5, dtype=tf.float32)
correct = tf.equal(predict, Y)
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
print("정확도 : {}".format(sess.run(accuracy, feed_dict={X: x_test,
                                                        Y : y_test})))


# In[150]:


predict = tf.cast(H > 0.5, dtype=tf.float32)

print(f"정확도 : {sess.run(predict,feed_dict={X : pp_x_df.iloc[:,1:]})}")
result = pd.DataFrame(sess.run(predict,feed_dict={X : pp_x_df.iloc[:,1:]}))
result.sum()


# In[149]:


pp_x_df["d15"] = result
pp_x_df.loc[:,["seq","d15"]]
pp_x_df["d15"].sum()


# # 모델 검증을 위한 k-fold cross validation 실시

# In[110]:


from sklearn.model_selection import cross_val_score, KFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

import warnings

warnings.simplefilter("ignore")

kfold = KFold(n_splits=10)
stratified_shuffle_split = StratifiedShuffleSplit(train_size=0.7,test_size=0.3,n_splits=10,random_state=0)

logreg = LogisticRegression()

scores = cross_val_score(logreg, x_train, y_train)
print("k-5 cross validation score : {}".format(scores))

scores_k10 = cross_val_score(logreg, x_train, y_train, cv=kfold)
print("k-10 cross validation score : {}".format(scores_k10))

print("k-fold cross validation mean score : {}".format(scores.mean()))

scores_shuffle = cross_val_score(logreg, x_train, y_train, cv=stratified_shuffle_split)
print("shuffle cross validation score : {}".format(scores_shuffle))


# # layer 추가해서 다시 모델 만들기

# In[ ]:


a_train_set["new"] = 0
a_train_set.loc[a_train_set["op_start"]==1,"new"] = 1
a_train_set.loc[a_train_set["op_end"]==1,"new"] = 2
a_train_set.loc[a_train_set["op_result"]==1,"new"] = 3


new_y_val = pd.get_dummies(a_train_set["new"])

x_train_data = a_train_set.iloc[:,24:47]
y_train_data = new_y_val


# In[ ]:


x_train_data.shape
train_num = int(x_train_data.shape[0] * 0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train_data[:train_num].values)
x_test = scaler.fit_transform(x_train_data[train_num:].values)
y_train = y_train_data[:train_num].values.reshape([-1,1])
y_test = y_train_data[train_num:].values.reshape([-1,1])


# In[ ]:



1
# Data loading
# 전처리 단계 (결측치, 이상치, 정규화, feature engineering)


# Model 정의
tf.reset_default_graph()

# 1. placeholder
X = tf.placeholder(shape=[None, 23], dtype=tf.float32)
Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# dropout_rate = tf.placeholder(dtype=tf.float32)


# Weight & bias 1
W1 = tf.Variable(tf.random_normal([23,100]),name="weight1")
b1 = tf.Variable(tf.random_normal([100]),name="bias1")
layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

# Weight & bias 2
W2 = tf.Variable(tf.random_normal([100,256]),name="weight2")
b2 = tf.Variable(tf.random_normal([256]),name="bias2")
layer2 = tf.sigmoid(tf.matmul(layer1,W2)+b2)

# Weight & bias 3
W3 = tf.Variable(tf.random_normal([256,1]),name="weight3")
b3 = tf.Variable(tf.random_normal([1]),name="bias3")

# Hypothesis
logit = tf.matmul(layer2, W3) + b3
H = tf.sigmoid(logit)

# cost function
# cost = tf.losses.softmax_cross_entropy(Y, H) # 순서가 바뀐다
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logit,
                                                                   labels = Y))

#
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

#
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[ ]:



for step in range(3000):
    _, cost_val = sess.run([train, cost], feed_dict={X : x_train,
                                                       Y : y_train})
    if step % 300 == 0:
        print(f"Cost값은 : {cost_val}")


# In[64]:


predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
print("정확도는 : {}".format(sess.run(accuracy, feed_dict={X : x_test,
                                                          Y : y_test})))

