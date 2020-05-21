#!/usr/bin/env python
# coding: utf-8

# # 모델 검증을 위한 k-fold cross validation 실시

# In[ ]:


# 분석용 데이터는 A1문제의 전처리 과정을 똑같이 거침


# In[ ]:


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


# 결과 :
# k-5 cross validation score : [0.99701865 0.99701865 0.99701865 0.99701865 0.99698398]
# k-10 cross validation score : [0.99611731 0.99486931 1.         0.98966928 0.99604798 1.
#  1.         0.99604798 0.99868266 0.99868266]
# k-fold cross validation mean score : 0.9970117173958262
# shuffle cross validation score : [0.99701865 0.99701865 0.99701865 0.99701865 0.99701865 0.99701865
#  0.99701865 0.99701865 0.99701865 0.99701865]

