import numpy as np
from collections import Counter
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer,load_boston,load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score,recall_score
import json

pd.options.display.max_columns = 999

X=load_breast_cancer()
df=pd.DataFrame(X.data,columns=X.feature_names)
Y=X.target
sc=StandardScaler()
sc.fit(df)

X=pd.DataFrame(sc.fit_transform(df))
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
d_train=lgb.Dataset(X_train, label=y_train)

params={}
params['learning_rate']=0.03
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='binary' #Binary target feature
params['metric']='binary_logloss' #metric for binary classification
params['max_depth']=10

clf=lgb.train(params,d_train,100) #train the model on 100 epocs
y_pred = [ 1 if y > 0.5 else 0 for y in clf.predict(X_test)]

print(f'test precision : {precision_score(y_test, y_pred)}\
        test recall : {recall_score(y_test, y_pred)}\
        test auc:{roc_auc_score(y_test, y_pred)}')

with open('gbdt.json', 'w') as gbdtfile:
    json.dump(clf.dump_model(), gbdtfile, indent = 4)