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



def test_binary():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMClassifier(n_estimators=50, silent=True)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5, verbose=False)
    y_pred = gbm.predict(X_test)
    y_pred = [ 1 if y > 0.5 else 0 for y in y_pred]

    print(f'test precision : {precision_score(y_test, y_pred)}\
        test recall : {recall_score(y_test, y_pred)}\
        test auc:{roc_auc_score(y_test, y_pred)}')

    with open('gbdt-data.txt', 'w') as gbdtdatafile:
        for p in X_test[:100].tolist():
            print(f"{' '.join([ str(n) for n in p])}", file=gbdtdatafile)

    with open('gbdt-data-pred-leaves.txt', 'w') as gbdtdata_pred_leaves_file:
        for p in gbm.predict(X_test[:100], pred_leaf=True).tolist():
            print(f"{' '.join([ str(n) for n in p])}", file=gbdtdata_pred_leaves_file)

    gbm.booster_.save_model('gbdt.txt')
    with open('gbdt.json', 'w') as gbdtfile:
        json.dump(gbm.booster_.dump_model(), gbdtfile, indent = 4)

if __name__ == '__main__':
    test_binary()
