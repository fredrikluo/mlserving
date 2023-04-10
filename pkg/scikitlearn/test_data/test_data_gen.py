import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression


def save_lr_to_file(lr, dirname):
    with open(os.path.join(dirname, 'lr.json'), 'w') as lr_file:
        json.dump({
            'init_params': lr.get_params(),
            'model_params': {
                    p:getattr(lr, p).tolist() for p in ('coef_', 'intercept_','classes_', 'n_iter_')
                }
            },
            lr_file,
            indent = 4)

x = np.arange(10).reshape(-1, 1)
y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(x, y)
p_pred = model.predict_proba(x)
print(p_pred)
save_lr_to_file(model, './')