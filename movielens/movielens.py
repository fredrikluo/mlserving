import numpy as np

from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score


movielens = fetch_movielens()

for key, value in movielens.items():
    print(key, type(value), value.shape)

train = movielens['train']
test = movielens['test']

model = LightFM(learning_rate=0.05, loss='warp')

model.fit_partial(train, item_features=movielens['item_features'], epochs=10)
train_precision = precision_at_k(model, train, item_features=movielens['item_features'],  k=10).mean()
test_precision = precision_at_k(model, test, item_features=movielens['item_features'], k=10).mean()

train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

item_biases, item_latent = model.get_item_representations(movielens['item_features'])
user_biases, user_latent = model.get_user_representations()

modelToSave = {
    'user_latent': user_latent,
    'user_biases': user_biases,
    'item_latent': item_latent,
    'item_biases': item_biases
}

print(user_latent.shape)
print(item_latent.shape)

print(test.row)
print(test.col)

for id in range(1, 100):
    uid = id
    iid = id
    predictions = (
                (user_latent[uid] * item_latent[iid]).sum()
                + user_biases[uid]
                + item_biases[iid]
            )

    test_predictions = model.predict(
            [uid], [iid], user_features=None, item_features=movielens['item_features'])

    print(predictions)
    assert np.allclose(test_predictions, predictions, atol=0.000001)


