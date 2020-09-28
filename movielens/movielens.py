import numpy as np

from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
import json

def trainTheModel():
    movielens = fetch_movielens()

    train = movielens['train']
    test = movielens['test']

    user_features = None
    item_features = movielens['item_features']

    model = LightFM(learning_rate=0.05, loss='warp')
    model.fit_partial(train, item_features=item_features, epochs=10)
    train_precision = precision_at_k(model, train, item_features=item_features,  k=10).mean()
    test_precision = precision_at_k(model, test, item_features=item_features, k=10).mean()

    train_auc = auc_score(model, train).mean()
    test_auc = auc_score(model, test).mean()

    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

    return model, user_features, item_features

def saveModelToFile(model, filename, user_features, item_features):
    item_biases, item_latent = model.get_item_representations(item_features)
    user_biases, user_latent = model.get_user_representations(user_features)

    modelToSave = {
        'user_latent': user_latent.tolist(),
        'user_biases': user_biases.tolist(),
        'item_latent': item_latent.tolist(),
        'item_biases': item_biases.tolist()
    }

    with open(filename, 'w') as modelFile:
        json.dump(modelToSave, modelFile, indent=4)

def verifyTheModel(model, user_features, item_features):
    item_biases, item_latent = model.get_item_representations(item_features)
    user_biases, user_latent = model.get_user_representations(user_features)

    for id in range(1, 100):
        uid = id
        iid = id
        predictions = (
                    (user_latent[uid] * item_latent[iid]).sum()
                    + user_biases[uid]
                    + item_biases[iid]
                )

        test_predictions = model.predict(
                [uid], [iid], user_features=user_features, item_features=item_features)

        assert np.allclose(test_predictions, predictions, atol=0.000001)

if __name__ == '__main__':
    model, user_features, item_features = trainTheModel()
    verifyTheModel(model, user_features, item_features)
    saveModelToFile(model, 'model.json', user_features, item_features)
