#!/usr/bin/env python3

import numpy as np

from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
import json
import sys
import argparse


def trainTheModel():
    movielens = fetch_movielens()

    train = movielens['train']
    test = movielens['test']

    user_features = None
    item_features = movielens['item_features']

    model = LightFM(learning_rate=0.05, loss='warp')
    model.fit_partial(train, item_features=item_features, epochs=10)
    train_precision = precision_at_k(
        model, train, item_features=item_features,  k=10).mean()
    test_precision = precision_at_k(
        model, test, item_features=item_features, k=10).mean()

    train_auc = auc_score(model, train).mean()
    test_auc = auc_score(model, test).mean()

    return model, user_features, item_features


def saveModelToFile(model, filename, user_features, item_features):
    item_biases, item_latent = model.get_item_representations(item_features)
    user_biases, user_latent = model.get_user_representations(user_features)

    numOfUser = user_latent.shape[0]
    numOfItem = item_latent.shape[0]
    modelToSave = {
        'user_latent': dict(zip(range(numOfUser), user_latent.tolist())),
        'user_biases': dict(zip(range(numOfUser), user_biases.tolist())),
        'item_latent': dict(zip(range(numOfItem), item_latent.tolist())),
        'item_biases': dict(zip(range(numOfItem), item_biases.tolist()))
    }

    with open(filename, 'w') as modelFile:
        json.dump(modelToSave, modelFile, indent=4)


def verifyTheModel(model, user_features, item_features):
    item_biases, item_latent = model.get_item_representations(item_features)
    user_biases, user_latent = model.get_user_representations(user_features)

    for id in range(0, 10):
        uid = 0
        iid = id
        predictions = (
            (user_latent[uid] * item_latent[iid]).sum()
            + user_biases[uid]
            + item_biases[iid]
        )

        test_predictions = model.predict(
            [uid], [iid], user_features=user_features, item_features=item_features)

        assert np.allclose(test_predictions, predictions, atol=0.000001)


def predictTopK(model, userId, topk, user_features, item_features):
    pred = model.predict(
        [userId],
        [iid for iid in range(0, item_features.shape[0])],
        user_features=user_features,
        item_features=item_features).tolist()

    return list(zip(sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)[:topk], sorted(pred, reverse=True)[:topk]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='predict top k for given user id')
    parser.add_argument('userid', metavar='N', type=int, help='userid')
    parser.add_argument('topk', metavar='N', type=int, help='topk')
    args = parser.parse_args()

    model, user_features, item_features = trainTheModel()
    verifyTheModel(model, user_features, item_features)
    saveModelToFile(model, 'model.json', user_features, item_features)

    print(json.dumps([{"id": str(id), "score": score} for id, score in predictTopK(
        model, args.userid, args.topk, user_features, item_features)], indent=4))
