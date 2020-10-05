#!/usr/bin/env python3

# Copyright (c) 2020 Luo Zhiyu
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np

from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
import json
import sys
import argparse
from annoy import AnnoyIndex
import os
import shutil


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


def saveAnnIndex(embeddings, filename, id2indexfilename, index2idfilename):
    # build the ann index file
    t = AnnoyIndex(10, 'dot')  # Length of item vector that will be indexed
    id2index = {}
    index2id = {}
    for i, v in enumerate(embeddings):
        t.add_item(i, v)
        id2index[str(i)] = i
        index2id[i] = str(i)
    t.build(100)  # 10 trees
    t.save(filename)

    with open(id2indexfilename, 'w') as mapping:
        json.dump(id2index, mapping, indent=4)

    with open(index2idfilename, 'w') as mapping:
        json.dump(index2id, mapping, indent=4)


def saveModelToFile(model, foldername, user_features, item_features):
    item_biases, item_latent = model.get_item_representations(item_features)
    user_biases, user_latent = model.get_user_representations(user_features)

    numOfUser = user_latent.shape[0]
    numOfItem = item_latent.shape[0]

    user_latent = user_latent.tolist()
    item_latent = item_latent.tolist()
    modelToSave = {
        'user_latent': dict(zip(range(numOfUser), user_latent)),
        'user_biases': dict(zip(range(numOfUser), user_biases.tolist())),
        'item_latent': dict(zip(range(numOfItem), item_latent)),
        'item_biases': dict(zip(range(numOfItem), item_biases.tolist()))
    }

    if os.path.isdir(foldername):
        shutil.rmtree(foldername)

    os.mkdir(foldername)

    with open(os.path.join(foldername, "model.json"), 'w') as modelFile:
        json.dump(modelToSave, modelFile, indent=4)

    saveAnnIndex(user_latent,
                 os.path.join(foldername, 'user_latent.ann'),
                 os.path.join(foldername, 'user_latent_id2index.json'),
                 os.path.join(foldername, 'user_latent_index2id.json'))
    saveAnnIndex(item_latent,
                 os.path.join(foldername, 'item_latent.ann'),
                 os.path.join(foldername, 'item_latent_id2index.json'),
                 os.path.join(foldername, 'item_latent_index2id.json'))


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
    saveModelToFile(model, 'model', user_features, item_features)

    print(json.dumps([{"id": str(id), "score": score} for id, score in predictTopK(
        model, args.userid, args.topk, user_features, item_features)], indent=4))
