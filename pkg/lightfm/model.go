// Copyright (c) 2020 Luo Zhiyu
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

package lightfm

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"sort"

	"github.com/fredrikluo/mlserving/pkg/embeddings"
)

// Prediction this stores a prediction
type Prediction struct {
	Score  float32 `json:"score"`
	ItemID string  `json:"id"`
}

// ModelData A saved model
type ModelData struct {
	UserLatent map[string][]float32 `json:"user_latent"`
	UserBiases map[string]float32   `json:"user_biases"`
	ItemLatent map[string][]float32 `json:"item_latent"`
	ItemBiases map[string]float32   `json:"item_biases"`
}

// Model A trained model
type Model struct {
	modelData ModelData
	itemIndex embeddings.AnnIndex
	userIndex embeddings.AnnIndex
}

// NewModel create a new model object
func NewModel() Model {
	return Model{}
}

// Load load the modeldata from file
func (model *Model) Load(folder string, dimensions int) error {
	modelfilename := folder + "/model.json"
	userLatentAnnFilename := folder + "/user_latent.ann"
	userIndex2IDFilename := folder + "/user_latent_index2id.json"
	itemLatentAnnFilename := folder + "/item_latent.ann"
	itemIndex2IDFilename := folder + "/item_latent_index2id.json"

	modelFile, err := ioutil.ReadFile(modelfilename)
	if err != nil {
		return err
	}

	modelData := ModelData{}
	err = json.Unmarshal([]byte(modelFile), &modelData)
	if err != nil {
		return err
	}

	model.modelData = modelData

	index, err := embeddings.NewAnnIndex(userLatentAnnFilename, userIndex2IDFilename, dimensions)
	if err != nil {
		return err
	}

	model.userIndex = *index

	index, err = embeddings.NewAnnIndex(itemLatentAnnFilename, itemIndex2IDFilename, dimensions)
	if err != nil {
		return err
	}

	model.itemIndex = *index

	return nil
}

func (model Model) calcPrediction(userLatent []float32, userBiases float32, itemLatent []float32, itemBiases float32) float32 {
	result := float32(0.0)
	for i := 0; i < len(userLatent); i++ {
		result += userLatent[i] * itemLatent[i]
	}

	return result + userBiases + itemBiases
}

// Predict items for specific userID
func (model Model) predict(userID string, topK int, candidates map[string][]float32) ([]Prediction, error) {
	userLatent, found := model.modelData.UserLatent[userID]
	if !found {
		return nil, fmt.Errorf("Can't find the user %s", userID)
	}

	ret := make([]Prediction, len(candidates))
	idx := 0
	for itemID, itemLatent := range candidates {
		ret[idx].Score = model.calcPrediction(userLatent,
			model.modelData.UserBiases[userID],
			itemLatent,
			model.modelData.ItemBiases[itemID])
		ret[idx].ItemID = itemID
		idx = idx + 1
	}

	//Sort and return topK
	sort.Slice(ret, func(i, j int) bool { return ret[i].Score > ret[j].Score })
	return ret[:topK], nil
}

// Predict items for specific userID
func (model Model) Predict(userID string, topK int) ([]Prediction, error) {
	return model.predict(userID, topK, model.modelData.ItemLatent)
}

// PredictFast predict fast with Approximate Nearest Neighbors algorithm
func (model Model) PredictFast(userID string, topK int) ([]Prediction, error) {
	id, found := model.userIndex.IdToIndex(userID)
	if !found {
		return nil, fmt.Errorf("can't find the user id %s", userID)
	}

	result := model.itemIndex.GetNnsByItem(id, topK*10, -1)

	candidate := map[string][]float32{}
	for _, r := range result {
		iid, found := model.itemIndex.IndexToId(r)
		if !found {
			log.Printf("can't find the item id, the index is %d", r)
			continue
		}

		candidate[iid] = model.modelData.ItemLatent[iid]
	}

	return model.predict(userID, topK, candidate)
}
