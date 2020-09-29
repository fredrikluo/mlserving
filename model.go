package lightfmserving

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"sort"
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

//Model A trained model
type Model struct {
	modelData ModelData
}

func newModel() Model {
	return Model{}
}

// load the model from file
func (model *Model) load(filename string) error {
	modelFile, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}

	modelData := ModelData{}
	err = json.Unmarshal([]byte(modelFile), &modelData)
	if err != nil {
		return err
	}

	model.modelData = modelData
	return nil
}

func (model Model) calcPrediction(userLatent []float32, userBiases float32, itemLatent []float32, itemBiases float32) float32 {
	result := float32(0.0)
	for i := 0; i < len(userLatent); i++ {
		result += userLatent[i] * itemLatent[i]
	}

	return result + userBiases + itemBiases
}

func (model Model) predict(userID string, topK int) ([]Prediction, error) {
	userLatent, found := model.modelData.UserLatent[userID]
	if !found {
		return nil, fmt.Errorf("Can't find the user %s", userID)
	}

	ret := make([]Prediction, len(model.modelData.ItemLatent))
	idx := 0
	for itemID, itemLatent := range model.modelData.ItemLatent {
		ret[idx].Score = model.calcPrediction(userLatent,
			model.modelData.UserBiases[userID],
			itemLatent,
			model.modelData.ItemBiases[itemID])
		ret[idx].ItemID = itemID
		idx = idx + 1
	}

	// Sort and return topK
	sort.Slice(ret, func(i, j int) bool { return ret[i].Score > ret[j].Score })
	return ret[:topK], nil
}

func (model Model) predictWithLSH(userID string, topK int) ([]Prediction, error) {
	return nil, nil
}
