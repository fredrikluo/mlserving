package lightfmserving

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"sort"

	"github.com/fredrikluo/lightfm-serving/annoyindex"
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
	index     annoyindex.AnnoyIndexDotProduct
	index2ID  map[int]string
	ID2index  map[string]int
}

//NewModel create a new model object
func NewModel() Model {
	return Model{}
}

func (model Model) getSampleItem() []float32 {
	for _, v := range model.modelData.ItemLatent {
		return v
	}

	return nil
}

//Load load the modeldata from file
func (model *Model) Load(filename string) error {
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

	// build the index
	model.index2ID = map[int]string{}
	model.ID2index = map[string]int{}

	vectorLength := len((*model).getSampleItem())
	model.index = annoyindex.NewAnnoyIndexDotProduct(vectorLength)
	idx := 0
	for key, value := range modelData.ItemLatent {
		model.index.AddItem(idx, value)
		model.index2ID[idx] = key
		model.ID2index[key] = idx
		idx = idx + 1
	}

	// TODO: number of trees should be tweakable
	model.index.Build(100)
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
	var result []int
	id, found := model.ID2index[userID]
	if !found {
		return nil, fmt.Errorf("can't find the user id %s", userID)
	}

	model.index.GetNnsByItem(id, topK*10, -1, &result)

	candidate := map[string][]float32{}
	for _, r := range result {
		iid := model.index2ID[r]
		candidate[iid] = model.modelData.ItemLatent[iid]
	}

	return model.predict(userID, topK, candidate)
}
