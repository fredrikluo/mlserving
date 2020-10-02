package lightfm

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"sort"

	"github.com/fredrikluo/mlserving/lightfm/annoyindex"
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

// AnnIndex Spotify Approximate Nearest Neighbors index
type AnnIndex struct {
	index    annoyindex.AnnoyIndexDotProduct
	index2ID map[int]string
	ID2index map[string]int
}

//Model A trained model
type Model struct {
	modelData ModelData
	itemIndex AnnIndex
	userIndex AnnIndex
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

func newAnnIndex(filename string, id2indexFilename string, index2idFilename string) (*AnnIndex, error) {
	index := AnnIndex{}
	index.index = annoyindex.NewAnnoyIndexDotProduct(10)
	if ret := index.index.Load(filename, true); ret == false {
		return nil, fmt.Errorf("failed to load %s", filename)
	}

	index.index2ID = map[int]string{}
	index.ID2index = map[string]int{}

	id2indexFile, err := ioutil.ReadFile(id2indexFilename)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal([]byte(id2indexFile), &index.ID2index)
	if err != nil {
		return nil, err
	}

	index2idFile, err := ioutil.ReadFile(index2idFilename)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal([]byte(index2idFile), &index.index2ID)
	if err != nil {
		return nil, err
	}

	return &index, nil
}

//Load load the modeldata from file
func (model *Model) Load(folder string) error {
	modelfilename := folder + "/model.json"
	userLatentAnnFilename := folder + "/user_latent.ann"
	userID2IndexFilename := folder + "/user_latent_id2index.json"
	userIndex2IDFilename := folder + "/user_latent_index2id.json"
	itemLatentAnnFilename := folder + "/item_latent.ann"
	itemID2IndexFilename := folder + "/item_latent_id2index.json"
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

	index, err := newAnnIndex(userLatentAnnFilename, userID2IndexFilename, userIndex2IDFilename)
	if err != nil {
		return err
	}

	model.userIndex = *index

	index, err = newAnnIndex(itemLatentAnnFilename, itemID2IndexFilename, itemIndex2IDFilename)
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
	var result []int
	id, found := model.itemIndex.ID2index[userID]
	if !found {
		return nil, fmt.Errorf("can't find the user id %s", userID)
	}

	model.itemIndex.index.GetNnsByItem(id, topK*10, -1, &result)

	candidate := map[string][]float32{}
	for _, r := range result {
		iid := model.itemIndex.index2ID[r]
		candidate[iid] = model.modelData.ItemLatent[iid]
	}

	return model.predict(userID, topK, candidate)
}
