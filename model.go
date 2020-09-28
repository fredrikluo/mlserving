package lightfmserving

import (
	"fmt"
	"sort"
)

// Prediction this stores a prediction
type Prediction struct {
	Prediction float32
	ItemID     string
}

//Model A trained model
type Model struct {
	UserLatent map[string][]float32
	UserBasis  map[string]float32
	ItemLatent map[string][]float32
	ItemBasis  map[string]float32

	// Those are just for debuggings
	// User features
	// Item features
}

// load the model from file
func (model Model) load(filename string) error {
	return nil
}

func (model Model) predict(userID string, topK int) ([]Prediction, error) {
	userLatent, found := model.UserLatent[userID]
	if !found {
		return nil, fmt.Errorf("Can't find the user %s", userID)
	}

	ret := make([]Prediction, len(model.ItemLatent))
	idx := 0
	for itemID, itemLatent := range model.ItemLatent {
		result := float32(0.0)
		for i := 0; i < len(userLatent); i++ {
			result += userLatent[i] * itemLatent[i]
		}

		result += model.UserBasis[userID] + model.ItemBasis[itemID]
		ret[idx].ItemID = itemID
		ret[idx].Prediction = result
		idx = idx + 1
	}

	// Sort and return topK
	sort.Slice(ret, func(i, j int) bool { return ret[i].Prediction > ret[i].Prediction })
	return ret[:topK], nil
}

func (model Model) predictWithLSH(userID string, topK int) ([]Prediction, error) {
	return nil, nil
}
