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
	UserRepresentation map[string][]float32
	ItemRepresentation map[string][]float32

	// Those are just for debuggings
	// User features
	// Item features
}

// load the model from file
func (model Model) load(filename string) error {
	return nil
}

func (model Model) predict(userID string, topK int) ([]Prediction, error) {
	userRepresentation, found := model.UserRepresentation[userID]
	if !found {
		return nil, fmt.Errorf("Can't find the user %s", userID)
	}

	ret := make([]Prediction, len(model.ItemRepresentation))
	idx := 0
	noComponents := len(userRepresentation) - 1
	for itemID, itemRepresentation := range model.ItemRepresentation {
		// Bias
		result := userRepresentation[noComponents] + itemRepresentation[noComponents]
		// Dot product on two list
		for i := 0; i < noComponents-1; i++ {
			result += userRepresentation[i] * itemRepresentation[i]
		}

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
