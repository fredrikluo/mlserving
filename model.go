package lightfmserving

// Prediction this stores a prediction
type Prediction struct {
	Prediction float32
	ItemID     string
}

//Model A trained model
type Model struct {
	UserEmbeddings map[string][]float32
	ItemEmbeddings map[string][]float32

	// Those are just for debuggings
	// User features
	// Item features
}

// load the model from file
func (model Model) load(filename string) error {
	return nil
}

func (model Model) predict(userID string, topK int) ([]Prediction, error) {
	userEmbedding, found := model.UserEmbeddings[userID]
	for itemId, itemEmbedding := range model.ItemEmbeddings {
		// Dot product on two list
	}

	return nil, nil
}

func (model Model) predictWithLSH(userID string, topK int) ([]Prediction, error) {
	return nil, nil
}
