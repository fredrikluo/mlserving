package lightfmserving

// Serving Light serving object to serve the model
type Serving struct {
	model Model
}

func newServingObject(modelFilename string) (*Serving, error) {
	model := Model{}
	if err := model.load(modelFilename); err != nil {
		return nil, err
	}
	return &Serving{model: model}, nil
}

// Recommend recommends a list of items for a user
// userID: the id of the user
// topK: how many items to recommend
// Return items that are recommended for the user and error which indicates
// there is error if it's not nil
func (serving Serving) Recommend(userID string, topK int) ([]Prediction, error) {
	return serving.model.predict(userID, topK)
}
