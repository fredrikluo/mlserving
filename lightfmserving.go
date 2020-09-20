package lightfmserving

// Serving Light serving object to serve the model
type Serving struct {
}

// Recommend recommends a list of items for a user
// userID: the id of the user
// topK: how many items to recommend
// Return items that are recommended for the user and error which indicates
// there is error if it's not nil
func (serving Serving) Recommend(userID string, topK int) ([]string, error) {
	return nil, nil
}

// Predict scores for user ID on itemIDs.
func (serving Serving) predict(userID string, itemIDs []string) ([]float32, error) {
	return nil, nil
}
