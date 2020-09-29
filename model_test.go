package lightfmserving

import (
	"fmt"
	"testing"
)

func TestPredict(t *testing.T) {
	model := newModel()
	err := (&model).load("movielens/model.json")
	ret, err := model.predict("0", 10)
	fmt.Printf("%#v, %#v", ret, err)
}

func TestCalculatePrediction(t *testing.T) {
	userLatent := []float32{0.04138161242008209, 0.6600756645202637, -0.19171953201293945, -0.5317293405532837, 0.9864006638526917, -0.39250457286834717, -0.3122827112674713, -0.6813488006591797, 0.3494027256965637, 0.4451221525669098}
	userBiases := float32(-4.509291648864746)
	itemLatent := []float32{0.5855547785758972, 0.5377424359321594, 0.17182695865631104, -0.36310696601867676, 0.7737079858779907, -0.2763769030570984, -0.4855107069015503, -0.6457154154777527, -0.8728089332580566, 0.7721853256225586}
	itemBiases := float32(1.4512581825256348)
	prediction := -1.0167253

	model := newModel()
	result := model.calcPrediction(userLatent, userBiases, itemLatent, itemBiases)
	fmt.Printf("pred: %v, pred calcuate: %v", prediction, result)
}
