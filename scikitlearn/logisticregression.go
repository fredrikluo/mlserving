package scikitlearn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
)

// ModelParams Model parameters for LR model
type ModelParams struct {
	Coef      [][]float64 `json:"coef_"`
	Intercept []float64   `json:"intercept_"`
	Classes   []int       `json:"classes_"`
}

// LogisticRegression  struct to hold model data
type LogisticRegression struct {
	Params ModelParams `json:"model_params"`
}

// Create a new LogisticRegression object
func newLogisticRegression() *LogisticRegression {
	return &LogisticRegression{}
}

// Load load the model data from a file
func (lr *LogisticRegression) Load(modelfilename string) error {
	modelFile, err := ioutil.ReadFile(modelfilename)
	if err != nil {
		return err
	}

	err = json.Unmarshal([]byte(modelFile), lr)
	if err != nil {
		return err
	}

	return nil
}

// Pred from the loaded model, it outputs probability of class 0 , probablity of class 2.
func (lr LogisticRegression) Pred(data []float64) (map[int]float64, error) {
	dimension := len(lr.Params.Coef[0])
	if len(data) != dimension {
		return map[int]float64{},
			fmt.Errorf("The model is trained with %d dimensions, but the input vector has %d dimensions",
				dimension,
				len(data))
	}

	z := 0.0
	for i := 0; i < dimension; i++ {
		z += lr.Params.Coef[0][i] * data[i]
	}

	z += lr.Params.Intercept[0]
	p := 1.0 / (1 + math.Exp(-1*z))
	return map[int]float64{lr.Params.Classes[0]: 1.0 - p, lr.Params.Classes[1]: p}, nil
}
