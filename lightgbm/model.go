// Copyright (c) 2020 Luo Zhiyu
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License

package lightgbm

import (
	"github.com/fredrikluo/leaves"
)

//Model A trained model
type Model struct {
	gbdt *leaves.Ensemble
}

//NewModel create a new model object
func NewModel() Model {
	return Model{}
}

//Load load the model from json file
func (model *Model) Load(modelfilename string) error {
	gbdt, err := leaves.LGEnsembleFromFile(modelfilename, false)
	gbdt = gbdt.EnsembleWithLeafPredictions()
	if err != nil {
		return nil
	}

	model.gbdt = gbdt
	return nil
}

// Predict use model to predict result
func (model Model) Predict(fvals []float64, predLeaf bool) (float64, []uint32, error) {
	if !predLeaf {
		pred := model.gbdt.PredictSingle(fvals, 0)
		return pred, nil, nil
	}

	predictions := make([]float64, model.gbdt.NOutputGroups())
	err := model.gbdt.Predict(fvals, 0, predictions)
	if err != nil {
		return 0, nil, err
	}

	leafIndices := make([]uint32, model.gbdt.NOutputGroups())
	for idx, leafIdx := range predictions {
		leafIndices[idx] = uint32(leafIdx)
	}

	return 0, leafIndices, nil
}

// NumberOfLeaves number of leaves for each estimator
func (model Model) NumberOfLeaves() []int {
	return model.gbdt.NLeaves()
}
