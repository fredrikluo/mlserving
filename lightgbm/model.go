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
	"encoding/json"
	"io/ioutil"
)

//Model A trained model
type Model struct {
	modelData gbdtModel
}

//NewModel create a new model object
func NewModel() Model {
	return Model{}
}

//Load load the model from json file
func (model *Model) Load(modelfilename string) error {
	modelFile, err := ioutil.ReadFile(modelfilename)
	if err != nil {
		return err
	}

	err = json.Unmarshal([]byte(modelFile), &model.modelData)
	if err != nil {
		return err
	}

	return nil
}

// Predict use model to predict result
func (model Model) Predict() error {
	return nil
}
