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
// the License.

package lightfm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os/exec"
	"testing"

	"github.com/stretchr/testify/assert"
)

func runShell(command string) (string, string, error) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd := exec.Command("bash", "-c", command)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	return stdout.String(), stderr.String(), err
}

func TestPredict(t *testing.T) {
	userID := "0"
	topK := 10
	out, _, err := runShell(fmt.Sprintf("cd movielens&&venv-run ./movielens.py %s %d", userID, topK))
	assert.Nil(t, err)

	predList := []Prediction{}
	err = json.Unmarshal([]byte(out), &predList)
	assert.Nil(t, err)

	model := NewModel()
	err = (&model).Load("movielens/model")
	ret, err := model.Predict(userID, topK)

	for i := 0; i < topK; i++ {
		assert.Equal(t, predList[i].ItemID, ret[i].ItemID)
		assert.True(t, math.Abs(float64(predList[i].Score-ret[i].Score)) < 0.0001)
	}

	ret, err = model.PredictFast(userID, topK)
	hitNumber := 0
	for i := 0; i < topK; i++ {
		if predList[i].ItemID == ret[i].ItemID &&
			math.Abs(float64(predList[i].Score-ret[i].Score)) < 0.0001 {
			hitNumber = hitNumber + 1
		}
	}

	// should hit > 90% of the topK
	assert.Truef(t, float32(hitNumber)/float32(topK) > 0.9, "hit for %d from %d", hitNumber, topK)
}

func TestCalculatePrediction(t *testing.T) {
	userLatent := []float32{0.04138161242008209, 0.6600756645202637, -0.19171953201293945, -0.5317293405532837, 0.9864006638526917, -0.39250457286834717, -0.3122827112674713, -0.6813488006591797, 0.3494027256965637, 0.4451221525669098}
	userBiases := float32(-4.509291648864746)
	itemLatent := []float32{0.5855547785758972, 0.5377424359321594, 0.17182695865631104, -0.36310696601867676, 0.7737079858779907, -0.2763769030570984, -0.4855107069015503, -0.6457154154777527, -0.8728089332580566, 0.7721853256225586}
	itemBiases := float32(1.4512581825256348)
	prediction := float32(-1.0167253)

	model := NewModel()
	result := model.calcPrediction(userLatent, userBiases, itemLatent, itemBiases)
	assert.True(t, math.Abs(float64(result-prediction)) < 0.000001)
}
