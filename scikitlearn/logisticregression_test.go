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

package scikitlearn

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPredict(t *testing.T) {
	lr := NewLogisticRegression()
	err := lr.Load("test_data/lr.json")
	assert.Nil(t, err)
	expected := [][]float64{
		{0.81999686, 0.18000314},
		{0.69272057, 0.3072794},
		{0.52732579, 0.4726742},
		{0.35570732, 0.6442926},
		{0.21458576, 0.7854142},
		{0.11910229, 0.8808977},
		{0.06271329, 0.9372867},
		{0.03205032, 0.9679496},
		{0.0161218, 0.9838782},
		{0.00804372, 0.9919562}}

	for i := 0; i < 10; i++ {
		vec := []float64{float64(i)}
		res, err := lr.Pred(vec)
		assert.Nil(t, err)
		assert.True(t, math.Abs(res[0]-expected[i][0]) < 0.001)
		assert.True(t, math.Abs(res[1]-expected[i][1]) < 0.001)
	}
}
