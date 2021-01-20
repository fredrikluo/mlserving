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

package lightgbm

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGBDTPredict(t *testing.T) {
	gbdt := NewModel()
	err := gbdt.Load("test_data/gbdt.txt")
	assert.Nil(t, err)

	fvals := []float64{12.47, 18.6, 81.09, 481.9, 0.09965, 0.1058, 0.08005, 0.03821, 0.1925, 0.06373, 0.3961, 1.044, 2.497, 30.29, 0.006953, 0.01911, 0.02701, 0.01037, 0.01782, 0.003586, 14.97, 24.64, 96.05, 677.9, 0.1426, 0.2378, 0.2671, 0.1015, 0.3014, 0.087}
	_, result, _ := gbdt.Predict(fvals, true)
	expected := []uint32{0, 0, 0, 0, 11, 14, 18, 12, 14, 12, 11, 10, 5, 13, 18, 11, 13, 11, 15, 22, 10, 13, 6, 1, 18, 6, 16, 14, 13, 3, 19, 15, 3, 8, 11, 13, 10, 14, 6}
	assert.Equal(t, result, expected)
}
