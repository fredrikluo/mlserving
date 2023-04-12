// Copyright (c) 2022 Luo Zhiyu
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

package embeddings

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"strconv"

	"github.com/fredrikluo/mlserving/internal/annoyindex"
)

// AnnIndex Spotify Approximate Nearest Neighbors index
type AnnIndex struct {
	index    annoyindex.AnnoyIndexDotProduct
	index2Id map[int]string
	id2index map[string]int
}

func NewAnnIndex(filename string, index2idFilename string) (*AnnIndex, error) {
	index := AnnIndex{}
	index.index = annoyindex.NewAnnoyIndexDotProduct(10)
	if ret := index.index.Load(filename, true); !ret {
		return nil, fmt.Errorf("failed to load %s", filename)
	}

	index2idFile, err := ioutil.ReadFile(index2idFilename)
	if err != nil {
		return nil, err
	}

	index2IdJson := map[string]string{}
	err = json.Unmarshal([]byte(index2idFile), &index2IdJson)
	if err != nil {
		return nil,
			fmt.Errorf("can't load index2idfile from json %v, with file %s", err, index2idFilename)
	}

	// build the index
	index.index2Id = map[int]string{}
	for k, v := range index2IdJson {
		if key, err := strconv.Atoi(k); err == nil {
			index.index2Id[key] = v
		}
	}

	// build the inverted index as well
	index.id2index = map[string]int{}
	for k, v := range index.index2Id {
		index.id2index[v] = k
	}

	return &index, nil
}

func (ann *AnnIndex) IndexToId(index int) (string, bool) {
	val, found := ann.index2Id[index]
	return val, found
}

func (ann *AnnIndex) IdToIndex(id string) (int, bool) {
	val, found := ann.id2index[id]
	return val, found
}

func (ann *AnnIndex) GetNnsByItem(index int, n int, search_k int) []int {
	var result []int
	ann.index.GetNnsByItem(index, n, search_k, &result)
	return result
}
