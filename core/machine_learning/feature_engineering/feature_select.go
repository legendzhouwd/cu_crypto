// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package feature_engineering

import (
	"github.com/PaddlePaddle/PaddleDTX/crypto/common/utils"
)

// FeatureSelect filter samples with selected features
// fileRows is sample content, the first row contains just names of features
// features is target feature list to be selected
func FeatureSelect(fileRows [][]string, features []string) [][]string {
	if len(features) == 0 {
		return [][]string{}
	}

	// find feature index list to be selected
	featuresIdx := findFeatureIdxList(fileRows, features)

	// copy sample values to a new slice
	newSamples := make([][]string, 0, len(fileRows))
	for i := 0; i < len(fileRows); i++ {
		row := make([]string, len(featuresIdx))
		for j := 0; j < len(featuresIdx); j++ {
			row[j] = fileRows[i][featuresIdx[j]]
		}
		newSamples = append(newSamples, row)
	}

	return newSamples
}

// find feature index list among all features
func findFeatureIdxList(fileRows [][]string, features []string) []int {
	var list []int
	for i := 0; i < len(fileRows[0]); i++ {
		if utils.StringInSlice(fileRows[0][i], features) {
			list = append(list, i)
		}
	}
	return list
}
