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
	"fmt"
	"sort"
	"strconv"
)

// EqualFrequencyBinning equal frequency binning for selected feature list
// fileRows is sample content, the first row contains just names of features
// binOpts are binning options, map from feature name to BinOpt
func EqualFrequencyBinning(fileRows [][]string, binOpts map[string]BinOpt) ([][]string, error) {
	processedFileRows := copySamples(fileRows)

	for f, opt := range binOpts {
		var err error
		processedFileRows, err = EqualFrequencyBinningForOne(processedFileRows, f, opt.BinNum)
		if err != nil {
			return nil, err
		}
	}

	return processedFileRows, nil
}

// EqualFrequencyBinningForOne binning for one selected feature
// fileRows is sample content, the first row contains just names of features
// feature is selected feature to be processed
// binNum is target number of bins
// return new samples, value of selected feature is a discrete number among [0,1,...,bins-1]
func EqualFrequencyBinningForOne(fileRows [][]string, feature string, binNum int) ([][]string, error) {
	// find feature index
	featureIdx, err := findFeatureIdx(fileRows, feature)
	if err != nil {
		return nil, fmt.Errorf("failed to findFeatureIdx for feature[%s]: %v", feature, err)
	}

	sampleNumPerBin := (len(fileRows) - 1) / binNum
	remainder := (len(fileRows) - 1) % binNum

	// sort feature value by ascending order
	list := make([]float64, 0, len(fileRows)-1)
	// map feature value to sample indices
	m := make(map[float64][]int)
	for i := 1; i < len(fileRows); i++ {
		value, err := strconv.ParseFloat(fileRows[i][featureIdx], 64)
		if err != nil {
			return nil, fmt.Errorf("parse value failed for sample[%d] feature[%s]: %v", i, feature, err)
		}
		list = append(list, value)
		m[value] = append(m[value], i)
	}
	sort.Float64s(list)

	// set new samples
	newRows := make([][]string, 0, len(fileRows))
	for i := 0; i < len(fileRows); i++ {
		newRows = append(newRows, fileRows[i])
	}

	// first 'remainder' bins have 'sampleNumPerBin'+1 samples, others have 'sampleNumPerBin' samples
	// binIdx is in {0, 1,..., binNum-1}
	binIdx := 0
	// denote samples number in current bin
	sampleNumThisBin := 0
	for i := 0; i < len(list); i++ {
		value := list[i]

		for _, v := range m[value] {
			newRows[v][featureIdx] = fmt.Sprintf("%d", binIdx)

			sampleNumThisBin++
			if (binIdx < remainder && sampleNumThisBin >= sampleNumPerBin+1) || (binIdx >= remainder && sampleNumThisBin >= sampleNumPerBin) {
				// reset sampleNumThisBin
				sampleNumThisBin = 0
				// bin index plus one
				binIdx++
			}
		}
		// list may continue duplicate values, handle those values only once
		delete(m, value)
	}

	return newRows, nil
}
