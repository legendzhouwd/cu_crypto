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
	"math"
	"strconv"
)

type BinOpt struct {
	BinNum int // target bins number
}

// EqualWidthBinning equal width binning for selected feature list
// fileRows is sample content, the first row contains just names of features
// binOpts are binning options, map from feature name to BinOpt
func EqualWidthBinning(fileRows [][]string, binOpts map[string]BinOpt) ([][]string, error) {
	processedFileRows := copySamples(fileRows)

	for f, opt := range binOpts {
		var err error
		processedFileRows, err = EqualWidthBinningForOne(processedFileRows, f, opt.BinNum)
		if err != nil {
			return nil, err
		}
	}

	return processedFileRows, nil
}

// EqualWidthBinningForOne binning for one selected feature
// fileRows is sample content, the first row contains just names of features
// feature is selected feature to be processed
// binNum is target number of bins
// return new samples, value of selected feature is a discrete number among [0,1,...,bins-1]
func EqualWidthBinningForOne(fileRows [][]string, feature string, binNum int) ([][]string, error) {
	// find feature index
	featureIdx, err := findFeatureIdx(fileRows, feature)
	if err != nil {
		return nil, fmt.Errorf("failed to findFeatureIdx for feature[%s]: %v", feature, err)
	}
	// find max and min
	var max = -math.MaxFloat64
	var min = math.MaxFloat64
	for i := 1; i < len(fileRows); i++ {
		value, err := strconv.ParseFloat(fileRows[i][featureIdx], 64)
		if err != nil {
			return nil, fmt.Errorf("parse value failed for sample[%d] feature[%s]: %v", i, feature, err)
		}
		if value > max {
			max = value
		}
		if value < min {
			min = value
		}
	}

	// compute width for each bin
	width := (max - min) / float64(binNum)
	// find bin index for each sample
	idxMap := make(map[int]int)
	for i := 1; i < len(fileRows); i++ {
		value, err := strconv.ParseFloat(fileRows[i][featureIdx], 64)
		if err != nil {
			return nil, fmt.Errorf("parse value failed for sample[%d] feature[%s]: %v", i, feature, err)
		}
		if value == max {
			idxMap[i] = binNum - 1
		} else {
			// compute bin index
			idxMap[i] = int((value - min) / width)
		}
	}

	newRows := make([][]string, 0, len(fileRows))
	// first row should contain all features
	newRows = append(newRows, fileRows[0])
	for i := 1; i < len(fileRows); i++ {
		row := make([]string, len(fileRows[0]))
		copy(row[0:], fileRows[i])
		row[featureIdx] = fmt.Sprintf("%d", idxMap[i])

		newRows = append(newRows, row)
	}

	return newRows, nil
}
