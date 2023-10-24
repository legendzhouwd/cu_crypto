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
	"strconv"
)

// specify several intervals to replace
type ReplaceOpt struct {
	Intervals []IntervalValue
}

// replace sample value in [left, right) with specified value
type IntervalValue struct {
	Left  float64
	Right float64
	Value string
}

// ReplaceIntervalsBySpecifiedValues replace interval values by specified values
// support several features replacement at one time, return processed samples
// fileRows is sample content, the first row contains just names of features
// repOpt are replacement options for selected features
func ReplaceIntervalsBySpecifiedValues(fileRows [][]string, repOpt map[string]ReplaceOpt) ([][]string, error) {
	processedFileRows := copySamples(fileRows)

	var err error
	for f, opt := range repOpt {
		processedFileRows, err = ReplaceIntervalsForOne(processedFileRows, f, opt)
		if err != nil {
			return nil, err
		}
	}

	return processedFileRows, nil
}

// ReplaceIntervalsForOne replace intervals by specified values for one feature
func ReplaceIntervalsForOne(fileRows [][]string, feature string, opt ReplaceOpt) ([][]string, error) {
	processedFileRows := copySamples(fileRows)

	// find feature index
	featureIdx, err := findFeatureIdx(fileRows, feature)
	if err != nil {
		return nil, fmt.Errorf("failed to findFeatureIdx for feature[%s]: %v", feature, err)
	}

	for i := 1; i < len(fileRows); i++ {
		sampleValue, err := strconv.ParseFloat(fileRows[i][featureIdx], 64)
		if err != nil {
			return nil, fmt.Errorf("parse value failed for sample[%d] feature[%s]: %v", i, feature, err)
		}
		for j := 0; j < len(opt.Intervals); j++ {
			if sampleValue >= opt.Intervals[j].Left && sampleValue < opt.Intervals[j].Right {
				processedFileRows[i][featureIdx] = opt.Intervals[j].Value
				break
			}
		}
	}
	return processedFileRows, nil
}
