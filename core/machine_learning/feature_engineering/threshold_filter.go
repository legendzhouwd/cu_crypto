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

const (
	FilterByGreater        = "g"
	FilterByGreaterOrEqual = "ge"
	FilterByEqual          = "e"
	FilterByNotEqual       = "ne"
	FilterByLess           = "l"
	FilterByLessOrEqual    = "le"
)

// filter condition
type FilterOpt struct {
	Threshold    float64
	FilterMethod string // "g", "ge", "e", "ne", "l" or "le"
}

// FilterByThreshold filter samples by given threshold and method
// support several features filter at one time, return processed samples
// fileRows is sample content, the first row contains just names of features
// filOpt are filter options for selected features
func FilterByThreshold(fileRows [][]string, filOpt map[string]FilterOpt) ([][]string, error) {
	processedFileRows := copySamples(fileRows)

	var err error
	for f, opt := range filOpt {
		processedFileRows, err = FilterByThresholdForOne(processedFileRows, f, opt)
		if err != nil {
			return nil, err
		}
	}

	return processedFileRows, nil
}

// FilterByThresholdForOne filter by threshold for one feature
func FilterByThresholdForOne(fileRows [][]string, feature string, opt FilterOpt) ([][]string, error) {
	var processedFileRows [][]string

	// find feature index
	featureIdx, err := findFeatureIdx(fileRows, feature)
	if err != nil {
		return nil, fmt.Errorf("failed to findFeatureIdx for feature[%s]: %v", feature, err)
	}

	firstRow := make([]string, len(fileRows[0]))
	copy(firstRow[0:], fileRows[0])
	processedFileRows = append(processedFileRows, firstRow)
	for i := 1; i < len(fileRows); i++ {
		sampleValue, err := strconv.ParseFloat(fileRows[i][featureIdx], 64)
		if err != nil {
			return nil, fmt.Errorf("parse value failed for sample[%d] feature[%s]: %v", i, feature, err)
		}

		satisfied := false
		if opt.FilterMethod == FilterByGreater {
			if sampleValue > opt.Threshold {
				satisfied = true
			}
		} else if opt.FilterMethod == FilterByGreaterOrEqual {
			if sampleValue >= opt.Threshold {
				satisfied = true
			}
		} else if opt.FilterMethod == FilterByEqual {
			if sampleValue == opt.Threshold {
				satisfied = true
			}
		} else if opt.FilterMethod == FilterByNotEqual {
			if sampleValue != opt.Threshold {
				satisfied = true
			}
		} else if opt.FilterMethod == FilterByLess {
			if sampleValue < opt.Threshold {
				satisfied = true
			}
		} else if opt.FilterMethod == FilterByLessOrEqual {
			if sampleValue <= opt.Threshold {
				satisfied = true
			}
		} else {
			return nil, fmt.Errorf("unsupported filter method")
		}

		if satisfied {
			row := make([]string, len(fileRows[0]))
			copy(row[0:], fileRows[i])
			processedFileRows = append(processedFileRows, row)
		}
	}
	return processedFileRows, nil
}
