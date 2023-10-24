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

const (
	FillMissingValuesByAverage        = "average"
	FillMissingValuesByMedian         = "median"
	FillMissingValuesBySpecifiedValue = "specified"
)

type FillOpt struct {
	Method string // "average", "median" or "specified"
	Value  string // cannot be empty when Method is "specified"
}

// FillMissingValues fills missing values of target features by average, median or specified value
// support filling several features at one time, return processed samples
// fileRows is sample content, the first row contains just names of features
// fillOpts are filling options for selected features
func FillMissingValues(fileRows [][]string, fillOpts map[string]FillOpt) ([][]string, error) {
	processedFileRows := copySamples(fileRows)

	var err error
	for f, opt := range fillOpts {
		if opt.Method == FillMissingValuesByAverage {
			processedFileRows, err = FillByAverage(processedFileRows, f)
		} else if opt.Method == FillMissingValuesByMedian {
			processedFileRows, err = FillByMedian(processedFileRows, f)
		} else {
			if len(opt.Value) == 0 {
				return nil, fmt.Errorf("feature[%s] value cannot be empty when filling by specified value", f)
			}
			processedFileRows, err = FillBySpecified(processedFileRows, f, opt.Value)
		}
		if err != nil {
			return nil, err
		}
	}

	return processedFileRows, nil
}

// FillByAverage fills missing values of target features by average
// fileRows is sample content, the first row contains just names of features
func FillByAverage(fileRows [][]string, feature string) ([][]string, error) {
	processedFileRows := copySamples(fileRows)

	// find feature index
	featureIdx, err := findFeatureIdx(fileRows, feature)
	if err != nil {
		return nil, fmt.Errorf("failed to findFeatureIdx for feature[%s]: %v", feature, err)
	}

	// find samples with missing value, and compute average
	var samplesToFill []int
	var total float64 = 0
	for i := 1; i < len(fileRows); i++ {
		if len(fileRows[i][featureIdx]) == 0 {
			samplesToFill = append(samplesToFill, i)
		} else {
			value, err := strconv.ParseFloat(fileRows[i][featureIdx], 64)
			if err != nil {
				return nil, fmt.Errorf("parse value failed for sample[%d] feature[%s]: %v", i, feature, err)
			}
			total += value
		}
	}
	// fill missing values by average
	average := total / float64(len(fileRows)-1-len(samplesToFill))
	for i := 0; i < len(samplesToFill); i++ {
		sample := samplesToFill[i]
		processedFileRows[sample][featureIdx] = strconv.FormatFloat(average, 'g', -1, 64)
	}
	return processedFileRows, nil
}

// FillByMedian fills missing values of target features by median
// fileRows is sample content, the first row contains just names of features
func FillByMedian(fileRows [][]string, feature string) ([][]string, error) {
	processedFileRows := copySamples(fileRows)

	// find feature index
	featureIdx, err := findFeatureIdx(fileRows, feature)
	if err != nil {
		return nil, fmt.Errorf("failed to findFeatureIdx for feature[%s]: %v", feature, err)
	}

	// find samples with missing value, and compute average
	var samplesToFill []int
	var existValues []float64
	for i := 1; i < len(fileRows); i++ {
		if len(fileRows[i][featureIdx]) == 0 {
			samplesToFill = append(samplesToFill, i)
		} else {
			value, err := strconv.ParseFloat(fileRows[i][featureIdx], 64)
			if err != nil {
				return nil, fmt.Errorf("parse value failed for sample[%d] feature[%s]: %v", i, feature, err)
			}
			existValues = append(existValues, value)
		}
	}
	// fill missing values by median
	median := getMedian(existValues)
	for i := 0; i < len(samplesToFill); i++ {
		sample := samplesToFill[i]
		processedFileRows[sample][featureIdx] = strconv.FormatFloat(median, 'g', -1, 64)
	}

	return processedFileRows, nil
}

// FillBySpecified fills missing values of target features by specified value
// fileRows is sample content, the first row contains just names of features
func FillBySpecified(fileRows [][]string, feature, value string) ([][]string, error) {
	processedFileRows := copySamples(fileRows)

	// find feature index
	featureIdx, err := findFeatureIdx(fileRows, feature)
	if err != nil {
		return nil, fmt.Errorf("failed to findFeatureIdx for feature[%s]: %v", feature, err)
	}

	// fill missing values with specified value
	for i := 1; i < len(fileRows); i++ {
		if len(fileRows[i][featureIdx]) == 0 {
			processedFileRows[i][featureIdx] = value
		}
	}

	return processedFileRows, nil
}

// find feature index among all features
func findFeatureIdx(fileRows [][]string, feature string) (int, error) {
	for i := 0; i < len(fileRows[0]); i++ {
		if fileRows[0][i] == feature {
			return i, nil
		}
	}
	return 0, fmt.Errorf("invalid feature, not exist")
}

// compute median of float64 slice
func getMedian(values []float64) float64 {
	// ascending order
	sort.Float64s(values)
	if len(values)%2 == 1 {
		middleIdx := len(values) / 2
		return values[middleIdx]
	}
	middle1Idx := len(values) / 2
	middle2Idx := len(values)/2 - 1
	return (values[middle1Idx] + values[middle2Idx]) / 2
}

// copy sample values to a new slice
func copySamples(fileRows [][]string) [][]string {
	newSamples := make([][]string, 0, len(fileRows))
	for i := 0; i < len(fileRows); i++ {
		row := make([]string, len(fileRows[0]))
		copy(row[0:], fileRows[i])
		newSamples = append(newSamples, row)
	}
	return newSamples
}

// PrintFileRows print sample content in human-readable format
func PrintFileRows(fileRows [][]string) {
	for i := 0; i < len(fileRows); i++ {
		for j := 0; j < len(fileRows[i]); j++ {
			fmt.Printf("%s\t", fileRows[i][j])
		}
		fmt.Printf("\n")
	}
	fmt.Printf("\n")
}
