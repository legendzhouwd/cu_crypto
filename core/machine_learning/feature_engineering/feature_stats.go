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
	"sort"
	"strconv"
)

type StatisticsInfo struct {
	Maximum  float64  // largest value
	Minimum  float64  // smallest value
	Mean     float64  // average value
	Mode     []string // values appear most frequently, a set may have multiple modes, or no mode(all values appear with same frequency)
	Median   float64  // median value
	StandDev float64  // standard deviation
}

// FeatureStatistics computes stats info for given feature
// fileRows is sample content, the first row contains just names of features
// valueIsStr is true if sample value type of the feature is 'string'
func FeatureStatistics(fileRows [][]string, feature string, valueIsStr bool) (StatisticsInfo, error) {
	if valueIsStr {
		return featureStatisticsStr(fileRows, feature)
	}
	return featureStatisticsFloat(fileRows, feature)
}

// only calculate modes
func featureStatisticsStr(fileRows [][]string, feature string) (StatisticsInfo, error) {
	valueTimesMap := make(map[string]int)

	// find feature index
	featureIdx, err := findFeatureIdx(fileRows, feature)
	if err != nil {
		return StatisticsInfo{}, fmt.Errorf("failed to findFeatureIdx for feature[%s]: %v", feature, err)
	}

	for i := 1; i < len(fileRows); i++ {
		value := fileRows[i][featureIdx]

		if _, exist := valueTimesMap[value]; !exist {
			valueTimesMap[value] = 1
		} else {
			valueTimesMap[value]++
		}
	}

	modes := calModes(valueTimesMap)

	return StatisticsInfo{
		Mode: modes,
	}, nil
}

func featureStatisticsFloat(fileRows [][]string, feature string) (StatisticsInfo, error) {
	var max = -math.MaxFloat64
	var min = math.MaxFloat64
	var total float64 = 0
	valueTimesMap := make(map[string]int)

	// find feature index
	featureIdx, err := findFeatureIdx(fileRows, feature)
	if err != nil {
		return StatisticsInfo{}, fmt.Errorf("failed to findFeatureIdx for feature[%s]: %v", feature, err)
	}

	featValues := make([]float64, 0, len(fileRows)-1)
	for i := 1; i < len(fileRows); i++ {
		value, err := strconv.ParseFloat(fileRows[i][featureIdx], 64)
		if err != nil {
			return StatisticsInfo{}, fmt.Errorf("parse value failed for sample[%d] feature[%s]: %v", i, feature, err)
		}
		featValues = append(featValues, value)

		if value > max {
			max = value
		}
		if value < min {
			min = value
		}
		total += value
		if _, exist := valueTimesMap[fmt.Sprintf("%f", value)]; !exist {
			valueTimesMap[fmt.Sprintf("%f", value)] = 1
		} else {
			valueTimesMap[fmt.Sprintf("%f", value)]++
		}
	}

	// calculate mean
	mean := total / float64(len(fileRows)-1)
	modes := calModes(valueTimesMap)

	sort.Float64s(featValues)
	median := featValues[len(featValues)/2]
	if len(featValues)%2 == 0 {
		median = (featValues[len(featValues)/2-1] + featValues[len(featValues)/2]) / 2
	}

	standDev := calStandDev(featValues, mean)

	return StatisticsInfo{
		Maximum:  max,
		Minimum:  min,
		Mean:     mean,
		Mode:     modes,
		Median:   median,
		StandDev: standDev,
	}, nil
}

// calculate modes
func calModes(valueTimesMap map[string]int) []string {
	var modes []string
	// get frequency list and rearrange in ascending order
	freqList := make([]int, 0, len(valueTimesMap))
	for _, times := range valueTimesMap {
		freqList = append(freqList, times)
	}
	sort.Ints(freqList)
	// if only one value, return it as mode
	if len(freqList) == 1 {
		for value, _ := range valueTimesMap {
			modes = append(modes, value)
			break
		}
	} else {
		// if all values appear with same frequency, there is no mode
		if freqList[0] != freqList[len(freqList)-1] {
			for value, times := range valueTimesMap {
				// find values that appear most frequently
				if times == freqList[len(freqList)-1] {
					modes = append(modes, value)
				}
			}
		}
	}

	return modes
}

// calculate standard deviation, sqrt(sum[(xi-x)^2])
func calStandDev(featValues []float64, mean float64) float64 {
	dev := 0.0
	for i := 0; i < len(featValues); i++ {
		dev += (featValues[i] - mean) * (featValues[i] - mean)
	}
	dev = dev / float64(len(featValues))
	return math.Sqrt(dev)
}
