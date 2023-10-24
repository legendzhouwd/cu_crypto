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

import "testing"

func TestFillMissingValues(t *testing.T) {
	fileRows := [][]string{
		{"f1", "f2", "f3", "f4", "f5"},
		{"", "20", "30", "40", "50"},
		{"11", "", "31", "41", "51"},
		{"12", "22", "", "42", "52"},
		{"13", "23", "", "43", "53"},
		{"14", "24", "", "44", "54"},
		{"", "25", "35", "45", "55"},
		{"16", "26", "36", "", "56"},
		{"", "27", "37", "47", "57"},
		{"18", "28", "38", "", "58"},
		{"19", "", "39", "49", "59"},
	}
	fillOpts := make(map[string]FillOpt)
	fillOpts["f1"] = FillOpt{
		Method: FillMissingValuesByAverage, //14.71
	}
	fillOpts["f2"] = FillOpt{
		Method: FillMissingValuesByMedian, //24.5
	}
	fillOpts["f3"] = FillOpt{
		Method: FillMissingValuesBySpecifiedValue,
		Value:  "33",
	}
	fillOpts["f4"] = FillOpt{
		Method: FillMissingValuesBySpecifiedValue,
		Value:  "45",
	}

	newFiles, err := FillMissingValues(fileRows, fillOpts)
	if err != nil {
		t.Error(err)
	}
	PrintFileRows(newFiles)
	PrintFileRows(fileRows)
}

func TestGetMedian(t *testing.T) {
	values := []float64{3, 16, 12, 5, 19, 6, 22, 1}
	// supposed to be 9
	median := getMedian(values)
	if median != 9 {
		t.Errorf("got %v, median supposed to be 9", median)
	}
}
