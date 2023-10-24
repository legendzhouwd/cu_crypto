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

func TestFilterByThreshold(t *testing.T) {
	fileRows := [][]string{
		{"f1", "f2", "f3", "f4", "f5", "f6"},
		{"10", "20", "30", "40", "50", "60"},
		{"11", "21", "31", "41", "51", "61"},
		{"12", "22", "32", "42", "52", "62"},
		{"13", "23", "33", "43", "53", "63"},
		{"14", "24", "35", "44", "54", "64"},
		{"15", "25", "35", "45", "55", "65"},
		{"16", "26", "35", "46", "56", "66"},
		{"17", "27", "35", "47", "57", "67"},
		{"18", "28", "35", "48", "58", "68"},
		{"19", "29", "35", "49", "59", "69"},
	}
	filOpts := make(map[string]FilterOpt)
	filOpts["f1"] = FilterOpt{
		Threshold:    11,
		FilterMethod: FilterByGreater,
	}
	filOpts["f2"] = FilterOpt{
		Threshold:    23,
		FilterMethod: FilterByGreaterOrEqual,
	}
	filOpts["f3"] = FilterOpt{
		Threshold:    35,
		FilterMethod: FilterByEqual,
	}
	filOpts["f4"] = FilterOpt{
		Threshold:    48,
		FilterMethod: FilterByLess,
	}
	filOpts["f5"] = FilterOpt{
		Threshold:    56,
		FilterMethod: FilterByLessOrEqual,
	}
	filOpts["f6"] = FilterOpt{
		Threshold:    64,
		FilterMethod: FilterByNotEqual,
	}

	newFiles, err := FilterByThreshold(fileRows, filOpts)
	if err != nil {
		t.Error(err)
	}
	PrintFileRows(newFiles)
	PrintFileRows(fileRows)
}
