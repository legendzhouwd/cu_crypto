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

func TestReplaceIntervalBySpecifiedValues(t *testing.T) {
	fileRows := [][]string{
		{"f1", "f2", "f3", "f4", "f5"},
		{"10", "20", "30", "40", "50"},
		{"11", "21", "31", "41", "51"},
		{"12", "22", "32", "42", "52"},
		{"13", "23", "33", "43", "53"},
		{"14", "24", "34", "44", "54"},
		{"15", "25", "35", "45", "55"},
		{"16", "26", "36", "46", "56"},
		{"17", "27", "37", "47", "57"},
		{"18", "28", "38", "48", "58"},
		{"19", "29", "39", "49", "59"},
	}
	repOpts := make(map[string]ReplaceOpt)
	repOpts["f1"] = ReplaceOpt{
		Intervals: []IntervalValue{{10, 12, "0"}, {12, 16, "1"}, {16, 20, "2"}},
	}
	repOpts["f2"] = ReplaceOpt{
		Intervals: []IntervalValue{{20, 26, "100"}, {26, 30, "200"}},
	}
	repOpts["f3"] = ReplaceOpt{
		Intervals: []IntervalValue{{32, 36, "test"}},
	}

	newFiles, err := ReplaceIntervalsBySpecifiedValues(fileRows, repOpts)
	if err != nil {
		t.Error(err)
	}
	PrintFileRows(newFiles)
	PrintFileRows(fileRows)
}
