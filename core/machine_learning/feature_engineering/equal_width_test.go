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

func TestEqualWidthBinning(t *testing.T) {
	fileRows := [][]string{
		{"f1", "f2", "f3", "f4", "f5"},
		{"10", "20", "30", "40", "1"},
		{"11", "21", "31", "41", "4"},
		{"12", "22", "32", "42", "23"},
		{"13", "23", "33", "43", "6"},
		{"14", "24", "34", "44", "34"},
		{"15", "25", "35", "45", "12"},
		{"16", "26", "36", "46", "66"},
		{"17", "27", "37", "47", "5"},
		{"18", "28", "38", "48", "88"},
		{"19", "29", "39", "49", "99"},
	}

	binOpts := make(map[string]BinOpt)
	binOpts["f2"] = BinOpt{
		BinNum: 3,
	}
	binOpts["f5"] = BinOpt{
		BinNum: 4,
	}
	newRows, err := EqualWidthBinning(fileRows, binOpts)
	if err != nil {
		t.Error(err)
	}
	PrintFileRows(fileRows)
	PrintFileRows(newRows)
}
