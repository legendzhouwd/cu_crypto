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
	"testing"
)

func TestFeatureSelect(t *testing.T) {
	fileRows := [][]string{
		{"f1", "f2", "f3", "f4", "f5"},
		{"10", "20", "30", "40", "50"},
		{"11", "21", "31", "41", "51"},
		{"12", "22", "32", "42", "52"},
		{"15", "23", "33", "43", "53"},
		{"15", "24", "35", "44", "54"},
		{"15", "25", "35", "45", "55"},
		{"19", "26", "35", "46", "56"},
		{"19", "27", "36", "47", "57"},
		{"19", "28", "36", "48", "58"},
		{"19", "29", "36", "49", "59"},
	}

	newRows := FeatureSelect(fileRows, []string{"f1", "f3", "f5"})

	PrintFileRows(fileRows)
	PrintFileRows(newRows)
}
