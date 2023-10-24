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
	"testing"
)

func TestFeatureStatistics(t *testing.T) {
	fileRows := [][]string{
		{"f1", "f2", "f3", "f4", "f5"},
		{"10", "20", "30", "40", "aa"},
		{"11", "21", "31", "41", "bb"},
		{"12", "22", "32", "42", "cc"},
		{"15", "23", "33", "43", "aa"},
		{"15", "24", "35", "44", "cc"},
		{"15", "25", "35", "45", "aa"},
		{"19", "26", "35", "46", "dd"},
		{"19", "27", "36", "47", "ee"},
		{"19", "28", "36", "48", "gg"},
		{"19", "29", "36", "49", "bb"},
	}

	stats, err := FeatureStatistics(fileRows, "f3", false)
	if err != nil {
		t.Error(err)
	}
	// supposed to be {36 30 33.9 [35 36] 35 2.11}
	fmt.Println(stats)

	stats, err = FeatureStatistics(fileRows, "f5", true)
	if err != nil {
		t.Error(err)
	}
	// supposed to be {   ["aa"]  }
	fmt.Println(stats)
}
