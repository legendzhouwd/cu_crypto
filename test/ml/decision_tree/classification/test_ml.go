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

package main

import (
	"encoding/csv"
	"encoding/json"
	"io/ioutil"
	"log"
	"strings"

	"github.com/legendzhouwd/cu_crypto/client/service/xchain"
	ml_common "github.com/legendzhouwd/cu_crypto/core/machine_learning/common"
	dt_class "github.com/legendzhouwd/cu_crypto/core/machine_learning/decision_tree/classification"
)

var xcc = new(xchain.XchainCryptoClient)

func main() {

	/*
		heart-disease是从UCI Heart Disease Data Set 官网下载的数据集
		数据属性说明：
		1）Age-年龄
		2）Sex-性别 （1 =男性； 0 =女性）
		3）Cp-胸痛类型（4个值，值1：典型的心绞痛，值2：非典型心绞痛，值3：非心绞痛，值4：无症状）
		4）Trestbps-患者入院时的静息血压（单位：mm Hg）
		5）Chol-血清胆固醇水平（单位：mg / dl）
		6）Fbs-空腹血糖（> 120 mg / dl ，1=真；0=假）
		7）Restecg-静息心电图结果（值0：正常，值1：有ST-T波异常（T波倒置和/或ST升高或降低> 0.05 mV），值2：根据Estes的标准显示可能或确定的左心室肥大）
		8）Thalach-达到的最大心率
		9）Exang-运动引起的心绞痛（1 =是； 0 =否）
		10）Oldpeak-运动相对于休息引起的ST压低
		11）Slope-最高运动ST段的斜率，（值1：上坡，值2：平坦，值3 ：下坡）
		12）Ca-萤光显色的主要血管数目（0-3）
		13）Thal-一种称为地中海贫血的血液疾病（3=正常；6=固定缺陷；7=可逆缺陷）
		14）Target- 患者是否患有心脏病。它是从0（不存在）到4的整数值。 Cleveland 数据库的实验集中在试图区分存在（值1、2、3、4）和不存在（值0）。
	*/

	// step 1: 导入训练数据集
	features, err := readFeaturesFromCSVFile("./testdata/train.csv")
	if err != nil {
		log.Printf("readFeaturesFromCSVFile failed: %v", err)
		return
	}
	dataSet := &ml_common.DTDataSet{
		Features: features,
	}

	// step 2: 模型训练
	contFeatures := []string{"Age", "Chol", "Trestbps", "Thalach", "Oldpeak"}
	label := "Target"
	cond := dt_class.StopCondition{
		SampleThreshold: 5,
		DepthThreshold:  12,
		GiniThreshold:   0.01,
	}
	regParam := 0.02
	tree, err := xcc.DecisionClassTreeTrain(dataSet, contFeatures, label, cond, regParam)
	if err != nil {
		log.Printf("DecisionTreeTrain failed: %v", err)
		return
	}
	model, _ := json.Marshal(tree)
	log.Printf("\n%s\n", model)

	// step 3: 预测
	verifyFeatures, err := readFeaturesFromCSVFile("./testdata/verify.csv")
	if err != nil {
		log.Printf("readFeaturesFromCSVFile failed: %v", err)
		return
	}
	verifyData := &ml_common.DTDataSet{verifyFeatures}

	results, err := xcc.DecisionClassTreePredict(verifyData, tree)
	if err != nil {
		log.Printf("DecisionTreePredict failed: %v", err)
		return
	}

	correct := 0
	total := len(verifyFeatures[0].Sets)
	for i := 0; i < total; i++ {
		predictResult := results[i]
		realResult := verifyFeatures[len(verifyFeatures)-1].Sets[i]
		log.Printf("predict result of [%d]: %v, real result: %v", i, predictResult, realResult)
		if predictResult == realResult {
			correct++
		}
	}
	log.Printf("accuracy: %f", float64(correct)/float64(total))
}

// readFeaturesFromCSVFile 从 csv 文件中读取样本特征
func readFeaturesFromCSVFile(path string) ([]*ml_common.DTDataFeature, error) {
	content, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	plainFile := string(content)
	r := csv.NewReader(strings.NewReader(plainFile))
	ss, _ := r.ReadAll()

	features, err := xcc.DecisionTreeImportFeatures(ss)
	if err != nil {
		return nil, err
	}
	return features, nil
}
