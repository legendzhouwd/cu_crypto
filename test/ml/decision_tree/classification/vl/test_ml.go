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
	"fmt"
	"io/ioutil"
	"log"
	"strings"

	"github.com/legendzhouwd/cu_crypto/client/service/xchain"
	ml_common "github.com/legendzhouwd/cu_crypto/core/machine_learning/common"
	dtc_vertical "github.com/legendzhouwd/cu_crypto/core/machine_learning/decision_tree/classification/mpc_vertical"
)

var (
	xcc = new(xchain.XchainCryptoClient)

	// 非标签方的模型
	modelA = make(map[string]*dtc_vertical.CTreeNode) // hash(hash(depth), index) -> node
	// 标签方的模型
	modelB = make(map[string]*dtc_vertical.CTreeNode)
)

func main() {
	// step 1: 导入训练数据集
	featuresA, err := readFeaturesFromCSVFile("./testdata/trainA.csv")
	if err != nil {
		log.Printf("readFeaturesFromCSVFile failed: %v", err)
		return
	}
	dataSetA := &ml_common.DTDataSet{
		Features: featuresA,
	}

	featuresB, err := readFeaturesFromCSVFile("./testdata/trainB.csv")
	if err != nil {
		log.Printf("readFeaturesFromCSVFile failed: %v", err)
		return
	}
	dataSetB := &ml_common.DTDataSet{
		Features: featuresB,
	}

	// 记录当前所有样本id列表
	var originalIDs []int
	for i := 0; i < len(dataSetA.Features[0].Sets); i++ {
		originalIDs = append(originalIDs, i)
	}

	// step 2: 模型训练
	contFeatures := []string{"Age", "Chol", "Trestbps", "Thalach", "Oldpeak"}
	label := "Target"
	cond := dtc_vertical.StopCondition{
		SampleThreshold: 5,
		DepthThreshold:  12,
		GiniThreshold:   0.01,
	}
	regParam := 0.02

	// B 判断根节点是否为叶子
	rootGini := xcc.DtcVLCalGiniByTagPart(dataSetB, originalIDs, nil, label)
	v, result := xcc.DtcVLDecideTerminateTagPart(dataSetB, originalIDs, nil, rootGini, 0, 0, label, cond)
	fmt.Printf("determine root node: %v, result: %v gini: %f\n", v, result, rootGini)

	rootNode := &dtc_vertical.CTreeNode{
		SampleIDList: originalIDs,
		Depth:        0,
		Index:        0,
		Gini:         rootGini,
	}
	// 将根节点部分信息存到 B 方
	rootNodeID := xcc.DtcVLNodeKeyByDepthIndex(0, 0)
	modelB[rootNodeID] = rootNode

	determineNodeByDepthIndex(dataSetA, dataSetB, rootNode, label, regParam, contFeatures, cond)

	// step 3: 剪枝
	pruneTreeB(dataSetB, regParam, label)
	printModels()

	// step 4: 分布式预测
	verifyFeaturesA, err := readFeaturesFromCSVFile("./testdata/verifyA.csv")
	if err != nil {
		log.Printf("readFeaturesFromCSVFile failed: %v", err)
		return
	}
	verifyDataA := &ml_common.DTDataSet{
		Features: verifyFeaturesA,
	}
	verifyFeaturesB, err := readFeaturesFromCSVFile("./testdata/verifyB.csv")
	if err != nil {
		log.Printf("readFeaturesFromCSVFile failed: %v", err)
		return
	}
	verifyDataB := &ml_common.DTDataSet{
		Features: verifyFeaturesB,
	}
	dataMapA := dataFeaturesToDataMaps(verifyDataA)
	dataMapB := dataFeaturesToDataMaps(verifyDataB)

	correct := 0
	for i := 0; i < len(dataMapA); i++ {
		predictResult := predictA(0, 0, dataMapA[i], dataMapB[i])
		realResult := verifyFeaturesB[len(verifyFeaturesB)-1].Sets[i]
		log.Printf("predict result of [%d]: %v, real result: %v", i, predictResult, realResult)
		if predictResult == realResult {
			correct++
		}
	}
	log.Printf("accuracy: %f", float64(correct)/float64(len(dataMapA)))
}

// 指定depth和index，确定该位置的树节点，若用A的特征划分，则存入modelA，否则存入modelB
// 入参 curNode 中包含 sampleIDList、depth、index、gini
func determineNodeByDepthIndex(allDataA, allDataB *ml_common.DTDataSet, curNode *dtc_vertical.CTreeNode, label string,
	regParam float64, contFeatures []string, stopCond dtc_vertical.StopCondition) {
	// 计算树节点对应的key，用于模型存储
	nodeKey := xcc.DtcVLNodeKeyByDepthIndex(curNode.Depth, curNode.Index)
	fmt.Printf("===== depth: %d, index: %d, ids: %d =====\n", curNode.Depth, curNode.Index, len(curNode.SampleIDList))

	// A&B 本地计算所有可能的特征、分割值、左右样本id列表，用于特征选择
	featuresA, splitsA, idsLeftA, idsRightA, _ := xcc.DtcVLPrepForFeatureSelect(allDataA, curNode.SampleIDList, label, contFeatures)
	featuresB, splitsB, idsLeftB, idsRightB, _ := xcc.DtcVLPrepForFeatureSelect(allDataB, curNode.SampleIDList, label, contFeatures)

	// A请求B计算Gini列表，计算每个可能的分割对应的Gini指数
	var minGiniA float64 = 1
	minGiniIdxA := 0
	for i := 0; i < len(idsLeftA); i++ {
		// TODO 网络交互
		giniA := xcc.DtcVLCalGiniByTagPart(allDataB, idsLeftA[i], idsRightA[i], label)
		if giniA < minGiniA {
			minGiniIdxA = i
			minGiniA = giniA
		}
	}

	// B 本地计算Gini列表，计算每个可能的分割对应的Gini指数
	var minGiniB float64 = 1
	minGiniIdxB := 0
	for i := 0; i < len(idsLeftB); i++ {
		giniB := xcc.DtcVLCalGiniByTagPart(allDataB, idsLeftB[i], idsRightB[i], label)
		if giniB < minGiniB {
			minGiniIdxB = i
			minGiniB = giniB
		}
	}

	// B 本地计算，选择两方中较小的Gini指数
	var isA bool
	var finalFeature string
	var finalSplit interface{}
	var finalContinuous bool
	var idsLeft []int
	var idsRight []int
	if minGiniA < minGiniB {
		isA = true
		finalFeature = featuresA[minGiniIdxA]
		finalSplit = splitsA[minGiniIdxA]

		idsLeft = idsLeftA[minGiniIdxA]
		idsRight = idsRightA[minGiniIdxA]
		fmt.Printf("A feature selected: %s, split: %v\n", finalFeature, finalSplit)
	} else {
		finalFeature = featuresB[minGiniIdxB]
		finalSplit = splitsB[minGiniIdxB]

		idsLeft = idsLeftB[minGiniIdxB]
		idsRight = idsRightB[minGiniIdxB]
		fmt.Printf("B feature selected: %s, split: %v\n", finalFeature, finalSplit)
	}
	// 判断是否为连续数值
	if _, v := finalSplit.(float64); v {
		finalContinuous = true
	}

	// A或者B记录树节点，作为模型的一部分
	if isA {
		modelA[nodeKey] = &dtc_vertical.CTreeNode{
			SampleIDList: curNode.SampleIDList,
			FeatureName:  finalFeature,
			SplitValue:   finalSplit,
			Continuous:   finalContinuous,
			Depth:        curNode.Depth,
			Index:        curNode.Index,
		}
	} else {
		modelB[nodeKey].FeatureName = finalFeature
		modelB[nodeKey].SplitValue = finalSplit
		modelB[nodeKey].Continuous = finalContinuous
	}

	// ----------------------- 处理子节点 -----------------------
	// B 计算左右节点的Gini
	giniLeft := xcc.DtcVLCalGiniByTagPart(allDataB, idsLeft, nil, label)
	giniRight := xcc.DtcVLCalGiniByTagPart(allDataB, idsRight, nil, label)

	nodeL := &dtc_vertical.CTreeNode{
		SampleIDList: idsLeft,
		Depth:        curNode.Depth + 1,
		Index:        curNode.Index * 2,
		Gini:         giniLeft,
	}
	nodeKeyL := xcc.DtcVLNodeKeyByDepthIndex(nodeL.Depth, nodeL.Index)
	modelB[nodeKeyL] = nodeL
	// 判断左节点是否为叶子节点
	if v, resultL := xcc.DtcVLDecideTerminateTagPart(allDataB, nodeL.SampleIDList, curNode.SampleIDList, nodeL.Gini, curNode.Gini, nodeL.Depth, label, stopCond); v {
		modelB[nodeKeyL].Result = resultL
	} else {
		determineNodeByDepthIndex(allDataA, allDataB, nodeL, label, regParam, contFeatures, stopCond)
	}

	nodeR := &dtc_vertical.CTreeNode{
		SampleIDList: idsRight,
		Depth:        curNode.Depth + 1,
		Index:        curNode.Index*2 + 1,
		Gini:         giniRight,
	}
	nodeKeyR := xcc.DtcVLNodeKeyByDepthIndex(nodeR.Depth, nodeR.Index)
	modelB[nodeKeyR] = nodeR
	// 判断右节点是否为叶子节点
	if v, resultR := xcc.DtcVLDecideTerminateTagPart(allDataB, nodeR.SampleIDList, curNode.SampleIDList, nodeR.Gini, curNode.Gini, nodeR.Depth, label, stopCond); v {
		modelB[nodeKeyR].Result = resultR
	} else {
		determineNodeByDepthIndex(allDataA, allDataB, nodeR, label, regParam, contFeatures, stopCond)
	}
}

// B 剪枝
func pruneTreeB(allDataB *ml_common.DTDataSet, regParam float64, label string) {
	prunedB := false
	for id, node := range modelB {
		// 跳过叶子节点
		if len(node.Result) != 0 {
			continue
		}

		leftNodeID := xcc.DtcVLNodeKeyByDepthIndex(node.Depth+1, node.Index*2)
		rightNodeID := xcc.DtcVLNodeKeyByDepthIndex(node.Depth+1, node.Index*2+1)

		// 判断子节点是否都在B，由于A不存叶子节点，若有子节点不在B，则该节点不是目标节点
		// 若子节点不是叶子节点，则该节点也不是目标节点
		if leftNode, v := modelB[leftNodeID]; !v || len(leftNode.Result) == 0 {
			continue
		}
		if rightNode, v := modelB[rightNodeID]; !v || len(rightNode.Result) == 0 {
			continue
		}

		// 判断是否要剪枝，若是，删除左右子节点，将该节点标记为叶子节点
		if xcc.DtcVLIfPruneNodeByTagPart(allDataB, modelB, label, node.Depth, node.Index, regParam) {
			prunedB = true

			delete(modelB, leftNodeID)
			delete(modelB, rightNodeID)

			// TODO 网络交互，如果节点的特征和分割值存在A方，则从模型A删除该节点
			if modelB[id].SplitValue == nil {
				delete(modelA, id)
			}

			t := xcc.DtcVLGetMaxLabelTypeTagPart(allDataB, node.SampleIDList, label)
			modelB[id].FeatureName = ""
			modelB[id].SplitValue = nil
			modelB[id].Result = t
		}
	}
	if prunedB {
		pruneTreeB(allDataB, regParam, label)
	}
	return
}

// A 预测
func predictA(depth, index int, dataA map[string]string, dataB map[string]string) string {
	result, depth, index, _ := xcc.DtcVLPredictionStep(modelA, dataA, depth, index)
	if len(result) != 0 {
		return result
	}
	// TODO 网络交互，通知B继续预测
	return predictB(depth, index, dataA, dataB)
}

// B 预测
func predictB(depth, index int, dataA map[string]string, dataB map[string]string) string {
	result, depth, index, _ := xcc.DtcVLPredictionStep(modelB, dataB, depth, index)
	if len(result) != 0 {
		return result
	}
	// TODO 网络交互，通知A继续预测
	return predictA(depth, index, dataA, dataB)
}

// 从 csv 文件中读取样本特征
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

// 打印 A 和 B 的模型
func printModels() {
	var nodesA []*dtc_vertical.CTreeNode
	var nodesB []*dtc_vertical.CTreeNode
	for _, v := range modelA {
		v.SampleIDList = nil
		nodesA = append(nodesA, v)
	}
	for _, v := range modelB {
		v.SampleIDList = nil
		nodesB = append(nodesB, v)
	}

	a, _ := json.Marshal(nodesA)
	b, _ := json.Marshal(nodesB)

	fmt.Printf("modelA: %s\n\n", a)
	fmt.Printf("modelB: %s\n", b)
}

// 将每个样本转化为 map[featureName] -> value
func dataFeaturesToDataMaps(data *ml_common.DTDataSet) []map[string]string {
	dataMap := make([]map[string]string, len(data.Features[0].Sets))
	for _, feature := range data.Features {
		for id, value := range feature.Sets {
			if dataMap[id] == nil {
				dataMap[id] = make(map[string]string)
			}
			dataMap[id][feature.FeatureName] = value
		}
	}
	return dataMap
}
