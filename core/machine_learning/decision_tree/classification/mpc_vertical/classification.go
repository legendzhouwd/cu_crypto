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

package classification

import (
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strconv"

	"github.com/legendzhouwd/cu_crypto/common/utils"
	"github.com/legendzhouwd/cu_crypto/core/hash"
	"github.com/legendzhouwd/cu_crypto/core/machine_learning/common"
)

// 基于二叉分类决策树的纵向联邦学习
/*
假设参与方A和B分别是非标签方和标签方，最终生成的决策树会有部分节点由A产生，部分节点由B产生
  				     --------
				    | part A |
				     --------
				      /   \
			 --------     --------
			| part B |   | part A |
			 --------     --------
			  /   \
        --------    --------
       | part A |  | part B |
        --------    --------
*/

// 决策树节点
type CTreeNode struct {
	SampleIDList []int       // 当前树节点的样本id列表
	FeatureName  string      // 当前节点使用的特征，若为叶子节点则该值为空
	Continuous   bool        // 该特征是否为连续值
	SplitValue   interface{} // 特征的分割值，可以是离散或连续值，用来分割样本，若为叶子节点则该值为空
	Result       string      // 该分支最终的决策值，若为非叶子节点该值为空（由标签方提供）
	Depth        int         // 节点所在的分支深度，Root深度为0
	Index        int         // 节点所在深度的节点位置，index < 2^depth
	Gini         float64     // 该节点的基尼指数（由标签方计算）
}

// 停止条件相关参数
type StopCondition struct {
	SampleThreshold int     // 节点样本数的阈值，节点的样本数小于该值则该节点标记为叶子节点
	DepthThreshold  int     // 节点深度的阈值，到达阈值则该节点标记为叶子节点
	GiniThreshold   float64 // 基尼指数震荡阈值，若节点基尼指数和父节点基尼指数振的差值小于该值，则该节点标记为叶子节点
}

// 为特征选择做准备，各方计算可能的特征选择和分割情况，用于后续计算Gini
// 返回特征名称列表、对应分割值列表、对应的左右子数据集id列表
// - dataset 全部样本
// - idList 当前样本id列表
// - label 目标特征
// - contFeatures 连续特征列表
func PrepForFeatureSelect(dataset *common.DTDataSet, idList []int, label string, contFeatures []string) ([]string, []interface{}, [][]int, [][]int, error) {
	curData := retrieveDatasetByIds(dataset, idList)

	var finalFeatures []string
	var finalSplitValues []interface{}
	var finalIDsLeft [][]int
	var finalIDsRight [][]int
	// 轮询所有特征
	for _, feature := range curData.Features {
		if feature.FeatureName == label {
			continue
		}
		// 找到特征的所有可能的分割值
		if utils.StringInSlice(feature.FeatureName, contFeatures) {
			values, err := getContAvailValues(curData, feature.FeatureName)
			if err != nil {
				return nil, nil, nil, nil, err
			}

			for i := 0; i < len(values)-1; i++ {
				// 取连续两个数值的均值作为分隔值
				splitValue := (values[i] + values[i+1]) / 2
				left, right, err := splitDataSetByFeature(curData, feature.FeatureName, splitValue, true)
				if err != nil {
					return nil, nil, nil, nil, err
				}

				finalFeatures = append(finalFeatures, feature.FeatureName)
				finalSplitValues = append(finalSplitValues, splitValue)
				finalIDsLeft = append(finalIDsLeft, left)
				finalIDsRight = append(finalIDsRight, right)
			}
		} else {
			values := getDiscAvailValues(curData, feature.FeatureName)
			for _, v := range values {
				left, right, _ := splitDataSetByFeature(curData, feature.FeatureName, v, false)

				finalFeatures = append(finalFeatures, feature.FeatureName)
				finalSplitValues = append(finalSplitValues, v)
				finalIDsLeft = append(finalIDsLeft, left)
				finalIDsRight = append(finalIDsRight, right)
			}
		}
	}
	return finalFeatures, finalSplitValues, finalIDsLeft, finalIDsRight, nil
}

// 利用特征值，将样本分割为两部分，左子节点对应 是/<=，右子节点对应 否/>
// 返回左右子树的样本id列表
// - dataset 训练样本集合
// - featureName 特征名称
// - splitValue 特征的分割值
// - isContinuous 特征是否为连续值
func splitDataSetByFeature(dataset *common.DTDataSet, featureName, splitValue interface{}, isContinuous bool) ([]int, []int, error) {
	var idsLeft []int
	var idsRight []int
	for _, feature := range dataset.Features {
		if feature.FeatureName == featureName {
			for id, v := range feature.Sets {
				if isContinuous {
					value, err := strconv.ParseFloat(v, 64)
					if err != nil {
						return nil, nil, fmt.Errorf("failed to parse string to float64, err: %v", err)
					}
					if value <= splitValue.(float64) {
						idsLeft = append(idsLeft, id)
					} else {
						idsRight = append(idsRight, id)
					}
				} else {
					if v == splitValue.(string) {
						idsLeft = append(idsLeft, id)
					} else {
						idsRight = append(idsRight, id)
					}
				}
			}
		}
	}

	return idsLeft, idsRight, nil
}

// 对于某个离散特征，获取训练样本中所有可能的取值
func getDiscAvailValues(dataset *common.DTDataSet, featureName string) []string {
	valueMap := make(map[string]bool)
	for _, feature := range dataset.Features {
		// 找到指定特征出现在样本中的所有取值
		if feature.FeatureName == featureName {
			for _, v := range feature.Sets {
				valueMap[v] = true
			}
		}
	}
	var values []string
	for key := range valueMap {
		values = append(values, key)
	}
	return values
}

// 对于某个连续特征，获取训练样本中所有可能的取值，并按照从小到大的顺序排列
func getContAvailValues(dataset *common.DTDataSet, featureName string) ([]float64, error) {
	valueMap := make(map[float64]bool)
	for _, feature := range dataset.Features {
		if feature.FeatureName == featureName {
			// 找到指定特征出现在样本中的所有取值
			for _, v := range feature.Sets {
				value, err := strconv.ParseFloat(v, 64)
				if err != nil {
					return nil, fmt.Errorf("failed to parse string to float64, err: %v", err)
				}
				valueMap[value] = true
			}
		}
	}
	var values []float64
	for key := range valueMap {
		values = append(values, key)
	}
	// 从小到大排序
	sort.Float64s(values)
	return values, nil
}

// 通过样本id列表，从所有样本中找到目标样本集合
func retrieveDatasetByIds(datasets *common.DTDataSet, idList []int) *common.DTDataSet {
	if len(idList) == 0 {
		return nil
	}

	var features []*common.DTDataFeature
	for _, feature := range datasets.Features {
		f := new(common.DTDataFeature)
		f.FeatureName = feature.FeatureName

		f.Sets = make(map[int]string)
		for k, v := range feature.Sets {
			if utils.IntInSlice(k, idList) {
				f.Sets[k] = v
			}
		}

		features = append(features, f)
	}

	return &common.DTDataSet{features}
}

// 标签方计算数据集分割后的的基尼指数
// - allDatasets 标签方的原始数据集，包含所有样本
// - idsLeft 左叶子样本id列表
// - idsRight 右叶子样本id列表
// - label 目标特征
func CalGiniByTagPart(allDatasets *common.DTDataSet, idsLeft, idsRight []int, label string) float64 {
	leftData := retrieveDatasetByIds(allDatasets, idsLeft)
	rightData := retrieveDatasetByIds(allDatasets, idsRight)
	return calMultiDataSetsGini([]*common.DTDataSet{leftData, rightData}, label)
}

// 计算节点指定标签的基尼指数，样本集合数可以为1
// 输入为[[D1][D2]] 或 [D], 计算样本子集的基尼系数
func calMultiDataSetsGini(datasets []*common.DTDataSet, label string) float64 {
	gini := 0.0
	// 特征集合总数
	dataSetsTotal := 0
	// 每个数据子集的基尼值
	dataSliceGinis := make(map[int]float64)
	// 每个数据子集的样本数
	dataSliceTotal := make(map[int]int)
	for index, dataSet := range datasets {
		if dataSet == nil {
			continue
		}
		for _, feature := range dataSet.Features {
			if feature.FeatureName != label {
				continue
			}
			// 计算每一个集合的基尼值, 即Gini(Di)
			dataSliceGini, dataSliceSetsNum := calSingleDataSetsGini(feature)
			// 计算特征集合总数, 即|D|
			dataSetsTotal += dataSliceSetsNum
			dataSliceGinis[index] = dataSliceGini
			dataSliceTotal[index] = dataSliceSetsNum
		}
	}
	// 计算基尼指数, 将 |Di|/|D| * Gini(Di) 求和
	for index, value := range dataSliceTotal {
		gini += (float64(value) / float64(dataSetsTotal)) * dataSliceGinis[index]
	}
	return gini
}

// 计算单个样本集基尼指数
func calSingleDataSetsGini(feature *common.DTDataFeature) (float64, int) {
	imp := 0.0
	// 计算指定特征值总数
	total := len(feature.Sets)
	// 计算该特征的分类总数
	counts := uniqueDataSetTypes(feature.Sets)
	for _, value := range counts {
		imp += math.Pow(float64(value)/float64(total), 2)
	}
	return 1 - imp, total
}

// 将输入的数据汇总(input dataSet)
// return Set{type1:type1Count,type2:type2Count ... typeN:typeNCount}
func uniqueDataSetTypes(sets map[int]string) map[string]int {
	results := make(map[string]int)
	for _, value := range sets {
		if _, ok := results[value]; !ok {
			results[value] = 0
		}
		results[value] += 1
	}
	return results
}

// 标签方：判断分支是否停止，若是，返回该分支最终结果
// - datasetsB 标签方的训练样本集合
// - curIdList 当前节点的样本id列表
// - fatherIdList 父节点的样本id列表
// - curGini 当前节点的Gini指数
// - fatherGini 父节点的Gini指数
// - curDepth 当前节点深度
// - label 目标特征
// - cond 分支停止条件
func DecideTerminateTagPart(datasetsB *common.DTDataSet, curIdList, fatherIdList []int, curGini, fatherGini float64,
	curDepth int, label string, cond StopCondition) (bool, string) {
	// 1.当前节点包含的样本集合为空, 当前节点标记为叶子节点，类别设置为其父节点所含样本最多的类别
	if len(curIdList) == 0 {
		// 类别设置为其父节点所含样本最多的类别
		fmt.Printf("==terminate== current feature is nil\n")
		return true, GetMaxLabelTypeTagPart(datasetsB, fatherIdList, label)
	}

	// 2.当前样本集属于同一类别，无需划分
	if labelType, isSame, _ := isSameTypePartB(datasetsB, curIdList, label); isSame {
		fmt.Printf("==terminate== current data has same type: %s\n", labelType)
		return true, labelType
	}

	// 3.判断 stopCondition...
	// 节点样本数小于阈值，则该节点标记为叶子节点, 类别设置为该节点所含样本最多的类别
	if len(curIdList) <= cond.SampleThreshold {
		fmt.Printf("==terminate== current data has fewer number than threshold: %d\n", len(curIdList))
		return true, GetMaxLabelTypeTagPart(datasetsB, curIdList, label)
	}
	// 节点深度小于阈值，则该节点标记为叶子节点, 类别设置为该节点所含样本最多的类别
	if cond.DepthThreshold != 0 && curDepth >= cond.DepthThreshold {
		fmt.Printf("==terminate== current node depth greater than threshold: %d\n", curDepth)
		return true, GetMaxLabelTypeTagPart(datasetsB, curIdList, label)
	}
	// 节点基尼指数小于阈值，则该节点标记为叶子节点, 类别设置为该节点所含样本最多的类别
	if curGini <= cond.GiniThreshold {
		fmt.Printf("==terminate== node gini smaller than threshold, current gini: %f\n", curGini)
		return true, GetMaxLabelTypeTagPart(datasetsB, curIdList, label)
	}

	return false, ""
}

// 标签方：判断目标样本集合是否属于同一类别
// 如果为true 返回指定labelType, 和 分类数 Set{type1:type1Count}
func isSameTypePartB(dataset *common.DTDataSet, idList []int, label string) (string, bool, map[string]int) {
	data := retrieveDatasetByIds(dataset, idList)
	return isSameType(data, label)
}

func isSameType(dataset *common.DTDataSet, label string) (string, bool, map[string]int) {
	result := make(map[string]int)
	for _, value := range dataset.Features {
		if value.FeatureName == label {
			result = uniqueDataSetTypes(value.Sets)
		}
	}
	if len(result) == 1 {
		labelType := ""
		for key := range result {
			labelType = key
		}
		return labelType, true, result
	}
	return "", false, result
}

// 标签方：获取指定样本集中取值最多的类, 如果最大值有多个，则随机选一个
// - dataSet 标签方的训练样本集合
// - idList 目标样本集合
func GetMaxLabelTypeTagPart(dataset *common.DTDataSet, idList []int, label string) string {
	data := retrieveDatasetByIds(dataset, idList)
	return getMaxLabelType(data, label)
}

func getMaxLabelType(dataset *common.DTDataSet, label string) string {
	labelType, isSame, labelCounts := isSameType(dataset, label)
	// 如果只有一个分类，直接返回该类别
	if isSame {
		return labelType
	}

	maxSetsNum := 0
	result := ""
	// 计算类别对应样本数最多的值, 如果最大值有多个，则label value取值最后一个对应的type
	for labelType, num := range labelCounts {
		if num >= maxSetsNum {
			maxSetsNum = num
			result = labelType
		}
	}
	return result
}

// 判断结束训练
// 对于二叉树，叶子节点个数 = 非叶子节点个数+1
func StopTrain(leafNum, nonLeafNum int) bool {
	return leafNum == nonLeafNum+1
}

// 标签方和非标签方都无法得到最终完整的模型，每一方只存储部分树节点
// 标签方和非标签方分别存储部分树节点的map：hash(hash(depth),index) -> CTreeNode
func NodeKeyByDepthIndex(depth, index int) string {
	hashDepth := hash.HashUsingSha256([]byte(strconv.FormatInt(int64(depth), 10)))
	key := hash.HashUsingSha256(append(hashDepth, []byte(strconv.FormatInt(int64(index), 10))...))
	return hex.EncodeToString(key)
}

// 标签方：计算指定节点的代价, cost = err_rate * len(samples)/len(all samples)
func calTreeNodeCostTagPart(dataset *common.DTDataSet, idList []int, label string) float64 {
	// 若该节点设置为叶子节点，找到该节点分类结果
	data := retrieveDatasetByIds(dataset, idList)
	t := getMaxLabelType(data, label)
	// 计算错误率
	rate := errRate(data, t, label)
	// 该节点样本数/全部样本数
	allSamplesNum := len(dataset.Features[0].Sets)
	return rate * float64(len(idList)) / float64(allSamplesNum)
}

func errRate(dataset *common.DTDataSet, result, label string) float64 {
	errNum := 0
	for _, feature := range dataset.Features {
		if feature.FeatureName == label {
			for _, v := range feature.Sets {
				if v != result {
					errNum++
				}
			}
		}
	}
	return float64(errNum) / float64(len(dataset.Features[0].Sets))
}

// 由标签方判断是否对目标节点剪枝
// - dataSet 标签方的训练样本
// - model 标签方的模型
// - label 目标特征
// - depth 目标节点的深度
// - index 目标节点所在深度的位置编号
// - regParam 泛化参数
func IfPruneNodeByTagPart(dataset *common.DTDataSet, model map[string]*CTreeNode, label string, depth, index int, regParam float64) bool {
	// 判断子节点是否都为叶子
	leftNodeID := NodeKeyByDepthIndex(depth+1, index*2)
	rightNodeID := NodeKeyByDepthIndex(depth+1, index*2+1)
	if len(model[leftNodeID].Result) == 0 || len(model[rightNodeID].Result) == 0 {
		return false
	}

	// 找到目标节点
	nodeID := NodeKeyByDepthIndex(depth, index)
	nodeCost := calTreeNodeCostTagPart(dataset, model[nodeID].SampleIDList, label)
	leftCost := calTreeNodeCostTagPart(dataset, model[leftNodeID].SampleIDList, label)
	rightCost := calTreeNodeCostTagPart(dataset, model[rightNodeID].SampleIDList, label)

	return nodeCost <= leftCost+rightCost+regParam
}

// 标签方和非标签方利用本地模型和样本进行预测，若在己方找到结果，则返回最终结果，否则返回当前寻址的depth、index，交给对方继续预测
func PredictionStep(model map[string]*CTreeNode, sample map[string]string, depth, index int) (string, int, int, error) {
	nodeID := NodeKeyByDepthIndex(depth, index)
	for {
		// 对于标签方B，可能存A方节点的Gini指数等信息，但分割值仍需请求A
		if node, v := model[nodeID]; v && (node.SplitValue != nil || len(node.Result) != 0) {
			if len(node.Result) != 0 {
				return node.Result, depth, index, nil
			}
			if !node.Continuous {
				if node.SplitValue == sample[node.FeatureName] {
					index = index * 2
				} else {
					index = index*2 + 1
				}
			}
			if node.Continuous {
				sampleValue, err := strconv.ParseFloat(sample[node.FeatureName], 64)
				if err != nil {
					return "", 0, 0, fmt.Errorf("failed to parse sample value with feature[%s], err: %v", node.FeatureName, err)
				}
				if sampleValue <= node.SplitValue.(float64) {
					index = index * 2
				} else {
					index = index*2 + 1
				}
			}
			return PredictionStep(model, sample, depth+1, index)
		} else {
			return "", depth, index, nil
		}
	}
}
