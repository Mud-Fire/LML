import numpy as np
import pandas as pd


# 准备数据，表里数据是按照课本样式存储的，这里进行一次预处理和编码操作
def prepareData():
    p = r'..\..\Datasets\waterMelon2.0_train.csv'
    rawdata = pd.read_csv(p, encoding="gbk")

    return rawdata


class DecisionTree:
    def __init__(self):
        self.model = None

    def fit(self, xTrain, yTrain=pd.Series()):
        if yTrain.size == 0:  # 如果不传，自动选择最后一列作为分类标签
            yTrain = xTrain.iloc[:, -1]
            xTrain = xTrain.iloc[:, :len(xTrain.columns) - 1]
        self.model = self.buildDecisionTree(xTrain, yTrain)
        # return self.model
        return self.model

    def cal_Gini(self, y):
        valRate = y.value_counts().apply(lambda x: x / y.size)  # 频次汇总 得到各个特征对应的概率
        print(valRate)
        valGini = 1 - np.inner(valRate, valRate)
        print(valGini)
        return valGini

    def buildDecisionTree(self, xTrain, yTrain):
        # 属性节点名称
        propNamesAll = xTrain.columns
        # 好瓜坏瓜计算数量
        yTrainCounts = yTrain.value_counts()
        # 当分支的瓜仅剩一类时，递归结束，返回叶子结点结果
        if yTrainCounts.size == 1:
            return yTrainCounts.index[0]
        # 计算当前递归中，传入节点的瓜的pk
        # 计算基尼指数
        giniIndexD = self.cal_Gini(yTrain)
        # 声明最大收益
        # maxGain = None
        # 声明当前最大收益的属性节点名
        # maxEntropyPropName = None
        # 声明最小基尼指数属性名
        minGini = None  # 声明最小基尼指数属性
        minGiniProbName = None

        # 计算该节点下其他节点的信息增益
        for propName in propNamesAll:
            # 该节点下的属性
            propDatas = xTrain[propName]
            # 计算该属性下，各分型值得频率
            propClassSummary = propDatas.value_counts().apply(lambda x: x / propDatas.size)  # 频次汇总 得到各个特征对应的概率
            # 声明属性各分型值基尼指数之和
            sumGiniByProp = 0
            for propClass, dvRate in propClassSummary.items():
                # 该属性下某分型下的好坏瓜
                yDataByPropClass = yTrain[xTrain[propName] == propClass]
                # 计算该属性分型的基尼指数
                giniDv = self.cal_Gini(yDataByPropClass)
                # 计算该属性分型后的信息熵之和
                sumGiniByProp += giniDv * dvRate
            # 计算该节点对样本集划分后的信息增益
            # gainEach = entropyD - sumEntropyByProp
            # 选择该迭代过程中的最小基尼指数和属性节点
            if minGini == None or sumGiniByProp > minGini:
                minGini = sumGiniByProp
                minGiniProbName = propName
        # print('select prop:', maxEntropyPropName, maxGain)

        # =========================================================
        # 计算出最大增益节点后，以此节点开启新的迭代
        # 计算最大增益节点的频率
        propDatas = xTrain[minGiniProbName]
        propClassSummary = propDatas.value_counts()
        retClassByProp = {}

        for propClass in propClassSummary.items():
            # 提取按照最小基尼指数的各分型的index对dataset进行划分
            whichIndex = xTrain[minGiniProbName] == propClass[0]
            if whichIndex.size == 0:
                continue
            xDataByPropClass = xTrain[whichIndex]
            yDataByPropClass = yTrain[whichIndex]
            del xDataByPropClass[minGiniProbName]  # 删除已经选择的属性列
            # 以该节点向下递归
            retClassByProp[propClass] = self.buildDecisionTree(xDataByPropClass, yDataByPropClass)
        return {'Node': minGiniProbName, 'Edge': retClassByProp}


if __name__ == '__main__':
    dataset_raw = prepareData()
    print(dataset_raw)
    wm_DT = DecisionTree()
    print(wm_DT.fit(dataset_raw))
