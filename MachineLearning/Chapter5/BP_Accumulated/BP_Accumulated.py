import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

codeMap = {}
codeMap['浅白'] = 0
codeMap['青绿'] = 0.5
codeMap['乌黑'] = 1
codeMap['蜷缩'] = 0
codeMap['稍蜷'] = 0.5
codeMap['硬挺'] = 1
codeMap['沉闷'] = 0
codeMap['浊响'] = 0.5
codeMap['清脆'] = 1
codeMap['模糊'] = 0
codeMap['稍糊'] = 0.5
codeMap['清晰'] = 1
codeMap['凹陷'] = 0
codeMap['稍凹'] = 0.5
codeMap['平坦'] = 1
codeMap['硬滑'] = 0
codeMap['软粘'] = 1
codeMap['否'] = 0
codeMap['是'] = 1


def prepareData(Path):
    rawdata = pd.read_csv(p, encoding="utf8")

    return rawdata


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_Acc(x):
    for i in range(len(x)):
        x[i] = sigmoid(x[i])
    return x


def predict(iX, v, w, theta, gamma):
    alpha = np.dot(iX, v)  # p101 line 2 from bottom, shape=m*q
    b = sigmoid_Acc(alpha - gamma)  # b=f(alpha-gamma), shape=m*q
    beta = np.dot(b, w)  # shape=(m*q)*(q*l)=m*l
    predictY = sigmoid_Acc(beta - theta.T)  # shape=m*l ,p102--5.3
    predictY[predictY >= 0.5] = 1
    predictY[predictY < 0.5] = 0
    predictY = predictY.astype("int")
    return predictY.T


# 程序入口
if __name__ == '__main__':

    p = r'..\..\Datasets\waterMelon3.0.csv'
    dataset = prepareData(p)
    dataset = dataset.replace(codeMap)

    y = dataset['好瓜']
    X = dataset.drop(['好瓜'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # 这里按照课本上的参数进行声明
    # m 为样本数量
    # d 为样本特征属性个数
    # l 为输出层节点个数
    # q 为隐藏层节点个数
    m, d = np.shape(X_train)
    # print(X_train)
    l = 1
    q = 5
    lr = 0.1
    maxTrain = 5000

    # 声明 θ、γ、v、w、
    theta = np.random.rand(l)
    gamma = np.random.randn(q)
    v = np.random.rand(d, q)
    w = np.random.rand(q, l)

    # 训练次数
    for _ in range(maxTrain):
        print(gamma.shape)
        # 标准BP按照每个样本运算一次，即更新各参数

        alpha = X_train.dot(v)
        b = sigmoid_Acc(alpha - gamma)
        belta = np.dot(b, w)
        # print(belta.shape)
        # print(theta.shape)
        # print((belta - theta).shape)
        y_ = np.squeeze(sigmoid_Acc(belta - theta.T))

        # 均方差
        E = 0.5 * np.dot((y_ - y_train), (y_ - y_train))

        g = y_ * (1 - y_) * (y_train - y_)
        g = g.reshape((1, len(g)))

        e1 = np.multiply(b, 1 - b)
        e2 = np.dot(w, g).T
        e = np.multiply(e1, np.squeeze(e2))
        print(e.shape)
        w = w + lr * np.dot(b.T, g.T)
        theta = theta - lr * g
        # print(X_train.shape)
        # print(e.shape)
        v = v + lr * np.dot(X_train.T, e)
        gamma = gamma - lr * e


    print("训练后：模型训练集预测结果\t:", np.squeeze(predict(X_train, v, w, theta, gamma)))
    print("训练集真实结果\t\t\t:", y_train)

    # print("训练后，模型测试集预测结果\t:", np.squeeze(predict(X_test, v, w, theta, gamma)))
    # print("测试集真实结果\t\t\t:", y_test)
