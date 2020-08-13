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

def predict(iX , v, w,theta, gamma):
    alpha = np.dot(iX, v)  # p101 line 2 from bottom, shape=m*q
    b = sigmoid(alpha - gamma)  # b=f(alpha-gamma), shape=m*q
    beta = np.dot(b, w)  # shape=(m*q)*(q*l)=m*l
    predictY = sigmoid(beta - theta)  # shape=m*l ,p102--5.3
    return predictY


# 程序入口
if __name__ == '__main__':

    p = r'..\..\Datasets\waterMelon3.0.csv'
    dataset = prepareData(p)
    dataset = dataset.replace(codeMap)

    y = dataset['好瓜']
    X = dataset.drop(['好瓜'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
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
    lr = 0.01
    maxTrain = 50

    # 声明 θ、γ、v、w、
    theta = np.random.rand(l)
    gamma = np.random.randn(q)
    v = np.random.rand(d, q)
    w = np.random.rand(q, l)

    # 训练次数
    for _ in range(maxTrain):
        # 标准BP按照每个样本运算一次，即更新各参数
        for i in range(m):
            alpha = X_train[i].dot(v)
            b = sigmoid(alpha + gamma)
            belta = np.dot(b, w)
            y_ = np.squeeze(sigmoid(belta + theta))

            # 均方差
            E = 0.5 * np.dot((y_ - y_train[i]), (y_ - y_train[i]))
            # print(E)

            g = y_ * (1 - y_) * (y_train[i] - y_)
            # print("++++++++++")
            e1 = np.multiply(b, 1 - b)
            e2 = np.dot(w, g)
            e = np.multiply(e1, np.squeeze(e2))
            # print("=======")
            w = w + lr * np.dot(b.reshape((q, 1)), g.reshape((1, l)))
            theta = theta - lr * g
            v = v + lr * np.dot(X_train[i].reshape(d, 1), e.reshape((1, q)))
            gamma = gamma - lr * e

    print(predict(X,v,w,theta,gamma))