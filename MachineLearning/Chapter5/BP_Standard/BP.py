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

    m, n = np.shape(X_train)
    d = n
    l = 1
    q = d + 1
    lr = 0.01
    maxTrain = 500

    theta = np.random.rand()
    gamma = np.random.randn(q)
    v = np.random.rand(n, q)
    w = np.random.rand(q, m)

    for _ in range(maxTrain):
        alpha = X_train.dot(v)
        b = sigmoid(alpha + gamma)
        belta = np.dot(b, w)
        y_ = np.squeeze(sigmoid(belta + theta))
        E = np.dot((y_ - y_train), (y_ - y_train))

        g1 = np.multiply(y_,(1-y_))
        g = np.multiply(g1,(y_train-y_))
        print(np.shape(g))
        e1 = np.multiply(b, (1 - b))
        print(np.shape(e1))
        e2 = np.dot(w, g)
        print(np.shape(e2))
        e = np.multiply(e1,e2.T)

        w += lr * np.multiply(g, b)
        print(w)


        stop()
