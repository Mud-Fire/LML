import numpy as np
import csv


class LogisticRegression(object):

    def __init__(self, learning_rate=0.1, max_iter=100, seed=None):
        self.seed = seed
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, x, y):
        np.random.seed(self.seed)
        self.w = np.random.normal(loc=0.0, scale=1.0, size=x.shape[1])
        self.b = np.random.normal(loc=0.0, scale=1.0)
        self.x = x
        self.y = y
        for i in range(self.max_iter):
            self._update_step()

    # ——————————————————————————————————————————————
    # 线性函数与几率函数的计算步骤
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # 线性函数在z = x·w+b
    # 对结果 z 取对率使用私有方法 _sigmoid()
    def _f(self, x, w, b):
        z = x.dot(w) + b
        return self._sigmoid(z)
    # ———————————————————————————————————————————————

    # predict_proba() 输出的是对率函数的值
    def predict_proba(self, x=None):
        if x is None:
            x = self.x
        y_pred = self._f(x, self.w, self.b)
        return y_pred

    # 与 prefict_proba() 不同，prefict()方法输出的是将对率结果以0.5 做截断点后的二分类结果
    def predict(self, x=None):
        if x is None:
            x = self.x
        y_pred_proba = self._f(x, self.w, self.b)
        y_pred = np.array([0 if y_pred_proba[i] < 0.5 else 1 for i in range(len(y_pred_proba))])
        return y_pred

    # 计算准确率，将输入的y 和 计算的得到y_pred 进行比较
    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred = self.predict()
        acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
        return acc

    #
    def loss(self, y_true=None, y_pred_proba=None):
        if y_true is None or y_pred_proba is None:
            y_true = self.y
            y_pred_proba = self.predict_proba()
        return np.mean(-1.0 * (y_true * np.log(y_pred_proba) + (1.0 - y_true) * np.log(1.0 - y_pred_proba)))

    def _calc_gradient(self):
        y_pred = self.predict()
        d_w = (y_pred - self.y).dot(self.x) / len(self.y)
        d_b = np.mean(y_pred - self.y)
        return d_w, d_b

    # 更新w 和 b
    def _update_step(self):
        d_w, d_b = self._calc_gradient()
        self.w = self.w - self.lr * d_w
        self.b = self.b - self.lr * d_b
        return self.w, self.b

if __name__ == '__main__':
    p = r'..\..\Datasets\waterMelon30a.csv'
    with open(p, encoding='utf-8') as f:
        data = np.loadtxt(f, str, delimiter=",")

    dataset = data[1:, 1:3]
    labelset = data[1:, 3]

    # 将数据集分成训练部分和测试部分
    tarinData = np.vstack((dataset[0:5, ], dataset[12:, ]))
    testData = dataset[5:12, ]
    tarinLabel = np.vstack((labelset[0:5, ], labelset[12:, ]))
    testLabel = labelset[5:12, ]
