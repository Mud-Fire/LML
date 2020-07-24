import numpy as np
import matplotlib.pyplot as plt


class LinearDiscriminant(object):

    def __init__(self, seed=None):
        self.seed = seed

    def mean_u(self, x):
        return np.mean(x, axis=0)

    # 计算某一类内散度矩阵Si
    # Sw = ΣSi
    # 一般用于二分类所以是 Sw = S0 + S1
    # 书里的样本属性是列向量公式是Σ(x - u)(x - u)T
    # 在编程中，很多样本的属性以行向量存储，所以这里是Σ(x - u)T(x - u)
    def Si(self, x):
        u = self.mean_u(x)
        ui = x - u
        si = np.zeros((len(u), len(u)))
        for i in range(len(ui[:, 0])):
            si = si + ui[i,].T * ui[i,]
        return si

    # 计算类间散度矩阵Sb
    def Sb(self, x0, x1):
        u0 = self.mean_u(x0)
        u1 = self.mean_u(x1)
        u = self.mean_u(np.vstack((x0, x1)))
        return (u0 - u1).T * (u0 - u1)

    def LDA(self, x0, x1):
        sw = self.Si(x0) + self.Si(x1)
        sb = self.Sb(x0, x1)
        # print(sw)
        eigenvalue, eignvector = np.linalg.eig(sw.I * sb)
        # 这里看一下使用np.lilnalg.eig()方法后的特征向量是横的还是竖的存储的
        # 特征向量是对应得列向量
        # print("sw.I * sb\t:\n", sw.I * sb)
        # print("eigenvalue\t:\n", eigenvalue)
        # print("eignvector\t:\n", eignvector)
        # print(sw.I * sb * eignvector[:, eignvector.argmax()-1 ])
        # print(eigenvalue[eignvector.argmax()-1, ].T * eignvector[: ,eignvector.argmax()-1])

        return eignvector[:, eignvector.argmax() - 1]


if __name__ == '__main__':
    p = r'..\..\Datasets\waterMelon30a.csv'
    with open(p, encoding='utf-8') as f:
        data = np.loadtxt(f, str, delimiter=",")

    dataset = data[1:, 1:3]
    labelset = data[1:, 3]

    Label_Point = 8
    # 讲数据集按label类型分成两部分
    x0 = np.mat(dataset[0:Label_Point, ]).astype(float)
    x1 = np.mat(dataset[Label_Point:, ]).astype(float)

    wmLDA = LinearDiscriminant()

    w = wmLDA.LDA(x0, x1)
    print(w)

    print(np.shape(x0[:, 0]), np.shape(x0[:, 1]))
    plt.plot(x0[:, 0], x0[:, 1], 'bo')
    plt.plot(x1[:, 0], x1[:, 1], 'ro')
    plt.plot((-1 * w[0], 0), (-1 * w[1], 0), c='red')

    # print(w.I)
    # y = np.multiply(x0[0, :], w.I)

    plt.show()
