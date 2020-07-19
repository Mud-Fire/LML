import numpy as np
import csv


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

