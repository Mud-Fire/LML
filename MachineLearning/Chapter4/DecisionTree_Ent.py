import numpy as np


# 准备数据，表里数据是按照课本样式存储的，这里进行一次预处理和编码操作
def prepareData():
    p = r'..\Datasets\waterMelon2.0.csv'
    with open(p, encoding='gbk') as f:
        data = np.loadtxt(f, str, delimiter=",")

    label_raw = data[0, 1:-1]
    dataset_raw = data[1:, 1:-1]
    return dataset_raw, label_raw


# 讲字符形式数据转换为数值型
def encodeData(dataset_raw, label_raw):
    for i in range(len(dataset_raw[0, :])):
        print(set(dataset_raw[:, i]))

    return dataset


if __name__ == '__main__':
    dataset_raw, label_raw = prepareData()
    print(dataset_raw)
    encodeData(dataset_raw, label_raw)
