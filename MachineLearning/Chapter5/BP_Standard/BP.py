import numpy as np
import pandas as pd


# class BP(object):


def prepareData(Path):
    rawdata = pd.read_csv(p, encoding="gbk")

    return rawdata


# 程序入口
if __name__ == '__main__':
    p = r'..\..\Datasets\waterMelon3.0.csv'
    dataset = prepareData(p)
    melon_color = pd.get_dummies(dataset['色泽'])
    print(melon_color)
    print(dataset)
