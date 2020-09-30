from __future__ import print_function

import pickle
import numpy as np

with open('../data/inputdata_DL.pkl', 'rb') as f: c3 = pickle.load(f)
x_full = c3[0]
y_full = c3[1]

y_head = y_full[0:100]
print("===============================================================")
print(len(x_full[0]))
print(x_full[0])
print(y_full[0:10])
print(y_head)

print(np.newaxis)

print(y_head[:, 0, np.newaxis])
print(y_head[:, 0])
print(y_head[:, 1])
del c3

print(np.arange(10))

# with open('../data/inputdata_conv.pkl', 'rb') as f: conv = pickle.load(f)
#
# x_full = conv[0]
# y_full = conv[1]
# print("===============================================================")
# print(len(x_full[:, 0]))
# print(x_full[0])
# print(y_full[0:10])
# del conv


# ===============================================================================================================
# 测试 cross_validated() 函数方法
# import optunity as opt
#
#
# def train(x, y, filler=''):
#     print(filler + 'Training data:')
#     for instance, label in zip(x, y):
#         print(filler + str(instance) + ' ' + str(label))
#
#
# def predict(x, filler=''):
#     print(filler + 'Testing data:')
#     for instance in x:
#         print(filler + str(instance))
#
#
# data = list(range(9))
# labels = [0] * 9
# print(data)
# print(labels)
#
#
# @opt.cross_validated(x=data, y=labels, num_folds=4)
# def cved(x_train, y_train, x_test, y_test):
#     train(x_train, y_train)
#     predict(x_test)
#     return 0.0
#
#
# cved()
# ===============================================================================================================