# -*- coding: utf-8 -*-
import numpy as np

# N 样本数量; D_in 输入属性值;
# H 隐藏层节点数; D_out 输出尺寸.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建一组随机输入输出数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

print(np.shape(x))
print(np.shape(y))

# 随机产生两层链接之间的w值
# w1 (1000, 1000)
# w2 (100 , 10)
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# testMaximum = np.random.randn(5,4)
# print(testMaximum)
# testMaximum_relu = np.maximum(testMaximum,0)
# print(testMaximum_relu)


# 学习率1xe-6
learning_rate = 1e-6

for t in range(500):
    # 向前传播: 计算w1，w2 预测出来的 label y
    # x 乘 w1 进入隐藏层
    h = x.dot(w1)

    # 留正值
    h_relu = np.maximum(h, 0)
    # h 乘 w2进入输出层
    y_pred = h_relu.dot(w2)

    # 计算损失函数值-平方差
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 后向传播 计算w1和w2梯度 降低损失函数的值

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)

    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 更新权值
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2