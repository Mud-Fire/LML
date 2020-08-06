import numpy as np


class BP(object):

    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def predict(self, input_vec):
        return self.activator(np.dot(input_vec, self.weights) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        # 每一次训练都是一次预测，并不断修正weight和bias
        for i in range(len(labels)):
            output = self.predict(input_vecs[i])
            self._update_weights(input_vecs[i], output, labels[i], rate)

    def _update_weights(self, input_vecs, output, labels, rate):
        delta = labels - output
        self.weights = self.weights + np.dot(rate * delta, input_vecs)
        self.bias += rate * delta


def f(x):
    return 1 if x > 0 else 0


def get_train_dataset():
    # 设置训练数据
    # 输入数据为[[1,1],[0,0],[1,0],[0,1]]
    input_vecs = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    # 对应标准输出数据为[1,0,0,0]
    labels = np.array([1, 0, 0, 0])
    return input_vecs, labels


def train_and_preceptron():
    # 感知机类的实体化为p
    p = Perceptron(2, f)
    # 获取设置的训练数据
    input_vecs, labels = get_train_dataset()
    # 对训练数据进行十次训练，设置每次调整比例为0.1
    p.train(input_vecs, labels, 10, 0.1)
    return p


# 程序入口
if __name__ == '__main__':
    # 获得程序的实体化操作后的对象
    perception = train_and_preceptron()
    print(perception.weights)
    print(perception.bias)
    # 测试
    print('1 and 1 = %d' % perception.predict([1, 1]))
    print('0 and 0 = %d' % perception.predict([0, 0]))
    print('1 and 0 = %d' % perception.predict([1, 0]))
    print('0 and 1 = %d' % perception.predict([0, 1]))