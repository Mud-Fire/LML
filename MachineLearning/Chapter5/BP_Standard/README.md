# Chapter 5 

### 西瓜书第五章课后习题
- 标准BP神经网络算法

### 数据集
- ..\..\Datasets\waterMelon3.0.csv
[waterMelon3.0](..\..\Datasets\)

        色泽	根蒂	敲声	纹理	脐部	触感	密度	含糖率	好瓜
        青绿	蜷缩	浊响	清晰	凹陷	硬滑	0.697	0.46	是
        乌黑	蜷缩	沉闷	清晰	凹陷	硬滑	0.774	0.376	是
        乌黑	蜷缩	浊响	清晰	凹陷	硬滑	0.634	0.264	是
        青绿	蜷缩	沉闷	清晰	凹陷	硬滑	0.608	0.318	是
        浅白	蜷缩	浊响	清晰	凹陷	硬滑	0.556	0.215	是
        青绿	稍蜷	浊响	清晰	稍凹	软粘	0.403	0.237	是
        乌黑	稍蜷	浊响	稍糊	稍凹	软粘	0.481	0.149	是
        乌黑	稍蜷	浊响	清晰	稍凹	硬滑	0.437	0.211	是
        乌黑	稍蜷	沉闷	稍糊	稍凹	硬滑	0.666	0.091	否
        青绿	硬挺	清脆	清晰	平坦	软粘	0.243	0.267	否
        浅白	硬挺	清脆	模糊	平坦	硬滑	0.245	0.057	否
        浅白	蜷缩	浊响	模糊	平坦	软粘	0.343	0.099	否
        青绿	稍蜷	浊响	稍糊	凹陷	硬滑	0.639	0.161	否
        浅白	稍蜷	沉闷	稍糊	凹陷	硬滑	0.657	0.198	否
        乌黑	稍蜷	浊响	清晰	稍凹	软粘	0.36	0.37	否
        浅白	蜷缩	浊响	模糊	平坦	硬滑	0.593	0.042	否
        青绿	蜷缩	沉闷	稍糊	稍凹	硬滑	0.719	0.103	否

### 主要逻辑代码：

- 使用sklearn包中的train_test_spit()方法划分了训练集和测试集，一行代码就可以

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

- 按照书上的符号进行参数声明

        # 声明 θ、γ、v、w、
        theta = np.random.rand(l)
        gamma = np.random.randn(q)
        v = np.random.rand(d, q)
        w = np.random.rand(q, l)

- 因为是标准bp，所以每个样本进行运算时，均要更新一次参数，这里设置了两层循环，一层为训练次数，内层为单此训练的样本数量。

        for _ in range(maxTrain):
            # 标准BP按照每个样本运算一次，即更新各参数
            for i in range(m):
            
- 前向计算过程：

        alpha = X_train[i].dot(v)
        b = sigmoid(alpha - gamma)
        belta = np.dot(b, w)
        y_ = np.squeeze(sigmoid(belta - theta))
        
- 课本(5.4) -(5.15)公式提高的参数更新过程：

        E = 0.5 * np.dot((y_ - y_train[i]), (y_ - y_train[i])) #公式(5.5)
        g = y_ * (1 - y_) * (y_train[i] - y_) #公式(5.10)
        # print("++++++++++")
        e1 = np.multiply(b, 1 - b)
        e2 = np.dot(w, g)
        e = np.multiply(e1, np.squeeze(e2)) # 公式(5.15)
        # print("=======")
        w = w + lr * np.dot(b.reshape((q, 1)), g.reshape((1, l))) # 公式(5.11)
        theta = theta - lr * g # 公式(5.12)
        gamma = gamma - lr * e # 公式(5.14)
        v = v + lr * np.dot(X_train[i].reshape((d, 1)), e.reshape((1, q))) # 公式(5.13)

- 使用训练后的参数计算样本，返回模型预测值：

        predict(X_train, v, w, theta, gamma)
        predict(X_test, v, w, theta, gamma)