# Linear Discriminant Analysis (LDA) 线性判别分析
- 书中习题3.5
- 用到的数据集是waterMelon30a.csv

这里的算法是针对二分类设计的，后续可能会更新多分类任务

线性判别方法利用投影技术，进行数据降维，找到类内散度比上类间散度(Sw/Sb)虽小的平面（直线L = wTx）方向

最后w的解为 Sw^(-1)Sb 的最大非零广义特征值对应得特征向量