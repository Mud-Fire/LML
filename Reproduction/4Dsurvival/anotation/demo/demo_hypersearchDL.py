import optunity
import lifelines
from lifelines.utils import concordance_index
import os, sys, pickle
import numpy as np

sys.path.insert(0, '../code')
from trainDL import *

with open('../data/inputdata_DL.pkl', 'rb') as f: c3 = pickle.load(f)

# 1000 * 11514
# 11514 <- (3 x 19 x 202) 一个心脏中'202'个顶点在'20'个时间片段内的'3'维坐标变化
x_full = c3[0]

# 1000 * 2
# 2 <- status & survival time
y_full = c3[1]
del c3

# 寻找学习模型的超参数

# x_data : 输入数据，网格点运动描述
# y_data : 输出数据，生存状态和生存时间
# method : 超参数搜索方法(optunity参数),"particle swarm"<-PSO
# nfolds : 交叉验证折数
# nevals : 超参数搜索的数量
# lrexp_range  : 学习率搜索范围，log10
# l1rexp_range : L1正则化值范围，log10
# dro_range    : drop out 范围
# units1_range : 编码器隐藏层 单元 数量
# units2_range : 编码器 中间层 单元数量
# alpha_range  : 公式（7） 重建损失和生存损失权重
# batch_size   :
# num_epochs   :

def hypersearch_DL(x_data, y_data, method, nfolds, nevals, lrexp_range, l1rexp_range, dro_range, 
                   units1_range, units2_range, alpha_range, batch_size, num_epochs):

    @optunity.cross_validated(x=x_data, y=y_data, num_folds=nfolds)
    def modelrun(x_train, y_train, x_test, y_test, lrexp, l1rexp, dro, units1, units2, alpha):
        cv_log = DL_single_run(xtr=x_train, ytr=y_train, units1=units1, units2=units2, dro=dro, lr=10**lrexp, 
                               l1r=10**l1rexp, alpha=alpha, batchsize=batch_size, numepochs=num_epochs)
        cv_preds = cv_log.model.predict(x_test, batch_size=1)[1]
        cv_C = concordance_index(y_test[:,1], -cv_preds, y_test[:,0])
        return cv_C

    optimal_pars, searchlog, _ = optunity.maximize(modelrun, num_evals=nevals, solver_name=method, lrexp=lrexp_range, 
                                                   l1rexp=l1rexp_range, dro=dro_range, units1=units1_range, 
                                                   units2=units2_range, alpha=alpha_range)

    print('Optimal hyperparameters : ' + str(optimal_pars))
    print('Cross-validated C after tuning: %1.3f' % searchlog.optimum)
    return optimal_pars, searchlog
    
    
opars, clog = hypersearch_DL(x_data=x_full, y_data=y_full, 
                             method='particle swarm', nfolds=6, nevals=50,
                             lrexp_range=[-6.,-4.5], l1rexp_range=[-7,-4], dro_range=[.1,.9], 
                             units1_range=[75,250],  units2_range=[5,20],  alpha_range=[0.3, 0.7],
                             batch_size=16, num_epochs=100)
