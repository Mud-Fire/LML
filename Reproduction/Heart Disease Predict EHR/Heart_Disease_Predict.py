import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 查看数据基本信息
dataset = pd.read_csv("dataset.csv")
dataset.info()
dataset.describe(include="all").to_csv("./doc/data_describe.csv")

# 对数据进行比较和可视化展示
# 相关系数矩阵
rcParams['figure.figsize'] = 20, 14
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()
# plt.show()
# 频率直方图
dataset.hist()
# plt.show()
# 查看分类类别以及各类的频率分布
rcParams['figure.figsize'] = 8, 6
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color=['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
# plt.show()

# 处理数据
# 使用get_dummies 对特征属性值进行重新编码
dataset = pd.get_dummies(dataset, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
dataset.info()
dataset.describe(include="all").to_csv("./doc/data_dummies.csv")
