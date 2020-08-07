# Heart Disease Predict by EHR dataset

EHR数据来源Kaggle [https://www.kaggle.com/ronitf/heart-disease-uci](https://www.kaggle.com/ronitf/heart-disease-uci)

**********************************
## 数据基本统计信息

    dataset.info()

查看dataset内容信息

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 303 entries, 0 to 302
    Data columns (total 14 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       303 non-null    int64  
     1   sex       303 non-null    int64  
     2   cp        303 non-null    int64  
     3   trestbps  303 non-null    int64  
     4   chol      303 non-null    int64  
     5   fbs       303 non-null    int64  
     6   restecg   303 non-null    int64  
     7   thalach   303 non-null    int64  
     8   exang     303 non-null    int64  
     9   oldpeak   303 non-null    float64
     10  slope     303 non-null    int64  
     11  ca        303 non-null    int64  
     12  thal      303 non-null    int64  
     13  target    303 non-null    int64  
    dtypes: float64(1), int64(13)
    memory usage: 33.3 KB

查看dataset的基本统计信息

    dataset.describe(include="all").to_csv("./doc/data_describe.csv")
    
统计结果：

![data_describe](./doc/data_describe.png)

**********************************
## 可视化的展现数据的特征

- 1、各特征值之间的相关系数矩阵

        rcParams['figure.figsize'] = 20, 14
        plt.matshow(dataset.corr())
        plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
        plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
        plt.colorbar()
        pylab.show()

相关系数矩阵：
![data_corr](./doc/data_corr.png)

    
