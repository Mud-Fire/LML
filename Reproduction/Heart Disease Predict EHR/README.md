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

&emsp;&emsp;相关系数矩阵：
![data_corr](./doc/data_corr.png)

- 2、各特征值频率直方图

        dataset.hist()
        plt.show()
        
&emsp;&emsp;频率直方图：
![data_hist](./doc/data_hist.png)

- 3、目标类别分类，以及各类的频率分布

        rcParams['figure.figsize'] = 7,6
        plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'green'])
        plt.xticks([0, 1])
        plt.xlabel('Target Classes')
        plt.ylabel('Count')
        plt.title('Count of each Target Class')
        plt.show()
        
&emsp;&emsp;类别分类频率图:

![data_target](./doc/data_target.png)

**********************************
## 数据处理
使用
        
        dataset = pd.get_dummies(dataset, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

对各属性值进行一次编码：
        
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 303 entries, 0 to 302
        Data columns (total 31 columns):
         #   Column     Non-Null Count  Dtype  
        ---  ------     --------------  -----  
         0   age        303 non-null    int64  
         1   trestbps   303 non-null    int64  
         2   chol       303 non-null    int64  
         3   thalach    303 non-null    int64  
         4   oldpeak    303 non-null    float64
         5   target     303 non-null    int64  
         6   sex_0      303 non-null    uint8  
         7   sex_1      303 non-null    uint8  
         8   cp_0       303 non-null    uint8  
         9   cp_1       303 non-null    uint8  
         10  cp_2       303 non-null    uint8  
         11  cp_3       303 non-null    uint8  
         12  fbs_0      303 non-null    uint8  
         13  fbs_1      303 non-null    uint8  
         14  restecg_0  303 non-null    uint8  
         15  restecg_1  303 non-null    uint8  
         16  restecg_2  303 non-null    uint8  
         17  exang_0    303 non-null    uint8  
         18  exang_1    303 non-null    uint8  
         19  slope_0    303 non-null    uint8  
         20  slope_1    303 non-null    uint8  
         21  slope_2    303 non-null    uint8  
         22  ca_0       303 non-null    uint8  
         23  ca_1       303 non-null    uint8  
         24  ca_2       303 non-null    uint8  
         25  ca_3       303 non-null    uint8  
         26  ca_4       303 non-null    uint8  
         27  thal_0     303 non-null    uint8  
         28  thal_1     303 non-null    uint8  
         29  thal_2     303 non-null    uint8  
         30  thal_3     303 non-null    uint8  
        dtypes: float64(1), int64(5), uint8(25)
        memory usage: 21.7 KB
