import numpy as np
import csv

p = r'..\..\Datasets\waterMelon30a.csv'
with open(p, encoding='utf-8') as f:
    data = np.loadtxt(f, str, delimiter=",")
    print(data[:5])
