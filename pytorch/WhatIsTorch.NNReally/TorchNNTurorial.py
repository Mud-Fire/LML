from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np

# 准备数据
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

# 获取数据
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# 查看数据
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)
