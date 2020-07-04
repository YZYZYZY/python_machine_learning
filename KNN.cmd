### KNN是唯一一个不需要模型的算法，训练数据集本身就是模型，感受一下skicit_learn中的模型封装，其实就是实现的一个类，类里面根据相应的算法需要实现一个个函数，但是有些函数是都有的，比如说fit,predict,score等。KNN也就是根据欧式距离来进行分类。


```python
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier(n_neighbors=6)
```


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
```

    D:\Anaconda\lib\importlib\_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    


```python
iris = datasets.load_iris()
```


```python
X = iris.data
y = iris.target
```


```python
X.shape
```




    (150, 4)




```python
y.shape
```




    (150,)



### train_test_split


```python
y
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])




```python
shuffle_indexes = np.random.permutation(len(X))
```


```python
shuffle_indexes
```




    array([100, 138,  81,  84,  23,  71,  24,  93,  25,  77,  57,  89, 142,
           110, 123, 101,  82,  72,  73,  80,  32,  15,  61, 146, 145,  34,
            37, 120, 105, 104,  52, 109, 117, 108,   6,  75,  55,  11, 106,
            45,  43,  69,  54, 129,   5, 122,  38,  85,   4,  16,  98,  22,
           103,  18,  66,  94,  76,  33, 133, 134, 140,  87,  63, 115,   9,
            91,  70,  78, 128, 131, 127,  90,  68,  39,  51, 139,  27,  67,
            56,  19,   8,  35,  14, 135,  53,   1,  60,  95, 119,  92, 125,
            83,  96,  58,  12,  74,  50,  62,  47, 121,  29,  36,   7,  64,
            44,  97,  10, 124, 147,   0, 118, 144,  88,  42, 112,   3,  21,
            49,  48, 114, 130,  30, 107,  59,  31,  20, 126,  65, 141,  17,
            41, 143,  28, 111,  40,   2, 116,  26, 149,  79,  86,  13, 102,
            46, 148, 137, 136, 132, 113,  99])




```python
test_radio = 0.2
test_size = int(len(x) * test_radio) 
```


```python
test_size
```




    30




```python
test_indexes = shuffle_indexes[:test_size]
train_indexes = shuffle_indexes[test_size:]
```


```python
X_train = X[train_indexes]
y_train = y[train_indexes]

X_test = X[train_indexes]
y_test = y[train_indexes]
```


```python
print(X_train.shape)
print(y_train.shape)
```

    (120, 4)
    (120,)
    


```python
print(X_test.shape)
print(y_test.shape)
```

    (120, 4)
    (120,)
    

### 使用sklearn自带的train_test_split


```python
from sklearn.model_selection import train_test_split
```


```python
X_train,y_train,X_test,y_test = train_test_split(X,y,test_size = 0.2)
```


```python
print(X_train.shape)
print(y_train.shape)
```

    (120, 4)
    (30, 4)
    


```python
print(X_test.shape)
print(y_test.shape)
```

    (120,)
    (30,)
    
