# Datapreprocessing

## Step 1: Import Libraries
```python  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  #for drawing  
```

## Step 2: Import Datasets
```python
dataset = pd.read_csv('Data.csv')
#df.iloc[ : , : ],first parameter selects rows and second parameter selects columns; 
X = dataset.iloc[ : , :-1].values   
Y = dataset.iloc[ : , 3].values
```

## Step 3: Fill Missing Datas
```python
from sklearn.preprocessing import Imputer
#strategy = mean, median, most_frequent; 
#axis = 0 or 1(Check the condition); ----------------------------------------------------------
imputer = Imputer(missing values = "NaN", strategy = "mean", axis = 0)
'''
Fit, Transform, Fit_transform;
Fit(): Method calculates the parameters μ and σ and saves them as internal objects. //求訓練集的均值,方差, .....為一個訓練過程
Transform(): Method using these calculated parameters apply the transformation to a particular dataset. //在fit的基礎上進行降維,標準化....
Fit_transform(): joins the fit() and transform() method for transformation of dataset. //包括了訓練又包含了轉換
必須先用fit_transform(trainData)，之後再transform(testData)
如果直接transform(testData)，程式會報錯
如果fit_transfrom(trainData)後，使用fit_transform(testData)而不transform(testData)，雖然也能歸一化，但是兩個結果不是在同一個“標準”下的，具有明顯差異。(一定要避免這種情況)
'''
imputer = imputer.fit(X[ : ,1:3])
X[ : ,1:3] = imputer.transform(X[ : ,1:3])
```

## Step 4: Encoding Categorical Data
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
'''
OneHotEncoder 無法對string編碼
#Sol1: Use LabelEncoder First;
#Sol2: Use get_dummies;
'''
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
#LabelEncoder轉出來的數字會有大小差別,所以如果要把數字大小解決就需要OneHotEncoder
```

## Step 5: Splitting Datasets
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

## Step 6: Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

## Reference
https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%201_Data%20PreProcessing.md 
