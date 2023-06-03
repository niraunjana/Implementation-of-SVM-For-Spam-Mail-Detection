# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary packages.

2.Read the given csv file and display the few contents of the data.

3.Assign the features for x and y respectively.

4.Split the x and y sets into train and test sets.

5.Convert the Alphabetical data to numeric using CountVectorizer.

6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7.Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: NIRAUNJANA GAYATHRI G R
RegisterNumber:  22008369
*/
import chardet
file='/content/spam (1).csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

import pandas as pd
data=pd.read_csv("/content/spam (1).csv",encoding="windows-1252")

print("Data Head ")
data.head()

print("data info")
data.info()

print("data.isnull()")
data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
print("y_pred")
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy")
accuracy

```

## Output:
 
 ![image](https://github.com/niraunjana/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119395610/c706154b-d3a2-4f7f-a6f5-250ea2d21e32)

 ![image](https://github.com/niraunjana/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119395610/986b7650-d809-4440-b2b3-0c18dabfef51)

![image](https://github.com/niraunjana/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119395610/836a0fa0-7d0f-445e-8451-73fad4ed3cec)

 ![image](https://github.com/niraunjana/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119395610/06ad00c3-01d8-4d87-a6c8-9b0e6ccf25af)

 ![image](https://github.com/niraunjana/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119395610/54f88219-b2ca-4bd8-be21-3454dbf9ca51)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
