# Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## Aim:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Use the standard libraries in python for Gradient Design.
2. Set variables for assigning dataset values.
3. Import LinearRegression from the sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for marks by using the representation of graph.
6. Compare the graphs and hence we obtain the Linear Regression for the given dataset. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SANJAI A
Register Number:  212220040142
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="yellow")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![simple linear regression model for predicting the marks scored](https://user-images.githubusercontent.com/95969295/196490458-6be73b1d-d607-498c-8fcb-368ce1b02c67.png)
![image](https://user-images.githubusercontent.com/95969295/196493624-19a2a8de-cb33-4c49-9227-6abe3938cfe2.png)

## Result:
Thus, the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
