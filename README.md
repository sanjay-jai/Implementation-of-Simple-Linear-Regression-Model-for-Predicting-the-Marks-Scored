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
import numpy as np
import pandas as pd
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()

X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print("X=",X)
print("Y=",Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,reg.predict(X_train),color='yellow')
plt.title('Training set(H vs S)',color='blue')
plt.xlabel('Hours',color='green')
plt.ylabel('Scores',color='green')

plt.scatter(X_test,Y_test,color='black')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Test set(H vs S)',color='green')
plt.xlabel('Hours',color='brown')
plt.ylabel('Scores',color='brown')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse) 
print('RMSE = ',rmse)


```
## Output:
![image](https://user-images.githubusercontent.com/95969295/202860194-a8220016-f134-4cbb-96d2-ec5fdc7c4bcb.png)

![image](https://user-images.githubusercontent.com/95969295/202860241-d53ce510-c5ca-43f3-a594-292b54ab36a0.png)

![image](https://user-images.githubusercontent.com/95969295/202860337-89cbb21b-0d61-48c4-89a4-9a58ffc3f91a.png)

![image](https://user-images.githubusercontent.com/95969295/202860384-05cfca5d-0049-454c-a3e8-0b4c45d74852.png)

![image](https://user-images.githubusercontent.com/95969295/202860423-0109070b-8b5c-41bd-8d13-a90b1365ed73.png)

![image](https://user-images.githubusercontent.com/95969295/202860455-3b3bc448-daca-429a-942d-fb349820eb00.png)

![image](https://user-images.githubusercontent.com/95969295/202860481-84789b57-b000-4533-9dc8-0b358fd14945.png)

![image](https://user-images.githubusercontent.com/95969295/202860504-231dc1c9-b36a-44e7-942f-7404ea60b54c.png)








## Result:
Thus, the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
