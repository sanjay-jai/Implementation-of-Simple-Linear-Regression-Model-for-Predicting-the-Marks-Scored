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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

dataset = pd.read_csv('/content/student_scores.csv')

dataset.head()

dataset.tail()
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred = reg.predict(X_test)
Y_pred

Y_test

plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae = mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```
## Output:
![image](https://user-images.githubusercontent.com/95969295/202352530-5d5c6ab9-c6b4-4f3c-af1d-ed951076744e.png)

![image](https://user-images.githubusercontent.com/95969295/202352662-66cf94e8-6680-4cb4-ad91-39b7b94eaeff.png)

![image](https://user-images.githubusercontent.com/95969295/202352727-5116f14d-3b11-40ef-a41d-c54e5834a670.png)

![image](https://user-images.githubusercontent.com/95969295/202352794-ed05df8c-9341-43d9-b859-e989e6ad2f9a.png)

![image](https://user-images.githubusercontent.com/95969295/202352916-4a46ccd0-8e42-42ca-a316-2aab1a83c202.png)

![image](https://user-images.githubusercontent.com/95969295/202352993-4873886a-b6ad-4121-afe5-c24c00ace280.png)

![image](https://user-images.githubusercontent.com/95969295/202353096-0f9f0ccf-a28a-493e-aadb-8c6fe38366af.png)

![image](https://user-images.githubusercontent.com/95969295/202353169-5cd2abd2-f7ee-41af-b168-d5c3eea92cb1.png)


## Result:
Thus, the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
