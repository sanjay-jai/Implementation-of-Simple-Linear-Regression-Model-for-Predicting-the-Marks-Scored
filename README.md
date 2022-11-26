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
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset = pd.read_csv('/content/student_scores.csv')

dataset.head()
dataset.tail()

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
X

Y = dataset.iloc[:,1].values
Y

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

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse) 
print('RMSE = ',rmse)


```
## Output:
![image](https://user-images.githubusercontent.com/95969295/204100034-dcd947dd-482c-429c-9e31-a8d02be2585d.png)

![image](https://user-images.githubusercontent.com/95969295/204100056-4aeafdc5-1ae7-426b-aae1-c7a6a58ce557.png)

![image](https://user-images.githubusercontent.com/95969295/204100069-c1c937b0-d7cc-4527-92d8-b5bd9657ae22.png)

![image](https://user-images.githubusercontent.com/95969295/204100083-ef69dc84-ec75-41ff-a034-4503602fcf23.png)

![image](https://user-images.githubusercontent.com/95969295/204100098-c3f32704-6fe9-4497-be52-f5becf2a442e.png)

![image](https://user-images.githubusercontent.com/95969295/204100122-dcdfe122-f1ac-4133-ac23-9e56cac08907.png)

![image](https://user-images.githubusercontent.com/95969295/204100139-d2510533-7fc4-4613-8fa2-f7ba04402687.png)

![image](https://user-images.githubusercontent.com/95969295/204100159-79c80e88-d3d0-4327-a863-622706d958d7.png)

![image](https://user-images.githubusercontent.com/95969295/204100182-d9e67f6c-6d10-4f54-96f5-cd164117f33c.png)

![image](https://user-images.githubusercontent.com/95969295/204100201-d43f6725-0e12-43e4-b38a-55432772faa4.png)

![image](https://user-images.githubusercontent.com/95969295/204100211-d76f3dc6-069d-41d2-a07b-8275007d36a5.png)









## Result:
Thus, the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
