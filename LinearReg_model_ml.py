from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('WHA_Data.csv')
df.describe()

from sklearn.model_selection import train_test_split
X = df.drop(columns = 'weight')
y = df['weight']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

print(X_train.head())
print(y_train.head())

import numpy as np
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train,y_train)
print('Intercept: ', reg.intercept_)
print('Slope: ', reg.coef_)

cv_results = cross_val_score(reg, X_train, y_train, cv = 5)
print(cv_results)

np.mean(cv_results)

y_pred = reg.predict(X_test)

print("R^2 = ",reg.score(X_test,y_test))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root mean square error = ", rmse)

print("Enter the feature values for weight prediction.\n")
k = int(input("Enter Age = "))
j = int(input("Enter Height = "))

y_new = (0.0840603*k)+(0.57185148*j)+reg.intercept_ # as per data set rmsr, redScore changes !

print("The predicted Weight of a person [⭐] = ",y_new,"Kg.")

print(type(X_test),type(X_train))

X_train.iloc[:,1]

print(X_train.head())

plt.scatter(X_train.iloc[:,1], y_train, color = 'red')

plt.scatter(X_test.iloc[:,1], y_pred, color = 'blue')

plt.ylabel("Target Variable[y] Weight of Person")
plt.xlabel("Feature Variable[x] Age and height")
plt.plot(X_test.iloc[:,1],y_pred,color = 'purple',linewidth = 3)
plt.show()

plt.scatter(X_train.iloc[:,1],y_train,color = 'orange')

plt.scatter(X_test.iloc[:,1],y_pred,color = 'green')

plt.ylabel("Target Variable[y] Weight of Person")
plt.xlabel("Feature Variable[x] Age and height")
plt.plot(X_test.iloc[:,1],y_pred,color = 'blue',linewidth = 3)
plt.show()

