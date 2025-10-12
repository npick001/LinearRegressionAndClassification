import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# data loading
train_data_location = 'dataset/Linear_Regression/train.csv'
test_data_location = 'dataset/Linear_Regression/test.csv'
train_data = np.genfromtxt(train_data_location, delimiter=",", names=True)
test_data = np.genfromtxt(test_data_location, delimiter=",", names=True)
X_train = train_data['x'].reshape(-1, 1) 
y_train = train_data['y'] 

# design a linear regression model for the linear regression dataset
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)

# calulate performance metrics
r2 = r2_score(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse) # root mean squared error
mae = mean_absolute_error(y_train, y_pred)

plt.figure(figsize=(10, 6))

# plot the data points and regression line
plt.scatter(X_train, y_train, alpha=0.6, color='blue', label='Data Points', s=50)
plt.plot(X_train, y_pred, color='red', linewidth=2, label='Linear Regression Line')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Linear Regression Model', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# add performance metrics to plot
plt.text(0.70, 0.05, f'R2 Score: {r2:.4f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.70, 0.10, f'MSE: {mse:.4f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.70, 0.15, f'RMSE: {rmse:.4f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.70, 0.20, f'MAE: {mae:.4f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.show()