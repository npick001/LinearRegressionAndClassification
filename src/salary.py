import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# data loading
data_location = 'dataset/salary/Salary_dataset.csv'
data = pl.read_csv(data_location)
X_train = data[['YearsExperience']]
y_train = data['Salary']

# model creation, fitting, and prediction
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)

# calculate performance metrics
r2 = r2_score(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse) # root mean squared error
mae = mean_absolute_error(y_train, y_pred)

plt.figure(figsize=(10, 6))

# plot the data points and regression line
plt.scatter(X_train, y_train, alpha=0.6, color='blue', label='Data Points', s=50)
plt.plot(X_train, y_pred, color='red', linewidth=2, label='Linear Regression Line')
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Salary ($)', fontsize=12)
plt.title('Salary vs Years of Experience', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# add performance metrics to plot
plt.text(0.70, 0.05, f'R2 Score: {r2:.4f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.70, 0.10, f'MSE: {mse:.4f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.70, 0.15, f'RMSE: {rmse:.4f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.70, 0.20, f'MAE: {mae:.4f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')

plt.savefig('results/salary_analysis_plot.png')
plt.tight_layout()
plt.show()