import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import os

# Send outputs to results file rather thanconsole
os.makedirs('results', exist_ok=True)
output_file = open('results/linear_regression_output.txt', 'w', encoding='utf-8')
sys.stdout = output_file
np.set_printoptions(linewidth=200, suppress=True, precision=6)

# data loading
train_data_location = 'dataset/Linear_Regression/train.csv'
test_data_location = 'dataset/Linear_Regression/test.csv'
train_data = pl.read_csv(train_data_location)
test_data = pl.read_csv(test_data_location)
X_train = train_data[['x']]
y_train = train_data['y']
X_test = test_data[['x']]
y_test = test_data['y']

# model creation, fitting, and prediction
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# calculate performance metrics for training set
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)

# calculate performance metrics for test set
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)

plt.figure(figsize=(15, 5))

# Create three subplots for comprehensive analysis
plt.subplot(1, 3, 1)
# Plot training data
plt.scatter(X_train, y_train, alpha=0.6, color='blue', label='Training Data', s=50)
plt.plot(X_train, y_train_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Training Set Performance', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Add training metrics to plot
plt.text(0.05, 0.95, f'R²: {train_r2:.4f}', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.05, 0.90, f'MSE: {train_mse:.4f}', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.05, 0.85, f'RMSE: {train_rmse:.4f}', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.05, 0.80, f'MAE: {train_mae:.4f}', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')

plt.subplot(1, 3, 2)
# Plot test data
plt.scatter(X_test, y_test, alpha=0.6, color='green', label='Test Data', s=50)
plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Test Set Performance', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Add test metrics to plot
plt.text(0.05, 0.95, f'R²: {test_r2:.4f}', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.05, 0.90, f'MSE: {test_mse:.4f}', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.05, 0.85, f'RMSE: {test_rmse:.4f}', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.05, 0.80, f'MAE: {test_mae:.4f}', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')

plt.subplot(1, 3, 3)
# Combined plot showing both datasets
plt.scatter(X_train, y_train, alpha=0.6, color='blue', label='Training Data', s=50)
plt.scatter(X_test, y_test, alpha=0.6, color='green', label='Test Data', s=50)
# Plot regression line using full range
x_combined = np.concatenate([X_train['x'].to_numpy(), X_test['x'].to_numpy()])
x_range = np.linspace(x_combined.min(), x_combined.max(), 100).reshape(-1, 1)
y_range_pred = model.predict(x_range)
plt.plot(x_range, y_range_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Combined Train/Test Visualization', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Add comparison metrics to plot
plt.text(0.05, 0.95, f'Train R²: {train_r2:.4f}', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.05, 0.90, f'Test R²: {test_r2:.4f}', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.05, 0.85, f'Generalization Gap:', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.05, 0.80, f'{abs(train_r2 - test_r2):.4f}', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.savefig('results/linear_regression_analysis_plot.png', dpi=300, bbox_inches='tight')

# Print detailed performance comparison
print("=== Linear Regression Performance Analysis ===")
print(f"Training Set Performance:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  MSE: {train_mse:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE: {train_mae:.4f}")
print(f"\nTest Set Performance:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  MSE: {test_mse:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE: {test_mae:.4f}")
print(f"\nGeneralization Analysis:")
print(f"  R² Difference (Train - Test): {train_r2 - test_r2:.4f}")
print(f"  MSE Difference (Test - Train): {test_mse - train_mse:.4f}")
if abs(train_r2 - test_r2) < 0.05:
    print("  Model generalizes well (low overfitting)")
elif train_r2 > test_r2:
    print("  Model may be overfitting (training performance > test performance)")
else:
    print("  Unusual: test performance > training performance")
    

# Close the output file and restore stdout
output_file.close()
sys.stdout = sys.__stdout__

print("Check 'results/linear_regression_output.txt' for detailed output.")
print("Combined analysis plot saved as 'results/linear_regression_combined_analysis.png'")

plt.show()