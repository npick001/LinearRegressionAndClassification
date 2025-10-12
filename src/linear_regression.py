import numpy as np
import sklearn as sk

# Training data
train_data_location = 'dataset/Linear_Regression/train.csv'
train_data = np.genfromtxt(train_data_location, delimiter=",", names=True)

# Test data
test_data_location = 'dataset/Linear_Regression/test.csv'
test_data = np.genfromtxt(test_data_location, delimiter=",", names=True)

print("Training data:")
for row in train_data[:10]:
    print(row)

print("\nTest data:")
for row in test_data[:10]:
    print(row)

print(f"\nTraining data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")