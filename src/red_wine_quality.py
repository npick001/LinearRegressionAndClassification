import numpy as np
import sklearn as sk

data_location = 'dataset/wine_quality/winequality-red.csv'

# Read in the wine quality data
data = np.genfromtxt(data_location, delimiter=",", names=True)

print("Wine quality data:")
for row in data[:10]:
    print(row)

print(f"\nData shape: {data.shape}")
print(f"Feature names: {data.dtype.names}")

# Extract features (all columns except 'quality') and target (quality)
feature_names = [name for name in data.dtype.names if name != 'quality']
print(f"Features: {feature_names}")
print(f"Target: quality")