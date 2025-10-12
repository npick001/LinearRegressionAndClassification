import numpy as np
import sklearn as sk

data_location = 'dataset/salary/Salary_dataset.csv'
output_dir = 'results/Task1/'

# read in the data - skip the first column (index) and use the correct column structure
data = np.genfromtxt(data_location, delimiter=",", names=True, usecols=(1, 2))

for row in data:
    print(row)

