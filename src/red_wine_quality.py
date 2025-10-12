import numpy as np
import sklearn as sk

data_location = 'dataset/wine_quality/winequality-red.csv'
data = np.genfromtxt(data_location, delimiter=",", names=True)



# design a linear regression model to perform a 3-fold cross-validation, perform a two-class classification.
# based on the 3-fold cross-validation, perform a two-class classification task:
# if the quality is < 5, the quality is 'High', otherwise the quality is 'Low'.