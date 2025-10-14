import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import sys
import os

# Send outputs to results file rather thanconsole
os.makedirs('results', exist_ok=True)
output_file = open('results/wine_quality_output.txt', 'w', encoding='utf-8')
sys.stdout = output_file

# longer line width is needed for viewing the data better
np.set_printoptions(linewidth=200, suppress=True, precision=6)

# data loading
data_location = 'dataset/wine_quality/winequality-red.csv'
data = pl.read_csv(data_location)


print("---------------------------------------------------------------")
print("Task A: 3-fold cross validation for linear regression")
print("---------------------------------------------------------------")
linear_model = LinearRegression()
x_train = data.select(pl.exclude('quality'))
y_train = data['quality']
cross_validation_score = cross_val_score(linear_model, x_train, y_train, cv=3, error_score='raise')
print("Linear Regression cross-validation scores:", cross_validation_score)


print("\n---------------------------------------------------------------")
print("Task B: Two-class classification based on the quality")
print("---------------------------------------------------------------")
binary_classification = np.where(y_train > 5, 'High', 'Low')
print("Binary classification labels (first 10):\n", binary_classification[:10])


print("\n---------------------------------------------------------------")
print("Task C: 3-fold cross validation for logistic regression")
print("---------------------------------------------------------------")
logistic_model = LogisticRegression(max_iter=1000)
cross_validation_score_logistic = cross_val_score(logistic_model, x_train, binary_classification, cv=3, error_score='raise')
print("Logistic Regression cross-validation scores:", cross_validation_score_logistic)


print("\n---------------------------------------------------------------")
print("Task D: Principal Component Analysis for the inputs and only keep first two principal components")
print("---------------------------------------------------------------")
print("First 10 rows of original x_train dataframe:\n", x_train.head(10))
print("\n")

# center the dataset and reduce the dimension to 2 using PCA
train_mean = x_train.mean()
print("Training data mean:", train_mean)
# with polars, its a bit harder to subtract a 1D df from a 2D df,
# so we have to do it column wise manually.
x_train_centered = x_train.with_columns(
    [(pl.col(col) - train_mean[col]) for col in x_train.columns]
)
print("\nFirst 10 rows of centered x_train_centered dataframe (x_train - train_mean):\n", x_train_centered.head(10))
pca = PCA(n_components=2)
x_train_reduced = pca.fit_transform(x_train_centered)
print("\n")


print("---------------------------------------------------------------")
print("PCA Results Analytics")
print("---------------------------------------------------------------")

feature_names = x_train.columns
print("Feature names:", feature_names)
print("\nFirst Principal Component (PC1) - explains {:.2%} of variance:".format(pca.explained_variance_ratio_[0]))
pc1_weights = pca.components_[0]
for i, (feature, weight) in enumerate(zip(feature_names, pc1_weights)):
    print(f"  {feature:20s}: {weight:8.6f}")

print("\nSecond Principal Component (PC2) - explains {:.2%} of variance:".format(pca.explained_variance_ratio_[1]))
pc2_weights = pca.components_[1]
for i, (feature, weight) in enumerate(zip(feature_names, pc2_weights)):
    print(f"  {feature:20s}: {weight:8.6f}")

print("\nMost important features by absolute weight:")
print("\nPC1 top contributors:")
pc1_importance = [(abs(weight), feature, weight) for feature, weight in zip(feature_names, pc1_weights)]
pc1_importance.sort(reverse=True)
for abs_weight, feature, weight in pc1_importance[:3]:
    print(f"  {feature:20s}: {weight:8.6f} (|{abs_weight:.6f}|)")

print("\nPC2 top contributors:")
pc2_importance = [(abs(weight), feature, weight) for feature, weight in zip(feature_names, pc2_weights)]
pc2_importance.sort(reverse=True)
for abs_weight, feature, weight in pc2_importance[:3]:
    print(f"  {feature:20s}: {weight:8.6f} (|{abs_weight:.6f}|)")

print(f"\nTotal variance explained by first 2 PCs: {pca.explained_variance_ratio_.sum():.2%}")
print(f"Remaining variance in {len(feature_names)-2} components: {1-pca.explained_variance_ratio_.sum():.2%}")


print("\n---------------------------------------------------------------")
print("Task E: Reconstruction Analysis using different numbers of PCs")
print("---------------------------------------------------------------")
n_components_list = [2, 3, 4]
mse_values = []
explained_variance_values = []

for index, n_comp in enumerate(n_components_list):
    print(f"\nReconstruction with {n_comp} principal components:")
    
    # PCA with n_comp components
    pca_n = PCA(n_components=n_comp)
    X_pca_n = pca_n.fit_transform(x_train_centered)
    X_reconstructed = pca_n.inverse_transform(X_pca_n)

    # MSE between original centered data and reconstructed data
    mse = mean_squared_error(x_train_centered.to_numpy(), X_reconstructed)
    mse_values.append(mse)
    
    # Store explained variance
    explained_var = pca_n.explained_variance_ratio_.sum()
    explained_variance_values.append(explained_var)
    
    print(f"  MSE: {mse:.6f}")
    print(f"  Explained variance: {explained_var:.4f} ({explained_var:.2%})")
    print(f"  Reconstruction error: {1-explained_var:.4f} ({(1-explained_var):.2%})")

print(f"\nMSE Values: {mse_values}")
print(f"Explained Variance Values: {explained_variance_values}")

plots = plt.figure(figsize=(16, 6))

# plot all data points and use their labels to color them (binary classification)
pca_plot = plots.add_subplot(1, 2, 1)
pca_plot.scatter(x_train_reduced[:, 0], x_train_reduced[:, 1], c=(binary_classification == 'High'), cmap='bwr', alpha=0.6, s=50)
pca_plot.set_xlabel(f'PC1: {pc1_importance[0][1]}', fontsize=11)
pca_plot.set_ylabel(f'PC2: {pc2_importance[0][1]}', fontsize=11)
pca_plot.set_title('PCA of Red Wine Quality Dataset', fontsize=12)

# MSE vs Number of Principal Components
mse_plot = plots.add_subplot(1, 2, 2)
mse_plot.plot(n_components_list, mse_values, 'bo-', linewidth=2, markersize=8, label='Reconstruction MSE')
mse_plot.set_xlabel('Number of Principal Components')
mse_plot.set_ylabel('Reconstruction MSE')
mse_plot.set_title('Reconstruction Error vs Number of PCs')
mse_plot.grid(True, alpha=0.3)
mse_plot.set_xticks(n_components_list)
mse_plot.legend()

# Add value labels on points
for i, (x, y) in enumerate(zip(n_components_list, mse_values)):
    mse_plot.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=10)

plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
plt.tight_layout()
plt.savefig('results/wine_quality_combined_analysis.png', dpi=300, bbox_inches='tight')

print("\nReconstruction Analysis Summary:")
print("=" * 50)
for i, n_comp in enumerate(n_components_list):
    print(f"{n_comp} PCs: MSE = {mse_values[i]:.6f}, Explained Variance = {explained_variance_values[i]:.2%}")

# Close the output file and restore stdout
output_file.close()
sys.stdout = sys.__stdout__

print("Check 'results/wine_quality_output.txt' for detailed output.")
print("Combined analysis plot saved as 'results/wine_quality_combined_analysis.png'")

plt.show()