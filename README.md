# Machine Learning (MSIM 607) Project 1

A comprehensive collection of machine learning implementations using Python and scikit-learn, featuring linear regression, classification, and Principal Component Analysis (PCA) across multiple datasets.

## 📁 Project Structure

```
Project1/
├── src/
│   ├── salary.py              # Salary prediction with comprehensive analysis
│   ├── linear_regression.py   # Train/test linear regression analysis
│   └── red_wine_quality.py    # Complete wine quality analysis with PCA
├── dataset/
│   ├── salary/
│   │   └── Salary_dataset.csv
│   ├── Linear_Regression/
│   │   ├── train.csv
│   │   └── test.csv
│   └── wine_quality/
│       └── winequality-red.csv
├── results/                    # All output files and plots
│   ├── wine_quality_output.txt
│   ├── wine_quality_combined_analysis.png
│   ├── salary_analysis_output.txt
│   ├── salary_analysis_plot.png
│   ├── linear_regression_output.txt
│   └── linear_regression_analysis.png
├── .vscode/
│   └── launch.json            # VS Code debug configurations for all scripts
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Required packages (see `requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Project1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Datasets

### 1. Salary Dataset
- **Source**: Kaggle
- **Features**: Years of Experience
- **Target**: Salary (USD)
- **Samples**: 30 data points
- **Task**: Linear regression with performance analysis

### 2. Linear Regression Dataset
- **Features**: X values
- **Target**: Y values
- **Samples**: 700+ training samples, 300+ test samples
- **Task**: Train/test linear regression with generalization analysis

### 3. Red Wine Quality Dataset
- **Source**: UCI Machine Learning Repository
- **Features**: 11 physicochemical properties (acidity, sulfur dioxide, alcohol, etc.)
- **Target**: Wine quality (3-8 scale)
- **Samples**: 1,599 red wines
- **Tasks**: Multiple ML approaches including regression, classification, and PCA

## 🔧 Usage

### Running Individual Scripts

#### Option 1: Using VS Code/Cursor Launch Configurations
1. Open the project in VS Code or Cursor
2. Press `Ctrl+Shift+D` to open Run and Debug panel
3. Select from available configurations:
   - **Salary Analysis** - Runs `salary.py`
   - **Red Wine Quality** - Runs `red_wine_quality.py` 
   - **Linear Regression** - Runs `linear_regression.py`
4. Press `F5` to run with debugging or `Ctrl+F5` to run without debugging

#### Option 2: Command Line
```bash
# Salary prediction analysis
python src/salary.py

# Linear regression analysis
python src/linear_regression.py

# Wine quality comprehensive analysis
python src/red_wine_quality.py
```

**Note**: All scripts automatically save detailed output to text files and plots to PNG files in the `results/` directory.

## 📈 Implementations

### 1. Salary Prediction (`salary.py`) ✅ **COMPLETE**
- **Algorithm**: Linear Regression (scikit-learn)
- **Features**: Years of Experience
- **Outputs**:
  - Comprehensive statistical analysis and model interpretation
  - Performance metrics (R²: ~0.95, RMSE: ~$5,500)
  - Professional visualization with model equation and metrics
  - Detailed text output saved to `results/salary_analysis_output.txt`

### 2. Linear Regression Analysis (`linear_regression.py`) ✅ **COMPLETE**
- **Algorithm**: Linear Regression (scikit-learn)
- **Features**: Single variable X
- **Analysis**: Training vs Test performance comparison
- **Outputs**:
  - Three-panel visualization (train, test, combined)
  - Generalization analysis and overfitting detection
  - Model coefficients and equation
  - Performance comparison between train/test sets

### 3. Wine Quality Analysis (`red_wine_quality.py`) ✅ **COMPLETE**
- **Comprehensive ML Pipeline** including:

#### Task A: Linear Regression (3-fold CV)
- **Performance**: R² scores around 0.27-0.35 (moderate fit)
- **Method**: 3-fold cross-validation on quality prediction

#### Task B: Binary Classification Setup
- **Classes**: High quality (>5) vs Low quality (≤5)
- **Distribution**: Balanced classification problem

#### Task C: Logistic Regression (3-fold CV)
- **Performance**: Accuracy ~70-75% (good classification)
- **Method**: 3-fold cross-validation on binary classification

#### Task D: Principal Component Analysis
- **Dimensionality Reduction**: 11 features → 2 principal components
- **Analysis**: PC1 dominated by total sulfur dioxide (94.7% variance)
- **Visualization**: Scatter plot colored by wine quality class
- **Feature Importance**: Detailed component analysis showing sulfur dioxide dominance

#### Task E: Reconstruction Analysis
- **PCA Reconstruction**: Testing 2, 3, and 4 principal components
- **MSE Analysis**: Quantifying reconstruction error vs dimensionality
- **Trade-off Visualization**: Error reduction with additional components
- **Results**: 99.5% variance explained with just 2 components

## 📊 Performance Summary

### Salary Prediction
- **R² Score**: 0.9569 (Excellent fit)
- **RMSE**: $5,407 (Low prediction error)
- **Model**: Salary = $9,449 × Experience + $25,792

### Linear Regression
- **Training R²**: High performance on training set
- **Test R²**: Generalization analysis
- **Overfitting Check**: Comparison between train/test metrics

### Wine Quality Analysis
- **Linear Regression CV**: 0.27-0.35 R² (moderate predictive power)
- **Logistic Regression CV**: 70-75% accuracy (good classification)
- **PCA Results**: 94.7% variance in first PC, 99.5% in first 2 PCs
- **Key Finding**: Sulfur dioxide levels are the dominant factor in wine characteristics

## 🎯 Key Features

### Automated Output Management
- **Text Outputs**: All console output automatically saved to results files
- **Plot Outputs**: High-resolution PNG plots saved automatically
- **UTF-8 Encoding**: Proper handling of special characters and symbols

### Advanced Analysis
- **Cross-Validation**: Robust model evaluation with 3-fold CV
- **PCA Analysis**: Component interpretation and reconstruction analysis
- **Visualization**: Professional multi-panel plots with detailed annotations
- **Statistical Reporting**: Comprehensive metrics and model interpretation

### Development Features
- **Polars Integration**: Modern DataFrame library for efficient data processing
- **Debug Configurations**: Ready-to-use VS Code launch configurations
- **Modular Design**: Each script is self-contained with comprehensive analysis

## 🛠️ Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
polars>=0.20.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
```

## � Insights and Findings

### Wine Quality Analysis
1. **Sulfur Dioxide Dominance**: Total and free sulfur dioxide explain 94.7% of data variance
2. **Classification Performance**: Logistic regression achieves 70-75% accuracy on quality classification
3. **Dimensionality**: Only 2 principal components capture 99.5% of data variance
4. **Quality Distribution**: Natural split between high (>5) and low (≤5) quality wines

### Model Performance
- **Salary**: Excellent linear relationship (R² = 0.96)
- **Linear Regression**: Good generalization from training to test data
- **Wine Classification**: Moderate success with physicochemical features alone

## 📄 License

This project is part of a machine learning course assignment demonstrating comprehensive ML workflows.

---

**Status**: All core implementations complete with comprehensive analysis, visualization, and automated output generation.
