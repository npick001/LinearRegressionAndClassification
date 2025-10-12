# Machine Learning (MSIM 607) Project 1

A collection of machine learning implementations using Python and scikit-learn, focusing on linear regression and classification tasks across multiple datasets.

## 📁 Project Structure

```
Project1/
├── src/
│   ├── salary.py              # Salary prediction using linear regression
│   ├── linear_regression.py   # Basic linear regression implementation
│   └── red_wine_quality.py    # Wine quality classification (in progress)
├── dataset/
│   ├── salary/
│   │   └── Salary_dataset.csv
│   ├── Linear_Regression/
│   │   ├── train.csv
│   │   └── test.csv
│   └── wine_quality/
│       └── winequality-red.csv
├── results/ 
├── .vscode/
│   └── launch.json            # VS Code/Cursor debug configurations
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
- **Target**: Salary
- **Samples**: 30+ data points
- **Task**: Linear regression to predict salary based on experience

### 2. Linear Regression Dataset
- **Features**: X values
- **Target**: Y values
- **Samples**: 700+ data points
- **Task**: Basic linear regression modeling

### 3. Red Wine Quality Dataset
- **Source**: UCI Machine Learning Repository
- **Features**: 11 physicochemical properties
- **Target**: Wine quality (0-10 scale)
- **Samples**: 1,599 red wines
- **Task**: Two-class classification (High/Low quality)

## 🔧 Usage

### Running Individual Scripts

#### Option 1: Using VS Code/Cursor Launch Configurations
1. Open the project in VS Code or Cursor
2. Press `Ctrl+Shift+D` to open Run and Debug panel
3. Select from available configurations:
   - **Task 1: Salary Analysis** - Runs `salary.py`
   - **Task 2: Red Wine Quality** - Runs `red_wine_quality.py`
   - **Task 3: Linear Regression** - Runs `linear_regression.py`
4. Press `F5` to run with debugging or `Ctrl+F5` to run without debugging

#### Option 2: Command Line
```bash
# Salary prediction
python src/salary.py

# Linear regression
python src/linear_regression.py

# Wine quality classification
python src/red_wine_quality.py
```

## 📈 Implementations

### 1. Salary Prediction (`salary.py`)
- **Algorithm**: Linear Regression (scikit-learn)
- **Features**: Years of Experience
- **Outputs**:
  - Training performance metrics (R², MSE, RMSE, MAE)
  - Scatter plot with regression line
  - Model coefficients

**Expected Performance**:
- R² Score: > 0.8 (Good fit)
- RMSE: < $6,000 (Reasonable prediction error)

### 2. Linear Regression (`linear_regression.py`)
- **Algorithm**: Linear Regression (scikit-learn)
- **Features**: Single variable X
- **Outputs**:
  - Performance metrics visualization
  - Data points with fitted line

### 3. Wine Quality Classification (`red_wine_quality.py`)
- **Status**: In development
- **Planned**: 3-fold cross-validation with two-class classification
- **Classes**: High quality (≥5) vs Low quality (<5)

## 📊 Performance Metrics

### Regression Tasks
- **R² Score**: Coefficient of determination (0-1, higher is better)
- **MSE**: Mean Squared Error (lower is better)
- **RMSE**: Root Mean Squared Error (in target units)
- **MAE**: Mean Absolute Error (in target units)

### Classification Tasks (Planned)
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 🛠️ Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
```

## 🔍 Data Description

Detailed data descriptions are available in:
- `dataset/data description.docx` - Comprehensive dataset documentation


## 📄 License

This project is part of a machine learning course assignment.

---

**Note**: This project is designed for educational purposes and demonstrates various machine learning concepts using real-world datasets.
