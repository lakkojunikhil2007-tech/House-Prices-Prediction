# House Price Prediction

## Overview

This project is a beginner-level Machine Learning regression project that predicts house prices using various housing features such as area, number of rooms, garage details, neighborhood information, and other property characteristics.

The project demonstrates a complete end-to-end ML workflow including:

* Data preprocessing
* Missing value handling
* Categorical encoding
* Model training
* Model evaluation
* Hyperparameter tuning

---

## Technologies Used

* Python
* Pandas
* NumPy
* scikit-learn

---

## Dataset

Dataset used:
House Prices - Advanced Regression Techniques from Kaggle

[Kaggle Dataset Link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques?utm_source=chatgpt.com)

---

## Project Workflow

### 1. Data Loading

* Loaded `train.csv` using pandas.

### 2. Data Exploration

* Checked dataset shape
* Inspected columns and datatypes
* Identified missing values

### 3. Data Cleaning

* Removed columns with excessive missing values:

  * PoolQC
  * MiscFeature
  * Alley
  * Fence

### 4. Missing Value Handling

* Filled numeric missing values using column mean.
* Filled categorical missing values using mode.

### 5. Feature Engineering

* Separated features (`X`) and target (`y`)
* Applied One-Hot Encoding using `pd.get_dummies()`

### 6. Train/Test Split

* Split dataset into training and testing sets using:

  * 80% training data
  * 20% testing data

### 7. Models Used

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

### 8. Model Evaluation

Used Mean Absolute Error (MAE) for evaluation.

---

## Results

| Model                   | MAE    |
| ----------------------- | ------ |
| Linear Regression       | ~18411 |
| Decision Tree Regressor | ~27698 |
| Random Forest Regressor | ~17528 |

Random Forest Regressor produced the best performance.

---

## Key Learnings

Through this project, I learned:

* Data preprocessing techniques
* Handling missing data
* One-Hot Encoding
* Train/Test splitting
* Regression models
* Model comparison
* Hyperparameter tuning
* Machine Learning workflow fundamentals

---

## Future Improvements

Possible future enhancements:

* Feature scaling
* Cross-validation
* GridSearchCV
* XGBoost
* Model deployment using Streamlit or Flask

---

## How to Run

1. Clone the repository
2. Install required libraries

```bash id="jlwm3r"
pip install pandas numpy scikit-learn
```

3. Run the Python file

```bash id="jlwm2q"
python main.py
```

---

## Author

Created by Lakkoju Nikhil Sai Phaneendra as a beginner Machine Learning project.
