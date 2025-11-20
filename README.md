# ATMS 523 - Module 6 Project
Random Forest Prediction, Feature Engineering, and Explainable AI for Monthly Tornado Counts  
Author: Nathan Makowski   
Date: 11/19/2025

## Project Overview
This project analyzes monthly tornado counts for selected Upper Midwest states and their relationship to large-scale climate indices. Random Forest regression and explainable AI tools (Permutation Importance and SHAP) are used to evaluate how well ENSO, PDO, NAO, and AO explain monthly tornado variability, and how seasonality (month of year) improves prediction skill.

This notebook is the final project for ATMS 523 Module 6.

## Datasets Used

### 1. Tornado Data (SPC, 1950-2021)
Source: https://www.spc.noaa.gov/wcm/data/1950-2021_actual_tornadoes.csv

States included in the count:
- Illinois (IL)
- Indiana (IN)
- Wisconsin (WI)
- Michigan (MI)
- Missouri (MO)
- Iowa (IA)
- Minnesota (MN)

### 2. Climate Indices (from Module 4)
Monthly data for:
- ENSO (NINO3.4)
- PDO
- NAO
- AO

These are merged with the tornado data into one monthly dataset.

## Project Tasks

### 1. Aggregate Tornado Counts
Compute monthly tornado counts across the seven states listed above.

### 2. Reconstruct Climate Index DataFrame
Use Module 4 code to reconstruct a monthly DataFrame containing ENSO, PDO, NAO, and AO.

### 3. Merge Tornado and Climate Data
Align by month. Months without tornadoes are assigned a count of zero.

### 4. Baseline Random Forest Model
Train a random forest using only the four climate indices.  
Outputs include:
- RMSE
- Pearson correlation
- Observed vs. predicted scatterplot

### 5. Add Month One-Hot Encoding
Add twelve month indicator columns and retrain the model.  
Compare RMSE and correlation against the baseline.

### 6. Feature Importance and Permutation Methods
Compute:
- Random Forest feature_importances_
- Multipass permutation importance (e.g., 30 repeats)

### 7. SHAP Analysis
Compute:
- SHAP summary plot
- SHAP dependence plot for ENSO

Note: TreeExplainer may require disabling additivity checks:
`explainer.shap_values(X, check_additivity=False)`

### 8. Observations and Interpretation
Summaries include:
- Whether climate indices alone show predictive skill
- How much seasonality improves the model
- Which features rank highest in importance
- Interpretation of SHAP patterns
- Limitations and future improvements

### 9. Save Models and Outputs
The notebook saves:
- Model files (joblib)
- Scalers
- Final merged CSV files

### 10. Diagnostics
Includes checks on:
- Number of months
- Tornado count summary
- Fraction of zero-tornado months

## Dependencies
- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib
- shap
- joblib

## License
This project is for educational use as part of ATMS 523 Module 6.

<p align="left">
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  </a>
</p>