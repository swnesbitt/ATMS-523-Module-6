#!/usr/bin/env python
# coding: utf-8

"""
Dara Procell
November 16, 2025
ATMS 523 - Module 6 Project
Tornado Prediction using Climate Indices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

import shap

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# STEP 1: Monthly Tornadoes for Illinois
# ============================================================================

# Load the SPC tornado dataset
tornado_url = 'https://www.spc.noaa.gov/wcm/data/1950-2021_actual_tornadoes.csv'
tornado_df = pd.read_csv(tornado_url)

# Convert date column to datetime
tornado_df['date'] = pd.to_datetime(tornado_df['date'])

# Filter for Illinois tornadoes only
il_tornadoes = tornado_df[tornado_df['st'] == 'IL'].copy()

# Create year and month columns
il_tornadoes['year'] = il_tornadoes['date'].dt.year
il_tornadoes['month'] = il_tornadoes['date'].dt.month
monthly_tornadoes = il_tornadoes.groupby(['year', 'month']).size().reset_index(name='tornado_count')
monthly_tornadoes['Date'] = pd.to_datetime(
    monthly_tornadoes[['year', 'month']].assign(day=1)
)
monthly_tornadoes = monthly_tornadoes.set_index('Date')
monthly_tornadoes = monthly_tornadoes.drop(['year', 'month'], axis=1)

# Make a complete time series and filling months with 0 tornadoes
date_range = pd.date_range(
    start=monthly_tornadoes.index.min(),
    end=monthly_tornadoes.index.max(),
    freq='MS'
)
monthly_tornadoes = monthly_tornadoes.reindex(date_range, fill_value=0)
monthly_tornadoes.index.name = 'Date'

# ============================================================================
# STEP 2: Merge Datasets
# ============================================================================

# Load climate indices from Module 4 Notebook 1
enso = pd.read_csv('https://www.atmos.illinois.edu/~snesbitt/soi.dat',
                   sep=r'\s+', header=None, skiprows=4, skipfooter=0, engine='python')
pdo = pd.read_csv('https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat',
                  sep=r'\s+', header=None, skiprows=2, engine='python')
nao = pd.read_csv('https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table',
                  sep=r'\s+', header=None, skiprows=1, engine='python')
ao = pd.read_csv('https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table',
                 sep=r'\s+', header=None, skiprows=1, engine='python')

# Process ENSO
enso_new = pd.DataFrame()
enso_new['Date'] = pd.date_range(
    start=datetime(int(enso[0].iloc[0]), 1, 1), # first year January
    end=datetime(int(enso[0].iloc[-1]), 12, 1), # last year December
    freq="MS"                                   # month start frequency
)
enso_new = enso_new.set_index('Date')
enso_new['ENSO'] = enso.loc[:, 1:].stack().values # pivot columns to rows

# Process NAO
nao_new = pd.DataFrame()
nao_new['Date'] = pd.date_range(
    start=datetime(int(nao[0].iloc[0]), 1, 1),
    end=datetime(int(nao[0].iloc[-1]), 12, 1),
    freq="MS"
)
nao_new = nao_new.set_index('Date')
nao_new['NAO'] = nao.loc[:, 1:].stack(dropna=False).values

# Process PDO
pdo_new = pd.DataFrame()
pdo_new['Date'] = pd.date_range(
    start=datetime(int(pdo[0].iloc[0]), 1, 1),
    end=datetime(int(pdo[0].iloc[-1]), 12, 1),
    freq="MS"
)
pdo_new = pdo_new.set_index('Date')
pdo_new['PDO'] = pdo.loc[:, 1:].stack(dropna=False).values

# Process AO
ao_new = pd.DataFrame()
ao_new['Date'] = pd.date_range(
    start=datetime(int(ao[0].iloc[0]), 1, 1),
    end=datetime(int(ao[0].iloc[-1]), 12, 1),
    freq="MS"
)
ao_new = ao_new.set_index('Date')
ao_new['AO'] = ao.loc[:, 1:].stack(dropna=False).values

# Merge all climate indices
newdf_all = pd.merge(enso_new, pdo_new, left_index=True, right_index=True)
newdf_all = pd.merge(newdf_all, nao_new, left_index=True, right_index=True)
newdf_all = pd.merge(newdf_all, ao_new, left_index=True, right_index=True)

newdf_all.loc[newdf_all['ENSO'] <= -9.9, 'ENSO'] = np.nan
newdf_all.loc[newdf_all['PDO'] > 90., 'PDO'] = np.nan
newdf_all.loc[newdf_all['NAO'] <= -99.9, 'NAO'] = np.nan
newdf_all.loc[newdf_all['AO'] <= -99.9, 'AO'] = np.nan

# Merge datasets
merged_data = pd.merge(monthly_tornadoes, newdf_all, 
                       left_index=True, right_index=True, 
                       how='inner')

merged_data = merged_data.dropna()


# ============================================================================
# STEP 3: Random Forest Model with Climate Indices Only
# ============================================================================

X_climate = merged_data[['ENSO', 'PDO', 'NAO', 'AO']]
y = merged_data['tornado_count']
X_train_climate, X_test_climate, y_train, y_test = train_test_split(
    X_climate, y, test_size=0.2, random_state=42
)
rf_climate = RandomForestRegressor(
    n_estimators=100,   # number of trees in forst
    max_depth=10,       # how deep each tree can go
    random_state=42,    # for repoducibility
    n_jobs=-1           # use all CPU cores
)
rf_climate.fit(X_train_climate, y_train)
y_pred_climate = rf_climate.predict(X_test_climate)

rmse_climate = np.sqrt(mean_squared_error(y_test, y_pred_climate))
corr_climate = np.corrcoef(y_test, y_pred_climate)[0, 1]

print("RMSE for 4 climate model:", rmse_climate)
print("Correlation coefficient:", corr_climate)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_climate, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Observed Tornado Count')
plt.ylabel('Predicted Tornado Count')
plt.title(f'Climate Indices Only\nRMSE: {rmse_climate:.2f}, Corr: {corr_climate:.3f}')
plt.savefig('q3_climate_indices_only.png', dpi=300, bbox_inches='tight')
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# STEP 4: Add One-Hot Encoding for Month
# ============================================================================

merged_data['month'] = merged_data.index.month
# transform categorical data (months) into binary columns
# prevents model from thinking months 12 is a greater value than month 1
month_dummies = pd.get_dummies(merged_data['month'], prefix='month')
X_with_month = pd.concat([X_climate, month_dummies], axis=1)
X_train_month, X_test_month, y_train, y_test = train_test_split(
    X_with_month, y, test_size=0.2, random_state=42
)

rf_month = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_month.fit(X_train_month, y_train)

y_pred_month = rf_month.predict(X_test_month)

rmse_month = np.sqrt(mean_squared_error(y_test, y_pred_month))
corr_month = np.corrcoef(y_test, y_pred_month)[0, 1]

print("RMSE for climate and month model:", rmse_month)
print("Correlation coefficient:", corr_month)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_month, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Observed Tornado Count')
plt.ylabel('Predicted Tornado Count')
plt.title(f'Climate Indices + Month\nRMSE: {rmse_month:.2f}, Corr: {corr_month:.3f}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q4_one_hot_encoding.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 5: Feature Importance and Permutation Analysis
# ============================================================================

feature_importance = pd.DataFrame({
    'feature': X_train_month.columns,
    'importance': rf_month.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

perm_importance = permutation_importance(
    rf_month, X_test_month, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Feature importance from model
top_features = feature_importance.head(15)
axes[0].barh(range(len(top_features)), top_features['importance'])
axes[0].set_yticks(range(len(top_features)))
axes[0].set_yticklabels(top_features['feature'])
axes[0].set_xlabel('Feature Importance')
axes[0].set_title('Random Forest Feature Importance (Top 15)')
axes[0].invert_yaxis()

# Plot 2: Permutation importance
perm_df = pd.DataFrame({
    'feature': X_train_month.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False).head(15)

axes[1].barh(range(len(perm_df)), perm_df['importance_mean'])
axes[1].set_yticks(range(len(perm_df)))
axes[1].set_yticklabels(perm_df['feature'])
axes[1].set_xlabel('Mean Decrease in R² Score')
axes[1].set_title('Permutation Feature Importance (Top 15)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('q5_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 6: SHAP Analysis - feature importance and feature effects
# ============================================================================

explainer = shap.TreeExplainer(rf_month)
shap_values = explainer.shap_values(X_test_month)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_month, show=False)
plt.title('SHAP Summary Plot: Feature Importance and Effects')
plt.tight_layout()
plt.savefig('q6_shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 7: SHAP Feature dependence for ENSO index
# ============================================================================

plt.figure(figsize=(10, 6))
shap.dependence_plot('ENSO', shap_values, X_test_month, show=False)
plt.title('SHAP Feature Dependence: ENSO Index')
plt.tight_layout()
plt.savefig('q7_shap_enso_dependence.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 8: Summary
# ============================================================================

print("Model Performance Comparison:")
print("  Climate Indices Only:")
print("    - RMSE:", rmse_climate)
print("    - Correlation:", corr_climate)
print("  Climate Indices + Month:")
print("    - RMSE:", rmse_month)
print("    - Correlation:", corr_month)
print("  Improvement with month encoding:")
print("    - RMSE reduction:", (rmse_climate - rmse_month),  ((rmse_climate - rmse_month)/rmse_climate*100))
print("    - Correlation increase:", (corr_month - corr_climate))

# climate index importance
climate_features = ['ENSO', 'PDO', 'NAO', 'AO']
climate_importance = feature_importance[feature_importance['feature'].isin(climate_features)]

print("1. Climate Index Importance Ranking:")
for idx, row in climate_importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

print("2. Seasonal Pattern Analysis:")
month_features = [f for f in feature_importance['feature'] if 'month_' in f]
important_months = feature_importance[feature_importance['feature'].isin(month_features)].head(5)
print("   Most important months for tornado prediction:")
for idx, row in important_months.iterrows():
    month_num = int(row['feature'].split('_')[1])
    month_name = pd.Timestamp(2000, month_num, 1).strftime('%B')
    print(f"   - {month_name}: {row['importance']:.4f}")

