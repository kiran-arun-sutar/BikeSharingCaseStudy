# Guide: Fixing Data Leakage and Improving Code Quality in the Notebook

## Overview
This document outlines the critical changes needed in `Bike Sharing Case Study.ipynb` to fix data leakage and improve code maintainability. A fixed Python version of the notebook is provided as `Bike Sharing Case Study.py` for reference.

## Critical Issue: Data Leakage

### Problem
The notebook uses `casual` and `registered` as features to predict `cnt`. However:
- `cnt = casual + registered` (exact mathematical relationship)
- At prediction time, you wouldn't know casual/registered counts before knowing the total count
- This makes the model unusable in practice despite showing high R-squared

### Required Changes

#### 1. Line ~386: Remove data leakage from num_vars
**Current:**
```python
num_vars = ['mnth', 'weekday', 'temp', 'hum', 'windspeed','casual','registered','cnt']
```

**Change to:**
```python
num_vars = ['mnth', 'weekday', 'temp', 'hum', 'windspeed','cnt']
```

#### 2. Line ~192: Update pairplot visualization
**Current:**
```python
sns.pairplot(dataset[['temp','hum','windspeed','casual','registered','cnt']])
```

**Change to:**
```python
sns.pairplot(dataset[['temp','hum','windspeed','cnt']])
```

#### 3. Line ~420: Update correlation pairplot
**Current:**
```python
sns.pairplot(df_train[['temp','casual','registered','cnt']])
```

**Change to:**
```python
sns.pairplot(df_train[['temp','cnt']])
```

#### 4. Line ~444: Change initial model feature
**Current:**
```python
X_train_lm = sm.add_constant(X_train[['registered']])
```

**Change to:**
```python
X_train_lm = sm.add_constant(X_train[['temp']])
```

#### 5. Line ~482: Update second model features
**Current:**
```python
X_train_lm = X_train[['registered', 'casual','temp']]
```

**Change to:**
```python
X_train_lm = X_train[['temp', 'yr', 'spring', 'summer', 'winter']]
```

#### 6. Line ~856: Update test set variable dropping
**Current:**
```python
X_test_m7 = X_test_m7.drop(["registered", "hum", "holiday", "summer","temp", "mnth"], axis = 1)
```

**Change to:**
```python
X_test_m7 = X_test_m7.drop(["hum", "holiday", "summer","temp", "mnth"], axis = 1)
```

## Code Quality Improvement: Reusable VIF Function

### Problem
The VIF calculation code is repeated 7+ times throughout the notebook (lines ~553-558, ~593-599, ~633-639, ~670-676, ~707-713, ~745-751, ~782-788).

### Solution
Add this function after the VIF import (around line 546):

```python
def calculate_vif(dataframe):
    """
    Calculate Variance Inflation Factor for all features in a dataframe.
    
    Parameters:
    dataframe: pandas DataFrame with numeric features
    
    Returns:
    DataFrame with Features and their VIF values, sorted by VIF descending
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif = pd.DataFrame()
    vif['Features'] = dataframe.columns
    vif['VIF'] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by="VIF", ascending=False)
    return vif
```

Then replace all 7 occurrences of the repetitive VIF calculation code with:
```python
vif = calculate_vif(X)  # or X_train depending on context
vif
```

## Expected Outcome

After these changes:
- Model will be practically usable (no data leakage)
- R-squared will likely drop to ~0.6-0.7 (realistic performance)
- Code will be more maintainable with reusable VIF function
- Model predictions will be based only on truly available features at prediction time

## Reference
See `Bike Sharing Case Study.py` for the complete corrected code implementation.
