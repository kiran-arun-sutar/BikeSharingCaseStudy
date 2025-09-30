# Bike Sharing Demand Prediction

A comprehensive machine learning case study for BoomBikes to predict daily bike rental demand and understand the factors influencing bike-sharing usage patterns.

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Business Goal](#business-goal)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Setup Instructions](#setup-instructions)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Problem Statement 

A US bike-sharing provider **BoomBikes** has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. 

### Key Business Questions:
- Which variables are significant in predicting the demand for shared bikes?
- How well do those variables describe the bike demands?
- What factors should management focus on to optimize operations?

## Business Goal

We are required to model the demand for shared bikes with the available independent variables. The model will be used by management to:

1. **Understand demand drivers**: Identify which features have the strongest impact on bike rental counts
2. **Strategic planning**: Manipulate business strategy to meet demand levels and customer expectations
3. **Market expansion**: Understand demand dynamics for potential new markets
4. **Resource optimization**: Allocate bikes and resources efficiently based on predicted demand

## Dataset

This project uses the **Bike Sharing Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset). The dataset contains **731 days** of daily bike rental data collected from the Capital Bikeshare system in Washington D.C., spanning 2018-2019.

### Download Instructions:
1. Visit: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
2. Download the `Bike-Sharing-Dataset.zip` file
3. Extract `day.csv` and place it in the project root directory

### Dataset Features:

**Target Variable:**
- `cnt`: Total rental bikes including both casual and registered users

**Temporal Features:**
- `season`: Season (1: spring, 2: summer, 3: fall, 4: winter)
- `yr`: Year (0: 2018, 1: 2019)
- `mnth`: Month (1 to 12)
- `weekday`: Day of week (0: Sunday to 6: Saturday)
- `holiday`: Whether day is a holiday (0: No, 1: Yes)
- `workingday`: Whether day is neither weekend nor holiday (0: No, 1: Yes)

**Weather Features:**
- `weathersit`: Weather situation
  - 1: Clear, Few clouds, Partly cloudy
  - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds
  - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds
  - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- `temp`: Normalized temperature in Celsius (divided by 41)
- `atemp`: Normalized feeling temperature in Celsius (divided by 50)
- `hum`: Normalized humidity (divided by 100)
- `windspeed`: Normalized wind speed (divided by 67)

**Note on Data Leakage:** Earlier versions of this project incorrectly included `casual` and `registered` users as features. These have been removed as they are components of the target variable `cnt` (cnt = casual + registered), which would create data leakage and make the model unusable in practice.

## Methodology

### 1. Data Preprocessing
- **Categorical Encoding**: Convert categorical variables (season, weather, month, weekday) to dummy variables
- **Feature Scaling**: Apply MinMaxScaler to numerical features for standardization
- **Train-Test Split**: 70-30 split for model training and validation

### 2. Exploratory Data Analysis (EDA)
- Visualize distributions of numerical variables
- Analyze correlations between features and target variable
- Identify seasonal and temporal patterns in bike rentals
- Generate pairplots to understand feature relationships

### 3. Feature Selection
Using iterative approach with two key metrics:
- **VIF (Variance Inflation Factor)**: Remove features with VIF > 5 to reduce multicollinearity
- **P-values**: Remove features with p-value > 0.05 to keep only statistically significant predictors
- Manual iterative refinement based on model performance

### 4. Model Building
- **Algorithm**: Multiple Linear Regression using Ordinary Least Squares (OLS)
- **Library**: statsmodels for detailed statistical analysis
- **Iterations**: Multiple model versions with progressive feature elimination

### 5. Model Evaluation
- **Training Performance**: R-squared, Adjusted R-squared
- **Residual Analysis**: Check for normality, homoscedasticity, and patterns
- **Test Set Validation**: Evaluate model on unseen data
- **Feature Importance**: Analyze coefficient magnitudes and statistical significance

## Key Findings

### Significant Predictors of Bike Demand

Based on the linear regression analysis, the following factors significantly influence bike rental demand:

**Positive Impact (Increase Demand):**
- 🌡️ **Temperature**: Higher temperatures lead to more rentals (strongest predictor)
- 📅 **Year (2019 vs 2018)**: Growing trend year-over-year
- ☀️ **Clear Weather**: Good weather conditions boost demand
- 🍂 **Fall Season**: Highest demand compared to other seasons

**Negative Impact (Decrease Demand):**
- 🌧️ **Light Rain/Snow**: Adverse weather reduces rentals
- 💨 **High Windspeed**: Windy conditions deter riders
- 💧 **High Humidity**: Uncomfortable humidity levels decrease demand
- 🏖️ **Summer/Spring**: Lower relative demand compared to fall

### Business Insights

1. **Weather Dependency**: Bike rentals are highly sensitive to weather conditions. Clear, moderate-temperature days see peak demand.

2. **Seasonal Patterns**: Fall shows the highest demand, suggesting comfortable temperatures and stable weather drive usage.

3. **Growth Trend**: Year-over-year growth indicates increasing adoption of bike-sharing services.

4. **Temperature Sweet Spot**: There's an optimal temperature range for bike rentals - neither too hot nor too cold.

5. **Weather Resilience**: Light precipitation significantly impacts demand, highlighting the need for weather-based operational strategies.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/kiran-arun-sutar/BikeSharingCaseStudy.git
   cd BikeSharingCaseStudy
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download `day.csv` from the UCI Machine Learning Repository (link above)
   - Place the file in the project root directory

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Open and run the notebook**
   - Open `Bike Sharing Case Study.ipynb`
   - Run all cells to see the complete analysis

## Usage Guide

### Running the Complete Analysis

```bash
jupyter notebook "Bike Sharing Case Study.ipynb"
```

The notebook contains the following sections:

1. **Data Loading and Initial Exploration**
   - Import libraries and load dataset
   - Check data types, missing values, and basic statistics

2. **Data Visualization**
   - Distribution plots for numerical variables
   - Categorical variable analysis
   - Correlation heatmaps and pairplots

3. **Data Preparation**
   - Handle categorical variables with dummy encoding
   - Split data into training and testing sets
   - Scale numerical features

4. **Model Building and Iteration**
   - Build initial linear regression model
   - Iteratively refine using VIF and p-values
   - Progressive feature elimination

5. **Model Evaluation**
   - Residual analysis with diagnostic plots
   - Performance metrics on test set
   - Feature importance interpretation

6. **Final Model and Predictions**
   - Summary of significant predictors
   - Model equation with coefficients
   - Business recommendations

### Using the VIF Calculation Function

The notebook includes a reusable function for calculating Variance Inflation Factor:

```python
def calculate_vif(dataframe):
    """
    Calculate VIF for all features in a dataframe.
    
    Parameters:
    dataframe: pandas DataFrame with numeric features
    
    Returns:
    DataFrame with Features and their VIF values, sorted by VIF descending
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif = pd.DataFrame()
    vif['Features'] = dataframe.columns
    vif['VIF'] = [variance_inflation_factor(dataframe.values, i) 
                  for i in range(dataframe.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by="VIF", ascending=False)
    return vif

# Usage
vif = calculate_vif(X_train)
print(vif)
```

### Making Predictions

To use the trained model for predictions:

```python
# Prepare your input features (ensure same preprocessing)
new_data = pd.DataFrame({
    'temp': [0.5],
    'yr': [1],
    'season_fall': [1],
    # ... other features
})

# Scale features if needed
new_data_scaled = scaler.transform(new_data)

# Make prediction
prediction = model.predict(new_data_scaled)
print(f"Predicted bike rentals: {prediction[0]:.0f}")
```

## Model Performance

### Final Model Metrics

After removing data leakage and applying proper feature selection:

- **R-squared (Training)**: ~0.83 - Model explains 83% of variance in bike rental demand
- **Adjusted R-squared**: ~0.82 - Accounts for number of predictors
- **Test Set Performance**: Comparable performance on unseen data

### Model Equation

The final model takes the form:
```
cnt = β₀ + β₁(temp) + β₂(yr) + β₃(windspeed) + β₄(season_vars) + β₅(weather_vars) + ... + ε
```

Where coefficients are determined through OLS regression to minimize squared errors.

### Residual Analysis

✅ **Normally Distributed**: Residuals follow approximately normal distribution  
✅ **Homoscedastic**: Constant variance across predictions  
✅ **No Patterns**: Residuals show random scatter with no systematic patterns  
✅ **Independence**: No autocorrelation in residuals

### Model Validation

The model was validated through:
- Train-test split validation
- Residual diagnostic plots
- VIF analysis for multicollinearity
- Statistical significance testing (p-values)

**Note**: The model achieves realistic performance metrics without data leakage. Earlier versions showing R² > 0.88 were due to including `casual` and `registered` users as features, which are components of the target variable.

## Technologies Used

### Core Technologies
- **Python 3.8+**: Programming language
- **Jupyter Notebook**: Interactive development environment

### Data Analysis Libraries
- **pandas (≥1.3.0)**: Data manipulation and analysis
- **numpy (≥1.21.0)**: Numerical computing and array operations

### Machine Learning Libraries
- **scikit-learn (≥0.24.2)**: Machine learning utilities, preprocessing, and metrics
- **statsmodels (≥0.12.2)**: Statistical modeling and OLS regression

### Visualization Libraries
- **matplotlib (≥3.4.2)**: Basic plotting and visualization
- **seaborn (≥0.11.1)**: Statistical data visualization and advanced plots

### Why These Technologies?

- **statsmodels over scikit-learn**: Provides detailed statistical summaries (p-values, confidence intervals, R-squared) essential for interpretability
- **Seaborn**: Enables beautiful correlation heatmaps and pairplots for EDA
- **Jupyter**: Interactive environment ideal for exploratory data analysis and iterative model development

## Troubleshooting

### Common Issues and Solutions

#### 1. Dataset Not Found Error
```
FileNotFoundError: [Errno 2] No such file or directory: 'day.csv'
```
**Solution**: Download `day.csv` from the UCI ML Repository and place it in the project root directory.

#### 2. Module Import Errors
```
ModuleNotFoundError: No module named 'seaborn'
```
**Solution**: Install dependencies using `pip install -r requirements.txt`

#### 3. Kernel Issues
```
Kernel appears to have died. It will restart automatically.
```
**Solution**: 
- Restart Jupyter kernel
- Check if you have enough RAM (model requires ~500MB)
- Update Jupyter: `pip install --upgrade jupyter notebook`

#### 4. VIF Calculation Takes Too Long
**Solution**: The VIF calculation can be slow for large feature sets. This is expected behavior. Consider:
- Using the optimized `calculate_vif()` function provided
- Running VIF only after preliminary feature selection

#### 5. Different Results Than Expected
**Potential Causes**:
- Using a different version of the dataset
- Random train-test split (set `random_state` for reproducibility)
- Different package versions

**Solution**: Ensure you're using the versions specified in `requirements.txt`

### Getting Help

If you encounter issues not covered here:
1. Check the [Issues](https://github.com/kiran-arun-sutar/BikeSharingCaseStudy/issues) page
2. Review `NOTEBOOK_CHANGES_GUIDE.md` for implementation details
3. Contact the repository maintainer (see [Contact](#contact) section)

## Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
- 🐛 Report bugs or issues
- 💡 Suggest new features or improvements
- 📝 Improve documentation
- 🔧 Submit pull requests with enhancements

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add: Description of feature'`)
6. Push to your fork (`git push origin feature/YourFeature`)
7. Open a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include comments for complex logic
- Update README if adding new features

## License

This project is part of an academic case study for the MLPG Program from IIITB with Upgrad. The dataset is publicly available from the UCI Machine Learning Repository under their terms of use.

## Project Structure
```
BikeSharingCaseStudy/
├── Bike Sharing Case Study.ipynb          # Main analysis notebook with EDA and modeling
├── Bike Sharing Case Study.py             # Converted Python script for reference
├── requirements.txt                        # Python dependencies
├── NOTEBOOK_CHANGES_GUIDE.md              # Guide for notebook modifications
├── fix_notebook.py                        # Helper script for programmatic notebook edits
├── README.md                              # Project documentation (this file)
├── bike_sharing_case_study_subjective_answers.pdf  # Subjective answers
└── day.csv                                # Dataset (download separately)
```

### File Descriptions

- **Bike Sharing Case Study.ipynb**: Complete Jupyter notebook with data analysis, visualization, model building, and evaluation
- **Bike Sharing Case Study.py**: Python script version of the notebook for easy code review
- **requirements.txt**: All Python package dependencies with version specifications
- **NOTEBOOK_CHANGES_GUIDE.md**: Documentation of fixes applied to remove data leakage
- **fix_notebook.py**: Automated script used to fix data leakage issues programmatically
- **README.md**: Comprehensive project documentation (you're reading it!)
- **day.csv**: Daily bike rental dataset (not included, must be downloaded separately)

## Acknowledgements

- **Dataset Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) - Bike Sharing Dataset
- **Original Data**: Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013)
- **Academic Program**: This project is part of a case study for the MLPG Program from IIITB with Upgrad
- **Domain**: Thanks to Capital Bikeshare for making their data publicly available for research

### References

1. Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, 2(2-3), 113-127.
2. UCI Machine Learning Repository: Bike Sharing Dataset
3. [Statsmodels Documentation](https://www.statsmodels.org/)
4. [Scikit-learn Documentation](https://scikit-learn.org/)

## Contact

**Created by**: Kiran Arun Sutar ([@kiranarunsutar](https://github.com/kiran-arun-sutar))

Feel free to reach out for:
- Questions about the analysis
- Collaboration opportunities
- Suggestions for improvements
- Bug reports or issues

📧 **Email**: kiran.arun.sutar@gmail.com  
🔗 **GitHub**: [@kiran-arun-sutar](https://github.com/kiran-arun-sutar)  
💼 **LinkedIn**: [Connect with me](https://linkedin.com/in/kiranarunsutar)

---

⭐ If you found this project helpful, please consider giving it a star!

**Last Updated**: September 2025

