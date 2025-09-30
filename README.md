# Bike Sharing Demand Prediction
Case Study for the bike sharing service BoomBikes

## Problem Statement 
A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. 

The company wants to know:
- Which variables are significant in predicting the demand for shared bikes.
- How well those variables describe the bike demands

## Business Goal
We are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market. 

## Dataset
This project uses the **Bike Sharing Dataset** from the UCI Machine Learning Repository. The dataset contains daily bike rental data with weather and seasonal information.

**Download the dataset:**
- Visit: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
- Download `day.csv` and place it in the project root directory

**Dataset Features:**
- `season`: Season (1: spring, 2: summer, 3: fall, 4: winter)
- `yr`: Year (0: 2018, 1: 2019)
- `mnth`: Month (1 to 12)
- `weathersit`: Weather situation (1: Clear, 2: Mist, 3: Light Snow/Rain)
- `temp`: Normalized temperature
- `hum`: Normalized humidity
- `windspeed`: Normalized wind speed
- `cnt`: Total rental bikes (target variable)

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
   - Run all cells to see the analysis

## Model Performance
The linear regression model achieves performance metrics that demonstrate the relationship between weather/seasonal features and bike rental demand.

## Technologies Used
- Python 3.8+
- Jupyter Notebook
- pandas - Data manipulation and analysis
- numpy - Numerical computing
- scikit-learn - Machine learning utilities
- statsmodels - Statistical modeling
- matplotlib - Data visualization
- seaborn - Statistical data visualization

## Project Structure
```
BikeSharingCaseStudy/
├── Bike Sharing Case Study.ipynb    # Main analysis notebook
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── day.csv                          # Dataset (download separately)
```

## Acknowledgements
This project is part of case study for the MLPG Program from IIITB with Upgrad.

## Contact
Created by [@kiranarunsutar] - feel free to contact me!

