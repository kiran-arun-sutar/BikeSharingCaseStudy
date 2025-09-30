#!/usr/bin/env python
# coding: utf-8

# ## Bike Sharing Case Study
# 
# Below are the steps performed to get the prediction for hte bike sharing case study.
# 
# 1. Reading the Data
# 2. Handling data issues <br>
#     2.1 Handling co-related variables <br>
#     2.2 Handling Categorial variables
# 3. Visualising and Understanding the Data
# 4. Data Preparation <br>
#     4.1 Dummy Variables <br>
# 5. Splitting the Data into Training and Testing Sets <br>
#     5.1 Rescaling Features
# 6. Building a linear model
# 7. Residual Analysis of the train data
# 8. Making Predictions Using the Final Model
# 9. Model Evaluation

# ### Step 1: Reading the Data
# 
# Let us first import NumPy and Pandas and read the dataset

# In[54]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[55]:


# import numpy and pandas 
import numpy as np
import pandas as pd


# In[56]:


# Read dataset

dataset = pd.read_csv("day.csv")


# In[57]:


# Check the head of the dataset
dataset.head()


# In[58]:


dataset.shape


# ### Step 2 Handling data issues
# 
# 1. There are no null values in the given dataset. 
# 2. The colume instant is just a sequence number. THis colume can be dropped.
# 3. The colume dteday is data colume, it can be converted to the Date format.

# In[59]:


dataset.info()


# In[60]:


dataset.describe()


# In[61]:


# dropping instant column
dataset.drop(['instant'], axis='columns', inplace=True)


# In[62]:


dataset.head()


# In[63]:


# converting dteday to data
dataset['dteday']= pd.to_datetime(dataset['dteday'])


# In[64]:


dataset.info()


# ### Step 2.2 - Handling redundent variables 
# On further observing the data we find that the dteday column is redundant. The data from this column is already convered by three other columns - yr, mnth, weekday. The presence of dteday colume does not provide us with any additional information except date of reading, which is not much useful for the prediction.
# 

# In[65]:


# Drop dteday
dataset.drop(['dteday'], axis='columns', inplace=True)
dataset.head(200).tail()


# In[107]:


# droping atemp as it looks redundent with temp
dataset.drop(['atemp'], axis='columns', inplace=True)


# ### Step 2.2 - Handling categorial variables
# 
# There are two main categorial variables in dataset – 
# 1.	Season – A season can be one of the 4 seasons from spring, summer, fall, winter. The season can have effect on the bike sharing as people will not prefer to use bike in bad weather. We expect to find overall higher or lower trend for a season.
# 2.	Weathersit – This variable shows the status of the weather conditions on given day. As season it would have effect on the number of users using the service as people would not prefer bike share in bad weather. As season gives us seasonal reading on the weather, which would stay consistent for weeks/months, this variable can provide us with the variation seen in the weather on daily basis.
# 
# Above variables are not ordered and hence would need to be categorized and then we would need to create the dummy variables for the same.
# Apart from these below additional variables can be treated as categorial variables.
# 1.	Yr – 2018 or 2019
# 2.	Mnth – Month of year for the reading (ordered 1-12)
# 3.	Weekday – Day of week (ordered 0-6)
# 4.	Holiday – if day is holiday or not (unordered binary)
# 5.	Workingday – weather day is working day or not (unordered binary)
# 

# In[66]:


# convert season to unordered categorial variable
def seasons_map(x):
    return x.map({1:'spring',2:'summer',3:'fall',4:'winter'})

dataset['season'] = dataset[['season']].apply(seasons_map)


# In[67]:


# convert weathersit to unordered categorial variable
def weather_map(x):
    return x.map({1:'clear',2:'mist',3:'snow',4:'rain'})

dataset['weathersit'] = dataset[['weathersit']].apply(weather_map)


# In[68]:


dataset.head()


# ### Step 3. Visualising and Understanding the Data
# 
# Here we are trying to find 
# - if there is some obvious multicollinearity going on
# - if some predictors directly have a strong association with the outcome variable

# In[69]:


# importing `matplotlib` and `seaborn`

import matplotlib.pyplot as plt
import seaborn as sns


# #### Visualising Numeric Variables
# 
# Let's make a pairplot of all the numeric variables

# In[74]:


# ploting pair plot of the important numric variables

sns.pairplot(dataset[['temp','hum','windspeed','cnt']])
plt.show()


# ### Inferences - 
# 1. registered users have highest impact on the cnt
# 2. casual users shouwing second most impact on the cnt
# 3. Temperature also looks co-related, number of users are higher when temp is more.
# 4. hum and windspeed are exibiting any singificunt influent on cnt.

# #### Visualising Categorical Variables
# 
# Let's make a boxplot for the categorical variables.

# In[76]:


plt.figure(figsize=(20, 12))
plt.subplot(2,2,1)
sns.boxplot(x = 'season', y = 'cnt', data = dataset)
plt.subplot(2,2,2)
sns.boxplot(x = 'weathersit', y = 'cnt', data = dataset)
plt.subplot(2,2,3)
sns.boxplot(x = 'workingday', y = 'cnt', data = dataset)
plt.subplot(2,2,4)
sns.boxplot(x = 'weekday', y = 'cnt', data = dataset)


# ### Inferences
# 
# 1. In fall shared bike uses are higher compared to other season.
# 2. People tend to use the shared bikes more in clear wheather
# 3. holidays and day of week seem to be not exibiting any significunt influence on the shared bike uses pattern

# ## Step 4: Data Preparation
# 

# In[108]:


dataset.info()


# In[109]:


dataset.head()


# ### Dummy Variables
# 
# The variable `season` has four levels. We have these levels in integers. 
# 
# Now we will convert them to `dummy variables`.

# In[81]:


# Get the dummy variables for the feature 'season' and store it in a new variable - 'status'
seasons = pd.get_dummies(dataset['season'])


# In[82]:


seasons.head()


# Now, we don't need four columns. We an drop the `fall` column, as the season can be identified with just the last three columns where — 
# - `000` will correspond to `fall`
# - `100` will correspond to `spring`
# - `010` will correspond to `summer`
# - `001` will correspond to `winter`

# In[83]:


# Let's drop the first column from seasons df using 'drop_first = True'

seasons = pd.get_dummies(dataset['season'], drop_first = True)


# In[84]:


# Add the results to the original dataset

dataset = pd.concat([dataset, seasons], axis = 1)


# In[85]:


# Drop 'season' as we have created the dummies for it

dataset.drop(['season'], axis = 1, inplace = True)


# In[86]:


dataset.head()


# In[87]:


dataset['weathersit'].value_counts()


# #### There is no value for Rain in weathersit

# In[88]:


# Get the dummy variables for the feature 'weathersit' and store it in a new variable - 'status'
weather = pd.get_dummies(dataset['weathersit'])


# In[89]:


weather.head()


# Now, we don't need three columns. We can drop the `clear` column, as the whater can be identified with just the last two columns where — 
# - `00` will correspond to `clear`
# - `01` will correspond to `snow`
# - `10` will correspond to `mist`

# In[90]:


# Let's drop the first column from whater df using 'drop_first = True'

weather = pd.get_dummies(dataset['weathersit'], drop_first = True)


# In[91]:


# Add the results to the original dataset

dataset = pd.concat([dataset, weather], axis = 1)


# In[95]:


# Drop 'weathersit' as we have created the dummies for it

dataset.drop(['weathersit'], axis = 1, inplace = True)


# In[93]:


dataset.head()


# ## Step 5: Splitting the Data into Training and Testing Sets
# 
# The first basic step for regression is performing a train-test split.

# In[248]:


from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(dataset, train_size = 0.7, test_size = 0.3, random_state = 100)


# ### Rescaling the Features 
# 
# We will use MinMax scaling.

# In[187]:


from sklearn.preprocessing import MinMaxScaler


# In[188]:


scaler = MinMaxScaler()


# In[189]:


# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['mnth', 'weekday', 'temp', 'hum', 'windspeed','cnt']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[190]:


df_train.head()


# In[191]:


df_train.describe()


# In[192]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# ### As expected, registered, casual and temp are the most correlated variables for the target variable

# In[193]:


# ploting most related numeric variables

sns.pairplot(df_train[['temp','cnt']])
plt.show()



# In[194]:


# Dividing into X and Y sets for the model building

y_train = df_train.pop('cnt')
X_train = df_train


# ## Step 6: Building a linear model
# 

# In[195]:


import statsmodels.api as sm

# Add a constant
X_train_lm = sm.add_constant(X_train[['temp']])

# Create a first fitted model
lr = sm.OLS(y_train, X_train_lm).fit()


# In[196]:


# Check the parameters obtained

lr.params


# In[197]:


# Let's visualise the data with a scatter plot and the fitted regression line
plt.scatter(X_train_lm.iloc[:, 1], y_train)
plt.plot(X_train_lm.iloc[:, 1], 0.0245 + 0.933*X_train_lm.iloc[:, 1], 'r')
plt.show()


# In[198]:


# Print a summary of the linear regression model obtained
print(lr.summary())


# ### Adding additional variables
# 

# In[199]:


# Assign all the feature variables to X
X_train_lm = X_train[['temp', 'yr', 'spring', 'summer', 'winter']]


# In[200]:


# Build a linear model

import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train_lm)

lr = sm.OLS(y_train, X_train_lm).fit()

lr.params


# In[201]:


# Check the summary
print(lr.summary())


# ### Adding all the variables to the model

# In[202]:


# Check all the columns of the dataframe

dataset.columns


# In[203]:


#Build a linear model

import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train)

lr_1 = sm.OLS(y_train, X_train_lm).fit()

lr_1.params


# In[204]:


print(lr_1.summary())


# The R-squared value is 1 which means we might be overfitting the regression line. Lets try to remove the features which are less relevent and try to achive the highest score with least number of variables.

# Looking at p-value it looks like yr is the insignificunt variables anso mnth also can be dropped.
# 
# Lets check VIF to take the decision.

# ### Checking VIF

# In[205]:


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

# Check for the VIF values of the feature variables
vif = calculate_vif(X_train)
vif


# ### Dropping the variable and updating the model
# 

# In[207]:



X = X_train  # Start with all available features (excluding casual/registered)


# In[208]:


# Build a third fitted model
X_train_lm = sm.add_constant(X)

lr_2 = sm.OLS(y_train, X_train_lm).fit()


# In[209]:


# Print the summary of the model
print(lr_2.summary())


# In[210]:


# Calculate the VIFs again for the new model using our reusable function

vif = calculate_vif(X)
vif


# ### Dropping the Variable and Updating the Model
# 
# As we can notice hum variable have high VIF value. let's drop this variable and update the model.

# In[211]:


# Dropping highly correlated variables and insignificant variables
X = X.drop('hum', 1)


# In[212]:


# Build a third fitted model
X_train_lm = sm.add_constant(X)

lr_3 = sm.OLS(y_train, X_train_lm).fit()


# In[213]:


# Print the summary of the model

print(lr_3.summary())


# In[214]:


# Calculate the VIFs again for the new model using our reusable function
vif = calculate_vif(X)
vif


# ### Dropping the Variable and Updating the Model
# 
# Now we can see holiday has high P value and the variable is insignificunt . let's drop this variable and update the model.

# In[215]:


X = X.drop('holiday', 1)


# In[216]:


# Build a fourth fitted model
X_train_lm = sm.add_constant(X)

lr_4 = sm.OLS(y_train, X_train_lm).fit()


# In[217]:


print(lr_4.summary())


# In[218]:


# Calculate the VIFs again for the new model using our reusable function
vif = calculate_vif(X)
vif


# ### Dropping the Variable and Updating the Model
# 
# Now we can see holiday has high P value and the variable is insignificunt . let's drop this variable and update the model.

# In[219]:


X = X.drop('summer', 1)


# In[220]:


# Build a fifth fitted model
X_train_lm = sm.add_constant(X)

lr_5 = sm.OLS(y_train, X_train_lm).fit()


# In[221]:


print(lr_5.summary())


# In[222]:


# Calculate the VIFs again for the new model using our reusable function
vif = calculate_vif(X)
vif


# ### Dropping the Variable and Updating the Model
# 
# Now we can see temp is having very high VIF. Let's drop this variable and update the model.
# 

# In[223]:


X = X.drop('temp', 1)


# In[224]:


# Build a sixth fitted model
X_train_lm = sm.add_constant(X)

lr_6 = sm.OLS(y_train, X_train_lm).fit()


# In[225]:


print(lr_6.summary())


# In[226]:


# Calculate the VIFs again for the new model using our reusable function
vif = calculate_vif(X)
vif


# ### Dropping the Variable and Updating the Model
# 
# Now we can see mnth is having very high VIF. Let's drop this variable and update the model.

# In[227]:


X = X.drop('mnth', 1)


# In[228]:


# Build a seventh fitted model
X_train_lm = sm.add_constant(X)

lr_7 = sm.OLS(y_train, X_train_lm).fit()


# In[229]:


print(lr_7.summary())


# In[230]:


# Calculate the VIFs again for the new model using our reusable function
vif = calculate_vif(X)
vif


# ## Step 7: Residual Analysis of the train data
# 
# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[249]:


y_train_cnt = lr_7.predict(X_train_lm)


# In[250]:


X_train_lm.shape


# In[251]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_cnt), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# ## Step 8: Making Predictions Using the Final Model
# 
# Now that we have fitted the model and checked the normality of error terms, it's time to go ahead and make predictions using the final, i.e. fourth model.

# #### Applying the scaling on the test sets

# In[252]:


df_test[num_vars] = scaler.transform(df_test[num_vars])


# In[253]:


df_test.describe()


# #### Dividing into X_test and y_test

# In[254]:


y_test = df_test.pop('cnt')
X_test = df_test


# In[255]:


# Adding constant variable to test dataframe
X_test_m7 = sm.add_constant(X_test)


# In[256]:


# Creating X_test_m4 dataframe by dropping variables from X_test_m4

X_test_m7 = X_test_m7.drop(["hum", "holiday", "summer","temp", "mnth"], axis = 1)
X_test_m7.shape


# In[257]:


X_test_m7.columns


# In[258]:


# Making predictions using the seventh model

y_pred_m7 = lr_7.predict(X_test_m7)


# ## Step 9: Model Evaluation
# 
# Let's now plot the graph for actual versus predicted values.

# In[260]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred_m7)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)     


# In[ ]:




