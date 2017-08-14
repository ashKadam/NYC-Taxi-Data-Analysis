
# coding: utf-8

# In[4]:

############################################################################################
#Purpose of this notebook is to analyze NYC taxi data provided on Kaggle.
# Ultimately I am trying to predict trip durations for given test data
# As this is first dataset I am working on from scratch, I will emphasize on EDA as well
# I am thinking of using GBM or Random Forest as my predictive modeling technique
############################################################################################
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import timedelta
import datetime as dt


# In[7]:

# Reading train and test data stored locally on machine.(Should change this while uploading on github)
train = pd.read_csv('F:/Python_Notebooks/NYCTaxiData/train/train.csv')
test = pd.read_csv('F:/Python_Notebooks/NYCTaxiData/test/test.csv')


# In[23]:

# let's find out how many entries are there in training and test data set
print("Numbers of entries in training data set : " + str(train.shape[0]) + " with " + str(train.shape[1]) + " attributes")
print("Numbers of entries in training data set : " + str(test.shape[0]) + " with " + str(test.shape[1]) + " attributes")


# In[11]:

# Above results suggests test data set has less attributes than training data.
# Taking pick at training and test data sets
train.head(3)


# In[25]:

# Test data ofcourse does not contain drop off time and trip duration
test.head(3)


# In[29]:

# We first need to find is there any missing data in given data set
# We will find out count for each attribute in dataset and compare it with number of samples in dataset 
# If number of entries for any attribute is less that total sample count then that attribute is missing entry -> missing value
print("Missing values in training data set") if train.count().min() != train.shape[0] else print("Training Data is A-ok")
print("Missing values in test data set") if test.count().min() != test.shape[0] else print("Test Data is A-ok")


# In[31]:

# Let's look for anamolies now...
# but before that I would like to know the data types of each attribute.
#It will help to decide the kind of approach to take in determining the outliers of data.
train.info()


# In[36]:

# Most of the data attributes are of numeric type. Hence mathematical operations 
# Outlier detection in trip duration and passenger count
f = plt.figure(figsize=(8,6))
plt.xlabel('Trip Index')
plt.ylabel('Trip duration')
x = range(train.shape[0])
y = np.sort(train.trip_duration.values)
plt.scatter(x,y)
plt.show()


# In[37]:

# It looks like there are some outliers in trip duration. 
# These could be valid entries where customers did went for long trip or there could be errors. 
# In either of cases, these outliers are disturbing data balance. Hence, I am removing these from further analysis. 
validQuantile = train.trip_duration.quantile(0.99)
train = train[train.trip_duration < validQuantile]

plt.figure(figsize=(8,6))
plt.xlabel('Trip Index')
plt.ylabel('Trip Duration')
x = range(train.shape[0])
y = np.sort(train.trip_duration.values)
plt.scatter(x,y)
plt.show()


# In[42]:

plt.figure(figsize=(12,8))
plt.xlabel('Trip Duration')
sb.distplot(train.trip_duration.values, bins=100)
plt.show()


# In[44]:

# This data is definately skewed. Let's fix this. 
plt.figure(figsize=(12,8))
plt.xlabel('Trip Duration')
sb.distplot(np.log(train.trip_duration.values), bins=100)
plt.show()


# In[60]:

# Number of passengrs could also be important factor. Check it.
plt.figure(figsize=(12,8))
plt.xlabel('Number of Passengers')
pass_count = train['passenger_count'].value_counts()
sb.barplot(pass_count.index, pass_count.values, alpha=1.0)
plt.show()


# In[61]:

# Now let's dive into more important attributes - pickup and drop off time and date. 
# Time and date need to be converted into proper python date and time format.
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])


# In[64]:

train['pickup_month'] = train['pickup_datetime'].dt.month
train['pickup_weekday'] = train['pickup_datetime'].dt.weekday
train['pickup_day'] = train['pickup_datetime'].dt.day
train['pickup_hour'] = train['pickup_datetime'].dt.hour

train['drop_month'] = train['dropoff_datetime'].dt.month
train['drop_weekday'] = train['dropoff_datetime'].dt.weekday
train['drop_day'] = train['dropoff_datetime'].dt.day
train['drop_hour'] = train['dropoff_datetime'].dt.hour


# In[71]:

pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]

plt.figure(figsize=(12,8))
plt.xlabel('Month')
plt.ylabel('Passenger Count')
sb.countplot(x='pickup_month',data=train, palette=pkmn_type_colors)
plt.show()

