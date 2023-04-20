#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Import required libraries and load the dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

wine_df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv')

# Step 2: Explore the data to gain insights and preprocess it
# Check for missing values
print(wine_df.isnull().sum())

# Check the distribution of the target variable
print(wine_df['quality'].value_counts())

# Convert the 'quality' column to binary values
wine_df['quality'] = wine_df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Split the data into input features and target variable
X = wine_df.drop('quality', axis=1)
y = wine_df['quality']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train and evaluate the model
# Create a Decision Tree classifier and fit it on the training data
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing data and calculate the accuracy score
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy}")

# Step 5: Make predictions on new data
# Create a sample data with physiochemical properties of a wine
sample_data = {
    'fixed acidity': [7.4],
    'volatile acidity': [0.7],
    'citric acid': [0],
    'residual sugar': [1.9],
    'chlorides': [0.076],
    'free sulfur dioxide': [11],
    'total sulfur dioxide': [34],
    'density': [0.9978],
    'pH': [3.51],
    'sulphates': [0.56],
    'alcohol': [9.4]
}
sample_df = pd.DataFrame(sample_data)

# Use the trained model to predict the quality of the wine
prediction = clf.predict(sample_df)
print(f"Predicted quality: {prediction}")


# In[2]:


# Step 1: Import required libraries and load the dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

insurance_df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset4/main/medical_cost_insurance.csv')

# Step 2: Explore the data to gain insights and preprocess it
# Check for missing values and data types
print(insurance_df.info())

# Check the statistics of the numerical columns
print(insurance_df.describe())

# Convert categorical columns to numerical using one-hot encoding
cat_cols = ['sex', 'smoker', 'region']
insurance_df = pd.get_dummies(insurance_df, columns=cat_cols)

# Split the data into input features and target variable
X = insurance_df.drop('charges', axis=1)
y = insurance_df['charges']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train and evaluate the model
# Create a Linear Regression model and fit it on the training data
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions on the testing data and evaluate the model
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Step 5: Make predictions on new data
# Create a sample data with information about a person
sample_data = {
    'age': [19],
    'bmi': [27.9],
    'children': [0],
    'sex_female': [0],
    'sex_male': [1],
    'smoker_no': [1],
    'smoker_yes': [0],
    'region_northeast': [0],
    'region_northwest': [0],
    'region_southeast': [1],
    'region_southwest': [0]
}
sample_df = pd.DataFrame(sample_data)

# Use the trained model to predict the insurance cost
prediction = reg.predict(sample_df)
print(f"Predicted insurance cost: {prediction}")


# In[ ]:




