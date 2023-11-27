# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


# %%
# Load training data
train_data = pd.read_csv("train.csv")

# Load test data
test_data = pd.read_csv("test.csv")

# %%
# Drop the "Id" column from both training and test datasets
train_data = train_data.drop('Id', axis=1)
test_data = test_data.drop('Id', axis=1)

# %%
# Display basic information about the training data
print(train_data.info())

# %%
# Display basic statistics of numerical features
print(train_data.describe())


# %%
# Align the columns between training and test datasets
train_data, test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)  # Use 'left' join for alignment

# %%
# Handling missing values for numerical columns
numeric_cols = train_data.select_dtypes(include=[np.number]).columns
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())
test_data[numeric_cols] = test_data[numeric_cols].fillna(train_data[numeric_cols].mean())  # Use the mean of training data for filling test data

# %%
# Handling missing values for categorical columns
categorical_cols = train_data.select_dtypes(include=[np.object]).columns
train_data[categorical_cols] = train_data[categorical_cols].fillna(train_data[categorical_cols].mode().iloc[0])
test_data[categorical_cols] = test_data[categorical_cols].fillna(test_data[categorical_cols].mode().iloc[0])

# %%
# Convert categorical variables to numerical using one-hot encoding
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

# %%
print(train_data)

# %%
print(test_data)

# %%
# Ensure that both datasets have the same columns after alignment
train_data, test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)  # Use 'left' join for alignment


# %%
# Split the training data into features and target variable
X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# %%
# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# %%
# Plotting the correlation matrix heatmap
correlation_matrix = train_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix Heatmap")
plt.show()

# %%
# Assuming you have already handled missing values in the test dataset
test_data = test_data.fillna(test_data.mean())

# Ensure the columns are aligned
test_data = test_data[train_data.columns.drop('SalePrice')]

# Predict house prices for the test dataset
predictions = model.predict(test_data)


# %%
# Remove the "Id" column from the test dataset if it exists
if 'Id' in test_data.columns:
    test_data = test_data.drop('Id', axis=1)

# Predict house prices for the test dataset
predictions = model.predict(test_data)

# Create a DataFrame with the predictions
output = pd.DataFrame({'Id': range(1461, 1461 + len(predictions)), 'SalePrice': predictions})

# Save the predictions to a CSV file
output.to_csv('house_price_predictions.csv', index=False)


# %%
print(output)

