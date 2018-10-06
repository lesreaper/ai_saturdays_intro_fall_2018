import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import timeit

start = timeit.default_timer()

dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

def preprocess_features(dataframe):
  selected_features = dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
  # Create a synthetic feature.
  selected_features["rooms_per_person"] = (
    dataframe["total_rooms"] /
    dataframe["population"])

  return selected_features

def preprocess_targets(dataframe):
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["median_house_value"] = (
    dataframe["median_house_value"] / 1000.0)
  return output_targets

X = preprocess_features(dataset)
y = preprocess_targets(dataset)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

new_test_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")
X_test = preprocess_features(new_test_data)
y_pred = regressor.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test[:3000], y_pred)))
stop = timeit.default_timer()
print('Time: ', stop - start)
