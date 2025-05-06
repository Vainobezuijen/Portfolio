import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

project_path = "/home/vaino/Documents/GitHub/ML_models"
project_path = os.path.abspath(project_path)
if project_path not in sys.path:
    sys.path.append(project_path)

from utils import *
from metrics import *
from Models.LinearRegression import LinearRegression
from plotting import plot_model
from Preprocessing.Standardizer import Standardizer

dataset = pd.read_csv('../Data/simple_dataset.csv')
dataset = dataset.dropna()

train, val, test = train_test_validate_split(dataset, train_split=0.8, validate_split=0.1, test_split=0.1, shuffle=False)

x_train = train[:, 0].reshape(-1,1)
y_train = train[:, 1]

x_val = val[:, 0].reshape(-1,1)
y_val = val[:, 1]

x_test = test[:, 0].reshape(-1,1)
y_test = test[:, 1]

standardizer = Standardizer()

x_train = standardizer.fit_transform(x_train)
x_test = standardizer.transform(x_test)
x_val = standardizer.transform(x_val)

lr = LinearRegression(loss_name='mse')
lr.fit(x_train,y_train,x_val,y_val)
lr.save_model('Saved_Models/model.pk1')

y_pred = lr.predict(x_test)

mse_value = RegressionMetrics.mae(y_test, y_pred)
rmse_value = RegressionMetrics.rmse(y_test, y_pred)
r_squared_value = RegressionMetrics.r_squared(y_test, y_pred)

plot_model(x_train, y_train, x_test, y_pred)

print(f"Mean Squared Error (MSE): {mse_value}")
print(f"Root Mean Squared Error (RMSE): {rmse_value}")
print(f"R-squared (Coefficient of Determination): {r_squared_value}")