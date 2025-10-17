import numpy as np
import os
import sys

project_path = "/home/vaino/Documents/GitHub/Portfolio/ModelsFromScratch"
project_path = os.path.abspath(project_path)
if project_path not in sys.path:
    sys.path.append(project_path)

from Utils.utils import *
from Utils.metrics import *
from Models.LinearRegression import LinearRegression
from Utils.plotting import plot_model
from Utils.Standardizer import Standardizer

dataset = np.genfromtxt('../Data/simple_dataset.csv', delimiter=',', skip_header=0, filling_values=np.nan)
dataset = dataset[~np.isnan(dataset).any(axis=1)]

X = dataset[:, 0]
y = dataset[:, 1]

x_train, y_train, x_val, y_val, x_test, y_test = train_test_validate_split(X=X, y=y, train_split=0.8, validate_split=0.1, test_split=0.1, shuffle=False)

x_train = x_train.reshape(-1, 1)
x_val = x_val.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

standardizer = Standardizer()

x_train = standardizer.fit_transform(x_train)
x_test = standardizer.transform(x_test)
x_val = standardizer.transform(x_val)

lr = LinearRegression(loss_name='mse')
lr.fit(x_train,y_train,x_val,y_val)
lr.save_model('Saved_Models/lr_model.pk1')

y_pred = lr.predict(x_test)

mse_value = RegressionMetrics.mae(y_test, y_pred)
rmse_value = RegressionMetrics.rmse(y_test, y_pred)
r_squared_value = RegressionMetrics.r_squared(y_test, y_pred)

plot_model(x_train, y_train, x_test, y_pred)

print(f"Mean Squared Error (MSE): {mse_value}")
print(f"Root Mean Squared Error (RMSE): {rmse_value}")
print(f"R-squared (Coefficient of Determination): {r_squared_value}")