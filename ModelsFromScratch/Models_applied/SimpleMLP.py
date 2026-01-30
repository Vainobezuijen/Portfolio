import numpy as np
import os
import sys

project_path = "/home/vaino/Documents/GitHub/Portfolio/ModelsFromScratch"
project_path = os.path.abspath(project_path)
if project_path not in sys.path:
    sys.path.append(project_path)

from Utils.utils import train_test_validate_split
from Utils.metrics import RegressionMetrics
from Utils.Standardizer import Standardizer
from Models.SimpleMLP import NeuralNet

dataset = np.genfromtxt("../Data/simple_dataset.csv", delimiter=",", skip_header=0, filling_values=np.nan)
dataset = dataset[~np.isnan(dataset).any(axis=1)]

X = dataset[:, 0].reshape(-1, 1)
y = dataset[:, 1]

x_train, y_train, x_val, y_val, x_test, y_test = train_test_validate_split(
    X=X, y=y, train_split=0.8, validate_split=0.1, test_split=0.1, shuffle=False
)

standardizer = Standardizer()
x_train = standardizer.fit_transform(x_train)
x_val = standardizer.transform(x_val)
x_test = standardizer.transform(x_test)

mlp = NeuralNet(n_inputs=1, n_hidden1=16, n_hidden2=8, n_outputs=1, learning_rate=0.01, epochs=2000, seed=42)
mlp.fit(x_train, y_train, verbose=True)

y_pred = mlp.predict(x_test).reshape(-1)

mse_value = RegressionMetrics.mse(y_test, y_pred)
rmse_value = RegressionMetrics.rmse(y_test, y_pred)
r_squared_value = RegressionMetrics.r_squared(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse_value}")
print(f"Root Mean Squared Error (RMSE): {rmse_value}")
print(f"R-squared (Coefficient of Determination): {r_squared_value}")
