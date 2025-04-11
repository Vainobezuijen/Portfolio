import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

dataset = pd.read_csv('../Data/dataset.csv')
dataset = dataset.dropna()

train, validate, test = train_test_validate_split(dataset, train_split=0.7, validate_split=0.15, test_split=0.15, shuffle=False)

x_train = train[:, 0].reshape(-1,1)
y_train = train[:, 1]

x_test = test[:, 0].reshape(-1,1)
y_test = test[:, 0]

x_train = standardize(x_train)
x_test = standardize(x_test)