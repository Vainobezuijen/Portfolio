import numpy as np

from sklearn import datasets
from Utils.utils import train_test_validate_split
from Utils.metrics import *
from Models.RandomForest import RandomForest

import os
import sys

project_path = "/home/vaino/Documents/GitHub/Portfolio/ModelsFromScratch"
project_path = os.path.abspath(project_path)
if project_path not in sys.path:
    sys.path.append(project_path)

data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, y_train, X_test, y_test = train_test_validate_split(
    X, y, train_split=0.8, test_split=0.2, random_state=42
    )

clf = RandomForest(n_trees=10, max_depth=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = ClassificationMetrics.accuracy(y_test, y_pred)
# recall = ClassificationMetrics.recall()
# precision = ClassificationMetrics.precision()
# f1 = ClassificationMetrics.f1()