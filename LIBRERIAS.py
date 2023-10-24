# Libreras Básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Algoritmo ROC CURVE
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# Datasets Sklearn

from sklearn.datasets import *

    #Dataset Cancer
X_cancer, y_cancer= load_breast_cancer (return_X_y=True)


    #Dataset Diabetes
X_diabetes, y_diabetes= load_diabetes(return_X_y=True)


    #Dataset Digit
X_digits, y_digits= load_digits(return_X_y=True)


    #Dataset Files
X_files, y_files= load_digits(return_X_y=True)


    #Dataset Iris
X_iris, y_iris= load_iris(return_X_y=True)


    #Dataset Linnerud
X_rud, y_rud= load_linnerud(return_X_y=True)


    #Dataset Wine
X_wine, y_wine= load_wine(return_X_y=True)