""" This module is meant for predicting examples with a pre-trained model """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
from datetime import datetime
import os
import shutil
import seaborn as sns
from tpot import TPOTRegressor
from pickle import load

# caml modules
import clean
import feature_engineering as feat
import models as mod
import plot_ as plot_
import evaluate

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor

from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='path to model folder')
    parser.add_argument('data_file', type=str,
                        help='path to example/data file')
    parser.add_argument("target", type=str,
                        help='target column name')

    args = parser.parse_args()
    model_path = args.model_path
    data_file = args.data_file
    target = args.target

    # load the model
    model = load(open(f'{model_path}/model.pkl', 'rb'))
    
    # load the scaler
    try:
        scaler = load(open(f'{model_path}/scaler.pkl', 'rb'))
    except:
        print("No scaler object")

    df_examples = pd.read_csv(str(data_file), index_col=0, header = 0)

    X_examples = df_examples.drop(str(target), axis=1).values.astype(np.float)
    y_examples = df_examples[str(target)].copy().values.astype(np.float)
    
    try:
        X_examples_transformed = scaler.transform(X_examples)
    except:
        X_examples_transformed = X_examples
        
    y_predicted = model.predict(X_examples_transformed)
    
    combined = [y_examples, y_predicted]
    
    df_predicted = pd.DataFrame(data=combined,
                                index=df_examples.index.tolist(),
                                columns=["True", "Predicted"]
                                )
    
    df_predicted.to_csv("predicted.csv")
        
####