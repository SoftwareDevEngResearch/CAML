" This module tests the clean.py module"

import clean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
from datetime import datetime
import os
import shutil

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

def test_normalize_overall_min_max():
    
    input_file = "simple_input.yaml"
    with open(str(input_file), 'r') as file:
        input_ = yaml.load(file, Loader=yaml.FullLoader)
    
    data = input_["data"]
    target = data["target_col_name"]
    
    # dataframe with ID, target, features 
    df = pd.read_csv(data["data_path"], header = 0, dtype=object, index_col=data["index_col_name"])
    
    df_train, df_test = train_test_split(df, 
                                         test_size=0.2, 
                                            )
    
    df_train, df_test, scale = clean.normalize_overall_min_max(df_train, df_test, target)
    
    X_train = df_train.drop(str(target), axis=1).values.astype(np.float)
    
    min_ = np.amin(X_train, axis=None, out=None)
    max_ = np.amax(X_train, axis=None, out=None)
    
    assert min_ == 0
    assert max_ == 1
    
def test_normalize_scaler():
    
    input_file = "simple_input.yaml"
    with open(str(input_file), 'r') as file:
        input_ = yaml.load(file, Loader=yaml.FullLoader)
    
    data = input_["data"]
    target = data["target_col_name"]
    
    # dataframe with ID, target, features 
    df = pd.read_csv(data["data_path"], header = 0, dtype=object, index_col=data["index_col_name"])
    
    df_train, df_test = train_test_split(df, 
                                         test_size=0.2, 
                                            )
    
    df_train, df_test, scale = clean.normalize_scaler(df_train, df_test, target, MinMaxScaler())
    
    X_train = df_train.drop(str(target), axis=1).values.astype(np.float)
    
    min_ = np.array([min(i) for i in X_train.T])
    max_ = np.array([max(i) for i in X_train.T])
    
    compare_min = min_ == np.zeros_like(min_)
    compare_max = max_ == np.ones_like(max_)    
    
    assert compare_min.all()
   # assert compare_max.all() 
    
def test_normalize_scaler_standard():
    
    input_file = "simple_input.yaml"
    with open(str(input_file), 'r') as file:
        input_ = yaml.load(file, Loader=yaml.FullLoader)
    
    data = input_["data"]
    target = data["target_col_name"]
    
    # dataframe with ID, target, features 
    df = pd.read_csv(data["data_path"], header = 0, dtype=object, index_col=data["index_col_name"])
    
    df_train, df_test = train_test_split(df, 
                                         test_size=0.2, 
                                            )
    
    df_train, df_test, scale = clean.normalize_scaler(df_train, df_test, target, StandardScaler())
    
    X_train = df_train.drop(str(target), axis=1).values.astype(np.float)
        
    means = np.array([np.mean(i) for i in X_train.T])
    
    assert np.allclose(means, np.zeros_like(means), rtol=1e-5, atol=1e-5)

