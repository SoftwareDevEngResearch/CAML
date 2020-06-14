" This module tests the feature_engineering module. "

import pytest
import feature_engineering as feat
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
from datetime import datetime
import os
import shutil
from sklearn.model_selection import train_test_split

def test_do_pca():

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
    
    X_train = df_train.drop(str(target), axis=1).values.astype(np.float)

    n_components = 5
    
    df_train_transformed, df_test_transformed, scaler = feat.do_pca(df_train, df_test, n_components, target)
    
    n_cols = len(df_train_transformed.columns)
    
    assert (n_cols - 1) == n_components
            
def test_smooth_spectra():
    
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
    
    n_points = 10
    
    X_train = df_train.drop(str(target), axis=1).values.astype(np.float)
    
    df_train_transformed = feat.smooth_spectra(df_train, target, n_points)
    
    X_train_smooth = df_train_transformed.drop(str(target), axis=1).values.astype(np.float)
    
    spectra = X_train[0]
    # check random point:
    i = 42
    intensity = np.mean(spectra[i-n_points : i+n_points])
            
    assert intensity == X_train_smooth[0][42]
    
