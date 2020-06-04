""" This module is meant for data cleaning """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def normalize_overall_min_max(df_train, df_test, target):
    ''' This function takes the train and test dataframes and 
    normalizes all values by the overall in and max'''
    
    X_train = df_train.drop(str(target), axis=1).values.astype(np.float)
    X_test = df_test.drop(str(target), axis=1).values.astype(np.float)
    
    features = list(df_train.columns)
    features.remove(target)
    index_list_train = list(df_train.index.values)
    index_list_test = list(df_test.index.values)

    min_ = np.amin(X_train, axis=None, out=None)
    max_ = np.amax(X_train, axis=None, out=None)
    
    X_train_norm = (X_train - min_) / (max_ - min_)
    
    # scale test set accordingly
    X_test_norm = (X_test - min_) / (max_ - min_)
    
    scale = [min_, max_]
    
    df_x_train = pd.DataFrame(data = X_train_norm
             , columns = features, index = index_list_train)
    df_x_test = pd.DataFrame(data = X_test_norm
             , columns = features, index = index_list_test)
    
    df_train_norm = pd.merge(df_train[[str(target)]], df_x_train, how='outer', left_index=True, right_index=True)
    df_test_norm = pd.merge(df_test[[str(target)]], df_x_test, how='outer', left_index=True, right_index=True)
    
    return df_train_norm, df_test_norm, scale

def normalize_scaler(df_train, df_test, target, scale_method):
    ''' This function takes the train and test dataframes and
    normalizes each individual feature by its mean and standard deviation.
    sclae mehtod may be MinMaxScaler or StandardScaler'''
    
    X_train = df_train.drop(str(target), axis=1).values.astype(np.float)
    X_test = df_test.drop(str(target), axis=1).values.astype(np.float)
    
    features = list(df_train.columns)
    features.remove(target)
    index_list_train = list(df_train.index.values)
    index_list_test = list(df_test.index.values)

    scaler = scale_method   
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    df_x_train = pd.DataFrame(data = X_train_norm
             , columns = features, index = index_list_train)
    df_x_test = pd.DataFrame(data = X_test_norm
             , columns = features, index = index_list_test)
    
    df_train_norm = pd.merge(df_train[[str(target)]], df_x_train, how='outer', left_index=True, right_index=True)
    df_test_norm = pd.merge(df_test[[str(target)]], df_x_test, how='outer', left_index=True, right_index=True)
    
    return df_train_norm, df_test_norm, scaler

def smooth_spectra(df, target, n):
    """ This function takes in a spectra dataframe and smooths noise by replacing each 
    intensity measurement with an average of the 'n' points to the left and right of it.
    Returns a dataframe of same dimensions. End points are averaged with whatever is around 
    if it exists. """
    
    X = df.drop(str(target), axis=1).values.astype(np.float)
    
    features = list(df.columns)
    features.remove(target)
    index_list= list(df.index.values)
    
    smoothed_spectra = np.array(X)
    
    for s, spectra in enumerate(X):
    
        for i in range(len(spectra)):
            
            if i < n:
                intensity = np.mean(spectra[0:i+n])
            else:
                try:
                    intensity = np.mean(spectra[i-n : i+n])
                except: 
                    intensity = np.mean(spectra[i-n : -1])
                
            smoothed_spectra[s][i] = intensity
            
    df_x_smoothed = pd.DataFrame(data = smoothed_spectra, 
                                 columns = features, index = index_list)
    
    df_smoothed= pd.merge(df[[str(target)]], df_x_smoothed, how='outer', left_index=True, right_index=True)
   
    return df_smoothed

def remove_outliers():
    """ This function takes in the train dataframe and removes outliers 
    based on a threshold """
    
    return









#