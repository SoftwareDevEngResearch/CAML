""" This module is meant for feature engineering and dimensionality reduction methods """

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from scipy.signal import find_peaks


def do_pca(df_train, df_test, n_components, target):
    """This function takes in train and test dataframes and performs PCA to reduce dims.
    It fits on the train and transforms the test df based on the train.
    ----------
        df_train : pandas dataframe
            dataframe of train set
        df_test : pandas dataframe
            dataframe of test set
        n_components : integer
            number of principal components to include in PCA transform.
            Max is one less than the total number of data points
        target : string
            name of target column in dataframes
    """
    
    features = list(df_train.columns)
    features.remove(target)
    x_train = df_train.loc[:, features].values.astype(np.float)
    x_test = df_test.loc[:, features].values.astype(np.float)
    index_list_train = list(df_train.index.values)
    index_list_test = list(df_test.index.values)
    
    pca = PCA(n_components=n_components)
        
    principalComponents_train = pca.fit_transform(x_train)

    col_list = []
    for c in range(1, 1 + n_components):
        col_list.append("PC_" + str(c))

    principalDf_train = pd.DataFrame(data = principalComponents_train
             , columns = col_list, index = index_list_train)
    
    principalComponents_test = pca.transform(x_test)
    
    principalDf_test = pd.DataFrame(data = principalComponents_test
             , columns = col_list, index = index_list_test)

    df_train_transformed = pd.merge(df_train[[str(target)]], principalDf_train, how='outer', left_index=True, right_index=True)
    df_test_transformed = pd.merge(df_test[[str(target)]], principalDf_test, how='outer', left_index=True, right_index=True)
        
#   variance = pca.explained_variance_ratio_
    return df_train_transformed, df_test_transformed, pca

def smooth_spectra(df, target, n):
    """ This function takes in a spectra dataframe and smooths noise by replacing each 
    intensity measurement with an average of the 'n' points to the left and right of it.
    Returns a dataframe of same dimensions. End points are averaged with whatever is around 
    if it exists.    
    ----------
        df : pandas dataframe
            dataframe of entire dataset
        target : string
            name of target column in dataframes
        n : integer
            number of points to average to the left and right of each point    
    """
    
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

def do_rfe(df_train, df_test, n_components, target):
    
    """This function takes in train and test dataframes and performs 
    Recursive Feature Elimination to reduce dims. It removes features
    based on the train and transforms the test df based on the train.
    
    [in construction]
    
    """
    
    return df_train_transformed, df_test_transformed