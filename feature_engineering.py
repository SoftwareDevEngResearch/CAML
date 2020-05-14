import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

""" This module is meant for feature engineering and dimensionality reduction methods """

def do_pca(df_train, df_test, n_components, target):
    """This function takes in train and test dataframes and performs PCA to reduce dims.
    It fits on the train and transforms the test df based on the train."""
    
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
    return df_train_transformed, df_test_transformed


def do_rfe(df_train, df_test, n_components, target):
    
    """This function takes in train and test dataframes and performs 
    Recursive Feature Elimination to reduce dims. It removes features
    based on the train and transforms the test df based on the train."""
    
    return df_train_transformed, df_test_transformed