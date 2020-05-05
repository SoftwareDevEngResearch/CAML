#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:01:50 2019

@author: mayermo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


def mk_feature_string(features, pca_n_comp):
    
    # save string of features for file naming
        feature_string = ''
        if len(features) < 3:
            for feature in features:
                feature_string = str(feature) + "_" + str(feature_string)
            feature_string = feature_string[:-1]
        else:
            feature_string = "{0}important_descr_funcgrps".format(len(features))
            
         # do principal component analysis if True/ pca_n_comp exists
        if pca_n_comp:           
            feature_string += "{0}PC_".format(pca_n_comp)
            
        return feature_string

def make_prop_df(descr_props, prop):
# dataframe with smiles, target
    try: # get target data from "smiles", either data file depending on target
        df = descr_props[[str(prop)]]
    except:
        raise ValueError("Not a valid property.")
    df = df.dropna()  # remove molecules with NaN prop value
    return df

def make_spectra_df(smiles_list):

    s = pd.read_csv('smiles-dict.csv', header=0)
    s = s.apply(lambda x: x.astype(str).str.lower()) #make all lowercase
    name_dict = dict(zip(s.smiles, s.Name))

    spectra_dict = {}
    for smiles in smiles_list:
        try:
            name = name_dict[str(smiles)]
            filepath = 'spectra/' + str(name) + '_absorbance.CSV'
            spectra = pd.read_csv(filepath, header = None, names=["wavelength", "absorbance"])
            abs_values = spectra.loc[:,'absorbance'].values #ignoring wavelengths, interpolation later?
            spectra_dict[str(smiles)] = [*abs_values]
        except:
            try:
                name = name_dict[str(smiles)]
                print('Fuel {0} not found in spectra folder'.format(name))
            except:
                pass #print('No spectra data for {0}'.format(smiles))

    wavelengths = spectra.loc[:,'wavelength'].values #assuming all wavelengths are the same
    col_names = [*wavelengths]
    df_spectra = pd.DataFrame.from_dict(spectra_dict, orient='index', columns=col_names)

    return df_spectra, wavelengths

def do_pca(df, target, n_comp):
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

    # Separating out the features
    features = list(df.columns[1:])
    x = df.loc[:, features].values
    smiles_list = list(df.index.values)

    # Separating out the target
   # y = df.loc[:,[str(target)]].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=n_comp)

    principalComponents = pca.fit_transform(x)

    col_list = []
    for c in range(1, 1 + n_comp):
        col_list.append("PC_" + str(c))

    principalDf = pd.DataFrame(data = principalComponents
             , columns = col_list, index = smiles_list)

    finalDf = pd.merge(df[[str(target)]], principalDf, how='outer', left_index=True, right_index=True) # merge along smiles index

    variance = pca.explained_variance_ratio_ # array

    return variance, finalDf

def make_df(target, descr_props, features, pca_n_comp):

    # make smiles-name dictionaries for use when spectra are included
    s = pd.read_csv('smiles-dict.csv', header=0)
    s = s.apply(lambda x: x.astype(str).str.lower()) #make all lowercase
 #   smiles_dict = dict(zip(s.Name, s.smiles))
   # name_dict = dict(zip(s.smiles, s.Name))

    df = make_prop_df(descr_props, target) # make dataframe with [fuel, target]
    smiles_list = list(df.index.values) # list of smiles that have target data

    if "spectra" in features: # make spectra dataframe
        df_spectra, wavelengths = make_spectra_df(smiles_list)
        df = pd.merge(df, df_spectra, how='outer', left_index=True, right_index=True)
        df = df.dropna()

    if "descriptors" in features: # make mordred descriptors dataframe
        df_descr = descr_props.loc[:, descr_props.columns.str.contains('mordred')]
        df = pd.merge(df, df_descr, how='outer', left_index=True, right_index=True) # merge along smiles index

        df = df.dropna(subset=[str(target)]) # drop row if target is missing from row

        df = df.apply(pd.to_numeric, args=('coerce',)) # change string values (errors from mordred) to NaN
        df = df.dropna(axis='columns') #drop columns if a value in a row is missing or null

        #remove columns where value is same for all molecules
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = list(nunique[nunique == 1].index)
        df = df.drop(cols_to_drop, axis=1)
        
    if "functional_groups" in features:
        df_fg = descr_props.loc[:, descr_props.columns.str.startswith('[')]
        df = pd.merge(df, df_fg, how='outer', left_index=True, right_index=True) # merge along smiles index
        
        df = df.dropna(subset=[str(target)]) # drop row if target is missing from row

        df = df.apply(pd.to_numeric, args=('coerce',)) # change string values (errors from mordred) to NaN
        df = df.dropna(axis='columns') #drop columns if a value in a row is missing or null

        #remove columns where value is same for all molecules
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = list(nunique[nunique == 1].index)
        df = df.drop(cols_to_drop, axis=1)
        
    for feature in features:
        if feature not in ["spectra", "descriptors", "functional_groups"]: # make df of smiles and global property feature
            try:
                df_prop = make_prop_df(descr_props, feature)
                df = pd.merge(df, df_prop, how='outer', left_index=True, right_index=True) # merge feature alond smiles index to main dataframe
                df = df.dropna(subset=[str(target), str(feature)]) # drop row if target or prop is missing from row
            except:
                print(f"Not valid feature: {feature}")

    df = df.apply(pd.to_numeric, args=('coerce',)) # change string values to NaN
    df = df.dropna(axis='columns') #drop columns if a value in a row is missing or null
    
    also_drop = [col for col in df.columns if "SpMAD" in str(col) and "SpMAD_DzZ" not in str(col)]
    
    # Drop features 
    df = df.drop(also_drop, axis=1)

    # do principal component analysis if True/ pca_n_comp exists
    if pca_n_comp:
        var, df = do_pca(df, target, pca_n_comp)
        print("PCA total explained valiance: ", sum(var))

    return df