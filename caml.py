import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
from datetime import datetime
import os
import shutil
import seaborn as sns

# caml modules
import clean
import feature_engineering as feat
import models as mod

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

""" This module is meant for control sequence """


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str,
                        help='name of input yaml file')

    args = parser.parse_args()
    input_file = args.input_file
    
    # unique ID is datetime
    id_num = str(datetime.now())[2:19]
    id_num = id_num.replace(':', '').replace(' ', '').replace('-', '')
    
    # define the name of the directory to be created
    path = os.getcwd() + "/request_" + str(id_num)
    
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
        
    shutil.copy(input_file, path) 
     
    with open(str(input_file), 'r') as file:
        input_ = yaml.load(file, Loader=yaml.FullLoader)
    
    data = input_["data"]
    cleaning = input_["cleaning"]
    validation = input_["validation"]
    feature_transformations = input_["feature_transformations"]
    models = input_["models"]
    
    target = data["target_col_name"]
    seed = validation["random_seed"]
    
    # dataframe with ID, target, features 
    df = pd.read_csv(data["data_path"], header = 0, dtype=object, index_col=data["index_col_name"])
    
    # split by validation request, use random seed
    if validation["stratified"] == True:
        stratify = df[str(target)].copy()
        stratified = True
    else:
        stratified = False
        
    # split into train, test, try stratified
    if stratified == True:
        try:
            df_train, df_test = train_test_split(df, 
                                                    test_size=validation["holdout_fraction"], 
                                                    random_state=validation["random_seed"],
                                                    stratify=stratify)
        except:
            print("Could not stratify train/test split because least populated class cannot have less than 2 members")
            df_train, df_test = train_test_split(df, 
                                                    test_size=validation["holdout_fraction"], 
                                                    random_state=seed
                                                    )
    else:
        df_train, df_test = train_test_split(df, 
                                                test_size=validation["holdout_fraction"], 
                                                random_state=seed
                                                )

    # plot requested distributions
    if 'overall distribution' in data["plot"]:
        
        target_train = df_train[str(target)].values.astype(np.float)
        target_test = df_test[str(target)].values.astype(np.float)
        
        labels = ['train', 'test']
        
        plt.hist([target_train, target_test], 
                 bins = 10, label=labels, stacked=True) # add auto bin number
        
        plt.ylabel('frequency')
        plt.xlabel(f"{target}")
        plt.legend(loc='upper right')
        plt.savefig(f"{path}/overall_dist.png")
        plt.clf()
        
    if 'train distribution' in data["plot"]:
        
        target_train = df_train[str(target)].values.astype(np.float)
        plt.hist([target_train], 
                 bins = 10, label=labels, stacked=True) # add auto bin number
        
        plt.ylabel('frequency')
        plt.xlabel(f"{target}")
        plt.savefig(f"{path}/train_dist.png")
        plt.clf()

    # k folds for train set:
    K = validation["Kfold"]["nsplits"]
    y_train = df_train[str(target)].copy().values
    X_train = df_train.drop(str(target), axis=1).values
    
    kf = KFold(n_splits = K, shuffle=True, random_state=seed)

    # clean, normalize train
    if cleaning['normalize'] == "overall min max":        
        df_train, df_test, scale = clean.normalize_overall_min_max(df_train, df_test, target)
       
    elif cleaning['normalize'] == "min max":        
        df_train, df_test, scale = clean.normalize_min_max(df_train, df_test, target, MinMaxScaler())
        
    elif cleaning['normalize'] == "standard scaler":        
        df_train, df_test, scale = clean.normalize_minmax_scaler(df_train, df_test, target, StandardScaler())
        
#    if stratified == True:
#        kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
#        for train_index, test_index in kf.split(X, y):
#            print("TRAIN:", train_index, "TEST:", test_index)
#            X_train, X_test = X[train_index], X[test_index]
#            y_train, y_test = y[train_index], y[test_index]
#    else:

    # do all feature transforms in list, save in list of dataframes
    df_trains_transformed = []
    df_tests_transformed = []
    for i, transformation in enumerate(feature_transformations):
        
        # do normalizing, then dim reduction
        if "dim_reduction" in transformation:
            if "PCA" in transformation["dim_reduction"]:
                try:
                    n_components = transformation["dim_reduction"]["PCA"]["number"]
                    if n_components == 'all':
                        n_components = int(len(df_train) - 1)
                    n_components = int(n_components)
                except ValueError:
                    print("Number of PC's not specified in dim reduction")
            
                df_train_transformed, df_test_transformed = feat.do_pca(df_train, df_test, n_components, target)
            
            else:
                df_train_transformed, df_test_transformed = df_train, df_test # no transformation
            
            df_trains_transformed.append(df_train_transformed)
            df_tests_transformed.append(df_test_transformed)
        
        
    #    for train_index, test_index in kf.split(X):
#      #  print("TRAIN:", train_index, "TEST:", test_index)
#        X_train, X_test = X[train_index], X[test_index]
#        y_train, y_test = y[train_index], y[test_index]

            
    # test, train, validation or test, k folds
    # save indices ?
    

    # plot target distributions of train and test stacked
    
    # plot 3D spectra
   
    # for regular sklearn model:
    # scale train data, transform validation set
    # if k-fold, scale k-1 train folds within each train fold
    
    
    

#=============================================================
    # read input file
    # make folder for this case study with unique #
    
    # overall data cleaning
    # data visualization of target data

    # possibly divide nodes for each case?
    # for each request/case:
    # - cleaning
    # - feature engineering
    # - data visualize
    # - split into train, (optional- validate), and test
    # - train model or TPOT search
    # - save each trained model as pkl file with associated metadata about input transformations
    # - plot train and test results
    # - output results summary file

    # Case study plots
    # case study output file
