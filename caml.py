""" This module is meant for control sequence """

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
import plot_ as plot_

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

from mpl_toolkits.mplot3d import Axes3D


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
    
    # make directory
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    
    # put copy of input request file into output folder    
    shutil.copy(input_file, path) 
     
    # load input file with yaml
    with open(str(input_file), 'r') as file:
        input_ = yaml.load(file, Loader=yaml.FullLoader)
    
    # dictionaries for each phase of ML process
    data = input_["data"]
    cleaning = input_["cleaning"]
    validation = input_["validation"]
    feature_transformations = input_["feature_transformations"]
    models = input_["models"]
    
    target = data["target_col_name"]
    seed = validation["random_seed"]
    
    output_comments = [] # error messages / modifications to output to file later, list of strings
    
    # dataframe with ID, target, features 
    df = pd.read_csv(data["data_path"], header = 0, dtype=object, index_col=data["index_col_name"])
    
##############################################
    # pre-split cleaning
    
    # handling nans, removing outliers before splitting into train, test, val
    
#################################################
    # split according to validation request, use random seed
        
    # split into train, test, try stratified
    if validation["stratified"] == True:
        try:
            stratified = True   
            df_train, df_test = train_test_split(df, 
                                                    test_size=validation["holdout_fraction"], 
                                                    random_state=seed,
                                                    stratify=df[str(target)].copy())
        except:
            stratified_error = "Could not stratify train/test split because least populated class has too few members. Proceeding without stratifying." 
            output_comments.append(stratified_error)
            print(stratified_error)
            
            df_train, df_test = train_test_split(df, 
                                                    test_size=validation["holdout_fraction"], 
                                                    random_state=seed
                                                    )
    else:
        df_train, df_test = train_test_split(df, 
                                                test_size=validation["holdout_fraction"], 
                                                random_state=seed
                                                )
        
    X = df.drop(str(target), axis=1).values.astype(np.float)
    y = df[str(target)].copy().values.astype(np.float)
    
    X_train = df_train.drop(str(target), axis=1).values.astype(np.float)
    y_train = df_train[str(target)].copy().values.astype(np.float)
    
    X_test = df_test.drop(str(target), axis=1).values.astype(np.float)
    y_test = df_test[str(target)].copy().values.astype(np.float)
    
    
###########################################
    # K fold splitting
        
    if "Kfold" in validation:
        
        try:
            K = validation["Kfold"]["nsplits"]
        except ValueError:
            print("nsplits for K-fold validation not given")
        
        if validation["Kfold"]["stratified"] == True: ### Fix for regression! use np.digitize for binning
            try:
                kf = StratifiedKFold(n_splits = K, shuffle=True, random_state=seed)
            except:
                kfold_error = "Could not stratify for K-fold splitting because least populated class has too few members. Proceeding without stratifying K-folds." 
                output_comments.append(kfold_error)
                print(kfold_error)
                
                kf = KFold(n_splits = K, shuffle=True, random_state=seed)
                
        elif validation["Kfold"]["stratified"] == False:
            kf = KFold(n_splits = K, shuffle=True, random_state=seed)
            
        
#########################################
    # plot requested distributions
    
    if 'overall distribution' in data["plot"]:
        
        target_train = df_train[str(target)].values.astype(np.float)
        target_test = df_test[str(target)].values.astype(np.float)
        
        labels = ['train', 'test']
        
        plt.hist([target_train, target_test], 
                 label=labels, stacked=True) # add auto bin number
        
        plt.ylabel('frequency')
        plt.xlabel(f"{target}")
        plt.legend(loc='upper right')
        plt.savefig(f"{path}/overall_dist.png")
        plt.clf()
        
    if 'train distribution' in data["plot"]:
        
        target_train = df_train[str(target)].values.astype(np.float)
        plt.hist([target_train]) 
        
        plt.ylabel('frequency')
        plt.xlabel(f"{target}")
        plt.savefig(f"{path}/train_dist.png")
        plt.clf()
        
    if 'test distribution' in data["plot"]:
        
        target_test = df_test[str(target)].values.astype(np.float)
        plt.hist([target_test]) # bins = 10,
        
        plt.ylabel('frequency')
        plt.xlabel(f"{target}")
        plt.savefig(f"{path}/test_dist.png")
        plt.clf()
        
    if "split distribution" in data["plot"]:
        
        if "Kfold" not in validation:
            plot_error = "Cannot plot split distribution if not using K-fold CV in validation"
            output_comments.append(plot_error)
            print(plot_error)
            
        else:
            
            folds = []
            try: # stratified kfold .split() takes X, y
                for train_index, test_index in kf.split(X_train, y_train):
                    y_test_ = y_train[test_index]
                    folds.append(y_test_)
            except: # kfold .split() takes only X
                for train_index, test_index in kf.split(X_train):
                    y_test_ = y_train[test_index]
                    folds.append(y_test_)
                    
            labels = [f"split {i+1}" for i in range(len(folds))]
            plt.hist(folds,
                     label=labels,
                     stacked = True)
            plt.ylabel('frequency')
            plt.xlabel(f"{target}")
            plt.legend(loc='upper right')
            plt.savefig(f"{path}/split_dist.png")
            plt.clf()
                        
       
    # plot 3D spectra
    if "3D spectra_" in data["plot"]:
        
        # To add: sort by target value, color gradient for target value or make target = y_
        # save higher quality image, fix fontsize / use tight layout, make interactive/moving
        # add train, test, split options to plot
        # add fuel names maybe? or remove numbers from y axis
        
        fig = plt.figure()    
        ax = fig.add_subplot(111, projection='3d')
        
        features = list(df.columns)
        features.remove(target)
        features = np.asarray(features)
        
        x_ = features.copy().astype(float)
        y_ = np.arange(len(df)).astype(float)
        
        X_,Y_ = np.meshgrid(x_,y_)
        Z = X.copy().astype(float)
            
        ax.plot_surface(X_, Y_, Z, rstride=1, cstride=1000, shade=True, lw=.5) 
         
#        ax.set_zlim(0, 5)
#        ax.set_xlim(-51, 51)
        ax.set_zlabel("Intensity")
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Fuel")
#        ax.view_init(20,-120)
        plt.savefig(f"{path}/3D_spectra.png")
        
        
    # plot heatmap of correlated features- spearman and pearson correlation coefficients

##############################################
    # post-split cleaning

    if cleaning["post_split_cleaning"]['normalize'] == "overall min max":        
        df_train, df_test, scale = clean.normalize_overall_min_max(df_train, df_test, target)
       
    elif cleaning["post_split_cleaning"]['normalize'] == "min max":        
        df_train, df_test, scale = clean.normalize_min_max(df_train, df_test, target, MinMaxScaler())
        
    elif cleaning["post_split_cleaning"]['normalize'] == "standard scaler":        
        df_train, df_test, scale = clean.normalize_minmax_scaler(df_train, df_test, target, StandardScaler())

##############################################
    # do all feature transforms in list, save in list of dataframes
    
    df_trains_transformed = []
    df_tests_transformed = []
    for i, transformation in enumerate(feature_transformations):
        
        if "PCA" in transformation:
            try:
                n_components = transformation["PCA"]["number"]
                if n_components == 'all':
                    n_components = int(len(df_train) - 1)
                n_components = int(n_components)
            except ValueError:
                print("Number of PC's not specified")
        
            df_train_transformed, df_test_transformed = feat.do_pca(df_train, df_test, n_components, target)
        
        elif "peaks" in transformation:
            # use scipy.find_peaks()
            pass
        
        else:
            df_train_transformed, df_test_transformed = df_train, df_test # no transformation
        
        df_trains_transformed.append(df_train_transformed)
        df_tests_transformed.append(df_test_transformed)
        
#####################################################
    # tune and train models
    
    tuned_models = []
    test_predictions = []
    
    for model in models:
        
        if model in ["Lasso"]:
            
            parameters = [{'alpha': np.arange(1, 300, 1)}]
            best_model = GridSearchCV(Lasso(), parameters, cv=kf)
            best_model.fit(X_train, y_train)
    
            tuned_model = best_model.best_estimator_
            tuned_model.fit(X_train, y_train)
            tuned_models.append(tuned_model)
            
            predictions = tuned_model.predict(X_test)
            test_predictions.append(predictions)

#####################################################
    # evaluate test performance
    
    test_errors = []
    
    if validation["metric"] == "MAE":
        
        for i, model in models:
            
            errors = np.abs(y_test - test_predictions[i])
            test_errors.append(errors)
            
#    elif validation["metric"] == "RMSE":
#        pass
    
#####################################################
    # make plots

    for i in range(len(models)):
        
        plot_.abs_error_hist(test_errors[i], models[i], target, path)
        plot_.test_parity_plot(y_test, test_predictions[i], test_errors[i], models[i], target)
    
#####################################################
    # output messages to file
    
    with open(f"{path}/log.txt", "w") as f:
        
        for line in output_comments:
            f.write(line)
        
#####################################################
    
    #    for train_index, test_index in kf.split(X):
#      #  print("TRAIN:", train_index, "TEST:", test_index)
#        X_train, X_test = X[train_index], X[test_index]
#        y_train, y_test = y[train_index], y[test_index]
            
    # test, train, validation or test, k folds
    # save indices ?
    
#    y_train = df_train[str(target)].copy().values.astype(np.float)
#    X_train = df_train.drop(str(target), axis=1).values.astype(np.float)
        
   
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
