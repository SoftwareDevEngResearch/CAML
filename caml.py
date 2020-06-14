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
from tpot import TPOTRegressor
from pickle import dump
from scipy.signal import find_peaks

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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

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
    
    # for testing, removes old request folders
    for fname in os.listdir(os.getcwd()):
        if fname.startswith("request"):
            shutil.rmtree(os.path.join(os.getcwd(), fname))
                
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
     
    # load input file with yaml
    with open(str(input_file), 'r') as file:
        input_ = yaml.load(file, Loader=yaml.FullLoader)
        
    # put copy of input request file into output folder    
    shutil.copy(input_file, path) 
    
    # dictionaries for each phase of ML process
    data = input_["data"]
    cleaning = input_["cleaning"]
    validation = input_["validation"]
    feature_transformations = input_["transformations"]["feature_transformations"]
    transform_names = input_["transformations"]["transform_names"]
    models = input_["models"]
    parameters_file = input_["validation"]["parameters_file"]
    
    # load input file with yaml
    with open(str(parameters_file), 'r') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
        
    # put copy of parameter file into output folder    
    shutil.copy(parameters_file, path) 

        
    target = data["target_col_name"]
    seed = validation["random_seed"]
    
    output_comments = [] # error messages / modifications to output to file later, list of strings
    
    # dataframe with ID, target, features 
    df = pd.read_csv(data["data_path"], header = 0, dtype=object, index_col=data["index_col_name"])
    
    
##############################################
    # pre-split cleaning - handling nans, removing outliers before splitting into train, test, val
    
    if df.isnull().values.any() == True: # if there is a missing value- feature or target
        
        if cleaning["pre_split_cleaning"]["nan"] == "remove example":
            df = df.dropna(how='any')
            
        elif cleaning["pre_split_cleaning"]["nan"] == "remove feature":
            df = df.dropna(axis='columns')
        
        else:
            # use imputer function from sklearn
            strategy = cleaning["pre_split_cleaning"]["nan"]
            try:
                imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
                df = imp.fit_transform(df)
            except:
                imputer_error = f"pre-split cleaning method {strategy} not valid"
                output_comments.append(imputer_error)
                print(imputer_error)
    
#################################################
    # split according to validation request, use random seed
        
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
        
        kf = KFold(n_splits = K, shuffle=True, random_state=seed)
            
        
#########################################
    # data plots
    
    data_plot_path = f"{path}/data_plots"
    os.mkdir(data_plot_path)
    
    if 'overall distribution' in data["plot"]:
        
        plot_.overall_dist(df_train, df_test, target, data_plot_path)
        
    if 'train distribution' in data["plot"]:
        
        plot_.set_dist(df_train, target, 'train', data_plot_path)
        
    if 'test distribution' in data["plot"]:
        
        plot_.set_dist(df_test, target, 'test', data_plot_path)
        
    if "Kfold distribution" in data["plot"]:
        
        if "Kfold" not in validation:
            plot_error = "Cannot plot K-fold distribution if not using K-fold CV in validation"
            output_comments.append(plot_error)
            print(plot_error)
            
        else:            
            plot_.kfold_dist(kf, target, X_train, y_train, data_plot_path)  
       
    if "3D spectra" in data["plot"]:
        
        plot_.spectra_3D(df, target, data_plot_path)
        
    if "2D spectra" in data["plot"]:
        
        plot_.spectra_2D(df_train, target, data_plot_path, "train")
        plot_.spectra_2D(df_test, target, data_plot_path, "test")
        
    if "all spectra" in data["plot"]:
        
        plot_.all_spectra(df, target, data_plot_path)

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
    scaler_objects = []  # save scaler objects to output
    
    for i, transformation in enumerate(feature_transformations):
     
        if "PCA" in transformation:
         
            try:
                n_components = transformation["PCA"]["number"]
                if n_components == 'all':
                    n_components = int(len(df_train) - 1)
                n_components = int(n_components)
            except ValueError:
                pca_error = "Number of PC's not specified"
                output_comments.append(pca_error)
                print(pca_error)
        
            df_train_transformed, df_test_transformed, scaler = feat.do_pca(df_train, df_test, n_components, target)
            
            plot_.PC_spectra(df_train_transformed, target, data_plot_path, f"train_{transform_names[i]}")
            plot_.PC_spectra(df_test_transformed, target, data_plot_path, f"test_{transform_names[i]}")

        elif "smoothing" in transformation:
            
            try:
                n_points = transformation["smoothing"]["n_points"]
            except ValueError:
                smoothing_error = "Number of points to use for smoothing not specified"
                output_comments.append(smoothing_error)
                print(smoothing_error)
                
            df_train_transformed = feat.smooth_spectra(df_train, target, n_points)
            df_test_transformed = feat.smooth_spectra(df_test, target, n_points)
            scaler = transformation # better way to do this for saving/evaluation ?
            
            plot_.spectra_2D(df_test_transformed, target, data_plot_path, f"test_{transform_names[i]}")
        
        elif "peaks" in transformation:
            # use scipy.find_peaks()
            pass
        
        elif "None" in transformation: # no transformation
            df_train_transformed, df_test_transformed = df_train, df_test
            scaler = 0
            
        else:
            raise ValueError("Not valid transformation")
            
        scaler_objects.append(scaler)
        df_trains_transformed.append(df_train_transformed)
        df_tests_transformed.append(df_test_transformed)
        
#####################################################
    # tune and train models
    
    all_tuned_models = []
    all_test_predictions = []
    all_train_predictions = []

    for t, transform in enumerate(feature_transformations):
        
        X_train_ = df_trains_transformed[t].drop(str(target), axis=1).values.astype(np.float)
        X_test_ = df_tests_transformed[t].drop(str(target), axis=1).values.astype(np.float)
        
        tuned_models = []
        test_predictions = []
        train_predictions = []
        
        for i, model in enumerate(models):
            
            model_path = f"{path}/{model}_{transform_names[t]}"
            os.mkdir(model_path)
            
            if model in ["Lasso"]:
                
                parameters_ = [parameters["Lasso"]]
                best_model = GridSearchCV(Lasso(), parameters_, cv=kf) # test that indices match
                best_model.fit(X_train_, y_train)    
                tuned_model = best_model.best_estimator_
                
            elif model in ["ElasticNet"]:
                
                parameters_ = [parameters["ElasticNet"]]
                best_model = GridSearchCV(ElasticNet(), parameters_, cv=kf) # test that indices match
                best_model.fit(X_train_, y_train)    
                tuned_model = best_model.best_estimator_
                
            elif model in ["RandomForest"]:
            
                parameters_ = parameters["RandomForest"]["parameters"]
                n_iter_rf = parameters["RandomForest"]["n_iter"]
                
                best_model = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = parameters_, 
                               n_iter = n_iter_rf, cv = kf, verbose=2, random_state=seed, n_jobs = -1, refit=True)
                best_model.fit(X_train, y_train)
                tuned_model = best_model.best_estimator_
                
            elif model in ["AdaBoost"]:
                
                parameters_ = parameters["AdaBoost"]["parameters"]
                n_iter_ab = parameters["AdaBoost"]["n_iter"]
                
                best_model = RandomizedSearchCV(estimator = AdaBoostRegressor(), param_distributions = parameters_, 
                               n_iter = n_iter_ab, cv = kf, verbose=2, random_state=seed, n_jobs = -1, refit=True)
                best_model.fit(X_train, y_train)
                tuned_model = best_model.best_estimator_
                
            elif model in ["PLS"]:
                
                parameters_ = [parameters["PLS"]]
                
                best_model = GridSearchCV(PLSRegression(), parameters_, cv=kf) # test that indices match
                best_model.fit(X_train_, y_train)    
                tuned_model = best_model.best_estimator_
                
            elif model in ["LinearRegression"]:
                
                tuned_model = LinearRegression()
                tuned_model.fit(X_train_, y_train) 
                
            elif model in ["TPOT", "tpot"]:
                                
                tpot = TPOTRegressor(generations=parameters["TPOT"]["generations"], 
                                     population_size=parameters["TPOT"]["population_size"], 
                                     verbosity=parameters["TPOT"]["verbosity"], 
                                     random_state=seed, 
                                     max_time_mins = parameters["TPOT"]["max_time_mins"], 
                                     n_jobs=parameters["TPOT"]["n_jobs"])
                
                tpot.fit(X_train_, y_train)                
                tpot.export(f'{model_path}/{model}_pipeline.py')
                
            elif model in ["dummy_average"]:
                
                tuned_model = DummyRegressor(strategy="mean")
                
            else: 
                model_error = f'{model} model not supported'
                output_comments.append(model_error)
                raise NameError(model_error)
                tuned_model = None
                
            try: 
                tuned_model.fit(X_train_, y_train)
                tuned_models.append(tuned_model)    
                
                test_predictions_m = tuned_model.predict(X_test_)
                test_predictions.append(test_predictions_m)
                
                train_predictions_m = tuned_model.predict(X_train_)
                train_predictions.append(train_predictions_m)
                
                # save the model 
                dump(tuned_model, open(f'{model_path}/model.pkl', 'wb'))

                # save the scaler
                try: 
                    dump(scaler_objects[t], open(f'{model_path}/scaler.pkl', 'wb'))
                except:
                    print("No scaler") # find way to save homemade scalers
                            
            except:
                model_error = "Could not tune and train {model} model"
                output_comments.append(model_error)
                raise ValueError(model_error)
                pass # add placeholder for undefined model in list ?

        all_tuned_models.append(tuned_models)
        all_test_predictions.append(test_predictions)
        all_train_predictions.append(train_predictions)


#####################################################
    # calculate test performance, absolute error
    
    all_test_errors = []
    all_train_errors = []
    
    all_test_performances = []
    all_train_performances = []
    
    for t, transform in enumerate(feature_transformations):
        
        test_errors = []
        train_errors = []
        
        test_performances = []
        train_performances = []
        
        for i, model in enumerate(models):
            
            test_errors_m = np.abs(y_test - all_test_predictions[t][i])
            test_errors.append(test_errors_m)
            
            train_errors_m = np.abs(y_train - all_train_predictions[t][i])
            train_errors.append(train_errors_m)
            
            test_performances.append(np.mean(test_errors_m))
            train_performances.append(np.mean(train_errors_m))
            
        all_test_errors.append(test_errors)
        all_train_errors.append(train_errors)
        
        all_test_performances.append(test_performances)
        all_train_performances.append(train_performances)
            
    
#####################################################
    # make performance plots
    
    for t, transform in enumerate(feature_transformations):
        
    #    plot_.bar_performances_by_algorithm(all_train_performances[t], all_test_performances[t], models, 
     #                            target, transform_names[t], path)
        
        plot_.box_performances_by_algorithm(all_train_errors[t], all_test_errors[t], models, 
                                 target, transform_names[t], path)

        for i in range(len(models)):
            
            model_path = f"{path}/{models[i]}_{transform_names[t]}"
            
            plot_.abs_error_hist(all_test_errors[t][i], models[i], transform_names[t], target, model_path, "Test")
            plot_.abs_error_hist(all_train_errors[t][i], models[i], transform_names[t], target, model_path, "Train")
            
            plot_.parity_plot(y_test, all_test_predictions[t][i], models[i], transform_names[t], target, model_path, "Test")
            plot_.parity_plot(y_train, all_train_predictions[t][i], models[i], transform_names[t], target, model_path, "Train")
            plot_.train_test_parity_plot(y_test, all_test_predictions[t][i], 
                                         y_train, all_train_predictions[t][i], 
                                         models[i], transform_names[t], target, 
                                         model_path)
            
    for i in range(len(models)):
        
        train_performances_ = [x[i] for x in all_train_performances]
        test_performances_ = [x[i] for x in all_test_performances]
    #    plot_.bar_performances_by_transform(train_performances_, test_performances_,
    #                                    models[i], transform_names, target, path)

        train_errors_ = [x[i] for x in all_train_errors]
        test_errors_ = [x[i] for x in all_test_errors]
        plot_.box_performances_by_transform(train_errors_, test_errors_,
                                        models[i], transform_names, target, path)
                    
#####################################################
        
    # output error messages to file
    
    with open(f"{path}/log.txt", "w") as f:
        
        for line in output_comments:
            f.write(line)
        
#####################################################