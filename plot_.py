""" This module includes plotting functions. """

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 
from matplotlib import cm
plt.tight_layout(h_pad = 3, w_pad=3)

####################################

def set_dist(df, target, set_, path):
    """This function takes in a dataframe and plots the distribution 
    of the target column. It is saved to the path
    ----------
        df : pandas dataframe
            dataframe of set to plot distribution
        target : string
            name of target column in dataframes
        set_ : string
            name of split. "train" or "test". Used for file naming
        path : string
            path where png file will be stored
    """
    
    y = df[str(target)].values.astype(np.float)
    plt.hist([y]) 
    
    plt.ylabel('frequency')
    plt.xlabel(f"{target}")
    plt.grid(axis = 'y') 
    plt.savefig(f"{path}/{set_}_dist.png", bbox_inches='tight', dpi=1200)
    plt.clf()

def overall_dist(df_train, df_test, target, path):    
    """This function takes in the train and test dataframes and plots both 
    target distributions stacked in a histogram, It is saved to the path
    ----------
        df_train : pandas dataframe
            dataframe of train set
        df_test : pandas dataframe
            dataframe of test set
        target : string
            name of target column in dataframes
        path : string
            path where png file will be stored
    """
    
    target_train = df_train[str(target)].values.astype(np.float)
    target_test = df_test[str(target)].values.astype(np.float)
    
    labels = ['train', 'test']
    
    plt.hist([target_train, target_test], 
             label=labels, stacked=True) # add auto bin number
    
    plt.ylabel('frequency')
    plt.xlabel(f"{target}")
    plt.legend(loc='upper right')
    plt.grid(axis = 'y') 
    plt.savefig(f"{path}/overall_dist.png", bbox_inches='tight', dpi=1200)
    plt.clf()
    
def kfold_dist(kf, target, X_train, y_train, path):
    """This function takes in the train data series and K-fold indices for 
    plotting the fold distribution. It is saved to the path
    ----------
        kf : skearn Kfold object
        X_train : 2D series
            feature data of train set
        y_train : 1D series
            target data of train set
        target : string
            name of target column in dataframes
        path : string
            path where png file will be stored
    """
    
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
    plt.grid(axis = 'y') 
    plt.savefig(f"{path}/kfold_dist.png", bbox_inches='tight', dpi=1200)
    plt.clf()
    
def spectra_3D(df, target, path):    
    """This function takes in a dataframe and plots the spectra in a 3D plot.
    It is saved to the path. [in construction]
    ----------
        df : pandas dataframe
            dataframe of spectra
        target : string
            name of target column in dataframes
        path : string
            path where png file will be stored
    """
    
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
    Z = df.drop(str(target), axis=1).values.astype(np.float)
        
    ax.plot_surface(X_, Y_, Z, rstride=1, cstride=1000, shade=True, lw=.1, alpha=0.4) 
     
    ax.set_zlabel("Intensity")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Fuel")
#        ax.view_init(20,-120)
    plt.savefig(f"{path}/3D_spectra.png", dpi=1200)
    plt.clf()
    
def spectra_2D(df, target, path, label):
    """This function takes in a spectra dataframe and plots the 
    staggered spectra on a 2D plot. It is saved to the path
    ----------
        df : pandas dataframe
            dataframe of set
        target : string
            name of target column in dataframes
        path : string
            path where png file will be stored
        label : string
            name of set. "train" or "test" for file naming.
    """
    
    X = df.drop(str(target), axis=1).values.astype(np.float)
   # y = df[str(target)].copy().values.astype(np.float)
    
    features = list(df.columns)
    features.remove(target)
    features = np.asarray(features).astype(np.float)
    
    for i, example in enumerate(X):
        
        intensity = X[i] + 0.2*i
        plt.plot(features, intensity, alpha = 0.5, linewidth=1, c='k')
        
    plt.ylabel("Intensity")
    plt.yticks([])
    plt.xlabel("Wavenumber")
    plt.savefig(f"{path}/spectra_2D_{label}.png", bbox_inches='tight', dpi=1200)
    plt.clf()

#####################################

def abs_error_hist(abs_errors, model_name, transform_name, target, path, dataset):
    """This function takes in the absolute errors from model predictions
    and plots the distribution of absolute error on a histogram. It is saved to the path
    ----------
        abs_errors : list 
            absolute errors
        model_name : string
            name of algorithm in yaml input file
        transform_name : string
            name of transform specified in transform_names in input file
        target : string
            name of target column in dataframes
        path : string
            path where png file will be stored
        dataset : string
            name of set. "train" or "test" for file naming.
    """

    plt.hist(abs_errors)
    
    plt.xlabel("Absolute error")
    plt.ylabel('Frequency')
    plt.savefig(f"{path}/{target}_{model_name}_{transform_name}_{dataset}_error_hist.png", bbox_inches='tight', dpi=1200)
    plt.clf()
    
    
def parity_plot(y, predictions, model_name, transform_name, target, path, dataset):
    '''  This function takes in the true and predicted values and plots a scatter
    plot alongside a y=x line for one dataset- train or test.
    ------------
    y : array-like
        the true values
    predictions : array-like
        the predicted values from evaluating the trianed model
    model_name : string
        name of algorithm in yaml input file
    transform_name : string
        name of transform specified in transform_names in input file
    target : string
        name of target column in dataframes
    path : string
        path where png file will be stored
    dataset : string
        name of set. "train" or "test" for file naming.
    '''
    
    abs_errors = np.abs(y - predictions)
    min_ = min([min(y), min(predictions)])
    max_ = max([max(y), max(predictions)])
    
    if dataset == "Train":
        color='r'
        marker="o"
    else: # dataset =="Test":
        color='b'
        marker="*"

    label = f"{dataset}, Average absolute error: {np.mean(abs_errors): .2f}"
    plt.scatter(y, predictions, facecolors='none', edgecolors=color, marker=marker, s= 80, label=label)
    plt.plot([min_, max_],[min_, max_])
    plt.grid()

    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.legend()
    plt.savefig(f"{path}/{target}_{model_name}_{transform_name}_{dataset}_parity.png", bbox_inches='tight', dpi=1200)
    plt.clf()
    
def train_test_parity_plot(y_test, y_test_pred, y_train, y_train_pred, model_name, transform_name, target, path):
    '''  This function takes in the true and predicted values and plots a scatter
    plot alongside a y=x line.
    ------------
    y_test : array-like
        the true test values
    y_test_pred : array-like
        the predicted test values
    y_train : array-like
        the true train values
    y_train_pred : array-like
        the predicted train values
    model_name : string
        name of algorithm in yaml input file
    transform_name : string
        name of transform specified in transform_names in input file
    target : string
        name of target column in dataframes
    path : string
        path where png file will be stored
    '''
    test_errors = np.abs(y_test - y_test_pred)
    train_errors = np.abs(y_train - y_train_pred)

    min_ = min([min(y_test), min(y_test_pred), min(y_train), min(y_train_pred)])
    max_ = max([max(y_test), max(y_test_pred), max(y_train), max(y_train_pred)])

    plt.scatter(y_train, y_train_pred, facecolors='none', edgecolors='r', marker='o', s= 80, label=f'Train, average absolute error: {np.mean(train_errors): .2f}')
    plt.scatter(y_test, y_test_pred, facecolors='none', edgecolors='b', marker='*', s= 80, label=f'Test, average absolute error: {np.mean(test_errors): .2f}')
    plt.plot([min_, max_],[min_, max_])
    plt.grid()
    plt.legend(loc="lower right")
    
#    range_ = max_ - min_
#    plt.text(min_+ 0.05*range_, max_ - 0.05*range_, f"Average absolute error in test set: {np.mean(test_errors): .2f}")
#    plt.text(min_+ 0.05*range_, max_ - 0.1*range_, f"Average absolute error in train set: {np.mean(train_errors): .2f}")

    plt.xlabel('True')
    plt.ylabel('Predicted')

    plt.savefig(f"{path}/{target}_{model_name}_{transform_name}_train_test_parity.png", bbox_inches='tight', dpi=1200)
    plt.clf()
    

def bar_performances_by_algorithm(train_performances, test_performances, models, target, transform, path):
    ''' This function plots the average absolute error of train and test sets for each 
    algorithm for a single transform
    ---------
    train_performances : array-like
        average absolute error in train set
    test_performances : array-like
        average absolute error in test set
    models : list
        list of model names as strings
    target : string
        name of target column in dataframes
    transform : string
        name of transform that is paired for each model. For file naming
    path : string
        path where png file will be stored
    
    '''
    
    # set width of bar
    barWidth = 0.25
     
    # Set position of bar on X axis
    r1 = np.arange(len(train_performances))
    r2 = [x + barWidth for x in r1]
     
    # Make the plot
    plt.bar(r1, train_performances, color='slateblue', width=barWidth, edgecolor='white', label='Train')
    plt.bar(r2, test_performances, color='forestgreen', width=barWidth, edgecolor='white', label='Test')
     
    # Add xticks on the middle of the group bars
    plt.xlabel('Model', fontweight='bold')
    plt.xticks([r + barWidth/2 for r in range(len(train_performances))], models)
    plt.tick_params(
        axis='x',
        bottom=False)
     
    # Create legend & Show graphic
    plt.legend()
    plt.ylabel('Average absolute error')
    plt.grid(axis = 'y')    
    plt.savefig(f"{path}/{target}_{transform}_performances_barplot.png", bbox_inches='tight', dpi=1200)
    plt.clf()
    
def box_performances_by_algorithm(train_errors, test_errors, models, target, transform, path):
    ''' This function plots the distribution of absolute error as boxplots of train and test sets for each 
    algorithm for a single transform
    ---------
    train_errors : array-like
        individual errors in each train set
    test_errors : array-like
        individual errors in each test set
    models : list
        list of model names as strings
    target : string
        name of target column in dataframes
    transform : string
        name of transform that is paired for each model. For file naming
    path : string
        path where png file will be stored
    
    '''
    
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    
    plt.figure()
    
  #  print(train_errors)
    train = plt.boxplot(train_errors, positions=np.array(range(len(train_errors)))*2.0-0.4, sym='', widths=0.5)
    test = plt.boxplot(test_errors, positions=np.array(range(len(test_errors)))*2.0+0.4, sym='', widths=0.5)
    set_box_color(train, '#D7191C') 
    set_box_color(test, '#2C7BB6')
    
    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Train')
    plt.plot([], c='#2C7BB6', label='Test')
    plt.legend()
    plt.ylabel('Absolute error')
    
    plt.xticks(range(0, len(models) * 2, 2), models)
    plt.xlim(-2, len(models)*2)
    plt.tight_layout()
    plt.grid(axis = 'y', alpha = 0.3)
    plt.savefig(f"{path}/{target}_{transform}_performances_boxplot.png", bbox_inches='tight', dpi=1200)
    plt.clf()


def bar_performances_by_transform(train_performances, test_performances, model_name, transform_names, target, path):
    ''' This function plots the average absolute error of train and test sets for each 
    transform for a single model/algorithm
    ---------
    train_performances : array-like
        average absolute error in train set
    test_performances : array-like
        average absolute error in test set    
    model_name: string
        name of model that is paired for each transform
    transform_names : list
        list of transform names as strings
    target : string
        name of target column in dataframes
    path : string
        path where png file will be stored'''
    transform_names = [name.replace("_"," ") for name in transform_names]
    
    # set width of bar
    barWidth = 0.25
     
    # Set position of bar on X axis
    r1 = np.arange(len(train_performances))
    r2 = [x + barWidth for x in r1]
     
    # Make the plot
    plt.bar(r1, train_performances, color='slateblue', width=barWidth, edgecolor='white', label='Train')
    plt.bar(r2, test_performances, color='forestgreen', width=barWidth, edgecolor='white', label='Test')
     
    # Add xticks on the middle of the group bars
    plt.xlabel('Transform', fontweight='bold')
    plt.xticks([r + barWidth/2 for r in range(len(train_performances))], transform_names)
    plt.tick_params(
        axis='x',
        bottom=False) 
     
    # Create legend & Show graphic
    plt.legend()
    plt.ylabel('Average absolute error')
    plt.grid(axis = 'y')    
    plt.savefig(f"{path}/{target}_{model_name}_performances_barplot.png", bbox_inches='tight', dpi=1200)
    plt.clf()

def box_performances_by_transform(train_errors, test_errors, model_name, transform_names, target, path):
    ''' This function plots the distribution of absolute error as boxplots of train and test sets for each 
    transformation for a single model/algorithm
    ---------
    train_errors : array-like
        individual errors in each train set
    test_errors : array-like
        individual errors in each test set
    model_name: string
        name of model that is paired for each transform
    transform_names : list
        list of transform names as strings
    target : string
        name of target column in dataframes
    path : string
        path where png file will be stored
    
    '''
    
    transform_names = [name.replace("_"," ") for name in transform_names]

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    
    plt.figure()
    
    train = plt.boxplot(train_errors, positions=np.array(range(len(train_errors)))*2.0-0.4, sym='', widths=0.5)
    test = plt.boxplot(test_errors, positions=np.array(range(len(test_errors)))*2.0+0.4, sym='', widths=0.5)
    set_box_color(train, '#D7191C') 
    set_box_color(test, '#2C7BB6')
    
    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Train')
    plt.plot([], c='#2C7BB6', label='Test')
    plt.legend()
    plt.ylabel('Absolute error')
    
    plt.xticks(range(0, len(transform_names) * 2, 2), transform_names)
    plt.xlim(-2, len(transform_names)*2)
    plt.tight_layout()
    plt.grid(axis = 'y', alpha = 0.3)
    plt.savefig(f"{path}/{target}_{model_name}_performances_boxplot.png", bbox_inches='tight', dpi=1200)
    plt.clf()

def PC_spectra(df, target, path, label):
    """This function takes in spectra dataframe that was transformed with PCA and plots the 
    staggered PCA-spectra on a 2D plot. It is saved to the path
    ----------
        df : pandas dataframe
            dataframe of set
        target : string
            name of target column in dataframes
        path : string
            path where png file will be stored
        label : string
            name of set. "train" or "test" for file naming.
    """    
    # spacing for PC's
    if label == "train":
        space = 10
    else:
        space = 5
    
    X = df.drop(str(target), axis=1).values.astype(np.float)
    
    features = list(df.columns)
    features.remove(target)
    features = np.asarray(features)
    
    for i, example in enumerate(X):
        
        intensity = X[i] + space*i
        plt.plot(features, intensity, alpha = 0.5, linewidth=1, c='k')
        
    plt.yticks([])
    plt.xticks([])
    plt.xlabel("Principal component")
    plt.savefig(f"{path}/PCA_spectra_2D_{label}.png", bbox_inches='tight', dpi=1200)
    plt.clf()
    
def all_spectra(df, target, path):    
    """This function takes in a dataframe and plots each spectra
    individually and stores the images in a folder in the path.
    ----------
        df : pandas dataframe
            dataframe of spectra set
        target : string
            name of target column in dataframes
        path : string
            path where folder of all spectra plots will be stored
    """
    
    # make folder for individual spectra
    spectra_path = f"{path}/spectra_plots"
    os.mkdir(spectra_path)
    
    X = df.drop(str(target), axis=1).values.astype(np.float)
    examples = df.index.values.tolist()
    
    features = list(df.columns)
    features.remove(target)
    features = np.asarray(features).astype(np.float)
    
    for i, spectra in enumerate(X):
        
        plt.plot(features, X[i])
        plt.xlabel("Wavenumber")
        plt.ylabel("Intensity")
        plt.title(f"{examples[i]}")
        
        plt.savefig(f"{spectra_path}/{examples[i]}_spectra.png", bbox_inches='tight', dpi=1200)
        plt.clf()
    
    
    
    
    
    
    
    
    
    
    #