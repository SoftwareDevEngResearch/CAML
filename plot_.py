""" This module includes plotting functions. """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import cm
plt.tight_layout(h_pad = 3, w_pad=3)

####################################

def set_dist(df, target, set_, path):
    
    y = df[str(target)].values.astype(np.float)
    plt.hist([y]) 
    
    plt.ylabel('frequency')
    plt.xlabel(f"{target}")
    plt.grid(axis = 'y') 
    plt.savefig(f"{path}/{set_}_dist.png", bbox_inches='tight', dpi=1200)
    plt.clf()

def overall_dist(df_train, df_test, target, path):
    
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
        
    ax.plot_surface(X_, Y_, Z, rstride=1, cstride=1000, shade=True, lw=.1) 
     
#        ax.set_zlim(0, 5)
#        ax.set_xlim(-51, 51)
    ax.set_zlabel("Intensity")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Fuel")
#        ax.view_init(20,-120)
    plt.savefig(f"{path}/3D_spectra.png", dpi=1200)
    plt.clf()
    
def spectra_2D(df, target, path, label):
    
#    fig = plt.figure()
#    df = df_train
#    df = df.sort_values(by=[str(target)]) # sort by target, ascending
    X = df.drop(str(target), axis=1).values.astype(np.float)
    y = df[str(target)].copy().values.astype(np.float)
    
#    color = [cm.cool(item/max(y)) for item in y]
    
#    X = X_train
    features = list(df.columns)
    features.remove(target)
    features = np.asarray(features).astype(np.float)
#    fig.axes.get_yaxis().set_visible(False)
#    plt.axis('off')
    
    for i, example in enumerate(X):
        
        intensity = X[i] + 0.2*i
        plt.plot(features, intensity, alpha = 0.5, linewidth=1, c='k')
        
    plt.ylabel("Intensity")
    plt.yticks([])
    plt.xlabel("Wavenumber")
#    plt.yticks(" ")
#    plt.savefig(f"see_.png", bbox_inches='tight', dpi=1200)
    plt.savefig(f"{path}/spectra_2D_{label}.png", bbox_inches='tight', dpi=1200)
    plt.clf()

#####################################

def abs_error_hist(abs_errors, model_name, transform_name, target, path, dataset):

    plt.hist(abs_errors)
    
    plt.xlabel("Absolute error")
    plt.ylabel('Frequency')
    plt.savefig(f"{path}/{target}_{model_name}_{transform_name}_{dataset}_error_hist.png", bbox_inches='tight', dpi=1200)
    plt.clf()
    
    
def parity_plot(y, predictions, model_name, transform_name, target, path, dataset):
    
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

#    range_ = max_ - min_
#    plt.text(min_+0.1*range_, max_-0.1*range_, f"Average absolute error: {np.mean(abs_errors): .2f}")

    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.legend()
    plt.savefig(f"{path}/{target}_{model_name}_{transform_name}_{dataset}_parity.png", bbox_inches='tight', dpi=1200)
    plt.clf()
    
def train_test_parity_plot(y_test, y_test_pred, y_train, y_train_pred, model_name, transform_name, target, path):

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
    

def performances_by_algorithm(train_performances, test_performances, models, target, transform, path):
    
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
    plt.savefig(f"{path}/{target}_{transform}_performances.png", bbox_inches='tight', dpi=1200)
    plt.clf()

def performances_by_transform(train_performances, test_performances, model_name, transform_names, target, path):
    
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
    plt.savefig(f"{path}/{target}_{model_name}_performances.png", bbox_inches='tight', dpi=1200)
    plt.clf()








