import plot_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
from datetime import datetime
import os
import shutil

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer


def test_set_dist():
    
    id_num = str(datetime.now())[2:19]
    id_num = id_num.replace(':', '').replace(' ', '').replace('-', '') 
    
    # define the name of the directory to be created
    path = os.getcwd() + "/test_request_" + str(id_num)
    
    # make directory
    os.mkdir(path)
    
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

    plot_.set_dist(df_test, target, 'test', path)
    
    pic_name = f"{path}/test_dist.png"
    
    isFile = os.path.isfile(pic_name) 
    
    assert isFile 
