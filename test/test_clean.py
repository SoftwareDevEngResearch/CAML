import caml
import clean

def test_normalize_overall_min_max(df_train, df_test, target):
    
    input_file = "../simple_input.yaml"
    with open(str(input_file), 'r') as file:
        input_ = yaml.load(file, Loader=yaml.FullLoader)
    
    data = input_["data"]
    target = data["target_col_name"]
    
    # dataframe with ID, target, features 
    df = pd.read_csv(data["data_path"], header = 0, dtype=object, index_col=data["index_col_name"])
    
    df_train, df_test = train_test_split(df, 
                                         test_size=0.2, 
                                            )
    
    df_train, df_test, scale = clean.normalize_overall_min_max(df_train, df_test, target)
    
    X_train = df_train.drop(str(target), axis=1).values.astype(np.float)
    X_test = df_test.drop(str(target), axis=1).values.astype(np.float)
    
    min_ = np.amin(X_train, axis=None, out=None)
    max_ = np.amax(X_train, axis=None, out=None)
    
    assert [(scale[0] == min_),
            (scale[1] == max_)
            ]
    
#def test_normalize_scaler(df_train, df_test, target, scale_method):
#    
#    input_file = "../simple_input.yaml"
#    with open(str(input_file), 'r') as file:
#        input_ = yaml.load(file, Loader=yaml.FullLoader)
#    
#    data = input_["data"]
#    target = data["target_col_name"]
#    
#    # dataframe with ID, target, features 
#    df = pd.read_csv(data["data_path"], header = 0, dtype=object, index_col=data["index_col_name"])
#    
#    df_train, df_test = train_test_split(df, 
#                                         test_size=0.2, 
#                                            )
#    
#    
#    X_train = df_train.drop(str(target), axis=1).values.astype(np.float)
#    X_test = df_test.drop(str(target), axis=1).values.astype(np.float)
#    
#    df_train, df_test, scale = clean.normalize_min_max(df_train, df_test, target, MinMaxScaler())
#    
#    X_train = df_train.drop(str(target), axis=1).values.astype(np.float)
#    X_test = df_test.drop(str(target), axis=1).values.astype(np.float)
#    
#    assert 

