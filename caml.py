import numpy as np
import pandas as pd
import argparse
import yaml

# caml modules
import clean

""" This module is meant for control sequence """

def read_input(input_file):
    
    with open(str(input_file), 'r') as file:
        input = yaml.load(file, Loader=yaml.FullLoader)
        
    return input

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str,
                        help='name of input yaml file')

    args = parser.parse_args()
    input_file = args.input_file

    input = read_input(input_file)

    print(input)

#=============================================================
    # read input file
    # make folder for this case study with unique #
    
    # overall data cleaning
    # data visualization of target data

    # possibly divide nodes?
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
