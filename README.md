# CAML
Case study for AutoML (CAML). Performs data cleaning and a variety of feature 
transformations for machine learning case studies. Uses the AutoML software TPOT, 
based in Scikit-Learn, for machine learning. Includes specialized options for 
spectral input data.


## Installation
Once the source files are downloaded, the following commands can be used to download 
the required packages.
```console
$ cd CAML
$ conda install numpy scipy scikit-learn pandas joblib pytorch
$ conda install -c anaconda pyyaml 
$ pip install tpot
```

## Running Optimus
Once installed, from the home directory, the code can be run with:
```console
$ caml.py --help
```

## Example
First unzip the csv file and ensure the path in the simple_input.yaml file 
matches where you want to store it.
An example can be run by using simple-input.yaml as the request file
```console
$ caml.py simple-input.yaml
```

Results will be output to a folder marked request with a datetime ID number.

## License
CAML is released under the MIT license; see LICENSE for details.