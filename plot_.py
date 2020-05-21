""" This module includes plotting functions. """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def abs_error_hist(test_abs_errors, model_name, target, path):

    plt.hist(test_abs_errors)
    
    plt.xlabel("Absolute error")
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{path}/{target}_{model_name}_error_hist.png", bbox_inches='tight')
    
    
def test_parity_plot(y_test, test_predictions, test_abs_errors, model_name, target):

    min_ = min([min(y_test), min(test_predictions)])
    max_ = max([max(y_test), max(test_predictions)])

    plt.scatter(y_test, test_predictions, facecolors='none', edgecolors='r', marker='o', s= 80)
    plt.plot([min_, max_],[min_, max_])
    plt.grid()

    plt.text(min_+5, max_-5, f"Average absolute error: {round(np.mean(test_abs_errors), 1)}")

    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.savefig(f"{path}/{target}_{model_name}_parity.png", bbox_inches='tight')