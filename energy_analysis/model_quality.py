import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


class Metrics:

    @staticmethod
    def print_all_metrics(y, y_hat):
        for func in [mean_absolute_error,
                     mean_absolute_percentage_error,
                     mean_squared_error,
                     r2_score]:
            print(f"{func.__name__}: {func(y, y_hat)}")
