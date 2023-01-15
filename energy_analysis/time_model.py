import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from prophet import Prophet

from energy_analysis.config import data_dir, output_dir


class TimeSeriesModel:
    """ This class is a wrapper around a prophet model.

    Prophet likes to have its data formatted in a special way, so the fit()
    and predict() methods in this class facilitate the translation.
    """

    def __init__(self, prophet_kwargs: dict = None):
        if prophet_kwargs is None:
            prophet_kwargs = dict(growth='linear',
                                  changepoints=None,
                                  n_changepoints=4,
                                  changepoint_range=0.8,
                                  yearly_seasonality=True,
                                  weekly_seasonality=False,
                                  daily_seasonality=True,
                                  holidays=None,
                                  seasonality_mode='additive',
                                  seasonality_prior_scale=10.0,
                                  holidays_prior_scale=1,
                                  changepoint_prior_scale=0.05,
                                  mcmc_samples=0,
                                  interval_width=0.80,
                                  uncertainty_samples=1000,
                                  stan_backend=None)
        self.m = Prophet(**prophet_kwargs)

    def fit(self, df: pd.DataFrame, time_col: str, data_col: str):
        # Prophet wants the time column to be 'ds' and the data column to be 'y'
        data = df[[time_col, data_col]].rename(columns={time_col: 'ds', data_col: 'y'})
        self.m.fit(data)

    def predict(self, df: pd.DataFrame, time_col: str):
        data = df[[time_col]].rename(columns={time_col: 'ds'})
        data = self.m.predict(data)
        df = df.merge(data, how='left', left_on=time_col, right_on='ds')
        return df

    def plot_components(self, df: pd.DataFrame, save_loc: str = None, title: str = ""):
        fig = self.m.plot_components(df)

        if title:
            plt.suptitle(title)
        plt.tight_layout()

        if save_loc:
            plt.savefig(os.path.join(output_dir, save_loc))
        plt.show()

        return fig

    def plot_prediction(self, df: pd.DataFrame, save_loc: str = None, title: str = ""):
        fig = self.m.plot(df)

        if title:
            plt.suptitle(title)
        fig.legend()
        plt.tight_layout()

        if save_loc:
            plt.savefig(os.path.join(output_dir, save_loc))
        plt.show()

        return fig
