import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from energy_analysis.config import data_dir, output_dir


class Prep:
    """
    Class for ingesting and preparing data for analysis
    """

    def __init__(self):
        # TODO
        #  Yes, I know that this class doesn't need to be a class.
        #  All of the methods could be static, or they could be standalone functions.
        #  Making it a class keeps everything all tidy together.
        #  And if we later want to introduce class variables, it's already set up for that.
        #  (And finally, this is how we did it at my old workplace, and consistency is key!  You'll see this throughout)
        pass

    def get_energy_data(self, file='sip_dam_data.csv') -> pd.DataFrame:
        energy_data = pd.read_csv(os.path.join(data_dir, file))
        energy_data['time'] = pd.to_datetime(energy_data['time'])

        return energy_data

    def plot_energy_data(self, energy_data: pd.DataFrame, save_loc: str = None, ylim: tuple = None):
        fig, ax = plt.subplots(figsize=(20, 5))

        # Seaborn likes a long (unpivoted) dataframe, so we melt it
        plot_frame = pd.melt(energy_data, id_vars='time')
        ax = sns.lineplot(data=plot_frame,
                          x='time', y='value', hue='variable', ax=ax, legend='full')

        ax.legend()
        ax.set_title('Energy Prices')
        if ylim:
            ax.set_ylim(*ylim)

        if save_loc:
            plt.savefig(os.path.join(output_dir, save_loc))
        plt.show()

    def get_weather_data(self, file='london_weather.csv') -> pd.DataFrame:
        weather_data = pd.read_csv(os.path.join(data_dir, file))
        weather_data['date'] = pd.to_datetime(weather_data['date'], format='%Y%m%d')

        # Truncate to period of interest
        # TODO Parameterize
        weather_data = weather_data[weather_data['date'] > '20160101']

        return weather_data

    def plot_weather_data(self, weather_data: pd.DataFrame, save_loc: str = None, ylim: tuple = None):
        fig, ax = plt.subplots(figsize=(20, 5))

        # Seaborn likes a long (unpivoted) dataframe, so we melt it
        plot_frame = pd.melt(weather_data[['date', 'max_temp', 'mean_temp', 'min_temp']], id_vars='date',
                             value_name='deg C')
        ax = sns.lineplot(data=plot_frame,
                          x='date', y='deg C', hue='variable', ax=ax, legend='full')

        ax.legend()
        ax.set_title('London Daily Temperatures')
        if ylim:
            ax.set_ylim(*ylim)

        if save_loc:
            plt.savefig(os.path.join(output_dir, save_loc))
        plt.show()
