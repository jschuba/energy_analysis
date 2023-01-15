

import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from energy_analysis import Prep, TimeSeriesModel, Metrics

prepper = Prep()

energy_data = prepper.get_energy_data()

# Plot, and then plot with the y axis zoomed in
prepper.plot_energy_data(energy_data, save_loc='energy_prices.png', ylim=None)
prepper.plot_energy_data(energy_data, save_loc='energy_prices_y_zoomed.png', ylim=(-100, 150))

# Print some basic statistics
print(energy_data.describe())
print(energy_data['time'].min(), energy_data['time'].max())

# Drop the NAN rows, because there are so few of them,
# and we won't loose other data by doing so (there is no other data)
energy_data = energy_data.dropna(axis=0)

# # Plot end of 2016, because it's very spiky
prepper.plot_energy_data(energy_data[(energy_data['time'] > '2016-10-01') & (energy_data['time'] < '2016-12-31')],
                         save_loc='energy_prices_end_of_2016.png', ylim=(-100, 500))
# # Plot the last few months only, because it's spiky
prepper.plot_energy_data(energy_data[energy_data['time'] > '2020-12-01'],
                         save_loc='energy_prices_2021.png', ylim=(-100, 500))
# Zoom in on a random week, where there are no spikes
prepper.plot_energy_data(energy_data[(energy_data['time'] > '2020-06-07') & (energy_data['time'] < '2020-06-13')],
                         save_loc='energy_prices_one_week.png', ylim=(0, 50))

# Lets smooth the SIP and then plot it again
energy_data['SIP Smooth'] = energy_data['SIP'].rolling(12).mean()
prepper.plot_energy_data(energy_data[(energy_data['time'] > '2020-06-07') & (energy_data['time'] < '2020-06-13')],
                         save_loc='energy_prices_one_week.png', ylim=(0, 50))

"""
Out of curiosity, I searched for London historical weather data, 
and I found this page, and downloaded the data as 'london_weather.csv'
https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data?resource=download
"""
weather_data = prepper.get_weather_data()
prepper.plot_weather_data(weather_data, save_loc='London Daily Temps.png')
# I'm not sure how to use this, yet.


"""
Before we try any tricks to tame the data, lets try fitting a time-series model.

I am assuming that since Habitat Energy is primarily concerned with battery storage, 
the short-term daily periodicity will be most important to understand.  
"""

# Leave out the last 3 months of data, as test_data
# This is a time-series model, so we cannot do a random split
training_data = energy_data[(energy_data['time'] < '2020-10-01')]
test_data = energy_data[(energy_data['time'] >= '2020-10-01')]

dam_model = TimeSeriesModel()
dam_model.fit(training_data, 'time', 'DAM')
training_data = dam_model.predict(training_data, 'time')
dam_model.plot_components(training_data, save_loc='DAM model components.png', title='DAM model components')
dam_model.plot_prediction(training_data, save_loc='DAM model prediction (training).png', title='DAM model prediction (training)')
test_data = dam_model.predict(test_data, 'time')
dam_model.plot_prediction(test_data, save_loc='DAM model prediction (test).png', title='DAM model prediction (test)')

all_data = pd.concat([training_data, test_data])
dam_model.plot_prediction(all_data, save_loc='DAM model prediction (all).png', title='DAM model prediction (all)')


print(f"DAM Training Metrics")
Metrics().print_all_metrics(training_data['DAM'], training_data['yhat'])
print(f"DAM Test Metrics")
Metrics().print_all_metrics(test_data['DAM'], test_data['yhat'])


"""
At this point, I have exhausted the time I allotted for this assignment (3 hours). 

The model is not great, when looking at the test period.  However the test period is the end of the dataset, and it is 
the most spiky period.  The model likely does better for the smoother periods.  Here are the metrics:

_______________________
DAM Training Metrics
mean_absolute_error: 6.910045990562776
mean_absolute_percentage_error: 25274251034332.344
mean_squared_error: 237.5107104845655
r2_score: 0.4090536190402515

DAM Test Metrics
mean_absolute_error: 19.746258387210897
mean_absolute_percentage_error: 42603304770586.586
mean_squared_error: 2771.2711598942105
r2_score: -0.04077551497373699
_______________________

MAE (mean absolute error) is probably the best metric for this data.  I usually prefer MAPE, but it doesn't do well 
with true values near zero, which we have here.  The negative r2 score is concerning, in the test period.  It implies 
that this period is totally unlike the period used to train the model.

++++++++++++++++++++++++++++++++
Next Steps / Ideas
++++++++++++++++++++++++++++++++

1)  Clean the data a bit.  De-spike the high spikes, perhaps.  
    I would need to consult a domain expert to determine if this is a good idea.  I don't like to remove 
    outliers without understanding why they are there.
2)  Tune model parameters.  For example, we can set the initial strength of the seasonal components. 
    Also, prophet has the capacity to fit holidays and additional regressors, which we are not using here. 
3)  Do the same with the SIP data.
4)  Stretch Goal:  Create an optimal control model to buy and sell energy throughout the day.  
    If you have capacity to buy energy, store it, and sell it later, this could be quite profitable. ::wink::
    Use the predicted prices to find good control inputs, and then see how those inputs would perform on the 
    actual data.
    



"""



