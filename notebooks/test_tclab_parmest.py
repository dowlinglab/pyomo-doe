from tclab_pyomo import TC_Lab_experiment, TC_Lab_data
import pandas as pd

import logging
import json
import matplotlib.pyplot as plt
import numpy as np

import pyomo.environ as aml

from pyomo.common.dependencies import pandas as pd
from os.path import join, abspath, dirname
import pyomo.contrib.parmest.parmest as parmest


file = '../data/tclab_sine_test.csv'
df = pd.read_csv(file)
df.head()

tc_data = TC_Lab_data(
    name="Sine Wave Test for Heater 1",
    time=df['Time'].values,
    T1=df['T1'].values,
    u1=df['Q1'].values,
    P1=200,
    TS1_data=None,
    T2=df['T2'].values,
    u2=df['Q2'].values,
    P2=200,
    TS2_data=None,
    Tamb=df['T1'].values[0],
)

# Create an experiment list
exp_list = []
exp_list.append(TC_Lab_experiment(data=tc_data, number_of_states=4))

pest = parmest.Estimator(exp_list, obj_function='SSE', tee=True)

# Parameter estimation with covariance
obj, theta = pest.theta_est()
print(obj)
print(theta)
