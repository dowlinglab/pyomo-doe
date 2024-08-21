from tclab_pyomo import TC_Lab_experiment, TC_Lab_data
import pandas as pd

import logging
import json
import matplotlib.pyplot as plt
import numpy as np

import pyomo.environ as aml

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

theta_vals = {'Ua': 0.05147278733764012, 'Ub': 0.0005342082856927798, 'inv_CpH': 0.14622879403418604, 'inv_CpS': 99.99999754623846, }

TC_lab_sine_wave_exp = TC_Lab_experiment(data=tc_data, theta_initial=theta_vals, sine_amplitude=20, sine_period=5)

test_model = TC_lab_sine_wave_exp.get_labeled_model()

solver = aml.SolverFactory('ipopt')

solver.solve(test_model, tee=True)
