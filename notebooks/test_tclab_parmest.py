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

other_T1 = [i for i in df['T1'].values]

time_delay = 0

new_list = other_T1[time_delay:]
for i in range(time_delay):
    new_list.append(df['T1'].values[-1])

tc_data = TC_Lab_data(
    name="Sine Wave Test for Heater 1",
    time=df['Time'].values,
    # T1=df['T1'].values,
    T1=new_list,
    u1=df['Q1'].values,
    P1=200,
    TS1_data=None,
    T2=df['T2'].values,
    u2=df['Q2'].values,
    P2=200,
    TS2_data=None,
    Tamb=df['T1'].values[0],
)

file = '../data/tclab_step_test.csv'
df2 = pd.read_csv(file)
df2.head()

tc_data2 = TC_Lab_data(
    name="Step Test for Heater 1",
    time=df2['Time'].values,
    T1=df2['T1'].values,
    u1=df2['Q1'].values,
    P1=200,
    TS1_data=None,
    T2=df2['T2'].values,
    u2=df2['Q2'].values,
    P2=200,
    TS2_data=None,
    Tamb=df2['T1'].values[0],
)

# Number of states
num_states = 4

# Create an experiment list
exp_list = []
exp_list.append(TC_Lab_experiment(data=tc_data, number_of_states=num_states))
exp_list.append(TC_Lab_experiment(data=tc_data2, number_of_states=num_states))

pest = parmest.Estimator(exp_list, obj_function='SSE', tee=True)

# Parameter estimation with covariance
obj, theta = pest.theta_est()
print(obj)
print(theta)

# REPARAM PRINTING
P1 = 200
alpha = 0.00016

inv_CpH = theta['beta_4'] / (P1 * alpha)
Ua = theta['beta_1'] / (inv_CpH)
Ub = theta['beta_2'] / (inv_CpH)
if num_states == 4:
    Uc = theta['beta_5'] / (inv_CpH)
else:
    theta['beta_5'] = 0.0
    Uc = 0.0
inv_CpS = theta['beta_3'] / (Ub)

print("Reparametrized Parameters")
print("beta 1: {:.8f} \nbeta_2: {:.8f}\nbeta_3: {:.8f}\nbeta_4: {:.8f}\nbeta_5: {:.8f}\n".format(theta['beta_1'], theta['beta_2'], theta['beta_3'], theta['beta_4'], theta['beta_5']))

print("Original Parameters")
print("inv_CpH: {:.8f} \ninv_CpS: {:.8f}\nUa: {:.8f}\nUb: {:.8f}\nUc: {:.8f}\n".format(inv_CpH, inv_CpS, Ua, Ub, Uc))

# U1_vals = np.array([aml.value(model.U1[t]) for t in model.t])
# U2_vals = [aml.value(model.U2[t]) for t in model.t]

# plt.plot(model.t, U1_vals, color='green', label='U1')
# plt.plot(model.t, df['Q1'].values, color='red', linestyle='--', label='U1 Data')
# plt.plot(model.t, U2_vals, color='blue', label='U2')
# plt.plot(model.t, df['Q2'].values, color='red', linestyle='--', label='U2 Data')
# plt.legend()
# plt.show()
# plt.clf()
# plt.close()

def plot_data_and_prediction(model, df):
    Th1_vals = [aml.value(model.Th1[t]) for t in model.t]
    Ts1_vals = [aml.value(model.Ts1[t]) for t in model.t]
    Th2_vals = [aml.value(model.Th2[t]) for t in model.t]
    Ts2_vals = [aml.value(model.Ts2[t]) for t in model.t]

    plt.plot(model.t, Ts1_vals, color='green', label='Ts1')
    plt.plot(model.t, df['T1'].values, color='green', linestyle='--', label='Ts1 Data')
    plt.plot(model.t, Ts2_vals, color='blue', label='Ts2')
    plt.plot(model.t, df['T2'].values, color='blue', linestyle='--', label='Ts2 Data')
    plt.plot(model.t, Th1_vals, color='black', label='Th1')
    plt.plot(model.t, Th2_vals, color='black', label='Th2')
    plt.legend()
    plt.show()
    plt.clf()
    plt.close()

    print((sum((Ts1_vals[i] - df['T1'].values[i])**2 for i, v in enumerate(df['Time'].values)) + sum((Ts2_vals[i] - df['T2'].values[i])**2 for i, v in enumerate(df['Time'].values))))

model = pest.ef_instance.Scenario0
# model = pest.ef_instance
plot_data_and_prediction(model, df)
model2 = pest.ef_instance.Scenario1
# model2 = pest.ef_instance
plot_data_and_prediction(model2, df2)