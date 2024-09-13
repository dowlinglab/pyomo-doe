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

import copy


def time_delay_data(data, time_delay):
    # # Uncomment for deleting data
    # new_T1 = copy.deepcopy(data['T1'].values)
    # new_T2 = copy.deepcopy(data['T2'].values)
    
    # new_data = copy.deepcopy(data.head(-time_delay))
    # new_data['T1'] = new_T1[time_delay:]
    # new_data['T2'] = new_T2[time_delay:]
    
    # Uncomment for shifting u to have 0s at beginning
    new_U1 = np.zeros(len(data['Q1'].values))
    new_U2 = np.zeros(len(data['Q2'].values))
    
    new_U1[time_delay:] = copy.deepcopy(data['Q1'].values)[:-time_delay]
    new_U2[time_delay:] = copy.deepcopy(data['Q2'].values)[:-time_delay]
    
    new_data = copy.deepcopy(data)
    new_data['Q1'] = new_U1
    new_data['Q2'] = new_U2
    
    return copy.deepcopy(new_data)

file = '../data/validation_experiment_env_2_sin_5_50_run_1.csv'
# file = '../data/validation_experiment_env_2_step_50_run_1.csv'
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

file = '../data/validation_experiment_env_2_step_50_run_1.csv'
# file = '../data/tclab_step_test.csv'
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
num_states = 2
run_single = True

# Create an experiment list
exp_list = []
if run_single:
    exp_list.append(TC_Lab_experiment(data=tc_data, number_of_states=num_states))
    exp_list.append(TC_Lab_experiment(data=tc_data2, number_of_states=num_states))
else:
    for k in [50, 2, 5, ]:
        track_Ts1_vals = pd.DataFrame(columns=['Ts1 1', 'Ts1 2', 'Ts1 3'])
        track_Ts2_vals = pd.DataFrame(columns=['Ts2 1', 'Ts2 2', 'Ts2 3'])
        for i in range(3):
            file = '../data/validation_experiment_env_2_sin_{}_50_run_{}.csv'.format(k, i + 1)
            df = pd.read_csv(file)
            
            df = time_delay_data(df, 8)
            
            tc_data = TC_Lab_data(
                name="Sine Wave Test {}, {} for Heater 1".format(k, i + 1),
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
            
            # track_Ts1_vals['Ts1 {}'.format(i + 1)] = df['T1'].values
            # track_Ts2_vals['Ts2 {}'.format(i + 1)] = df['T2'].values
            
            exp_list.append(TC_Lab_experiment(data=copy.deepcopy(tc_data), number_of_states=num_states))


# exp_list.append(TC_Lab_experiment(data=tc_data2, number_of_states=num_states))

# pest = parmest.Estimator(exp_list, obj_function='SSE', tee=True)

# # Parameter estimation with covariance
# obj, theta = pest.theta_est()
# print(obj)
# print(theta)

# print(exp_list)

orders = []
values = []

# MULTIPLE IN SAME INSTANCE
for i in range(1):
    exp_list = []

    # Make experiments
    exp_list.append(TC_Lab_experiment(data=tc_data, number_of_states=num_states))
    exp_list.append(TC_Lab_experiment(data=tc_data2, number_of_states=num_states))

    # Run Parmest
    try:
        pest = parmest.Estimator(exp_list, obj_function='SSE', tee=False)

        # Parameter estimation with covariance
        obj, theta = pest.theta_est()
        print(obj)
        print(theta)

        new_orders = []
        for i in pest.ef_instance._C_EF_:
            new_orders.append(str(pest.ef_instance._C_EF_[i].body))
        
        values.append(theta)

        orders.append(new_orders)
    except:
        new_orders = []
        new_orders.append('FAILURE')
        values.append('FAILURE')

        orders.append(new_orders)


    


# # REPARAM PRINTING
# P1 = 200
# alpha = 0.00016

# inv_CpH = theta['beta_4'] / (P1 * alpha)
# Ua = theta['beta_1'] / (inv_CpH)
# Ub = theta['beta_2'] / (inv_CpH)
# if num_states == 4:
#     Uc = theta['beta_5'] / (inv_CpH)
# else:
#     theta['beta_5'] = 0.0
#     Uc = 0.0
# inv_CpS = theta['beta_3'] / (Ub)

# inv_CpH = 1 / theta['inv_CpH']
# Ua = theta['Ua']
# Ub = theta['Ub']
# if num_states == 4:
#     Uc = theta['beta_5'] / (inv_CpH)
# else:
#     theta['beta_5'] = 0.0
#     Uc = 0.0
# inv_CpS = 1 / theta['inv_CpS']

# # print("Reparametrized Parameters")
# # print("beta 1: {:.8f} \nbeta_2: {:.8f}\nbeta_3: {:.8f}\nbeta_4: {:.8f}\nbeta_5: {:.8f}\n".format(theta['beta_1'], theta['beta_2'], theta['beta_3'], theta['beta_4'], theta['beta_5']))

# print("Original Parameters")
# print("inv_CpH: {:.8f} \ninv_CpS: {:.8f}\nUa: {:.8f}\nUb: {:.8f}\nUc: {:.8f}\n".format(inv_CpH, inv_CpS, Ua, Ub, Uc))

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

def plot_data_and_prediction(model):
    Th1_vals = [aml.value(model.Th1[t]) for t in model.t]
    Ts1_vals = [aml.value(model.Ts1[t]) for t in model.t]
    if num_states == 4:
        Th2_vals = [aml.value(model.Th2[t]) for t in model.t]
        Ts2_vals = [aml.value(model.Ts2[t]) for t in model.t]
    
    Ts1_data = [v for k,v in model.experiment_outputs.items() if 'Ts1' in k.name]
    Ts2_data = [v for k,v in model.experiment_outputs.items() if 'Ts2' in k.name]

    plt.plot(model.t, Ts1_vals, color='green', label='Ts1')
    plt.plot(model.t, Ts1_data, color='green', linestyle='--', label='Ts1 Data')
    if num_states == 4:
        plt.plot(model.t, Ts2_vals, color='blue', label='Ts2')
        plt.plot(model.t, Ts2_data, color='blue', linestyle='--', label='Ts2 Data')
        plt.plot(model.t, Th1_vals, color='black', label='Th1')
        plt.plot(model.t, Th2_vals, color='black', label='Th2')
    plt.legend()
    plt.show()
    plt.clf()
    plt.close()

    SSE = sum((Ts1_vals[i] - Ts1_data[i])**2 for i, v in enumerate(model.t))
    if num_states == 4:
        SSE += sum((Ts2_vals[i] - Ts2_data[i])**2 for i, v in enumerate(model.t))

    print(SSE)

# model = pest.ef_instance.Scenario0
# model = pest.ef_instance.Scenario2
# model = pest.ef_instance
# plot_data_and_prediction(model)
# model2 = pest.ef_instance.Scenario1
# model2 = pest.ef_instance
# plot_data_and_prediction(model2)

# Print all 3 model scenarios
# model = pest.ef_instance.Scenario0
# plot_data_and_prediction(model)
# model = pest.ef_instance.Scenario1
# plot_data_and_prediction(model)
# model = pest.ef_instance.Scenario2
# plot_data_and_prediction(model)
# model = pest.ef_instance.Scenario3
# plot_data_and_prediction(model)
# model = pest.ef_instance.Scenario4
# plot_data_and_prediction(model)
# model = pest.ef_instance.Scenario5
# plot_data_and_prediction(model)
# model = pest.ef_instance.Scenario6
# plot_data_and_prediction(model)
# model = pest.ef_instance.Scenario7
# plot_data_and_prediction(model)
# model = pest.ef_instance.Scenario8
# plot_data_and_prediction(model)

# track_Ts1_vals['std'] = track_Ts1_vals.std(axis=1)
# track_Ts2_vals['std'] = track_Ts2_vals.std(axis=1)
# print(track_Ts1_vals.std(axis=1))
# print(track_Ts2_vals.std(axis=1))

# track_Ts1_vals.to_csv('Ts1_for_sin_5_50_all_runs.csv', index=False)
# track_Ts2_vals.to_csv('Ts2_for_sin_5_50_all_runs.csv', index=False)
