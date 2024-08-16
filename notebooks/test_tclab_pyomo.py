from tclab_pyomo import TC_Lab_experiment, TC_Lab_data
import pandas as pd

import logging
import json
import matplotlib.pyplot as plt
import numpy as np

import pyomo.environ as aml

file = '../data/tclab_sine_test.csv'
# file = './data/tclab_step_test.csv'
df = pd.read_csv(file)
df.head()

ax = df.plot(x='Time', y=['T1', 'T2'], xlabel='Time (s)', ylabel='Temperature (°C)')

plt.show()
plt.clf()
plt.close()

ax = df.plot(x='Time', y=['Q1', 'Q2'], xlabel='Time (s)', ylabel='Heater Power (%)')

plt.show()
plt.clf()
plt.close()

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

TC_lab_exp_instance = TC_Lab_experiment(data=tc_data, theta_initial=theta_vals)

m = TC_lab_exp_instance.get_labeled_model()

solver = aml.SolverFactory('ipopt')

solver.solve(m, tee=True)

Th1_vals = [aml.value(m.Th1[t]) for t in m.t]
Ts1_vals = [aml.value(m.Ts1[t]) for t in m.t]
t_vals = [i for i in m.t]

plt.plot(t_vals, Th1_vals)
plt.plot(t_vals, Ts1_vals)
plt.legend(['H', 'S'])
plt.show()
plt.clf()
plt.close()

from pyomo.contrib.doe import DesignOfExperiments

test_DoE = DesignOfExperiments(experiment=TC_lab_exp_instance, 
                               step=1e-2, 
                               scale_nominal_param_value=True, 
                               objective_option="trace", 
                               tee=True, 
                               logger_level=logging.INFO)

prior_FIM = test_DoE.compute_FIM(method='kaug')

print(prior_FIM)

test_DoE.prior_FIM = prior_FIM

test_DoE.run_doe(results_file="test_solve.json")

with open("test_solve.json") as f:
    d = json.load(f)

plt.plot(range(len(d['Experiment Design'])), d['Experiment Design'])
plt.show()
plt.clf()
plt.close()

TC_lab_exp_instance2 = TC_Lab_experiment(data=tc_data, theta_initial=theta_vals)

solver = aml.SolverFactory('ipopt')
# solver.options['bound_push'] = 1E-10
solver.options['halt_on_ampl_error'] = 'yes'
# solver.options['linear_solver'] = 'ma57'

test_DoE_second_exp = DesignOfExperiments(experiment=TC_lab_exp_instance2, 
                                          step=1e-2, 
                                          scale_nominal_param_value=True, 
                                          objective_option="determinant", 
                                          prior_FIM=prior_FIM, 
                                          tee=True,
                                          solver=solver,
                                          logger_level=logging.INFO,)

test_DoE_second_exp.run_doe(results_file="test_solve_d_opt.json")

with open("test_solve_d_opt.json") as f:
    d2 = json.load(f)

plt.plot(range(len(d2['Experiment Design'])), d2['Experiment Design'])
plt.show()
plt.clf()
plt.close()