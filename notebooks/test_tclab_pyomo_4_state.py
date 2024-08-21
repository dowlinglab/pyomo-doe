from tclab_pyomo import TC_Lab_experiment, TC_Lab_data
import pandas as pd

import logging
import json
import matplotlib.pyplot as plt
import numpy as np

import pyomo.environ as aml

file = '../data/tclab_sine_test.csv'
# file2 = './data/tclab_step_test.csv'
df = pd.read_csv(file)
df.head()

# ax = df.plot(x='Time', y=['T1', 'T2'], xlabel='Time (s)', ylabel='Temperature (°C)')

# plt.show()
# plt.clf()
# plt.close()

# ax = df.plot(x='Time', y=['Q1', 'Q2'], xlabel='Time (s)', ylabel='Heater Power (%)')

# plt.show()
# plt.clf()
# plt.close()

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

theta_vals = {'Ua': 0.065846, 'Ub': 0.018052, 'inv_CpH': 0.14278, 'Uc': 0.018515, 'inv_CpS': 1 / 0.318, }

theta_vals = {'Ua': 0.065846, "Ub": 0.0148, 'inv_CpH': 0.14278, 'Uc': 0.018515, 'inv_CpS': 1 / 0.318, }

TC_lab_exp_instance = TC_Lab_experiment(data=tc_data, theta_initial=theta_vals, number_of_states=4)

m = TC_lab_exp_instance.get_labeled_model()

Th1_vals = [aml.value(m.Th1[t]) for t in m.t]
Ts1_vals = [aml.value(m.Ts1[t]) for t in m.t]
t_vals = [i for i in m.t]

from pyomo.contrib.doe import DesignOfExperiments

test_DoE = DesignOfExperiments(experiment=TC_lab_exp_instance, 
                               step=1e-2,
                               scale_constant_value=1,
                               scale_nominal_param_value=True, 
                               objective_option="trace", 
                               tee=True, 
                               logger_level=logging.INFO)

# test_DoE = DesignOfExperiments(experiment=TC_lab_exp_instance, 
                               # step=1e-2,
                               # scale_nominal_param_value=True, 
                               # objective_option="trace", 
                               # tee=True, 
                               # logger_level=logging.INFO)

prior_FIM = test_DoE.compute_FIM(method='kaug')

print(prior_FIM)

test_DoE.prior_FIM = prior_FIM

test_DoE.run_doe(results_file="test_solve.json")

with open("test_solve.json") as f:
    d = json.load(f)

plt.plot(range(901), d['Experiment Design'][:901], label='u1')
plt.plot(range(901), d['Experiment Design'][901:], label='u2')
plt.legend()
plt.show()
plt.clf()
plt.close()

TC_lab_exp_instance2 = TC_Lab_experiment(data=tc_data, theta_initial=theta_vals, number_of_states=4)

solver = aml.SolverFactory('ipopt')
# solver.options['bound_push'] = 1E-10
# solver.options['halt_on_ampl_error'] = 'yes'
solver.options['linear_solver'] = 'ma57'
# solver.options['max_iter'] = 9

test_DoE_second_exp = DesignOfExperiments(experiment=TC_lab_exp_instance2, 
                                          step=1e-2,
                                          scale_constant_value=1,
                                          scale_nominal_param_value=True, 
                                          objective_option="determinant", 
                                          prior_FIM=prior_FIM,  # CHANGE BACK FROM NUMBER 2 TO PLAIN
                                          L_diagonal_lower_bound=1e-6,
                                          tee=True,
                                          solver=solver,
                                          logger_level=logging.INFO,)


# test_DoE_second_exp = DesignOfExperiments(experiment=TC_lab_exp_instance2, 
                                          # step=1e-3,
                                          # scale_constant_value=1,
                                          # scale_nominal_param_value=True, 
                                          # objective_option="determinant", 
                                          # prior_FIM=prior_FIM2,  # CHANGE BACK FROM NUMBER 2 TO PLAIN
                                          # L_diagonal_lower_bound=1e-6,
                                          # tee=True,
                                          # solver=solver,
                                          # logger_level=logging.INFO,)

test_DoE_second_exp.run_doe(results_file="test_solve_d_opt.json")

with open("test_solve_d_opt.json") as f:
    d2 = json.load(f)

plt.plot(range(901), d['Experiment Design'][:901], label='u1')
plt.plot(range(901), d['Experiment Design'][901:], label='u2')
plt.legend()
plt.show()
plt.clf()
plt.close()