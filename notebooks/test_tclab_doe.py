from tclab_pyomo import TC_Lab_experiment, TC_Lab_data
import pandas as pd

from idaes.core.util import DiagnosticsToolbox

import logging
import json
import matplotlib.pyplot as plt
import numpy as np

import pyomo.environ as aml

from pyomo.contrib.doe import DesignOfExperiments

# Set number of states
number_of_states = 2

# Specify previous data files
data_files = ['../data/validation_experiment_env_2_sin_5_50_run_{}.csv'.format(i + 1) for i in range(1)]
# data_files = ['../data/validation_experiment_env_2_step_50_run_{}.csv'.format(i + 1) for i in range(1)]

skip = 6

# Make data objects for the data files
tc_data_objs = [0, ]
for ind, file in enumerate(data_files):
    df = pd.read_csv(file)
    
    tc_data_objs[ind] = TC_Lab_data(
        name="Sine Wave Test {} for Heater 1".format(ind + 1),
        time=df['Time'].values[::skip],
        T1=df['T1'].values[::skip],
        u1=df['Q1'].values[::skip],
        P1=200,
        TS1_data=None,
        T2=df['T2'].values[::skip],
        u2=df['Q2'].values[::skip],
        P2=200,
        TS2_data=None,
        Tamb=df['T1'].values[0],
    )

# Specify the initial guess for parameters, reparametrized (from parmest)
#theta_vals = {'beta_1': 0.00612210, 'beta_2': 0.00444966, 'beta_3': 0.04704280, 'beta_4': 0.00590863, 'beta_5': 0.00247446,}
if number_of_states == 4:
    theta_vals = {'Ua': 0.03158196, "Ub": 0.01390962, 'inv_CpH': 0.17983526, 'Uc': 0.01401547, 'inv_CpS': 3.45005093, }
else:
    theta_vals = {'Ua': 0.041705, "Ub": 0.009441, 'inv_CpH': 0.165909, 'Uc': 0.01340118, 'inv_CpS': 5.835756, }

# Gather initial FIM
n_param = 4 + 1 * (number_of_states == 4)
FIM = np.zeros((n_param, n_param))
for data in tc_data_objs:
    TC_lab_exp_instance = TC_Lab_experiment(data=data, theta_initial=theta_vals, number_of_states=number_of_states)
    
    # Calculate FIM and add to current FIM for prior
    TC_Lab_DoE = DesignOfExperiments(experiment=TC_lab_exp_instance, 
                                     step=1e-2,
                                     scale_constant_value=1,
                                     scale_nominal_param_value=True, 
                                     objective_option="trace", 
                                     tee=True, 
                                     logger_level=logging.INFO)
    
    prior_FIM = TC_Lab_DoE.compute_FIM(method='sequential')
    FIM += prior_FIM

# # For one experiment
# df = pd.read_csv(data_files[0])
    
# tc_data_obj = TC_Lab_data(
    # name="Sine Wave Test {} for Heater 1".format(ind + 1),
    # time=df['Time'].values,
    # T1=df['T1'].values,
    # u1=df['Q1'].values,
    # P1=200,
    # TS1_data=None,
    # T2=df['T2'].values,
    # u2=df['Q2'].values,
    # P2=200,
    # TS2_data=None,
    # Tamb=df['T1'].values[0],
# )

# TC_lab_exp_instance = TC_Lab_experiment(data=tc_data_obj, theta_initial=theta_vals, number_of_states=number_of_states)

# # Diagnostics run on the model (found that it wasn't square)
# # dt = DiagnosticsToolbox(TC_lab_exp_instance.get_labeled_model())
# # dt.report_structural_issues()
# # dt.display_underconstrained_set()

# TC_Lab_DoE = DesignOfExperiments(experiment=TC_lab_exp_instance, 
                                 # step=1e-2,
                                 # scale_constant_value=1,
                                 # scale_nominal_param_value=True, 
                                 # objective_option="trace", 
                                 # tee=True, 
                                 # logger_level=logging.INFO)

# prior_FIM = TC_Lab_DoE.compute_FIM(method='sequential')
# FIM += prior_FIM

print(FIM)

eig, eig_vec = np.linalg.eig(FIM)
for i in range(4):
    print('Eigenvalue:\n{:.5f}'.format(eig[i]))
    print('Eigenvector: ')
    print(eig_vec[:, i])

# Solver for optimal DoE solve
solver = aml.SolverFactory('ipopt')
# solver.options['bound_push'] = 1E-10
# solver.options['halt_on_ampl_error'] = 'yes'
solver.options['linear_solver'] = 'ma57'
# solver.options['acceptable_tol'] = 1e-8
# solver.options['max_iter'] = 9)

TC_lab_exp_instance = TC_Lab_experiment(data=tc_data_objs[0], theta_initial=theta_vals, number_of_states=number_of_states)

TC_Lab_DoE = DesignOfExperiments(experiment=TC_lab_exp_instance, 
                                 step=1e-2,
                                 scale_constant_value=1,
                                 scale_nominal_param_value=True, 
                                 objective_option="trace",
                                 prior_FIM=prior_FIM,                                
                                 tee=True,
                                 solver=solver,                                 
                                 logger_level=logging.INFO)

TC_Lab_DoE.run_doe(results_file="test_solve_A_opt_{}_states_1_prior_no_reparam.json".format(number_of_states))

with open("test_solve_A_opt_{}_states_1_prior_no_reparam.json".format(number_of_states)) as f:
    d = json.load(f)

plt.plot(range(len(d['Experiment Design'][:901])), d['Experiment Design'][:901], label='u1')
# plt.plot(range(901), d['Experiment Design'][901:], label='u2')
plt.legend()
plt.show()
plt.clf()
plt.close()

print(TC_Lab_DoE.results['FIM'])
