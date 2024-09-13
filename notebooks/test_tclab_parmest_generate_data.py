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

from pyomo.contrib.doe import DesignOfExperiments


def run_parmest_instance(df, num_states=2, tee=True):
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
    
    exp_list = []
    exp_list.append(TC_Lab_experiment(data=copy.deepcopy(tc_data), number_of_states=num_states))
    
    pest = parmest.Estimator(exp_list, obj_function='SSE', tee=tee)
    
    return pest.theta_est()


def run_FIM_instance(df, theta_init=None, num_states=2, tee=True):
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
    
    if theta_init is not None:
        TC_exp = TC_Lab_experiment(data=copy.deepcopy(tc_data), theta_initial=theta_init, number_of_states=num_states)
    else:
        TC_exp = TC_Lab_experiment(data=copy.deepcopy(tc_data), number_of_states=num_states)
    
    solver = aml.SolverFactory('ipopt')
    # solver.options['bound_push'] = 1E-10
    # solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['linear_solver'] = 'ma57'
    # solver.options['max_iter'] = 9)
    
    TC_Lab_DoE = DesignOfExperiments(experiment=TC_exp, 
                                     step=1e-2,
                                     scale_constant_value=1,
                                     scale_nominal_param_value=True, 
                                     objective_option="trace",                                
                                     tee=tee,
                                     solver=solver,                                 
                                     logger_level=logging.INFO)

    prior_FIM = TC_Lab_DoE.compute_FIM(method='sequential')
    
    return prior_FIM


# # Declaring num states
# num_states = 4

# beta_1_vals = []
# beta_2_vals = []
# beta_3_vals = []
# beta_4_vals = []
# if num_states == 4:
    # beta_5_vals = []
# # Run all experiments
# for k in [50, 2, 5, ]:
    # for i in range(3):
        # file = '../data/validation_experiment_env_2_sin_{}_50_run_{}.csv'.format(k, i + 1)
        
        # obj, theta = run_parmest_instance(file, num_states)
        # beta_1_vals.append(theta['beta_1'])
        # beta_2_vals.append(theta['beta_2'])
        # beta_3_vals.append(theta['beta_3'])
        # beta_4_vals.append(theta['beta_4'])
        # if num_states == 4:
            # beta_5_vals.append(theta['beta_5'])

# # Step experiments
# for i in range(3):
    # file = '../data/validation_experiment_env_2_step_50_run_{}.csv'.format(i + 1)
    
    # obj, theta = run_parmest_instance(file, num_states)
    # beta_1_vals.append(theta['beta_1'])
    # beta_2_vals.append(theta['beta_2'])
    # beta_3_vals.append(theta['beta_3'])
    # beta_4_vals.append(theta['beta_4'])
    # if num_states == 4:
            # beta_5_vals.append(theta['beta_5'])
    
# # Save to a file
# frame = pd.DataFrame(columns=['beta_1', 'beta_2', 'beta_3', 'beta_4'])

# frame['beta_1'] = beta_1_vals
# frame['beta_2'] = beta_2_vals
# frame['beta_3'] = beta_3_vals
# frame['beta_4'] = beta_4_vals
# if num_states == 4:
    # frame['beta_5'] = beta_5_vals

# frame.to_csv('ParmEst_results_from_single_experiments_{}_states.csv'.format(num_states), index=False)


if __name__=='__main__':
    # Declaring num states
    num_states = 2

    A_vals = []
    D_vals = []
    E_vals = []
    # Run all experiments
    for k in [50, 2, 5, ]:
        for i in range(3):
            file = '../data/validation_experiment_env_2_sin_{}_50_run_{}.csv'.format(k, i + 1)
            df = pd.read_csv(file)
            
            FIM = run_FIM_instance(df, num_states)
            A_vals.append(np.log10(np.trace(FIM)))
            D_vals.append(np.log10(np.linalg.det(FIM)))
            E_vals.append(np.log10(min(np.linalg.eig(FIM)[0])))

    # Step experiments
    for i in range(3):
        file = '../data/validation_experiment_env_2_step_50_run_{}.csv'.format(i + 1)
        df = pd.read_csv(file)
        
        FIM = run_FIM_instance(file, num_states)
        FIM = run_FIM_instance(file, num_states)
        A_vals.append(np.log10(np.trace(FIM)))
        D_vals.append(np.log10(np.linalg.det(FIM)))
        E_vals.append(np.log10(min(np.linalg.eig(FIM)[0])))
        
    # Save to a file
    frame2 = pd.DataFrame(columns=['beta_1', 'beta_2', 'beta_3', 'beta_4'])

    frame2['A'] = A_vals
    frame2['D'] = D_vals
    frame2['E'] = E_vals

    frame2.to_csv('FIM_results_from_single_experiments_{}_states.csv'.format(num_states), index=False)