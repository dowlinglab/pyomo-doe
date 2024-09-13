# Functions to compute FIM and run the parmest instance
from test_tclab_parmest_generate_data import run_FIM_instance, run_parmest_instance

# TC Lab specific objects
from tclab_pyomo import TC_Lab_experiment, TC_Lab_data

# Requirements for analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# In case we need pyomo
import pyomo.environ as aml

import copy
import time as t


# New function to set time delay
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


# Declare settings for runs
candidate_time_delays = range(10)
num_states = 2

run = []
period = []
delays = []
# Data collection
A_vals = []
D_vals = []
E_vals = []
ME_vals = []

obj_vals = []
beta_1_vals = []
beta_2_vals = []
beta_3_vals = []
beta_4_vals = []
if num_states == 4:
    beta_5_vals = []

# Try to find an optimal time delay
for tau in candidate_time_delays:
    time_delay = tau + 1
    t0 = t.time()
    # Sine waves
    for k in [50, 2, 5, ]:
        for i in range(3):
            file = '../data/validation_experiment_env_2_sin_{}_50_run_{}.csv'.format(k, i + 1)
            df = pd.read_csv(file)
            
            # Run stats
            run.append(i + 1)
            if k == 50:
                period.append(0.5)
            else:
                period.append(k)
            delays.append(time_delay)
            # Shift data by a time delay
            shifted_df = time_delay_data(df, time_delay)
            
            # Parameter estimation
            obj, theta = run_parmest_instance(shifted_df, num_states, tee=False)
            obj_vals.append(obj)
            beta_1_vals.append(theta['beta_1'])
            beta_2_vals.append(theta['beta_2'])
            beta_3_vals.append(theta['beta_3'])
            beta_4_vals.append(theta['beta_4'])
            if num_states == 4:
                beta_5_vals.append(theta['beta_5'])
            
            # Reparameterization backtracking for theta_initial
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
            
            theta_init = {"Ua": Ua, "Ub": Ub, "inv_CpH": inv_CpH, "inv_CpS": inv_CpS, "Uc": Uc,}
            
            # Computing FIM
            FIM = run_FIM_instance(shifted_df, theta_init, num_states, tee=False)
            A_vals.append(np.log10(np.trace(FIM)))
            D_vals.append(np.log10(np.linalg.det(FIM)))
            E_vals.append(np.log10(min(np.linalg.eig(FIM)[0])))
            ME_vals.append(np.log10(np.linalg.cond(FIM)))

    # Step experiments
    for i in range(3):
        file = '../data/validation_experiment_env_2_step_50_run_{}.csv'.format(i + 1)
        df = pd.read_csv(file)
        
        # Run stats
        run.append(i + 1)
        period.append(0)
        delays.append(time_delay)
        
        # Shift data by a time delay
        shifted_df = time_delay_data(df, time_delay)
        
        # Performing Parmaeter Estimation
        obj, theta = run_parmest_instance(shifted_df, num_states, tee=False)
        obj_vals.append(obj)
        beta_1_vals.append(theta['beta_1'])
        beta_2_vals.append(theta['beta_2'])
        beta_3_vals.append(theta['beta_3'])
        beta_4_vals.append(theta['beta_4'])
        if num_states == 4:
                beta_5_vals.append(theta['beta_5'])
        
        # Reparameterization backtracking for theta_initial
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
        
        theta_init = {"Ua": Ua, "Ub": Ub, "inv_CpH": inv_CpH, "inv_CpS": inv_CpS, "Uc": Uc,}
        
        # Computing FIM
        FIM = run_FIM_instance(shifted_df, theta_init, num_states, tee=False)
        A_vals.append(np.log10(np.trace(FIM)))
        D_vals.append(np.log10(np.linalg.det(FIM)))
        E_vals.append(np.log10(min(np.linalg.eig(FIM)[0])))
        ME_vals.append(np.log10(np.linalg.cond(FIM)))
    
    t1 = t.time()
    print('Time delay of {} took {:.2f} seconds.'.format(time_delay, t1 - t0))

# Save the data
frame = pd.DataFrame(columns=['beta_1', 'beta_2', 'beta_3', 'beta_4'])

frame['beta_1'] = beta_1_vals
frame['beta_2'] = beta_2_vals
frame['beta_3'] = beta_3_vals
frame['beta_4'] = beta_4_vals
if num_states == 4:
    frame['beta_5'] = beta_5_vals
frame['A'] = A_vals
frame['D'] = D_vals
frame['E'] = E_vals
frame['ME'] = ME_vals
frame['obj'] = obj_vals
frame['delay'] = delays
frame['run'] = run
frame['period'] = period

frame.to_csv('Surveying_time_delay_0_through_10_u_to_0.csv', index=False)