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

TC_lab_exp_instance = TC_Lab_experiment(data=tc_data, theta_initial=theta_vals)

m = TC_lab_exp_instance.get_labeled_model()

from pyomo.contrib.doe import DesignOfExperiments

my_df = pd.DataFrame(columns=['Constant Scaling', 'Scale Param Bool', 'Optimal FIM Resulting Condition Number'])
k_vals = []
scale_vals = []
param_scale = []

# Loop to check condition number of FIM for various scale factors
for i in range(5):
    scale = 10 ** (-i)
    for j in [True, ]:
        test_DoE = DesignOfExperiments(experiment=TC_lab_exp_instance, 
                                       step=1e-2,
                                       scale_constant_value=scale,
                                       scale_nominal_param_value=j, 
                                       objective_option="trace", 
                                       tee=False, 
                                       logger_level=logging.ERROR)

        FIM = test_DoE.compute_FIM(method='kaug')
        
        test_DoE.prior_FIM = FIM

        test_DoE.run_doe(results_file="test_solve.json")

        with open("test_solve.json") as f:
            d = json.load(f)

        plt.plot(range(len(d['Experiment Design'])), d['Experiment Design'])
        plt.savefig('testing_FIM_condition_number_solve_{}_{}.png'.format(i, int(j)), format='png', dpi=450)
        plt.clf()
        plt.close()
        
        print(i, j)
        print(np.log(np.linalg.cond(d['FIM'])))
        
        k_vals.append(np.log(np.linalg.cond(d['FIM'])))
        scale_vals.append(scale)
        param_scale.append(j)

my_df['Optimal FIM Resulting Condition Number'] = k_vals
my_df['Constant Scaling'] = scale_vals
my_df['Scale Param Bool'] = param_scale

my_df.to_csv('diagnose_k_A-opt_run_statistics.csv', index=False)
