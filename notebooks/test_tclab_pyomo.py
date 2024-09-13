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

theta_vals = {'Ua': 0.05147278733764012, 'Ub': 0.0005342082856927798, 'inv_CpH': 0.14622879403418604, 'inv_CpS': 99.99999754623846, }

# Values without the Th1 fitting objective function value?
# theta_vals = {'inv_CpH': 0.148000, 'inv_CpS': 25.098339, 'Ub': 0.002072, 'Ua': 0.051500}

theta_vals = {'Ua': 0.065846, 'Ub': 0.018052, 'inv_CpH': 0.14278, 'inv_CpS': 0.018515, }

TC_lab_exp_instance = TC_Lab_experiment(data=tc_data, theta_initial=theta_vals)

m = TC_lab_exp_instance.get_labeled_model()

# solver = aml.SolverFactory('ipopt')

# solver.solve(m, tee=True)

Th1_vals = [aml.value(m.Th1[t]) for t in m.t]
Ts1_vals = [aml.value(m.Ts1[t]) for t in m.t]
t_vals = [i for i in m.t]

# plt.plot(t_vals, Th1_vals)
# plt.plot(t_vals, Ts1_vals)
# plt.legend(['H', 'S'])
# plt.show()
# plt.clf()
# plt.close()

from pyomo.contrib.doe import DesignOfExperiments

test_DoE = DesignOfExperiments(experiment=TC_lab_exp_instance, 
                               step=1e-2,
                               scale_constant_value=1e-3,
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

prior_FIM2 = test_DoE.compute_FIM(method='kaug')

# True with 1e-3 constant scaling
# [[ 9.98610287 -0.10656653 -1.23767564 -0.10849711]
# [-0.10656653  0.02741872  0.1017977   0.02754532]
# [-1.23767564  0.1017977   0.62748393  0.10269213]
# [-0.10849711  0.02754532  0.10269213  0.02767323]]
prior_FIM1 = np.array(
[[ 9.98610287, -0.10656653, -1.23767564, -0.10849711],
[-0.10656653,  0.02741872,  0.1017977,   0.02754532],
[-1.23767564,  0.1017977,   0.62748393,  0.10269213],
[-0.10849711,  0.02754532,  0.10269213,  0.02767323],]
)



prior_FIM = np.array([[ 9986102.87166119,  -106566.52635245, -1237675.63607192,  -108497.11144836],
                      [ -106566.52635245,    27418.72154292,   101797.69950907,    27545.32287426],
                      [-1237675.63607192,   101797.69950907,   627483.93440371,   102692.13341657],
                      [ -108497.11144836,    27545.32287426,   102692.13341657,    27673.2310662 ]])

print(prior_FIM2)

test_DoE.prior_FIM = prior_FIM2  # SWITCH BACK TO NOT HAVE A 1

test_DoE.run_doe(results_file="test_solve.json")

with open("test_solve.json") as f:
    d = json.load(f)

plt.plot(range(len(d['Experiment Design'])), d['Experiment Design'])
plt.show()
plt.clf()
plt.close()

print(np.log(np.linalg.cond(prior_FIM2)))
print(np.log(np.linalg.cond(prior_FIM1)))
print(np.log(np.linalg.cond(prior_FIM)))

print(np.linalg.eig(prior_FIM2))
print(np.linalg.eig(d['FIM']))


#####################################################################
# # Performing full factorial exploration of sine wave test
# # Make design ranges to compute the full factorial design
# TC_lab_sine_wave_exp = TC_Lab_experiment(data=tc_data, theta_initial=theta_vals, sine_amplitude=20, sine_period=5)

# test_DoE_sine_wave = DesignOfExperiments(experiment=TC_lab_sine_wave_exp, 
                                         # step=1e-2, 
                                         # scale_nominal_param_value=True, 
                                         # objective_option="trace", 
                                         # tee=False, 
                                         # logger_level=logging.INFO)

# design_ranges = {"u1_amplitude": [15, 45, 4], "u1_period": [1, 8, 8]}

# # Compute the full factorial design with the sequential FIM calculation
# test_DoE_sine_wave.compute_FIM_full_factorial(design_ranges=design_ranges, method="sequential")

# # Plot the results
# test_DoE_sine_wave.draw_factorial_figure(
    # sensitivity_design_variables=["u1_period", "u1_amplitude"],
    # fixed_design_variables = {},
    # title_text="Sine Wave Test",
    # xlabel_text="Amplitude [% power]",
    # ylabel_text="Period [min]",
    # figure_file_name="example_TCLab_compute_FIM_seq",
    # log_scale=False,
# )


# # Create initial point from previous experiment as the initial point
# new_u1_values = d['Experiment Design']

# tc_data2 = TC_Lab_data(
    # name="Step Test for Heater 1",
    # time=df['Time'].values,
    # T1=df['T1'].values,
    # u1=new_u1_values,
    # P1=200,
    # TS1_data=None,
    # T2=df['T2'].values,
    # u2=df['Q2'].values,
    # P2=200,
    # TS2_data=None,
    # Tamb=df['T1'].values[0],
# )
#################################################################################

tc_data2 = TC_Lab_data(
    name="Sine Wave Test for Heater 1",
    time=df['Time'].values,
    T1=df['T1'].values,
    u1=[0.5] * len(df['Q1'].values),  # SWITCH THIS BACK TO MAKE NOT STEP TEST
    P1=200,
    TS1_data=None,
    T2=df['T2'].values,
    u2=df['Q2'].values,
    P2=200,
    TS2_data=None,
    Tamb=df['T1'].values[0],
)


TC_lab_exp_instance2 = TC_Lab_experiment(data=tc_data, theta_initial=theta_vals)

solver = aml.SolverFactory('ipopt')
# solver.options['bound_push'] = 1E-10
# solver.options['halt_on_ampl_error'] = 'yes'
solver.options['linear_solver'] = 'ma57'
# solver.options['max_iter'] = 9

test_DoE_second_exp = DesignOfExperiments(experiment=TC_lab_exp_instance2, 
                                          step=1e-2,
                                          scale_constant_value=1e-3,
                                          scale_nominal_param_value=True, 
                                          objective_option="determinant", 
                                          prior_FIM=prior_FIM2,  # CHANGE BACK FROM NUMBER 2 TO PLAIN
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

plt.plot(range(len(d2['Experiment Design'])), d2['Experiment Design'])
plt.show()
plt.clf()
plt.close()