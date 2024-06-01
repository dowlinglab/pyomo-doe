import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pandas as pd

import shutil
import sys
import os.path
import os
import re

import subprocess

### -------------- Part 1: Install software -------------- ###

###### The code below was adapted from IDAES
# And this covered under the IDAES license
# https://github.com/IDAES/idaes-pse/blob/main/scripts/colab_helper.py


def _check_available(executable_name):
    """Utility to check in an executable is available"""
    return shutil.which(executable_name) or os.path.isfile(executable_name)


def _update_path():
    """Add idaes executables to PATH"""
    if not re.search(re.escape("/root/.idaes/bin/"), os.environ["PATH"]):
        os.environ["PATH"] = "/root/.idaes/bin/:" + os.environ["PATH"]


def _print_single_solver_version(solvername):
    """Print the version for a single solver
    Arg:
        solvername: solver executable name (string)
    """
    v = subprocess.run([solvername, "-v"], check=True, capture_output=True, text=True)
    print(v.stdout)
    print(v.stderr)


def _print_solver_versions():
    """Print versions of solvers in idaes get-extensions

    This is the primary check that solvers installed correctly and are callable
    """

    # This does not work for cbc and clp; calling --version with these solvers,
    # enters their scripting language mode.
    for s in ["ipopt", "k_aug", "couenne", "bonmin", "ipopt_l1", "dot_sens"]:
        _print_single_solver_version(s)


# Install software if on Google colab
if "google.colab" in sys.modules:

    verbose = True

    # Install IDAES
    try:
        import idaes

        print("idaes was found! No need to install.")
    except ImportError:
        print("Installing idaes via pip...")
        v = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "idaes_pse"],
            check=True,
            capture_output=True,
            text=True,
        )
        if verbose:
            print(v.stdout)
            print(v.stderr)
        print("idaes was successfully installed")
        v = subprocess.run(
            ["idaes", "--version"], check=True, capture_output=True, text=True
        )
        print(v.stdout)
        print(v.stderr)

    # Install Ipopt
    if not _check_available("ipopt"):
        print("Running idaes get-extensions to install Ipopt, k_aug, and more...")
        v = subprocess.run(
            ["idaes", "get-extensions"], check=True, capture_output=True, text=True
        )
        if verbose:
            print(v.stdout)
            print(v.stderr)
        _update_path()
        print("Checking solver versions:")
        _print_solver_versions()

    # Check if correct version of Pyomo is installed
    def _check_pyomo_installed_old():
        '''
        This is no longer needed because improvements to Pyomo.DoE have been merged into
        the next release of Pyomo. This function is kept for reference.
        '''

        try:
            v = subprocess.run(
                ["pyomo", "--version"], check=True, capture_output=True, text=True
            )
            if "pyomo-doe-fixes" in v.stdout:
                reinstall_pyomo = False
                print("Correct version of Pyomo.DoE is installed.")
            else:
                reinstall_pyomo = True
        except FileNotFoundError:
            reinstall_pyomo = True

        return reinstall_pyomo

    # Install updated version of Pyomo
    # No longer needed because improvements to Pyomo.DoE have been 
    # merged into the next release of Pyomo
    '''
    if _check_pyomo_installed_old():
        print("Installing updated version of Pyomo.DoE...")
        print("  (this takes up to 5 minutes)")
        v = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "git+https://github.com/adowling2/pyomo.git@pyomo-doe-fixes",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if verbose:
            print(v.stdout)
            print(v.stderr)
        _check_pyomo_installed()
    '''

    import idaes

    print("Finished installing software")

###### End note

### -------------- Part 2: Load libraries -------------- ###

# Need to import IDAES for Ipopt
# This is important for running on local machines
import idaes

from pyomo.contrib.doe import (
    ModelOptionLib,
    # DesignOfExperiments,
    # MeasurementVariables,
    # DesignVariables,
)

from pyomo.environ import (
    ConcreteModel,
    Var,
    Param,
    Constraint,
    TransformationFactory,
    SolverFactory,
    Objective,
    minimize,
    value,
    Suffix,
    Expression,
    sin,
)
from pyomo.dae import DerivativeVar, ContinuousSet, Simulator

# https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth=3)

from dataclasses import dataclass

### -------------- Part 3: Handling experimental data -------------- ###


@dataclass
class TCLabExperiment:
    """Class for storing data from a TCLab experiment."""

    name: str  # Name of the experiment (optional)
    time: np.array  # Time stamp for measurements, [seconds]
    T1: np.array  # Temperature of heater 1, [degC]
    u1: np.array  # Heater 1 power setting, [0-100]
    P1: float  # Power setting for heater 1, [W]
    TS1_data: np.array  # Setpoint data for temperature of sensor 1, [degC]
    T2: np.array  # Temperature of heater 2, [degC]
    u2: np.array  # Heater 2 power setting, [0-100]
    P2: float  # Power setting for heater 2, [W]
    TS2_data: np.array  # Setpoint data for temperature of sensor 1, [degC]
    Tamb: float  # Ambient temperature, [degC]

    def to_data_frame(self):
        """Convert instance of this class to a pandas DataFrame."""

        df = pd.DataFrame(
            {
                "time": self.time,
                "T1": self.T1,
                "u1": self.u1,
                "P1": self.P1,
                "TS1_data": self.TS1_data,
                "T2": self.T2,
                "u2": self.u2,
                "P2": self.P2,
                "TS2_data": self.TS2_data,
                "Tamb": self.Tamb,
            }
        )

        return df


### -------------- Part 4: Construct Pyomo Model -------------- ###


def create_model(
    data=None,  # TCLabExperiment instance
    alpha=0.00016,  # Conversion factor for TCLab (fixed parameter)
    theta={
        "Ua": 0.0535,
        "Ub": 0.0148,
        "inv_CpH": 1 / 6.911,
        "inv_CpS": 1 / 0.318,
        "Uc": 0.001,
    },  # initial guess
    mode='simulate',  # Mode of operation,
    number_of_states=2,  # Number of states in the model
    obj_weight_optimize=0.1,  # Weight in the tracking objective function
    obj_weight_observe=1.0,  # Weight in the observation objective function
    obj_weight_estimate=0.01,  # Weight in the estimation objective function
    time_finite_difference='BACKWARD',  # Finite difference scheme
    integrate_to_initialize=True,  # Integrate to initialize
    sine_period=None,  # Optional argument for sensitivity analysis of sine ID test
    sine_amplitude=None,  # Optional argument for sensitivity analysis of sine ID test
):
    """ """

    m = ConcreteModel()

    if number_of_states == 2:
        m.four_states = False
    elif number_of_states == 4:
        m.four_states = True
    else:
        raise ValueError("number_of_states must be 2 or 4.")

    # Support data as either TCLabExperiment or DataFrame instance
    if isinstance(data, pd.DataFrame):
        # Dataframe
        Tamb = data.Tamb.values[0]
        P1 = data.P1.values[0]
        P2 = data.P2.values[0]
        time = data.time.values

        u1 = data.u1.values
        T1 = data.T1.values
        TS1_data = data.TS1_data.values

        u2 = data.u2.values
        T2 = data.T2.values
        TS2_data = data.TS2_data.values

    else:
        Tamb = data.Tamb
        P1 = data.P1
        P2 = data.P2
        time = data.time

        u1 = data.u1
        T1 = data.T1
        TS1_data = data.TS1_data

        u2 = data.u2
        T2 = data.T2
        TS2_data = data.TS2_data

    valid_modes = ['optimize', 'parmest', 'doe']

    if mode not in valid_modes:
        raise ValueError("mode needs to be one of" + valid_modes + ".")

    if mode == 'doe' and sine_amplitude is not None and sine_period is not None:

        sine_period_max = 10  # minutes
        sine_period_min = 10 / 60  # minutes

        assert sine_amplitude <= 50, "Sine amplitude must be less than 50."
        assert sine_amplitude >= 0, "Sine amplitude must be greater than 0."

        assert sine_period <= sine_period_max, "Sine period must be less than " + str(
            sine_period_max
        )
        assert (
            sine_period >= sine_period_min
        ), "Sine period must be greater than " + str(sine_period_min)

        # Create a copy to prevent overwriting the original data
        u1 = u1.copy()

        # Calculate parameterized control signal for u1
        u1 = 50 + sine_amplitude * np.sin(2 * np.pi / (sine_period * 60) * time)

    Tmax = 85.0  # Maximum temperature (K)

    # create the time set
    m.t = ContinuousSet(
        initialize=time
    )  # make sure the experimental time grid are discretization points
    # define the heater and sensor temperatures as variables
    m.Th1 = Var(m.t, bounds=[0, Tmax], initialize=Tamb)
    m.Ts1 = Var(m.t, bounds=[0, Tmax], initialize=Tamb)

    if m.four_states:
        m.Th2 = Var(m.t, bounds=[0, Tmax], initialize=Tamb)
        m.Ts2 = Var(m.t, bounds=[0, Tmax], initialize=Tamb)

    def helper(my_array):
        '''
        Method that builds a dictionary to help initialization.
        Arguments:
            my_array: an array
        Returns:
            data: a dict {time: array_value}
        '''
        # ensure that the dimensions of array and time data match
        assert len(my_array) == len(time), "Dimension mismatch."
        data2 = {}
        for k, t in enumerate(time):
            if my_array[k] is not None:
                data2[t] = my_array[k]
            else:
                # Replace None with 0
                data2[t] = 0
        return data2

    # for the simulate and observe modes
    if mode in ['parmest']:
        # control decision is a parameter initialized with the input control data dict
        m.U1 = Param(m.t, initialize=helper(u1), default=0)

        if m.four_states:
            m.U2 = Param(m.t, initialize=helper(u2), default=0)

    else:
        # otherwise (optimize, doe) control decisions are variables
        m.U1 = Var(m.t, bounds=(0, 100), initialize=helper(u1))

        if m.four_states:
            m.U2 = Var(m.t, bounds=(0, 100), initialize=helper(u2))

    # define parameters that do not depend on mode
    m.Tamb = Param(initialize=Tamb)
    m.P1 = Param(initialize=P1)
    m.alpha = Param(initialize=alpha)
    m.P2 = Param(initialize=P2)

    # for the simulate, optimize, observe modes
    if mode in ['optimize']:
        # Ua, Ub, CpH, and CpS are parameters
        m.Ua = Param(initialize=theta["Ua"])
        m.Ub = Param(initialize=theta["Ub"])
        # 1/CpH and 1/CpS parameters
        m.inv_CpH = Param(initialize=theta["inv_CpH"])
        m.inv_CpS = Param(initialize=theta["inv_CpS"])

        if m.four_states:
            m.Uc = Param(initialize=theta["Uc"])

    else:
        # otherwise, Ua, Ub, CpH, and CpS are variables
        m.Ua = Var(initialize=theta["Ua"], bounds=(1e-5, 2.0))
        m.Ub = Var(initialize=theta["Ub"], bounds=(1e-5, 2.0))
        if m.four_states:
            m.Uc = Var(initialize=theta["Uc"], bounds=(1e-5, 2.0))
        # 1/CpH and 1/CpS variables
        if mode == 'doe':
            expand_bounds = 1.01
        else:
            expand_bounds = 1.0
        m.inv_CpH = Var(
            initialize=theta["inv_CpH"],
            bounds=(1e-2 / expand_bounds, 100 * expand_bounds),
        )
        m.inv_CpS = Var(
            initialize=theta["inv_CpS"],
            bounds=(1e-1 / expand_bounds, 100 * expand_bounds),
        )

    # define variables for change in temperature wrt to time
    m.Th1dot = DerivativeVar(m.Th1, wrt=m.t)
    m.Ts1dot = DerivativeVar(m.Ts1, wrt=m.t)

    if m.four_states:
        m.Th2dot = DerivativeVar(m.Th2, wrt=m.t)
        m.Ts2dot = DerivativeVar(m.Ts2, wrt=m.t)

    # define differential equations (model) as constraints
    # moved Cps to the right hand side to diagnose integrator
    if not m.four_states:
        m.Th_ode = Constraint(
            m.t,
            rule=lambda m, t: m.Th1dot[t]
            == (
                m.Ua * (m.Tamb - m.Th1[t])
                + m.Ub * (m.Ts1[t] - m.Th1[t])
                + m.alpha * m.P1 * m.U1[t]
            )
            * m.inv_CpH,
        )

        m.Ts_ode = Constraint(
            m.t,
            rule=lambda m, t: m.Ts1dot[t] == (m.Ub * (m.Th1[t] - m.Ts1[t])) * m.inv_CpS,
        )

    else:
        m.Th1_ode = Constraint(
            m.t,
            rule=lambda m, t: m.Th1dot[t]
            == (
                m.Ua * (m.Tamb - m.Th1[t])
                + m.Ub * (m.Ts1[t] - m.Th1[t])
                + m.Uc * (m.Ts2[t] - m.Th1[t])
                + m.alpha * m.P1 * m.U1[t]
            )
            * m.inv_CpH,
        )

        m.Ts1_ode = Constraint(
            m.t,
            rule=lambda m, t: m.Ts1dot[t] == (m.Ub * (m.Th1[t] - m.Ts1[t])) * m.inv_CpS,
        )

        m.Th2_ode = Constraint(
            m.t,
            rule=lambda m, t: m.Th2dot[t]
            == (
                m.Ua * (m.Tamb - m.Th2[t])
                + m.Ub * (m.Ts2[t] - m.Th2[t])
                + m.Uc * (m.Th1[t] - m.Th2[t])
                + m.alpha * m.P2 * m.U2[t]
            )
            * m.inv_CpH,
        )

        m.Ts2_ode = Constraint(
            m.t,
            rule=lambda m, t: m.Ts2dot[t] == (m.Ub * (m.Th2[t] - m.Ts2[t])) * m.inv_CpS,
        )

    if integrate_to_initialize:

        m.var_input = Suffix(direction=Suffix.LOCAL)

        if u1 is not None:
            # initialize with data
            m.var_input[m.U1] = helper(u1)
        else:
            # otherwise initialize control decision of 0
            m.var_input[m.U1] = {0: 0}

        if m.four_states:
            if u2 is not None:
                # initialize with data
                m.var_input[m.U2] = helper(u2)
            else:
                # otherwise initialize control decision of 0
                m.var_input[m.U2] = {0: 0}

        # Simulate to initialize
        # Makes the solver more efficient
        sim = Simulator(m, package='scipy')
        tsim, profiles = sim.simulate(
            numpoints=100, integrator='vode', varying_inputs=m.var_input
        )
        sim.initialize_model()

    # for the optimize mode, set point data is a parameter
    if mode == 'optimize':
        m.Tset1 = Param(m.t, initialize=helper(TS1_data))

        if m.four_states:
            m.Tset2 = Param(m.t, initialize=helper(TS2_data))

        # otherwise, we are not using it

    # for the estimate mode, experimental data is a parameter
    if mode == 'parmest':
        m.Ts1_measure = Param(m.t, initialize=helper(T1))

        if m.four_states:
            m.Ts2_measure = Param(m.t, initialize=helper(T2))

        # otherwise, we are not using it

    # apply backward finite difference to the model
    TransformationFactory('dae.finite_difference').apply_to(
        m, scheme=time_finite_difference, nfe=len(time) - 1
    )

    if mode == 'optimize':
        # defining the tracking objective function
        if not m.four_states:
            m.obj = Objective(
                expr=sum(
                    (m.Ts1[t] - m.Tset1[t]) ** 2
                    + obj_weight_optimize * (m.Th1[t] - m.Tset1[t]) ** 2
                    for t in m.t
                ),
                sense=minimize,
            )

        else:
            m.obj = Objective(
                expr=sum(
                    (m.Ts1[t] - m.Tset1[t]) ** 2
                    + obj_weight_optimize * (m.Th1[t] - m.Tset1[t]) ** 2
                    + (m.Ts2[t] - m.Tset2[t]) ** 2
                    + obj_weight_optimize * (m.Th2[t] - m.Tset2[t]) ** 2
                    for t in m.t
                ),
                sense=minimize,
            )

    if mode == 'parmest':
        m.FirstStageCost = Expression(expr=0)
        m.SecondStageCost = Expression(
            expr=sum(
                (m.Ts1[t] - m.Ts1_measure[t]) ** 2
                + obj_weight_estimate * (m.Th1[t] - m.Ts1_measure[t]) ** 2
                for t in m.t
            )
        )
        m.Total_Cost_Objective = Objective(
            expr=m.FirstStageCost + m.SecondStageCost, sense=minimize
        )

    if mode == 'doe' and sine_amplitude is not None and sine_period is not None:

        # Add measurement control decision variables
        m.u1_period = Var(
            initialize=sine_period, bounds=(sine_period_min, sine_period_max)
        )  # minutes
        m.u1_amplitude = Var(initialize=sine_amplitude)  # % power

        # Add constraint to calculate u1
        m.u1_constraint = Constraint(
            m.t,
            rule=lambda m, t: m.U1[t]
            == 50 + m.u1_amplitude * sin(2 * np.pi / (m.u1_period * 60) * value(t)),
        )

    # initial conditions
    # For moving horizion we check if t=0 is in the horizon t data and fix initial conditions
    if time[0] == 0:
        if TS1_data is not None and TS1_data[0] is not None:
            # Initialize with first temperature measurement
            m.Th1[0].fix(TS1_data[0])
            m.Ts1[0].fix(TS1_data[0])
        else:
            # Initialize with ambient temperature
            m.Th1[0].fix(m.Tamb)
            m.Ts1[0].fix(m.Tamb)

        if m.four_states:
            if TS2_data is not None and TS2_data[0] is not None:
                # Initialize with first temperature measurement
                m.Th2[0].fix(TS2_data[0])
                m.Ts2[0].fix(TS2_data[0])
            else:
                # Initialize with ambient temperature
                m.Th2[0].fix(m.Tamb)
                m.Ts2[0].fix(m.Tamb)

    # otherwise, we will use the 'set_initial_conditions'

    # for the optimize mode, add constraints to fix the control input at the beginning and end of the horizon
    # this is because in backward finite difference, u[0] has not impact on the solution
    # likewise, for forward finite difference, u[-1] has no impact on the solution
    if mode == 'optimize':

        if time_finite_difference == 'BACKWARD':
            # Remember that Pyomo is 1-indexed, which means '1' is the first element of the time set
            m.first_u = Constraint(expr=m.U1[m.t.at(1)] == m.U1[m.t.at(2)])

        if time_finite_difference == 'FORWARD':
            m.last_u = Constraint(expr=m.U1[m.t.at(-1)] == m.U1[m.t.at(-2)])

    return m


### -------------- Part 5: Extract and visualize results -------------- ###


def extract_results(model, name="Pyomo results"):
    """Extract results from the Pyomo model"""

    time = np.array([value(t) for t in model.t])
    Th1 = np.array([value(model.Th1[t]) for t in model.t])
    Ts1 = np.array([value(model.Ts1[t]) for t in model.t])
    U1 = np.array([value(model.U1[t]) for t in model.t])
    P1 = value(model.P1)
    if not model.four_states:
        Th2 = None
        Ts2 = None
        U2 = None
    else:
        Th2 = np.array([value(model.Th2[t]) for t in model.t])
        Ts2 = np.array([value(model.Ts2[t]) for t in model.t])
        U2 = np.array([value(model.U2[t]) for t in model.t])
    P2 = model.P2
    Tamb = model.Tamb

    return TCLabExperiment(name, time, Th1, U1, P1, Ts1, Th2, U2, P2, Ts2, Tamb)


def extract_plot_results(tc_exp_data, model):
    """Extract and plot the results of the Pyomo model

    Arguments:
    ----------
    tc_exp_data: experimental data, TCLabExperiment instance
    model: Pyomo model

    Returns:
    --------
    solution: solution from Pyomo model, extracted and stored in TCLabExperiment instance

    """

    # For convenience, save in a shorter variable name
    if tc_exp_data is not None:
        exp = tc_exp_data
    else:
        exp = TCLabExperiment(
            None, None, None, None, None, None, None, None, None, None, None
        )

    mod = extract_results(model)

    # create figure
    plt.figure(figsize=(10, 6))

    # subplot 1: temperatures
    plt.subplot(2, 1, 1)

    colors = {
        'T1': 'orange',  # data
        'T2': 'green',  # data
        'Th1': 'red',  # model
        'Ts1': 'blue',  # model
        'Th2': 'purple',  # model
        'Ts2': 'brown',  # model
        'u1_data': 'orange',  # data
        'u2_data': 'green',  # data
        'u1_mod': 'red',  # model
        'u2_mod': 'purple',  # model
    }

    LW = 3.0  # line width

    four_states = (mod.TS2_data is not None) and (mod.T2 is not None)

    if exp.T1 is not None:
        plt.scatter(
            exp.time,
            exp.T1,
            marker='o',
            label="$T_{S,1}$ measured",
            alpha=0.5,
            color=colors["T1"],
        )
    if mod.TS1_data is not None:
        plt.plot(
            mod.time, mod.TS1_data, label="$T_{S,1}$ predicted", color=colors["Ts1"]
        )
    if mod.T1 is not None:
        plt.plot(
            mod.time,
            mod.T1,
            label="$T_{H,1}$ predicted",
            color=colors["Th1"],
            linestyle='--',
        )
    if exp.T2 is not None:
        plt.scatter(
            exp.time,
            exp.T2,
            marker='s',
            label="$T_{S,2}$ measured",
            alpha=0.5,
            color=colors["T2"],
        )
    if mod.TS2_data is not None:
        plt.plot(
            mod.time, mod.TS2_data, label="$T_{S,2}$ predicted", color=colors["Ts2"]
        )
    if mod.T2 is not None:
        plt.plot(
            mod.time,
            mod.T2,
            label="$T_{H,2}$ predicted",
            color=colors["Th2"],
            linestyle='--',
        )

    plt.ylabel('Temperature (°C)')

    if four_states:
        nc = 2
    else:
        nc = 1
    plt.legend(ncol=nc)

    # subplot 2: control decision
    plt.subplot(2, 1, 2)
    if exp.u1 is not None:
        plt.scatter(
            exp.time,
            exp.u1,
            marker='o',
            label="$u_1$ measured",
            color=colors['u1_data'],
            alpha=0.5,
        )
    if mod.u1 is not None:
        plt.plot(mod.time, mod.u1, label="$u_1$ predicted", color=colors["u1_mod"])
    if exp.u2 is not None:
        plt.scatter(
            exp.time,
            exp.u2,
            marker='s',
            label="$u_2$ measured",
            color=colors["u2_data"],
            alpha=0.5,
        )
    if mod.u2 is not None:
        plt.plot(mod.time, mod.u2, label="$u_2$ predicted", color=colors["u2_mod"])

    plt.ylabel('Heater Power (%)')
    plt.xlabel('Time (s)')
    plt.legend(ncol=nc)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Model parameters:")
    print("Ua =", round(value(model.Ua), 4), "Watts/degC")
    print("Ub =", round(value(model.Ub), 4), "Watts/degC")
    if model.four_states:
        print("Uc =", round(value(model.Uc), 4), "Watts/degC")
    print("CpH =", round(1 / value(model.inv_CpH), 4), "Joules/degC")
    print("CpS =", round(1 / value(model.inv_CpS), 4), "Joules/degC")

    if hasattr(model, 'u1_period'):
        print("u1_period =", round(value(model.u1_period), 2), "minutes")
    if hasattr(model, 'u1_amplitude'):
        print("u1_amplitude =", round(value(model.u1_amplitude), 4), "% power")

    print(" ")  # New line

    return mod


def results_summary(result):
    print("======Results Summary======")
    print("Four design criteria log10() value:")
    print("A-optimality:", np.log10(result.trace))
    print("D-optimality:", np.log10(result.det))
    print("E-optimality:", np.log10(result.min_eig))
    print("Modified E-optimality:", np.log10(result.cond))
    print("\nFIM:\n", result.FIM)

    eigenvalues, eigenvectors = np.linalg.eig(result.FIM)

    print("\neigenvalues:\n", eigenvalues)

    print("\neigenvectors:\n", eigenvectors)
