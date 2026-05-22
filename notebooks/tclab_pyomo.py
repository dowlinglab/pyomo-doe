import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pandas as pd

import shutil
import sys
import os.path
import os
import re
import json

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
        ### Install newest version of IDAES
        v = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "idaes_pse"],
            check=True,
            capture_output=True,
            text=True,
        )
        ### Pin to a specific version of IDAES
        '''
        v = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-qv", "idaes_pse==2.7.0"],
            check=True,
            capture_output=True,
            text=True,
        )
        '''
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
    def _check_pyomo_installed():
        '''
        Check whether Pyomo was installed from the workshop git branch.
        '''

        try:
            import importlib.metadata as md
        except ImportError:
            return False

        expected_url = "https://github.com/dowlinglab/pyomo.git"
        expected_revision = "pyomo-doe-workshop-2026"

        try:
            dist = md.distribution("pyomo")
        except md.PackageNotFoundError:
            return False

        direct_url_text = dist.read_text("direct_url.json")
        if not direct_url_text:
            return False

        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            return False

        vcs_info = direct_url.get("vcs_info", {})
        if vcs_info.get("vcs") != "git":
            return False

        if vcs_info.get("requested_revision") != expected_revision:
            return False

        if direct_url.get("url") != expected_url:
            return False

        print("Correct git branch of Pyomo.DoE is installed.")
        return True

    # Install updated version of Pyomo
    
    if not _check_pyomo_installed():
        print("Installing updated version of Pyomo.DoE...")
        print("  (this takes up to 5 minutes)")
        v = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "git+https://github.com/dowlinglab/pyomo.git@pyomo-doe-workshop-2026",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if verbose:
            print(v.stdout)
            print(v.stderr)
        if not _check_pyomo_installed():
            raise RuntimeError(
                "Pyomo was installed, but not from the expected git branch."
            )
    

    import idaes

    print("Finished installing software")

###### End note

### -------------- Part 2: Load libraries -------------- ###

# Need to import IDAES for Ipopt
# This is important for running on local machines
# TODO: uncomment this
# import idaes
# from idaes.core.util import DiagnosticsToolbox

from pyomo.contrib.parmest.graphics import profile_likelihood_plot
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe import DesignOfExperiments

from pyomo.environ import (
    ConcreteModel,
    Var,
    Param,
    Constraint,
    TransformationFactory,
    SolverFactory,
    Objective,
    minimize,
    value as pyovalue,
    Suffix,
    Expression,
    PositiveReals,
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
class TC_Lab_data:
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

### -------------- Part 3.1: Helper function for initializing the model -------------- ###
def helper(my_array, time):
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

### -------------- Part 4 v 2: Create Experiment object -------------- ###
class TC_Lab_experiment(Experiment):
    def __init__(
        self,
        data,
        alpha=0.00016,
        theta_initial=None,
        number_of_states=2,
        reparam=False,
        measurement_error=0.25,
    ):
        """
        Arguments
        ---------
        data: TC_Lab_Data object
        alpha: float, Conversion factor for TCLab (fixed parameter)
        theta_initial: dictionary, initial guesses for the unknown parameters
        number_of_states: number of states in the heat transfer model (must be 2 or 4), default: 2
        measurement_error: float, constant measurement error of sensor 1, default: 0.25 deg C
        
        """
        self.data = data
        
        if theta_initial is None:
            self.theta_initial={
                "Ua": 0.0535,
                "Ub": 0.0148,
                "inv_CpH": 1 / 6.911,
                "inv_CpS": 1 / 0.318,
                "Uc": 0.001,
            }
        else:
            self.theta_initial = theta_initial
        
        # TODO: Move alpha to the data object?
        self.alpha = alpha
        
        # Make sure that the number of states is either 2 or 4
        if number_of_states not in [2, 4, ]:
            raise ValueError("number_of_states must be 2 or 4.")
        self.number_of_states = number_of_states
        
        self.reparam = reparam
        self.measurement_error = measurement_error
        
        self.model = None
    
    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment()
        return self.model
    
    def create_model(self):
        """
        Method to create an unlabled model of the TC Lab system.
        
        """
        m = self.model = ConcreteModel()
        
        #########################################
        # Begin model constants definition
        m.Tamb = Param(initialize=self.data.Tamb)
        m.P1 = Param(initialize=self.data.P1)
        m.alpha = Param(initialize=self.alpha)
        m.P2 = Param(initialize=self.data.P2)
        
        m.Tmax = 85  # Maximum temparture (Deg C)
        
        # End model constants
        #########################################
        
        ################################
        # Defining state variables
        m.t = ContinuousSet(initialize=self.data.time)
        
        # Temperature states for the fins
        m.Th1 = Var(m.t, bounds=[0, m.Tmax], initialize=m.Tamb.value)
        m.Ts1 = Var(m.t, bounds=[0, m.Tmax], initialize=m.Tamb.value)

        if self.number_of_states == 4:
            m.Th2 = Var(m.t, bounds=[0, m.Tmax], initialize=m.Tamb.value)
            m.Ts2 = Var(m.t, bounds=[0, m.Tmax], initialize=m.Tamb.value)
        
        # Derivatives of the temperature state variables
        m.Th1dot = DerivativeVar(m.Th1, wrt=m.t)
        m.Ts1dot = DerivativeVar(m.Ts1, wrt=m.t)

        if self.number_of_states == 4:
            m.Th2dot = DerivativeVar(m.Th2, wrt=m.t)
            m.Ts2dot = DerivativeVar(m.Ts2, wrt=m.t)
        
        # End state variable definition
        ################################

        ####################################
        # Defining experimental inputs
        
        # Add control variables (experimental design decisions)
        m.U1 = Var(m.t, bounds=(0, 100), initialize=helper(self.data.u1, self.data.time))
        m.U1.fix()  # Fixed for parameter estimation

        if self.number_of_states == 4:
            m.U2 = Var(m.t, bounds=(0, 100), initialize=helper(self.data.u2, self.data.time))
            m.U2.fix()  # Fixed for parameter estimation
        
        # End experimental input definition
        ####################################
        
        ####################################
        # Defining unknown model parameters
        # (estimated during parameter estimation)
        
        # Heat transfer coefficients
        if not self.reparam:
            m.Ua = Var(initialize=self.theta_initial["Ua"], bounds=(1e-6, 0.1))
            m.Ua.fix()
            m.Ub = Var(initialize=self.theta_initial["Ub"], bounds=(0.01, 0.05))
            m.Ub.fix()
            
            if self.number_of_states == 4:
                m.Uc = Var(initialize=self.theta_initial["Uc"], bounds=(0, 1e4))
                m.Uc.fix()
            
            # Inverse of the heat capacity coefficients (1/CpH and 1/CpS)
            m.inv_CpH = Var(initialize=self.theta_initial["inv_CpH"], bounds=(0.1, 0.4))
            m.inv_CpH.fix()
            m.inv_CpS = Var(initialize=self.theta_initial["inv_CpS"], bounds=(1, 10))
            m.inv_CpS.fix()
        else:
            if all(k in self.theta_initial for k in ("beta_1", "beta_2", "beta_3", "beta_4")):
                m.beta_1 = Var(initialize=self.theta_initial["beta_1"], bounds=(1e-3, 1))
                m.beta_1.fix()
                m.beta_2 = Var(initialize=self.theta_initial["beta_2"], bounds=(5e-3, 0.1))
                m.beta_2.fix()
                m.beta_3 = Var(initialize=self.theta_initial["beta_3"], bounds=(1e-3, 1))
                m.beta_3.fix()
                m.beta_4 = Var(initialize=self.theta_initial["beta_4"], bounds=(1e-3, 1))
                m.beta_4.fix()
            else: 
                # REPARAMETRIZATION
                m.beta_1 = Var(initialize=self.theta_initial["Ua"] * self.theta_initial["inv_CpH"], bounds=(0.01, 10))
                m.beta_1.fix()
                m.beta_2 = Var(initialize=self.theta_initial["Ub"] * self.theta_initial["inv_CpH"], bounds=(0.01, 10))
                m.beta_2.fix()
                m.beta_3 = Var(initialize=self.theta_initial["Ub"] * self.theta_initial["inv_CpS"], bounds=(0.01, 10))
                m.beta_3.fix()
                m.beta_4 = Var(initialize=self.alpha * pyovalue(m.P1) * self.theta_initial["inv_CpH"], bounds=(0.01, 10))
                m.beta_4.fix()

            if self.number_of_states == 4:
                m.beta_5 = Var(initialize=self.theta_initial["Uc"] / self.theta_initial["inv_CpH"], bounds=(0, 1e6))
                m.beta_5.fix()
        
        # End unknown parameter definition
        ####################################
        
        ################################
        # Defining model equations
        
        # First fin energy balance
        @m.Constraint(m.t)
        def Th1_ode(m, t):
            if not self.reparam:
                rhs_expr = (m.Ua * (m.Tamb - m.Th1[t]) + m.Ub * (m.Ts1[t] - m.Th1[t]) + m.alpha * m.P1 * m.U1[t]) * m.inv_CpH
            else:
                # REPARAM
                rhs_expr = m.beta_1 * (m.Tamb - m.Th1[t]) + m.beta_2 * (m.Ts1[t] - m.Th1[t]) + m.beta_4 * m.U1[t]
                        
            # If we use the 4-state model, we add heat transfer from sensor 2 to the energy balance on fin 1
            if self.number_of_states == 4:
                if not self.reparam:
                    rhs_expr += (m.Uc * (m.Th2[t] - m.Th1[t])) * m.inv_CpH
                else:
                    # REPARAM
                    rhs_expr += m.beta_5 * (m.Th2[t] - m.Th1[t])
            
            return m.Th1dot[t] == rhs_expr
        
        # First sensor energy balance
        @m.Constraint(m.t)
        def Ts1_ode(m, t):
            if not self.reparam:
                return m.Ts1dot[t] == (m.Ub * (m.Th1[t] - m.Ts1[t])) * m.inv_CpS
            else:
                # REPARAM
                return m.Ts1dot[t] == m.beta_3 * (m.Th1[t] - m.Ts1[t])
        
        # Second fin/sensor (only active for the 4-state model
        if self.number_of_states == 4:
            # Second fin energy balance
            @m.Constraint(m.t)
            def Th2_ode(m, t):
                if not self.reparam:
                    return m.Th2dot[t] == (m.Ua * (m.Tamb - m.Th2[t]) + m.Ub * (m.Ts2[t] - m.Th2[t]) + m.Uc * (m.Th1[t] - m.Th2[t]) + m.alpha * m.P2 * m.U2[t]) * m.inv_CpH
                else:
                    # REPARAM
                    return m.Th2dot[t] == m.beta_1 * (m.Tamb - m.Th2[t]) + m.beta_2 * (m.Ts2[t] - m.Th2[t]) + m.beta_5 * (m.Th1[t] - m.Th2[t]) + m.beta_4 * m.U2[t]
           
            # Second sensor energy balance
            @m.Constraint(m.t)
            def Ts2_ode(m, t):
                if not self.reparam:
                    return m.Ts2dot[t] == (m.Ub * (m.Th2[t] - m.Ts2[t])) * m.inv_CpS
                else:
                    # REPARAM
                    return m.Ts2dot[t] == m.beta_3 * (m.Th2[t] - m.Ts2[t])

        # End model equation definition
        ################################
        
        return m
    
    def finalize_model(self):
        """
        Finalizing the TC Lab model. Here, we will set the 
        experimental conditions and discretize the dae model.
        
        """
        m = self.model
        
        ####################################
        # Set initial conditions
        if self.data.time[0] == 0:
            if self.data.TS1_data is not None and self.data.TS1_data[0] is not None:
                # Initialize with first temperature measurement
                m.Th1[0].fix(self.data.TS1_data[0])
                m.Ts1[0].fix(self.data.TS1_data[0])
            else:
                # Initialize with ambient temperature
                m.Th1[0].fix(m.Tamb)
                m.Ts1[0].fix(m.Tamb)

            if self.number_of_states == 4:
                if self.data.TS2_data is not None and self.data.TS2_data[0] is not None:
                    # Initialize with first temperature measurement
                    m.Th2[0].fix(self.data.TS2_data[0])
                    m.Ts2[0].fix(self.data.TS2_data[0])
                else:
                    # Initialize with ambient temperature
                    m.Th2[0].fix(m.Tamb)
                    m.Ts2[0].fix(m.Tamb)

        # End initial conditions definition
        ####################################
        
        #########################################
        # Initialize the model using integration
        m.var_input = Suffix(direction=Suffix.LOCAL)

        if self.data.u1 is not None:
            # initialize with data
            m.var_input[m.U1] = helper(self.data.u1, self.data.time)
        else:
            # otherwise initialize control decision of 0
            m.var_input[m.U1] = {0: 0}

        if self.number_of_states == 4:
            if self.data.u2 is not None:
                # initialize with data
                m.var_input[m.U2] = helper(self.data.u2, self.data.time)
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
        
        TransformationFactory('dae.finite_difference').apply_to(
            m, scheme='BACKWARD', nfe=len(self.data.time) - 1
        )
            
        # End dynamic model initialization
        #########################################
        
        # TODO: Add "optimize" mode equations OUTSIDE of the get_labeled_model workflow
    
    def label_experiment(self):
        """
        Annotating (labeling) the model with experimental 
        data, associated measurement error, experimental 
        design decisions, and unknown model parameters.

        """
        m = self.model
        
        #################################
        # Labeling experiment outputs
        # (experiment measurements)
        
        m.experiment_outputs = Suffix(direction=Suffix.LOCAL)
        # Add sensor 1 temperature (m.Ts1) to experiment outputs
        m.experiment_outputs.update((m.Ts1[t], self.data.T1[ind]) for ind, t in enumerate(self.data.time))
        if self.number_of_states == 4:
            m.experiment_outputs.update((m.Ts2[t], self.data.T2[ind]) for ind, t in enumerate(self.data.time))
        
        # End experiment outputs
        #################################
        
        #################################
        # Labeling unknown parameters
        
        m.unknown_parameters = Suffix(direction=Suffix.LOCAL)
        # Add labels to all unknown parameters with nominal value as the value
        if not self.reparam:
            m.unknown_parameters.update((k, k.value) for k in [m.Ua, m.Ub, m.inv_CpH, m.inv_CpS])
            if self.number_of_states == 4:
                m.unknown_parameters[m.Uc] = m.Uc.value
        else:
        # REPARAM
            m.unknown_parameters.update((k, k.value) for k in [m.beta_1, m.beta_2, m.beta_3, m.beta_4])
            if self.number_of_states == 4:
                m.unknown_parameters[m.beta_5] = m.beta_5.value
        
        # End unknown parameters
        #################################
        
        #################################
        # Labeling experiment inputs
        # (experiment design decisions)
        
        m.experiment_inputs = Suffix(direction=Suffix.LOCAL)
        # Add experimental input label for control variable (m.U1)
        m.experiment_inputs.update((m.U1[t], None) for t in self.data.time)
        if self.number_of_states == 4:
            m.experiment_inputs.update((m.U2[t], None) for t in self.data.time)
        
        # End experiment inputs
        #################################
        
        #################################
        # Labeling measurement error
        # (for experiment outputs)
        
        m.measurement_error = Suffix(direction=Suffix.LOCAL)
        # Add sensor 1 temperature (m.Ts1) measurement error (assuming constant error of 0.25 deg C)
        m.measurement_error.update((m.Ts1[t], self.measurement_error) for t in self.data.time)
        if self.number_of_states == 4:
            m.measurement_error.update((m.Ts2[t], 1) for ind, t in enumerate(self.data.time))
        
        # End measurement error
        #################################
        

### -------------- Part 5: Extract and visualize results -------------- ###


def extract_results(model, name="Pyomo results", number_of_states=2):
    """Extract results from the Pyomo model"""

    time = np.array([pyovalue(t) for t in model.t])
    Th1 = np.array([pyovalue(model.Th1[t]) for t in model.t])
    Ts1 = np.array([pyovalue(model.Ts1[t]) for t in model.t])
    U1 = np.array([pyovalue(model.U1[t]) for t in model.t])
    P1 = pyovalue(model.P1)
    if not number_of_states == 4:
        Th2 = None
        Ts2 = None
        U2 = None
    else:
        Th2 = np.array([pyovalue(model.Th2[t]) for t in model.t])
        Ts2 = np.array([pyovalue(model.Ts2[t]) for t in model.t])
        U2 = np.array([pyovalue(model.U2[t]) for t in model.t])
    P2 = model.P2
    Tamb = model.Tamb

    return TC_Lab_data(name, time, Th1, U1, P1, Ts1, Th2, U2, P2, Ts2, Tamb)


def extract_plot_results(tc_exp_data, model, number_of_states=2, reparam=False):
    """Extract and plot Pyomo or DoE optimize_experiments results.

    Arguments:
    ----------
    tc_exp_data: experimental data (`TC_Lab_data`) or list of experiments
    model: Pyomo model, DoE results dict, or object with `results` attribute
    number_of_states: int, number of states, default: 2

    Returns:
    --------
    For a Pyomo model input, returns one `TC_Lab_data` object.
    For optimize_experiments results, returns a list of `TC_Lab_data` objects
    (one per optimized experiment).
    """

    empty_exp = TC_Lab_data(
        None, None, None, None, None, None, None, None, None, None, None
    )

    doe_results = model if isinstance(model, dict) else None
    if doe_results is None:
        try:
            doe_results = getattr(model, "results", None)
        except Exception:
            doe_results = None
    if (
        not isinstance(doe_results, dict)
        or "solution" not in doe_results
        or "param_scenarios" not in doe_results["solution"]
    ):
        doe_results = None

    # Branch 1: multi-experiment DoE optimize_experiments() results
    if doe_results is not None:
        param_scenarios = doe_results["solution"].get("param_scenarios", [])
        if len(param_scenarios) == 0:
            raise ValueError("No parameter scenarios found in optimize_experiments results.")

        scenario = param_scenarios[0]
        experiments = scenario.get("experiments", [])
        if len(experiments) == 0:
            raise ValueError("No experiment entries found in optimize_experiments results.")

        if tc_exp_data is None:
            exp_list = []
        elif isinstance(tc_exp_data, (list, tuple)):
            exp_list = list(tc_exp_data)
        else:
            exp_list = [tc_exp_data]
        if len(exp_list) not in (0, 1, len(experiments)):
            raise ValueError(
                "Number of provided tc_exp_data entries does not match the number "
                "of optimized experiments."
            )

        mod_results = []

        # create figure
        plt.figure(figsize=(10, 6))
        ax_temp = plt.subplot(2, 1, 1)
        ax_u = plt.subplot(2, 1, 2)

        cmap = plt.get_cmap("tab10")
        try:
            exp_blocks = model.model.param_scenario_blocks[0].exp_blocks
        except Exception as err:
            raise ValueError(
                "Multi-experiment plotting requires a DesignOfExperiments object "
                "with solved model blocks at "
                "model.model.param_scenario_blocks[0].exp_blocks[i]."
                "fd_scenario_blocks[0]."
            ) from err
        if len(exp_blocks) < len(experiments):
            raise ValueError(
                "Model contains fewer experiment blocks than optimize_experiments "
                "results entries."
            )

        for i, exp_result in enumerate(experiments):
            exp_data = empty_exp if len(exp_list) == 0 else exp_list[min(i, len(exp_list) - 1)]
            exp_id = exp_result.get("exp_id", i)
            suffix = f" (exp {exp_id+1})"
            color = cmap(i % 10)

            try:
                exp_model = exp_blocks[i].fd_scenario_blocks[0]
            except Exception as err:
                raise ValueError(
                    f"Could not access experiment block {i} at "
                    "model.model.param_scenario_blocks[0].exp_blocks[i]."
                    "fd_scenario_blocks[0]."
                ) from err

            if not hasattr(exp_model, "t"):
                raise ValueError(
                    f"Experiment block {i} does not contain a time set `t`."
                )

            mod_i = extract_results(
                exp_model,
                name=f"Pyomo DoE results exp {exp_id}",
                number_of_states=number_of_states,
            )
            mod_results.append(mod_i)

            if exp_data.T1 is not None and exp_data.time is not None:
                ax_temp.scatter(
                    exp_data.time,
                    exp_data.T1,
                    marker='o',
                    label="$T_{S,1}$ measured" + suffix,
                    alpha=0.4,
                    color=color,
                )
            if mod_i.TS1_data is not None:
                ax_temp.plot(
                    mod_i.time,
                    mod_i.TS1_data,
                    label="$T_{S,1}$ predicted" + suffix,
                    color=color,
                )
            if mod_i.T1 is not None:
                ax_temp.plot(
                    mod_i.time,
                    mod_i.T1,
                    label="$T_{H,1}$ predicted" + suffix,
                    color=color,
                    linestyle=':',
                )
            if exp_data.T2 is not None and exp_data.time is not None:
                ax_temp.scatter(
                    exp_data.time,
                    exp_data.T2,
                    marker='s',
                    label="$T_{S,2}$ measured" + suffix,
                    alpha=0.4,
                    color=color,
                )
            if mod_i.TS2_data is not None:
                ax_temp.plot(
                    mod_i.time,
                    mod_i.TS2_data,
                    label="$T_{S,2}$ predicted" + suffix,
                    color=color,
                    linestyle='--',
                )
            if mod_i.T2 is not None:
                ax_temp.plot(
                    mod_i.time,
                    mod_i.T2,
                    label="$T_{H,2}$ predicted" + suffix,
                    color=color,
                    linestyle='-.',
                )

            if exp_data.u1 is not None and exp_data.time is not None:
                ax_u.scatter(
                    exp_data.time,
                    exp_data.u1,
                    marker='o',
                    label="$u_1$ measured" + suffix,
                    color=color,
                    alpha=0.4,
                )
            if mod_i.u1 is not None:
                ax_u.plot(
                    mod_i.time,
                    mod_i.u1,
                    label="$u_1$ optimized" + suffix,
                    color=color,
                )
            if exp_data.u2 is not None and exp_data.time is not None:
                ax_u.scatter(
                    exp_data.time,
                    exp_data.u2,
                    marker='s',
                    label="$u_2$ measured" + suffix,
                    color=color,
                    alpha=0.4,
                )

        ax_temp.set_ylabel('Temperature (°C)')
        temp_handles, _ = ax_temp.get_legend_handles_labels()
        if len(temp_handles) > 0:
            ax_temp.legend(ncol=2 if len(experiments) > 1 else 1)
        ax_temp.grid(True)

        ax_u.set_ylabel('Heater Power (%)')
        ax_u.set_xlabel('Time (s)')
        control_handles, _ = ax_u.get_legend_handles_labels()
        if len(control_handles) > 0:
            ax_u.legend(ncol=2 if len(experiments) > 1 else 1)
        ax_u.grid(True)

        plt.tight_layout()
        plt.show()

        return mod_results

    # Branch 2: original single-model extraction and plotting
    # For convenience, save in a shorter variable name
    if tc_exp_data is not None:
        exp = tc_exp_data
    else:
        exp = empty_exp

    if not hasattr(model, "t"):
        raise ValueError(
            "Single-experiment plotting expects a solved Pyomo model with time set `t`. "
            "For multi-experiment plotting, pass a DesignOfExperiments object or "
            "results from optimize_experiments()."
        )

    mod = extract_results(model, number_of_states=number_of_states)

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
    plt.grid(True)

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

    if reparam:
        print("Beta_1 =", round(pyovalue(model.beta_1), 4), "Watts/Joules")
        print("Beta_2 =", round(pyovalue(model.beta_2), 4), "Watts/Joules")
        print("Beta_3 =", round(pyovalue(model.beta_3), 4), "Watts/Joules")
        print("Beta_4 =", round(pyovalue(model.beta_4), 4), "°C.Watts/(Joules.%)")
        if number_of_states == 4:
            print("Beta_5 =", round(pyovalue(model.beta_5), 4), "Watts/Joules")
    else:
        print("Ua =", round(pyovalue(model.Ua), 4), "Watts/°C")
        print("Ub =", round(pyovalue(model.Ub), 4), "Watts/°C")
        if number_of_states == 4:
            print("Uc =", round(pyovalue(model.Uc), 4), "Watts/°C")
        print("CpH =", round(1 / pyovalue(model.inv_CpH), 4), "Joules/°C")
        print("CpS =", round(1 / pyovalue(model.inv_CpS), 4), "Joules/°C")

    print(" ")  # New line

    return mod


def results_summary(result, reparam=False):
    eigenvalues, eigenvectors = np.linalg.eig(result)

    min_eig = min(eigenvalues)

    print("======Results Summary======")
    print("Five design criteria log10() value:")
    print("Pseudo A-optimality:", np.log10(np.trace(result)))
    try:
        print("A-optimality:", np.log10(np.trace(np.linalg.inv(result))))
    except np.linalg.LinAlgError:
        print("A-optimality: Matrix is singular, cannot compute inverse.")
    print("D-optimality:", np.log10(np.linalg.det(result)))
    print("E-optimality:", np.log10(min_eig))
    print("Modified E-optimality:", np.log10(np.linalg.cond(result)))
    print("\nFIM:\n", result)

    print("\neigenvalues:\n", eigenvalues)

    # print("\neigenvectors:\n", eigenvectors)
    if reparam:
        params = ["beta_1", "beta_2", "beta_3", "beta_4"]
    else:
        params = ["Ua", "Ub", "inv_CpH", "inv_CpS"]

    eigvec_df = pd.DataFrame(
        eigenvectors,
        index=params,
        columns=[f"eigvec_{i+1}" for i in range(eigenvectors.shape[1])]
    )    
    print("\nEigenvector matrix:\n", eigvec_df.round(4))
    
### ------ Part 5b: Extract and visualize multistart sampling and profile likelihood results ----- ###
def extract_multistart_sampling(
    results_df,
    results_df_lhs,
    results_df_sobol,
    x_col="Ua",
    y_col="inv_CpS",
    figsize=(18, 5),
    alpha=0.75,
    show=True,
):
    """
    Plot starting theta values for different multistart sampling methods.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results from random uniform multistart sampling.
    results_df_lhs : pandas.DataFrame
        Results from Latin hypercube multistart sampling.
    results_df_sobol : pandas.DataFrame
        Results from Sobol multistart sampling.
    x_col : str, optional
        Column name to plot on the x-axis. Default is "Ua".
    y_col : str, optional
        Column name to plot on the y-axis. Default is "inv_CpS".
    figsize : tuple, optional
        Figure size. Default is (18, 5).
    alpha : float, optional
        Scatter point transparency. Default is 0.75.
    show : bool, optional
        Whether to call plt.show(). Default is True.

    Returns
    -------
    fig, axs
        Matplotlib figure and axes objects.
    """

    sampling_results = [
        {
            "df": results_df,
            "label": "Random Uniform",
            "title": "Random Uniform Sampling",
            "color": "blue",
        },
        {
            "df": results_df_lhs,
            "label": "Latin Hypercube",
            "title": "Latin Hypercube Sampling",
            "color": "green",
        },
        {
            "df": results_df_sobol,
            "label": "Sobol",
            "title": "Sobol Sampling",
            "color": "orange",
        },
    ]

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    for ax, method in zip(axs, sampling_results):
        theta_df = method["df"][[x_col, y_col]].copy()

        ax.scatter(
            theta_df[x_col],
            theta_df[y_col],
            label=method["label"],
            color=method["color"],
            alpha=alpha,
        )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(method["title"])

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axs

import scipy.stats as stats
import matplotlib.pyplot as plt


def plot_profile_likelihood(
    profile_results,
    alpha=0.95,
    xlims=None,
    ylims=None,
):
    """
    Plot profile likelihood curves with optional x/y limits for each parameter.

    Parameters
    ----------
    profile_results : object
        Profile likelihood results from parmest.
    
    alpha : float, optional
        Confidence level used for the profile likelihood plot.
        Default is 0.95.
    
    xlims : list of tuple or None, optional
        x-axis limits for each parameter subplot.
        Example:
            xlims = [
                (0, 10),
                (0, 5),
                None,
                (1, 3),
            ]
        If None, all plots use default full x-axis range.
    
    ylims : list of tuple or None, optional
        y-axis limits for each parameter subplot.
        Example:
            ylims = [
                (0, 20),
                (0, 20),
                (0, 20),
                (0, 20),
            ]
        If None, all plots use default full y-axis range.

    Returns
    -------
    fig, axes
        The matplotlib figure and axes objects.
    """

    threshold = stats.chi2.ppf(float(alpha), df=1)

    print(
        f"Chi-squared threshold for {alpha:.0%} confidence interval: "
        f"{threshold:.4f}"
    )

    fig, axes = profile_likelihood_plot(
        profile_results,
        alpha=alpha,
        show=False,
        ylabel=r"$2\left(\Phi_{\mathrm{PL},i}(\theta_i)-\Phi(\hat{\theta})\right)$",
    )

    # Flatten axes in case axes is a 2D array from plt.subplots(..., squeeze=False)
    axes_flat = axes.flatten()

    # Label the threshold dashed line only on the top subplot
    top_ax = axes_flat[0]
    for line in top_ax.lines:
        if line.get_linestyle() == "--":
            line.set_label("threshold")
            break

    top_ax.legend(loc="best", prop={"size": 12})

    if xlims is not None:
        for ax, xlim in zip(axes_flat, xlims):
            if xlim is not None:
                ax.set_xlim(*xlim)

    if ylims is not None:
        for ax, ylim in zip(axes_flat, ylims):
            if ylim is not None:
                ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return fig, axes

### ---------- Part 6: Extract the original parameters and covariance ---------- ###

def reformulate_parameters(orig_params, alpha, P1):
    """
    Function to recover the reformulated beta parameter
    values from the original model parameters.

    Arguments
    ---------
    orig_params: dict
        Keys are original parameter names and values are
        the parameter estimates. Expected keys are:
        "Ua", "Ub", "inv_CpH", and "inv_CpS"

    alpha: float
        Alpha value for beta_4

    P1: float
        P1 value for beta_4

    Returns
    -------
    reform_theta_vals: dict
        Keys are reformulated beta parameter names and values are
        the parameter estimates. Returned keys are:
        "beta_1", "beta_2", "beta_3", and "beta_4"
    """

    Ua = orig_params["Ua"]
    Ub = orig_params["Ub"]
    inv_CpH = orig_params["inv_CpH"]
    inv_CpS = orig_params["inv_CpS"]

    reform_theta_vals = {
        "beta_1": Ua * inv_CpH,
        "beta_2": Ub * inv_CpH,
        "beta_3": Ub * inv_CpS,
        "beta_4": alpha * P1 * inv_CpH,
    }

    return reform_theta_vals


def recover_original_parameters(reform_params, alpha, P1):
    """
    Function to recover the original parameter
    values from the model after estimation.

    Arguments
    ---------
    reform_params: dict,
        Keys are reformulated parameter names and values are
        the parameter estimates
    alpha: float,
        alpha value for beta 4
    P1: float,
        P1 value for beta 4

    Returns
    -------
    orig_theta_vals: dict,
        keys are original parameter names and values are
        the parameter estimates
    """

    # Recover the values
    CpH = alpha * P1 / reform_params["beta_4"]
    Ua = CpH * reform_params["beta_1"]
    Ub = CpH * reform_params["beta_2"]
    CpS = Ub / reform_params["beta_3"]

    # Store them in a dictionary
    orig_theta_vals = {"Ua": Ua, "Ub": Ub, "inv_CpH": 1/CpH, "inv_CpS": 1/CpS}

    return orig_theta_vals


def recover_original_covariance(reform_params, cov_reform, alpha, P1):
    """
        Computes the covariance matrix of the original parameters

        Parameters
        ----------
        reform_params: dict,
            Keys are reformulated parameter names and values are
            the parameter estimates
        cov_reform: Pandas.DataFrame,
            Covariance matrix of the reformulated parameters
        alpha: float,
            alpha value
        P1: float,
            P1 value

        Returns
        -------
        cov_orig: Pandas.DataFrame,
            covariance matrix of the original parameters
    """

    # derivatives of Ua with respect to the reformulated parameters
    dUa_dbeta_1 = alpha * P1 / reform_params['beta_4']
    dUa_dbeta_2 = 0
    dUa_dbeta_3 = 0
    dUa_dbeta_4 = - alpha * P1 * reform_params['beta_1'] / (reform_params['beta_4'] ** 2)
    dUa_dbeta = [dUa_dbeta_1, dUa_dbeta_2, dUa_dbeta_3, dUa_dbeta_4]

    # derivatives of Ub with respect to the reformulated parameters
    dUb_dbeta_1 = 0
    dUb_dbeta_2 = alpha * P1 / reform_params['beta_4']
    dUb_dbeta_3 = 0
    dUb_dbeta_4 = - alpha * P1 * reform_params['beta_2'] / (reform_params['beta_4'] ** 2)
    dUb_dbeta = [dUb_dbeta_1, dUb_dbeta_2, dUb_dbeta_3, dUb_dbeta_4]

    # derivatives of inv_CpH with respect to the reformulated parameters
    dinv_CpH_dbeta_1 = 0
    dinv_CpH_dbeta_2 = 0
    dinv_CpH_dbeta_3 = 0
    dinv_CpH_dbeta_4 = 1 / (alpha * P1)
    dinv_CpH_dbeta = [dinv_CpH_dbeta_1, dinv_CpH_dbeta_2, dinv_CpH_dbeta_3, dinv_CpH_dbeta_4]

    # derivatives of inv_CpS with respect to the reformulated parameters
    dinv_CpS_dbeta_1 = 0
    dinv_CpS_dbeta_2 = (- reform_params['beta_3'] * reform_params['beta_4'] /
                        (alpha * P1 * (reform_params['beta_2'] ** 2)))
    dinv_CpS_dbeta_3 = reform_params['beta_4'] / (alpha * P1 * reform_params['beta_2'])
    dinv_CpS_dbeta_4 = reform_params['beta_3'] / (alpha * P1 * reform_params['beta_2'])
    dinv_CpS_dbeta = [dinv_CpS_dbeta_1, dinv_CpS_dbeta_2, dinv_CpS_dbeta_3, dinv_CpS_dbeta_4]

    dtheta_dbeta = np.zeros((4, 4))
    dtheta_dbeta[0, :] = dUa_dbeta
    dtheta_dbeta[1, :] = dUb_dbeta
    dtheta_dbeta[2, :] = dinv_CpH_dbeta
    dtheta_dbeta[3, :] = dinv_CpS_dbeta

    # compute the covariance matrix of the original parameters
    cov_theta = dtheta_dbeta @ cov_reform @ dtheta_dbeta.T
    cov_orig = pd.DataFrame(
        cov_theta.to_numpy(),
        index=["Ua", "Ub", "Inv_CpH", "inv_CpS"],
        columns=["Ua", "Ub", "Inv_CpH", "inv_CpS"], )

    return cov_orig


def extract_trace_covariance(cov, method):
    """Computes the trace of the covariance matrix

    Parameters
    ----------
    cov: Pandas.DataFrame,
        Covariance matrix of the parameters
    method: str,
        The method used to compute the covariance matrix
    """

    # check if the covariance matrix is positive semi-definite
    eigen_values = np.linalg.eigvalsh(cov)
    if any(eig_val < -1e-10 for eig_val in eigen_values):
        print(f"\nWARNING: The covariance matrix from {method} is "
              f"not positive semi-definite.\n", cov)
        print(f"The trace of the covariance matrix from the {method} "
              f"method is:", format(np.trace(cov), ".3e"))
    else:
        print(f"\nThe covariance matrix from the {method} method is:\n",
              cov)
        print(f"The trace of the covariance matrix from the {method} "
              f"method is:", format(np.trace(cov), ".3e"))
