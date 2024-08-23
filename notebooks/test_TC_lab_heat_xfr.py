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
    sin,
    PositiveReals,
)
from pyomo.dae import DerivativeVar, ContinuousSet, Simulator

import matplotlib.pyplot as plt

def run_heat_model(Ub, inv_CpH, inv_CpS):
    m = ConcreteModel()
            
    #########################################
    # Begin model constants definition
    m.Tamb = Param(initialize=30)

    m.Tmax = 85  # Maximum temparture (Deg C)

    # End model constants
    #########################################

    ################################
    # Defining state variables
    m.t = ContinuousSet(initialize=[0, 100])

    # Temperature states for the fins
    m.Th1 = Var(m.t, bounds=[0, m.Tmax], initialize=30)
    m.Ts1 = Var(m.t, bounds=[0, m.Tmax], initialize=30)

    m.Th1[0].fix(50.0)
    m.Ts1[0].fix(25.0)

    # Derivatives of the temperature state variables
    m.Th1dot = DerivativeVar(m.Th1, wrt=m.t)
    m.Ts1dot = DerivativeVar(m.Ts1, wrt=m.t)

    # End state variable definition
    ################################

    ####################################
    # Defining unknown model parameters
    # (estimated during parameter estimation)

    # Heat transfer coefficients
    m.Ub = Var(initialize=Ub, bounds=(0, 1e4))
    m.Ub.fix()

    # Inverse of the heat capacity coefficients (1/CpH and 1/CpS)
    m.inv_CpH = Var(initialize=inv_CpH, bounds=(0, 1e6))
    m.inv_CpH.fix()
    m.inv_CpS = Var(initialize=inv_CpS, bounds=(0, 1e3))
    m.inv_CpS.fix()

    # End unknown parameter definition
    ####################################

    ################################
    # Defining model equations

    # First fin energy balance
    @m.Constraint(m.t)
    def Th1_ode(m, t):
        rhs_expr = (m.Ub * (m.Ts1[t] - m.Th1[t])) * m.inv_CpH
        
        return m.Th1dot[t] == rhs_expr

    # First sensor energy balance
    @m.Constraint(m.t)
    def Ts1_ode(m, t):
        return m.Ts1dot[t] == (m.Ub * (m.Th1[t] - m.Ts1[t])) * m.inv_CpS

    # End model equation definition
    ################################

    TransformationFactory('dae.finite_difference').apply_to(
            m, scheme='BACKWARD', nfe=100
        )
    
    return m

def plot_results(m):
    Th_vals = [pyovalue(m.Th1[t]) for t in m.t]
    Ts_vals = [pyovalue(m.Ts1[t]) for t in m.t]

    plt.plot(m.t, Th_vals, label='Th')
    plt.plot(m.t, Ts_vals, label='Ts')
    plt.legend()
    plt.show()
    plt.clf()
    plt.close()

inv_CpS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in range(10):
    m = run_heat_model(0.015, 1, inv_CpS[i])
    
    solver = SolverFactory('ipopt')
    solver.solve(m)

    plot_results(m)