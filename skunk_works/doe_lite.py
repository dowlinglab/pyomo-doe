import pyomo.environ as pyo

from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet

import numpy as np

import pandas as pd

from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment # Load the example

# For IPOPT executable
import idaes

# Required if reading in files.
import json


def doe_lite(experiment, objective="A"):
    ''' A lightweight version of Pyomo.DoE.

    Arguments:
        experiment: an Experiment object
        objective:
            None: skip objective and return after converging Jacobian constraints
            "A": A-optimality, trace(FIM)
            "D": D-optimality, logdet(FIM)
    
    Returns:
        model: the model object
        jac: the Jacobian matrix
        fim: the Fisher information matrix
            
    '''

    ## Step 1: Create and converge the experiment model

    # Create a Pyomo model
    model = experiment.get_labeled_model()

    # Loop over the design variables, fix them
    for v in model.experiment_inputs:
        v.fix()

    print("Solving the square model with design variables fixed...")
    # Solve the model
    solver = pyo.SolverFactory('ipopt')
    results1 = solver.solve(model, tee=True)

    print("\nDone.\n\n")

    ## Step 2: Assemble the Jacobians via Symbolic Difference

    # Parameters
    # Create an empty component set
    param_set = ComponentSet()

    # Loop over the unknown model parameters
    for p in model.unknown_parameters.keys():
        param_set.add(p)

    # Assemble into a list
    
    param_list = list(param_set)

    # Measurements (outputs)
    # Create an empty component set
    output_set = ComponentSet()

    # Loop over the model outputs
    for o in model.experiment_outputs.keys():
        output_set.add(o)

    # Assemble into a list
    output_list = list(output_set)

    # Constraints and Variables
    # Create empty component sets
    con_set = ComponentSet() # These will be all constraints in the Pyomo model
    var_set = ComponentSet() # These will be all Pyomo variables in the Pyomo model

    # Loop over the active model constraints
    for c in model.component_data_objects(pyo.Constraint, descend_into=True, active=True):

        # Add constraint c to the constraint set
        con_set.add(c)

        # Loop over the variables in the constraint c
        # Note: changed this to include_fixed=True
        # Changed back to False to fix problem degree of freedom issues
        for v in identify_variables(c.body, include_fixed=False):
            # Add variable v to the variable set
            var_set.add(v)

    # recall that the parameters are fixed, so we did not
    # get them above. Let's add them now.
    for p in model.unknown_parameters.keys():
        var_set.add(p)

    # Assemble into lists
    con_list = list(con_set)
    var_list = list(var_set)

    # Assemble Jacobian 
    # Create an empty dictionary
    jac_dict = {}

    # Enumerate over the constraints
    for i,c in enumerate(con_list):
        # Check we only have equality constraints... otherwise this gets more complicated
        assert c.equality, "This function only works with equality constraints"
        
        # Perform symbolic differentiation
        der_map = reverse_sd(c.body)

        # Loop over the Pyomo variables, which includes 
        # parameters, measurements, control decisions
        for j,v in enumerate(var_list):
            # Check if the variable is in the derivative map
            if v in der_map:
                # Record the expression 
                deriv = der_map[v]
            else:
                # Otherwise, record 0
                deriv = 0
            # Save results in the Jacobian dictionary
            jac_dict[(i, j)] = deriv


    ## Build the constraints to compute the Jacobian

    # Create empty lists
    param_index = []
    model_var_index = []
    measurement_index = []
    # Adding a `included` suffix to only
    # take outputs that are unfixed. This
    # makes indices match.
    model.me_included = pyo.Suffix(direction=pyo.Suffix.LOCAL)

    # Loop over the variables and determine which ones 
    # (and associated indices) are (a) parameters or 
    # (b) measurements
    # TODO: Does this considered fixed variables?
    # How does that change things? We fix all of our
    # experiment inputs and unknown parameters.
    for i, v in enumerate(var_set):
        # Check if the variable is a parameter
        if v in param_set:
            # If yes, record its index
            param_index.append(i)
        else:
            # Otherwise, it is a model variable
            model_var_index.append(i)

            # Check if the model variable is a measurement
            if v in output_set:
                # If yes, record its index
                measurement_index.append(i)
                model.me_included[v] = model.measurement_error[v]

    # Using the lists of indices to create Pyomo Sets
    model.param_index = pyo.Set(initialize=param_index)
    model.measurement_index = pyo.Set(initialize=measurement_index)
    model.constraint_index = pyo.Set(initialize=range(len(con_list)))
    model.var_index = pyo.Set(initialize=model_var_index)

    # Define a Pyomo variable for the Jacobian of the model variables 
    # (everything except parameters) with respect to the model parameters
    model.jac_variables_wrt_param = pyo.Var(model.var_index, model.param_index, initialize=0)

    # Calculate the Jacobian using the chain rule and total derivative definitions
    #
    # Prior comment:
    # This has an index mistake... jac_dict includes the parameters, but var_index skips them
    # We need to be more careful about the indices
    #
    # New reflection:
    # var_index is built from the indices in var_list, which includes the parameters
    # I think this is okay
    @model.Constraint(model.constraint_index, model.param_index)
    def jacobian_constraint(model, i, j):
        return jac_dict[(i,j)] == -sum(model.jac_variables_wrt_param[k,j] * jac_dict[(i,k)] for k in model.var_index)
    
    # Step 3: Solve the model with the Jacobian constraints, extract the Jacobian
    results2 = solver.solve(model, tee=True)

    def get_jac():
        jac = np.zeros((len(measurement_index), len(param_index)))

        for i,y in enumerate(model.measurement_index):
            for j,p in enumerate(model.param_index):
                # print(f"Jacobian of {var_list[y]} with respect to {var_list[p]}: {model.jac_variables_wrt_param[y,p].value}")
                jac[i,j] = model.jac_variables_wrt_param[y,p].value

        row_names = [str(var_list[y]) for y in model.measurement_index]
        col_names = [str(var_list[p]) for p in model.param_index]

        # print("jac = ", jac)
        # print("row_names = ", row_names)
        # print("col_names = ", col_names)

        jac_df = pd.DataFrame(jac, index=row_names, columns=col_names)

        return jac_df
    
    jac = get_jac()

    print("\nDone.\n\n")

    # Step 4: Compute the Fisher information matrix (FIM)

    print("Computing the Fisher information matrix (FIM)...")

    model.fim = pyo.Var(model.param_index, model.param_index, initialize=1)
    # Update this to include the measurement error
    # TODO: Think of more creative way to skip fixed stuff?
    @model.Constraint(model.param_index, model.param_index)
    def fim_constraint(model, i, j):
        return model.fim[i,j] == sum((1 / model.measurement_error[val]) * model.jac_variables_wrt_param[model.measurement_index[ind + 1], i] * model.jac_variables_wrt_param[model.measurement_index[ind + 1], j] for ind, val in enumerate(model.me_included))

    # model.fim_constraint.pprint()
    model.T.pprint()
    results3 = solver.solve(model, tee=True)

    def get_fim():
        # Extract the FIM matrix
        fim = np.zeros((len(model.param_index), len(model.param_index)))
        for i, c in enumerate(model.param_index):
            for j, d in enumerate(model.param_index):
                fim[i, j] = model.fim[c, d].value
        
        # Grab the parameter names
        col_names = [str(var_list[c]) for c in model.param_index]

        # Store in a pandas dataframe
        fim_df = pd.DataFrame(fim, index=col_names, columns=col_names)

        return fim_df
    
    fim = get_fim()

    print("\nDone.\n\n")

    if objective is None:
        return model, jac, fim
    else:

        print("Solving DoE optimization problem with {objective}-optimality objective")

        # Unfix the experiment design decisions
        for v in model.experiment_inputs:
            v.unfix()

    if objective == "A":
        @model.Objective(sense=pyo.maximize)
        def trace_fim(model):
            return sum(model.fim[i,i] for i in model.param_index)
    elif objective == "D":
        
        fim_array = fim.to_numpy()

        # Calculate the eigenvalues of the FIM matrix
        eig = np.linalg.eigvals(fim_array)

        # If the smallest eigenvalue is (practically) negative, add a diagonal matrix to make it positive definite
        small_number = 1e-10
        if min(eig) < small_number:
            fim_array = fim_array + np.eye(len(model.param_index)) * (
                small_number - min(eig)
            )

        # Compute the Cholesky decomposition of the FIM matrix
        L = np.linalg.cholesky(fim_array)

        model.L = pyo.Var(
                        model.param_index, model.param_index, initialize=0
                    )

        # loop over parameter name
        for i, c in enumerate(model.param_index):
            for j, d in enumerate(model.param_index):
                # fix the 0 half of L matrix to be 0.0
                if i < j:
                    model.L[c, d].fix(0.0)
                # Give LB to the diagonal entries
                elif i == j:
                    # Set the lower bound for the diagonal entries
                    # to be a small number
                    model.L[c, d].setlb(1E-10)
                            

        # Initialize the Cholesky matrix
        for i, c in enumerate(model.param_index):
            for j, d in enumerate(model.param_index):
                model.L[c, d].value = L[i, j]

        def cholesky_imp(m, c, d):
            """
            Calculate Cholesky L matrix using algebraic constraints
            """
            # If the row is greater than or equal to the column, we are in the
            # lower triangle region of the L and FIM matrices.
            # This region is where our equations are well-defined.
            if list(m.param_index).index(c) >= list(m.param_index).index(d):
                return m.fim[c, d] == sum(
                    m.L[c, m.param_index.at(k + 1)]
                    * m.L[d, m.param_index.at(k + 1)]
                    for k in range(list(m.param_index).index(d) + 1)
                )
            else:
                # This is the empty half of L above the diagonal
                return pyo.Constraint.Skip

        model.cholesky_cons = pyo.Constraint(
            model.param_index, model.param_index, rule=cholesky_imp
        )

        model.logdet_FIM = pyo.Objective(
            expr=2 * sum(pyo.log10(model.L[j, j]) for j in model.param_index),
            sense=pyo.maximize,
        )
    else:
        raise ValueError("Objective must be None, 'A' or 'D'")
        
    results4 = solver.solve(model, tee=True)
    print("\nDone.\n\n")

    jac = get_jac()
    fim = get_fim()

    return model, jac, fim


def perform_reactor_doe():
    # import json

    # Copied from the json file in the example
    # TODO: Use the json file instead of copying the data. Need to figure out how to load the file without hardcoding the path.
    # Changed the control points ot see if FIM changes at initial point.
    data_ex = {"CA0": 5.0, "CA_bounds": [1.0, 5.0], "CB0": 0.0, "CC0": 0.0, "t_range": [0.0, 1.0], "control_points": {0: 500, 0.125: 450, 0.25: 400, 0.375: 350, 0.5: 300, 0.625: 300, 0.75: 300, 0.875: 300, 1: 300}, "T_bounds": [300, 700], "A1": 84.79, "A2": 371.72, "E1": 7.78, "E2": 15.05}

    experiment = ReactorExperiment(data=data_ex, nfe=10, ncp=3)

    model, jac, fim = doe_lite(experiment, objective="D")

    print(jac)
    print(fim)

if __name__ == "__main__":
    perform_reactor_doe()