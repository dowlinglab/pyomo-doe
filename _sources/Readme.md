# Optimizing Experiments with Pyomo.DoE 

Welcome to the interaction tutorial workshop for parameter estimation and model-based design of experiments in the Pyomo ecosystem!

![Pyomo workflow](./images/pyomo_workflow_new.png)

These materials were created by [Prof. Alexander Dowling](https://dowlinglab.nd.edu/people/professor-alexander-w-dowling/) at the University of Notre Dame. Special thanks to [Prof. Jeff Kantor](https://engineering.nd.edu/news/in-memoriam-jeffrey-kantor-former-vice-president-associate-provost-and-dean/), [Maddie Watson](https://dowlinglab.nd.edu/people/madelynn-watson/), [Molly Dougher](https://dowlinglab.nd.edu/people/molly-dougher/), and [Hailey Lynch](https://dowlinglab.nd.edu/people/hailey-lynch/) for assistance with the TCLab models and activities. Pyomo.DoE was developed by [Jialu Wang](https://dowlinglab.nd.edu/people/jialu-wang/) and Alexander Dowling with assistance from [John Siirola](https://www.sandia.gov/ccr/staff/john-daniel-siirola/), [Bethany Nicholson](https://scholar.google.com/citations?user=WxqNQ6IAAAAJ&hl=en), [Miranda Mundt](https://ieeexplore.ieee.org/author/37089520396), [Hailey Lynch](https://dowlinglab.nd.edu/people/hailey-lynch/), and [Dan Laky](https://dowlinglab.nd.edu/people/daniel-laky/). Pyomo.DoE was refactored over the summer of 2024 by [Dan Laky](https://dowlinglab.nd.edu/people/daniel-laky/) and [Prof. Alexander Dowling](https://dowlinglab.nd.edu/people/professor-alexander-w-dowling/), who both also updated the workshop materials to coincide with the refactored versions of both Pyomo.DoE and parmest.

## American Control Conference (ACC) 2026 Workshop

**Model Building and Digital Twin Construction using Pyomo.DoE, a Pythonic Tool for Optimal Experimental Design**

Digital twins rely on high-quality data to achieve predictive capability, yet experiments are often expensive and constrained. Optimal experimental design (OED) provides a principled framework for selecting experiments that maximally reduce model uncertainty, and can be naturally posed as an optimal control problem constrained by dynamic system models and experimental limitations.
This workshop introduces Pyomo.DoE, an open-source, Python-based, equation-oriented framework for model building and optimal experimental design within the Pyomo modeling ecosystem. Participants will learn how classical information-based design criteria—including D-, A-, E-, and ME-optimality derived from the Fisher Information Matrix—can be embedded in nonlinear dynamic optimization problems and solved using modern numerical optimization tools. The formulation treats experimental inputs (e.g., control trajectories, sampling times, operating conditions) as decision variables in an optimal control problem that explicitly balances information gain and experimental feasibility.

The workshop will begin with a brief introduction to algebraic modeling and dynamic optimization in Pyomo, followed by a hands-on workflow for parameter estimation, uncertainty quantification, and optimal experiment design. Through interactive examples, participants will construct predictive models, design informative experiments, and explore how different optimality criteria target distinct sources of uncertainty.

This workshop is intended for researchers and practitioners interested in digital twins, system identification, and model-based experiment design. The workshop highlights how modern computational optimization tools enables scalable, rigorous, and automated experimental design for complex dynamic systems.

[*Register Here*](https://acc2026.a2c2.org/registration)

We are scheduled for a [half-day afternoon workshop](https://acc2026.a2c2.org/program/workshop-listing#session-6-17) on **Tuesday, May 26, 2026**. The schedule below assumes a 1pm start time and will be updated once the conference schedule is finalized.

| Time | Topic |
| ---- | -------- |
| 1:00 pm  | *Welcome and Overview* |
| 1:15 pm | **Dynamic Modeling and Optimization in Pyomo** |
| | [TC Lab Model](./notebooks/tclab_model.ipynb) |
| | [Simulation in Pyomo](./notebooks/pyomo_simulation.ipynb) |
| | Hands-on Examples |
| 2:00 pm | *Break* |
| 2:15 pm | **Parameter Estimation** |
| | [Parmest package overview](./notebooks/parmest.ipynb) |
| | [Parmest Exercise](./notebooks/parmest_exercise.ipynb) |
| | Hands-on Examples | 
| 3:15 pm | *Break* |
| 3:30 pm | **Optimal Experiment Design** |
| | [Pyomo.DoE exploratory analysis](./notebooks/doe_exploratory_analysis.ipynb) |
| | [Pyomo.DoE optimal experimental design](./notebooks/doe_optimize.ipynb) |
| | [Pyomo.DoE Exercise](./notebooks/doe_exercise.ipynb) |
| | Hands-on Examples |
| 4:45 pm | *Adjourn* |

## What will I learn in this workshop?

In this workshop, we will learn how to develop digital twin models in the open-source Pyomo ecosystem. Specifically, we will learn how to use three Pyomo-based toolkits:
* `pyomo.dae` for modeling and discretization of (partial) differential algebriac equations to facilitate dynamic optimization
* `ParmEst` for parameter estimation and uncertainty quantification
* `Pyomo.DoE` for model-based design of experiments

## What do I need to complete the tutorial?

This tutorial assumes the audience is familiar with basic Python programming. (New to Python? Check out [this](https://lectures.scientific-python.org/index.html) and similar online resources.) The tutorial is designed to run in Google Colab. The `tclab_pyomo.py` file contains the Pyomo model for our motivating system as well as utilities to install software on Colab.

Alternatively, pariticipants can run the tutorial locally on their computer. Use the following command to create a new conda environment:

```
conda create -n summer2026 -c conda-forge -c IDAES-PSE python=3.11 idaes-pse pandas numpy matplotlib scipy ipykernel
```

Then install the optimization solvers, including `Ipopt` with HSL linear algebra and `k_aug`:

```
idaes get-extensions
```

Note: `k_aug` is not distributed for macOS users with an Intel processor. Instead, you will either need to compile yourself or skip a few sections of the tutorial. `k_aug` is an optional dependency for `Pyomo.DoE`.

Next, download the files for this tutorial:

```
git clone git@github.com:dowlinglab/pyomo-doe.git
```

## How do I learn more about Pyomo.DoE?

The [Pyomo.DoE documentation](https://pyomo.readthedocs.io/en/stable/contributed_packages/doe/doe.html) is a great information and a different set of examples. Also see our tutorial notebook for the [reaction kinetics example](https://colab.research.google.com/github/Pyomo/pyomo/blob/main/pyomo/contrib/doe/examples/fim_doe_tutorial.ipynb).

If you use Pyomo.DoE, please cite our paper:

```{admonition} Pyomo.DoE paper
Wang and Dowling, 2022. Pyomo.DoE: An open-source package for model-based design of experiments in Python. AIChE Journal, 68(12), e17813. https://doi.org/10.1002/aic.17813
```

New to Pyomo? Check out these great resources:
* [Pyomo documentation](https://pyomo.readthedocs.io/en/stable/)
* [ND Pyomo Cookbook](https://jckantor.github.io/ND-Pyomo-Cookbook/README.html)
* [Pyomo textbook](https://link.springer.com/book/10.1007/978-3-030-68928-5)
* [Hands-On Mathematical Optimization with Python](https://mobook.github.io/MO-book/intro.html)

## Past Tutorials

European Symposium on Computer Aided Process Engineering and International Symposium on Process Systems Engineering:
* [Workshop slides](https://raw.githubusercontent.com/dowlinglab/pyomo-doe/main/slides/ESCAPE34_PSE24_Workshop.pdf)
* [Presentation slides](https://raw.githubusercontent.com/dowlinglab/pyomo-doe/main/slides/ESCAPE34_PSE24_Presentation.pdf)
