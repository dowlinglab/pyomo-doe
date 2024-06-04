# Optimizing Experiments with Pyomo.DoE 

Welcome to the interaction tutorial workshop for parameter estimation and model-based design of experiments in the Pyomo ecosystem!

![Pyomo workflow](./images/pyomo_workflow.png)

These materials were created by [Prof. Alexander Dowling](https://dowlinglab.nd.edu/people/professor-alexander-w-dowling/) at the University of Notre Dame. Special thanks to [Prof. Jeff Kantor](https://engineering.nd.edu/news/in-memoriam-jeffrey-kantor-former-vice-president-associate-provost-and-dean/), [Maddie Watson](https://dowlinglab.nd.edu/people/madelynn-watson/), [Molly Dougher](https://dowlinglab.nd.edu/people/molly-dougher/), and [Hailey Lynch](https://dowlinglab.nd.edu/people/hailey-lynch/) for assistance with the TCLab models and activities. Pyomo.DoE was developed by [Jialu Wang](https://dowlinglab.nd.edu/people/jialu-wang/) and Alexander Dowling with assistance from [John Siirola](https://www.sandia.gov/ccr/staff/john-daniel-siirola/), [Bethany Nicholson](https://scholar.google.com/citations?user=WxqNQ6IAAAAJ&hl=en), [Miranda Mundt](https://ieeexplore.ieee.org/author/37089520396), [Hailey Lynch](https://dowlinglab.nd.edu/people/hailey-lynch/), and [Dan Laky](https://dowlinglab.nd.edu/people/daniel-laky/).

## ESCAPE/PSE 2024 Workshop Schedule

Thank you for joining the workshop at the ESCAPE/PSE meeting in Florence, Italy on June 2, 2024.
* [Workshop slides](https://raw.githubusercontent.com/dowlinglab/pyomo-doe/main/slides/ESCAPE34_PSE24_Workshop.pdf)
* [Presentation slides](https://raw.githubusercontent.com/dowlinglab/pyomo-doe/main/slides/ESCAPE34_PSE24_Presentation.pdf)

| Time | Topic |
| ---- | -------- |
| 1:00 pm  | *Welcome and Overview* |
| 1:05 pm | **Modeling and Optimization in Pyomo** |
| | [](./notebooks/tclab_model.ipynb)
| | [](./notebooks/pyomo_simulation.ipynb) |
| 1:30 pm | **Parameter Estimation** |
| | [](./notebooks/parmest.ipynb) |
| | [](./notebooks/parmest_exercise.ipynb) |
| 2:10 pm | *Break* |
| 2:20 pm | **Optimal Experiment Design** |
| | [](./notebooks/doe_exploratory_analysis.ipynb)
| | [](./notebooks/doe_optimize.ipynb)
| | [](./notebooks/doe_exercise.ipynb)
| 3:00 pm | *Adjourn* |

## What will I learn in this workshop?

Digital twins refer to a new perspective on predictive modeling, where a mathematical model (often grounded in engineering science fundamentals) is continuously updated with as new data from the corresponding physical system become available. Thus a digital twin mimics the behaviors of its corresponding physical system. Often digital twins are developed and deployed for a specific purpose, e.g., optimizing maintain schedules, process monitoring for improved safety, optimal control of complex systems.

In this workshop, we will learn how to develop digital twin models in the open-source Pyomo ecosystem. Specifically, we will learn how to use two Pyomo-based toolkits:
* `ParmEst` for parameter estimation and uncertainty quantification
* `Pyomo.DoE` for model-based design of experiments



## What do I need to complete the tutorial?

This tutorial assumes the audience is familiar with basic Python programming. (New to Python? Check out [this](https://lectures.scientific-python.org/index.html) and similar online resources.) The tutorial is designed to run in Google Colab. The `tclab_pyomo.py` file contains the Pyomo model for our motivating system as well as utilities to install software on Colab.

Alternatively, pariticipants can run the tutorial locally on their computer. Use the following command to create a new conda environment:

```
conda create -n summer2024 -c conda-forge -c IDAES-PSE python=3.10 idaes-pse pandas numpy matplotlib scipy ipykernel
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