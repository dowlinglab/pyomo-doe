# Pyomo.DoE Workshop

Welcome to the interaction tutorial workshop for parameter estimation and model-based design of experiments in the Pyomo ecosystem!

## What will I learn in this workshop?

Digital twins refer to a new perspective on predictive modeling, where a mathematical model (often grounded in engineering science fundamentals) is continously updated with as new data from the corresponding physical system become available. Thus a digital twin mimics the behaviors of its corresponding physical system. Often digital twins are developed and deployed for a specific purpose, e.g., optimizing maintain schedules, process monitoring for improved safety, optimal control of complex systems.

In this workshop, we will learn how to develop digital twin models in the open-source Pyomo ecosystem.

![Pyomo workflow](./images/pyomo_workflow.png)

Specifically, we will learn how to use two Pyomo-based toolkits:
* `ParmEst` for parameter estimation and uncertainty quantification
* `Pyomo.DoE` for model-based design of experiments

## What do I need to complete the tutorial?

This tutorial assumes the audience is familar with basic Python programming. (New to Python? Check out [this](https://lectures.scientific-python.org/index.html) and similar online resources.) The tutorial is designed to run in Google Colab. The `tclab_pyomo.py` file contains the Pyomo model for our motivating system as well as utilities to install software on Colab.

Alternatively, pariticipants can run the tutorial locally on their computer. Use the following commands to create a new conda environment:

```
conda create -n summer2024 -c conda-forge -c IDAES-PSE python=3.10 idaes-pse pandas numpy matplotlib scipy ipykernel
idaes get-extensions
pip uninstall pyomo
git+https://github.com/adowling2/pyomo.git@pyomo-doe-fixes
```

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