# Skunk Works Notes

This folder is a scratchpad for working notes and small write-ups. The current focus is the TCLab reformulation discussion and the local solver setup on this laptop.

## Files In This Folder

### Root-Level Notes

- `README.md`: this file
- `reformulating_tclab_second_order_as_delayed_single_state_system.md`: the write-up describing the delayed single-state reformulation of the TCLab model
- `tclab_reformulated.py`: the standalone TCLab model and estimation script

### TCLab Results

The TCLab estimation script writes its output to `skunk_works/tclab_reformulated_results/`.

That directory currently contains:

- `tclab_reformulated_console.log`: captured console output from the most recent script run
- `tclab_reformulated_fit_summary.csv`: summary table of fitted values and fit metrics
- `tclab_reformulated_covariance_summary.csv`: summary table of covariance values and propagated covariance
- `physics_delay2_SSE_fit.png`: fit plot for the physics parameterization with `delay_order=2`
- `physics_delay2_SSE_cov.png`: covariance heatmap for the physics parameterization with `delay_order=2`
- `io_delay2_SSE_fit.png`: fit plot for the input-output parameterization with `delay_order=2`
- `io_delay2_SSE_cov.png`: covariance heatmap for the input-output parameterization with `delay_order=2`
- `physics_delay3_SSE_fit.png`: fit plot for the physics parameterization with `delay_order=3`
- `physics_delay3_SSE_cov.png`: covariance heatmap for the physics parameterization with `delay_order=3`
- `io_delay3_SSE_fit.png`: fit plot for the input-output parameterization with `delay_order=3`
- `io_delay3_SSE_cov.png`: covariance heatmap for the input-output parameterization with `delay_order=3`

## What We Have Learned So Far

There are two distinct Ipopt installations on this machine:

1. `~/.idaes/bin/ipopt`
2. `~/opt/ipopt-hsl-pyomo/ipopt/bin/ipopt`

There is also a `k_aug` binary alongside the `~/.idaes` Ipopt install:

- `~/.idaes/bin/k_aug`

## Conda Environments We Found

The following conda environments are present:

- `summer2026`
- `ipopt-hsl-pyomo`
- `pyomo-doe-maint`
- several unrelated environments used for other projects

### `summer2026`

This looks like the workshop attendee environment described in the repo README.

- Python 3.11
- `idaes-pse` installed
- `pyomo` installed
- `pandas`, `numpy`, `matplotlib`, `scipy`, and `ipykernel` installed
- no `ipopt` conda package on its own
- `ipopt` and `k_aug` are not on `PATH` in a default activation

So `summer2026` is a plausible workshop base env, but it is not currently the cleanest solver-ready env by itself.

### `ipopt-hsl-pyomo`

This env is the custom Ipopt stack used in the crystallization work.

- `ipopt` is installed as a conda package in the env metadata
- `cyipopt` is installed
- `/opt/anaconda3/envs/ipopt-hsl-pyomo/bin/ipopt` is a symlink to the custom local build
- that local build lives at `~/opt/ipopt-hsl-pyomo/ipopt/bin/ipopt`
- the binary is linked against `libcoinhsl`, so it is HSL-enabled

This is the strongest match we have found so far for GreyBox / `cyipopt` workflows.

### `pyomo-doe-maint`

This is the maintainer/build environment from the workshop repo documentation.

- Python 3.11
- `idaes-pse`
- notebook-processing dependencies
- `nodejs`
- `jupyter-book`

This env is for site and maintainer tasks, not necessarily the best choice for solver-heavy workshop runs.

## Solver Lookup Notes

The local solver helpers in the crystallization project prefer Ipopt in this order:

1. `IPOPT_EXEC`
2. `IPOPT_PREFIX/bin/ipopt`
3. `~/opt/ipopt-hsl-pyomo/ipopt/bin/ipopt`
4. `~/.idaes/bin/ipopt`
5. `ipopt` on `PATH`

That means the machine has multiple valid solver entry points, and the exact conda env we choose later may depend on which notebook or script we are running.

## Current Takeaway

We do have a locally compiled Ipopt available for HSL / `cyipopt` work.

The open question is not whether Ipopt exists, but which environment we should activate for the workshop material in the next step.
