# Skunk Works Notes

This folder is a scratchpad for working notes and small write-ups. The current focus is the TCLab reformulation discussion and the local solver setup on this laptop.

## Files In This Folder

### Root-Level Notes

- `README.md`: this file
- `reformulating_tclab_second_order_as_delayed_single_state_system.md`: the write-up describing the delayed single-state reformulation of the TCLab model
- `tclab_reformulated.py`: the standalone TCLab model and estimation script

If you want the main narrative for the planned approach, start with
`reformulating_tclab_second_order_as_delayed_single_state_system.md`.

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

## TCLab Multistart Findings

We ran the TCLab sweep with 10 multistart restarts per model using the sine-wave dataset from `parmest.ipynb`.

Observed outcomes:

- `delay_order=2`, physics parameterization: the multistart search landed in a poor basin with a large delay estimate and a weak fit.
- `delay_order=2`, input-output parameterization: the multistart search recovered the good fit that matched the earlier single-start result.
- `delay_order=3`, physics parameterization: the multistart search again landed in a poor basin with a very large delay.
- `delay_order=3`, input-output parameterization: the fit was also poor and the delay estimate remained very large.

| Delay Order | Parameterization | Best Objective | Fit Quality | Notes |
| --- | --- | --- | --- | --- |
| 2 | physics (`Ua`, `Cp`, `theta`) | 47,314.3 | Poor | Very large delay; weak fit |
| 2 | input-output (`K`, `tau`, `theta`) | 465.151 | Good | Recovered the earlier good basin |
| 3 | physics (`Ua`, `Cp`, `theta`) | 47,314.3 | Poor | Very large delay; weak fit |
| 3 | input-output (`K`, `tau`, `theta`) | 47,179.3 | Poor | Very large delay; weak fit |

Best-fit parameter values:

| Delay Order | Parameterization | `Ua` / `K` | `Cp` / `tau` | `theta` |
| --- | --- | --- | --- | --- |
| 2 | physics (`Ua`, `Cp`, `theta`) | 0.043394 | 3.918313 | 894383.427664 |
| 2 | input-output (`K`, `tau`, `theta`) | 0.769616 | 165.052133 | 16.034829 |
| 3 | physics (`Ua`, `Cp`, `theta`) | 0.043394 | 3.918313 | 894383.427664 |
| 3 | input-output (`K`, `tau`, `theta`) | 0.725447 | 86.737615 | 1504.354826 |

This tells us:

- the delayed TCLab formulation is still quite sensitive to initialization
- the `K, tau, theta` form is currently the most promising one for the `delay_order=2` case
- the `delay_order=3` cases likely need better bounds, smarter initialization, or a different multistart strategy before they are useful for parameter estimation
- covariance estimates become extremely large when the solver lands near the upper delay bound, so those covariance results should be interpreted cautiously

The sweep log is saved at:

- `tclab_reformulated_console.log`

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

## Recommended Next Steps

1. Inspect the multistart restart CSV files to see which sampled starts consistently lead to the good `delay_order=2` basin.
2. Try a second multistart sampling method, likely `latin_hypercube`, to see whether the restart coverage improves.
3. Tighten or re-think the delay bounds and initial guesses, especially for `delay_order=3`.
4. Consider separating the `delay_order=2` and `delay_order=3` comparisons into their own runs so the logs and results are easier to read.
5. Once the multistart behavior is stable, move the workflow into a notebook wrapper for interactive use.

## Hibernation Notes

This experiment is in a good pause state now.

What to remember when resuming:

- The main script lives at `skunk_works/tclab_reformulated.py`.
- The latest multistart sweep results live in `skunk_works/tclab_reformulated_results/`.
- The console log for the last sweep is `skunk_works/tclab_reformulated_results/tclab_reformulated_console.log`.
- The best-performing case from the 10-restart sweep was `delay_order=2` with the `K, tau, theta` parameterization.
- The poor-basin cases are still worth revisiting, but only after we improve initialization or bounds.

Suggested resume plan:

1. Read the multistart restart CSV files first, especially for the `delay_order=2` physics case and the `delay_order=3` runs.
2. Try `--multistart-method latin_hypercube` with the same `--n-restarts 10` sweep.
3. If the `delay_order=3` cases are still poor, tighten the scope to `delay_order=2` only until the restart strategy is improved.
4. Once the multistart behavior looks stable, migrate the workflow into a notebook and keep the script as the batch runner.
