# Developer Guide

This guide is for contributors and maintainers of the workshop materials. The top-level [Readme.md](./Readme.md) is a landing page for workshop attendees.

## Purpose

This repository contains:

- the workshop source materials
- the MyST/Jupyter Book website source
- the GitHub Pages deployment configuration

The public workshop site is published automatically from `main` by GitHub Actions.

## Maintainer environment

Create the maintainer environment with:

```bash
conda env create -f environment-maintainer.yml
conda activate pyomo-doe-maint
idaes get-extensions
npm install -g mystmd
```

The maintainer environment includes:

- Python 3.11
- `idaes-pse`
- notebook-processing dependencies
- `nodejs`
- `mystmd` (provides the `myst` CLI)
- `jupyter-book`
- `black[jupyter]` for formatting notebook sources

## Notebook formatting

Notebook formatting is handled with Black's Jupyter support inside the maintainer environment.

- install: `conda run -n pyomo-doe-maint python -m pip install "black[jupyter]"`
- format the notebooks that are included in `myst.yml` with `black`
- skip notebooks that are commented out in `myst.yml`

## Notebook execution

For a long local execution check of the active website notebooks, use:

```bash
conda run --no-capture-output -n pyomo-doe-maint python scripts/run_notebooks_from_myst.py
```

Notes:

- the runner reads the top-level [myst.yml](./myst.yml)
- notebooks that are commented out in `myst.yml` are skipped
- student exercise notebooks in `notebooks/` are skipped by default because they intentionally contain incomplete cells
- solution notebooks in `exercise_solutions/` are still executed
- the runner prepends `~/.idaes/bin` to `PATH` so IDAES-installed solver tools such as `k_aug` and `dot_sens` can be found
- use `--include-exercises` to include the student exercise notebooks anyway
- use `--list-only` to print the resolved notebook set without executing it

`conda run` captures subprocess output by default, so use `--no-capture-output` if you want to see `[RUN]`, `[PASS]`, and `[FAIL]` lines in real time.

## Refactor Notes

We are treating [notebooks/doe_exploratory_analysis.ipynb](./notebooks/doe_exploratory_analysis.ipynb) as an obsolete archival notebook. It stays out of the active site TOC and is no longer part of the supported workflow.

Current cleanup plan:

- remove the parameterized sine-wave test path from [notebooks/tclab_pyomo.py](./notebooks/tclab_pyomo.py)
- keep the active notebooks in [myst.yml](./myst.yml) working without any sine-specific branches
- use pytest to cover the shared `tclab_pyomo.py` module and the active notebook set

Testing approach:

- unit tests should focus on `tclab_pyomo.py` behavior that is shared by the active notebooks
- notebook tests should verify that every active notebook listed in `myst.yml` is still parseable and that the site build succeeds
- full notebook execution is handled by `scripts/run_notebooks_from_myst.py` as a separate long local check
- because solver binaries are not guaranteed in every environment, we are not relying on full notebook execution in pytest right now

## Site build workflow

Every site build has three stages:

1. preprocess notebooks
2. package the custom theme from the vendored theme source
3. build the website

Those stages are encoded in the repo scripts so maintainers do not need to remember the exact command sequence.

## Theme architecture

The custom theme workflow has two layers:

1. Theme source subtree:
   - [vendor/myst-theme](./vendor/myst-theme)
2. Generated distributable theme artifact:
   - [themes/pyomo-book-theme-dist](./themes/pyomo-book-theme-dist)

`myst.yml` points to the packaged artifact, not the raw source subtree.

This split is necessary because MyST expects a packaged template directory containing:

- `template.yml`
- `server.js`
- `package.json`
- `package-lock.json`
- `public/`
- `build/`

The raw subtree is appropriate for development, but not for direct consumption by `site.template`.

## Maintainer scripts

### `scripts/build_theme_dist.sh`

```bash
bash scripts/build_theme_dist.sh
```

Use this when:

- the vendored `vendor/myst-theme` subtree has changed
- you want to refresh the packaged theme artifact manually

What it does:

- installs or updates the vendored theme workspace dependencies
- builds the required workspace packages
- builds the production book theme
- assembles a distributable theme in `themes/pyomo-book-theme-dist`
- generates `package-lock.json` for the packaged theme

### `scripts/build_local.sh`

```bash
bash scripts/build_local.sh
```

Use this for the normal interactive local preview workflow.

What it does:

- runs notebook preprocessing
- rebuilds the packaged theme
- builds the site with the local preview flow
- starts the local preview server

Default ports:

- site UI: `http://localhost:4300`
- content server: `http://localhost:4301`

Environment overrides:

- `BOOK_PORT`
- `BOOK_SERVER_PORT`

### `scripts/preview_html.sh`

```bash
bash scripts/preview_html.sh local
```

or:

```bash
bash scripts/preview_html.sh pages
```

Use this for static HTML preview.

Modes:

- `local`
  - builds the site without `BASE_URL`
  - serves the static output at `http://localhost:8000/`
- `pages`
  - builds the site with `BASE_URL=/<repo-name>`
  - serves the static output under a repository subpath so it behaves like GitHub Pages

Environment override:

- `HTML_PREVIEW_PORT`

### `scripts/publish.sh`

```bash
bash scripts/publish.sh
```

Use this as a local CI-style sanity check before pushing.

What it does:

- preprocesses notebooks
- rebuilds the packaged theme
- runs `myst build --html`

It does not deploy anything directly. GitHub Pages deployment happens through the GitHub Actions workflow after you push to `main`.

## Publishing with GitHub Actions

Publishing is configured in:

- [.github/workflows/deploy.yml](./.github/workflows/deploy.yml)

The workflow:

1. checks out the repo
2. installs Python and Node tooling
3. preprocesses notebooks
4. rebuilds the packaged custom theme
5. runs `myst build --html`
6. uploads `./_build/html`
7. deploys the uploaded artifact to GitHub Pages

GitHub Pages settings for this repository should use:

- `Source: GitHub Actions`

## Updating the vendored theme

The theme source is maintained in the separate [`dowlinglab/myst-theme`](https://github.com/dowlinglab/myst-theme) fork and imported here using `git subtree`.

Typical workflow:

1. Make source-level theme changes in the fork.
2. Test those changes there locally.
3. Pull the updated source into `vendor/myst-theme`.
4. Run `bash scripts/build_theme_dist.sh`.
5. Test the website locally with `bash scripts/build_local.sh` or `bash scripts/preview_html.sh pages`.
6. Commit both the subtree update and the regenerated packaged theme artifact.

## Immportant files for maintainers

- [myst.yml](./myst.yml)
- [environment-maintainer.yml](./environment-maintainer.yml)
- [scripts/process_notebooks.py](./scripts/process_notebooks.py)
- [scripts/run_notebooks_from_myst.py](./scripts/run_notebooks_from_myst.py)
- [scripts/build_theme_dist.sh](./scripts/build_theme_dist.sh)
- [scripts/build_local.sh](./scripts/build_local.sh)
- [scripts/preview_html.sh](./scripts/preview_html.sh)
- [scripts/publish.sh](./scripts/publish.sh)
- [.github/workflows/deploy.yml](./.github/workflows/deploy.yml)
