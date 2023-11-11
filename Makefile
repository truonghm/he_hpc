.ONESHELL:
SHELL = /bin/bash
CONDA_ENV_PATH=.conda/he_hpc
CONDA_HOME_PATH=$(HOME)/miniconda3

## Create conda env (python 3.10) using environment.yml
env: 
	source $(CONDA_HOME_PATH)/bin/activate; conda create -p $(CONDA_ENV_PATH) --no-default-packages --no-deps python=3.10 -y; conda env update -p $(CONDA_ENV_PATH) --file environment.yml
	touch .conda/.gitignore
	echo "*" > .conda/.gitignore

## Remove old conda env and create a new one
env-reset:
	rm -rf $(CONDA_ENV_PATH)
	make env

render:
	quarto render