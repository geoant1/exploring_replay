# Exploring replay
This repository containes code and data for the paper 'Exploring replay' currently under review.

## Contents
- [Overview](#overview)
- [Software requirements](#software-requirements)
- [Installation guide](#installation-guide)
- [License](#lisense)

# Overview
The repository is structured as follows:
- `paper/code` contains all models and maze configurations
- - `paper/code/bandit` contains code for the bandit simulations
  - `paper/code/maze` contains code for the Tolman maze simulations. There is also a jupyter notebook `demo.ipynb` which shows how to run the code as well as the expected output
  - `paper/code/mazes` contains maze configurations
- `paper/figures_code` contains scripts which can be run to generate the data and any individual figure in the manuscript. Each individual figure has its own folder. Note that the figures were additionally assembled using Inkscape
- `paper/source_data` contains the source data excel spreadsheet with the raw data plotted in the figures

# Software requirements
## OS requirements
The code was developed and tested on Linux Manjaro (5.10.186-1)
## Python dependencies 
All the code was written in `Python 3.9.7` with the following external packages: 
```
numpy 
matplotlib
seaborn
scipy
jupyterlab
```
## Latex dependencies
The bandit visualisations were done by writing Latex files with the `tikz` package using custom Python code (all present within this repository). To create pdf visualisations of the trees as shown in the paper, use `pdflatex tex_file_name.tex` where the latter is the name of the file.

# Installation guide
## Install from github

```sh
git clone https://github.com/geoant1/exploring_replay.git
cd exploring_replay
python -m venv .env

# Linux/macos
source .env/bin/activate
# Windows
.env/Scripts/activate

pip install -r requirements.txt
```
## Running

After the necessary installations, the individual scripts present in `paper/figures_code` can be run to generate the figures in the paper.

# License
This projected is covered by the **MIT license**
