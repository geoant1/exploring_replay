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
  - `paper/code/maze` contains code for the Tolman maze simulations
  - `paper/code/mazes` contains maze configurations
- `paper/figures_code` contains scripts which can be run to generate the data and any individual figure in the manuscript. Each individual figure has its own folder. Note that the figures were additionally assembled using Inkscape.

# Software requirements
## OS requirements
The code was developed and tested on Linux Manjaro (5.10.186-1)
## Python dependencies 
All the code was written in `Python 3.9.5` with the following external packages: 
```
numpy 
matplotlib
seaborn
scipy
```
## Latex dependencies
The bandit visualisations were done by writing Latex files using custom Python code (all present within this repository). To create pdf visualisations of the trees as shown in the paper, use `pdflatex tex_file_name.tex` where the latter is the name of the file.

# Installation guide
## Install from github
`git clone https://github.com/geoant1/exploring_replay.git`

# License
This projected is covered by the **MIT license**
