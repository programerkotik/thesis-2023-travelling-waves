# ECoG-electrode signals relationship in cat visual cortex
A thesis project conducted in 2022/2023 at the Computational Systems Neuroscience Group (CSNG) of Charles University's Faculty of Mathematics and Physics. The project aims to study relationships between ECoG and electrode signals in the visual cortex of cats.
## Project organisation
The project should be organized as follows:
```
├── tmp  <-------------------# Not used files, files to be deleted
├── cfg <--------------------# Project-wide configuration (like conda setup)
├── doc  <-------------------# Any documents, later files to put latex code
├── exp  <-------------------# The main experiments folder
│   ├── 27-09-2022-exp <-----# An example Experiment
│   │   ├── tmp   <----------# Any temporary files not supposed to be saved
│   │   ├── dat  <-----------# Any data generated by workflows
│   │   ├── exp_dscr.md <----# Short experiment description
│   │   ├── plotting.ipynb
│   │   ├── functions.py
│   │   ├── params.yml
│   │   └── run.cmd <--------# One file to run full experiment
├── raw  <-------------------# Project-wide raw data
│	  ├── 10-09-2022-data
│   │   ├── dat_dscr.md <----# Short data description
│   │   └── data  <-----------# Data folder
├── res  <-------------------# Project-wide results
│	  ├── 27-09-2022-res <-----# An example Results
│   │   ├── res_dscr.md <----# Short results description
│   │   └── plots.png
├── README.md
└── src  <-----------------# Project-wide code
```
The data is organized in two layers:

1. Functional organization: Data is grouped by function
2. Chronological organization: Experiments, data and results are also sequenced by time

The raw folder is also organized by:

- EcoG/electrodes: Data is separated by the specific intracranial electrodes used
- Sampling rate: Data is separated based on the sampling frequency
```
├── raw  <-------------------# Project-wide raw data
   └── 10-09-2022-data
        ├── data_dscr.md <----# Short data description
        └── data  <-----------# Data folder
            ├── 20Khz
            └── 1Khz
```
