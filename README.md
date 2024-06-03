# MSc Thesis notebook repo - David Haasnoot
Code for Thesis which runs the requires [eWaterCycle](https://www.ewatercycle.org/) platform to be installed, easily done [with these instructions](https://github.com/eWaterCycle/ewatercycle). 
Some notebook require geospatial packages, namely `geopandas` and `contexitly` which aren't included in the eWatercycle environment. 
I would advise to make a new environment for this.
Runs models in eWaterCycle environment, applies DA to said models


# Data
This Repo contains data from the CAMELS USA data set which can be found [here](https://ral.ucar.edu/solutions/products/camels). 
Current hosting of this data is merely to keep it organised for myself and own use. Please download the data for your use case from the original owner. 
On completion of this thesis all data will be removed from github. 
To the notebooks more FAIR the data for catchment 1620500 will remain hosted here. This is merely to make the notebooks useable.
Once a FAIR way of acessing the camels data datset is created, this will be removed. 

# File system
This started off a as WIP repo, so it does contain _all_ files used in running and analysing code, but the downside is the (earlier) code is messy and not organised. 
In some cases seperate repos have been used, these are explained. Most of the (unorganised code) is also here though.

## layout
- `0-29_development`: Notebooks from start to the end of development stages. Syntax and good practices still change a lot here
- `30-43_application_and_analysis`: This is the start of the experiment phase, moving away from notebooks and to `.py` files.
Everything else (Forcing, Observations, notebooks '1-5'): demonstration as included at the end of the thesis report. 

## other code repos
- Those used for running experiments are practically identitical but save you from cloning this whole repo:
    - [run_experiment](https://github.com/Daafip/run_experiment) runs experiment for 671 catchmetns with a provided hyperparameter set storing min, max and weighted mean. 
    - [run_experiment_best](https://github.com/Daafip/run_experiment_best) same as run experiment but also stores the highest weight
    - [run_calibration](https://github.com/Daafip/run_calibration) Runs a monte carlo calibration for 500 iterations across 671 catchments

- Model repos:
    - [HBV BMI](https://github.com/Daafip/HBV-bmi) Basic Model Interface code for the HBV which is run by the ewatercycle-HBV
    - [ewatercycle-HBV](https://github.com/Daafip/ewatercycle-hbv) Wrapper for the HBV BMI which interfaces with the user, with [documentation](https://ewatercycle-hbv.readthedocs.io/en/latest/index.html). 
    - [Lorenz BMI](https://github.com/Daafip/lorenz-bmi) Basic Model Interface code for the HBV which is run by the ewatercycle-lorenz
    - [ewatercycle-lorenz](https://github.com/Daafip/ewatercycle-lorenz) Wrapper for the Lorenz model BMI which interfaces with the user, with [documentation](https://ewatercycle-lorenz.readthedocs.io/en/latest/index.html).
 
- Main framework:
    - [eWaterCycle-DA](https://github.com/Daafip/eWaterCycle-DA) Main framework made in this thesis is found here with [documentation](https://ewatercycle-da.readthedocs.io/en/latest)


## Useful notes:
```bash
wsl --shutdown
netsh winsock reset
netsh int ip reset all
netsh winhttp reset proxy
ipconfig /flushdns
netsh winsock reset
shutdown /r

```
`jupyter nbconvert "mynotebook.ipynb" --to python`
