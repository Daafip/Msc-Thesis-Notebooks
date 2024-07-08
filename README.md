# MSc Thesis notebook repo - David Haasnoot
Code for Thesis: 'Analysing deficiencies in hydrological models using data assimilation.' which runs the requires [eWaterCycle](https://www.ewatercycle.org/) platform to be installed, easily done [with these instructions](https://github.com/eWaterCycle/ewatercycle). <br>
Some notebook require geospatial packages, namely `geopandas` and `contexitly` which aren't included in the eWatercycle environment. <br>
I would advise to make a new environment for the geospatial analysis.<br>

The code runs models in eWaterCycle environment, applies data assimilation to said models and generates results.<br>
For more info see the thesis hosted on [repository.tudelft.nl](https://repository.tudelft.nl)


# Data
This Repo contains data from the CAMELS USA data set which can be found [here](https://ral.ucar.edu/solutions/products/camels). 
Current hosting of this data is merely to keep it organised for myself and own use. Please download the data for your use case from the original owner. 
On completion of this thesis all data will be removed from github. 
To the notebooks more FAIR the data for catchment 1620500 will remain hosted here. This is merely to make the notebooks useable.
Once a FAIR way of acessing the camels data datset is created, this will be removed. 

# File system
This started off a as WIP repo, so it does contain _all_ files used in running and analysing code, but the downside is the (earlier) code is messy and not organised. 
In some cases seperate repos have been used, these are explained. Most of the (unorganised code) is also here though.
Warning this repo is LARGE! Cloning it all in not recomended. Optimisation will take place later to remove forcing and large model runs. For now it also serves as a backup. 

## layout
- `0-29_development`: Notebooks from start to the end of development stages. Syntax and good practices still change a lot here. The subdirectory also contains a readme with explanation of the folders. 
- `30-43_application_and_analysis`: This is the start of the experiment phase, moving away from notebooks and to `.py` files.  The subdirectory also contains a readme with explanation of the folders. 
- Everything else (Forcing, Observations, notebooks '1-5'): demonstration as included at the end of the thesis report. Three examples of running DA are show. Then then the lorenz model is shown and lastly using caravan data in eWaterCycle. 
    1. A classical example where all (hyper)parameters are defined beforehand. 
    1. A on the fly data assimilation where the (hyper)parameter for the data assimilation experiment are added after the model has already run.
    1. A calibration run for a hydrological model saving all the states and parameters specified.
    1. Show case use of the Lorenz model in ewatercycle, which can be found [here](https://github.com/Daafip/ewatercycle-lorenz). 
    1. The Carvan dataset was made accessible in eWatercycle. An interactive map show casing the catchments can be found at [https://www.ewatercycle.org/caravan-map/](https://www.ewatercycle.org/caravan-map/). The pull request can be found [here](https://github.com/eWaterCycle/ewatercycle/pull/407)

# Other Repositories created during this thesis
Alongside the notebooks shown in this repo the models and framework used were developed as a part of this thesis. <br>
Some of the experiment code was also split into different repos to make switching between running locally and on SURF easier. 

## experiments
- Those used for running experiments are practically identitical but save you from cloning this whole repo:
    - [run_experiment](https://github.com/Daafip/run_experiment) runs experiment for 671 catchmetns with a provided hyperparameter set storing min, max and weighted mean. 
    - [run_experiment_best](https://github.com/Daafip/run_experiment_best) same as run experiment but also stores the highest weight
    - [run_calibration](https://github.com/Daafip/run_calibration) Runs a monte carlo calibration for 500 iterations across 671 catchments

## model
- Model repos:
    - [HBV BMI](https://github.com/Daafip/HBV-bmi) Basic Model Interface code for the HBV which is run by the ewatercycle-HBV
    - [ewatercycle-HBV](https://github.com/Daafip/ewatercycle-hbv) Wrapper for the HBV BMI which interfaces with the user, with [documentation](https://ewatercycle-hbv.readthedocs.io/en/latest/index.html). 
    - [Lorenz BMI](https://github.com/Daafip/lorenz-bmi) Basic Model Interface code for the HBV which is run by the ewatercycle-lorenz
    - [ewatercycle-lorenz](https://github.com/Daafip/ewatercycle-lorenz) Wrapper for the Lorenz model BMI which interfaces with the user, with [documentation](https://ewatercycle-lorenz.readthedocs.io/en/latest/index.html).
 
 ## Framework
 - Data assimilation framework:
    - [eWaterCycle-DA](https://github.com/Daafip/eWaterCycle-DA) Main framework made in this thesis is found here with [documentation](https://ewatercycle-da.readthedocs.io/en/latest)
