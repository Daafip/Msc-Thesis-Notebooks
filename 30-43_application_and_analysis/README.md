# MSc Thesis notebook repo - David Haasnoot \  `0-29_development`

## layout

- `30-43_application_and_analysis`: This is the start of the experiment phase, moving away from notebooks and to `.py` files.
    - `nbs_30_rewrite_to_function`: Here we move away from the notebook approach we had and instead run the function as `30.82_Camels_itterate.py` for the first time
    - `nbs_31_traditional_callibration`: This run traditional calibration in `run_trad_callibration.py`, sorry for the typo. 
    - `nbs_32_Use_Neff`: Here we adjust the particle filter scheme to use a threshold to resample, only if `N` < `N_eff` will resampling take place. $N_ef = f_n_particles * N$ 
    - `nbs_33_run_diff_sig_s`: here we change the hyperparameter which causes perturbation to be different for each of model parameters/states: sigma of the state vector = `sig_s`
    - `nbs_34_run_lower_sf`: More variation in `sig_s`
    - `nbs_35_debug_crashing`: Here we had issues with WSL consuming too much RAM in combination with DASK. Fixed it by switching to linux
    - `nbs_36_sigma_as_factor_of_value`: More changes to how the pertubation in the state vector is added: `sig_s` or `sig_p` is now changed to being a factor of the range (max-min) of the parameters space.
    - `nbs_37_synthetic_change_param`: Synthetic experiment, though not really used in the end
    - `nbs_38_repeat_many_catchments`: Formalises the work done in `nbs_36` and runs it for a large number of catchments
    - `nbs_39_formalise_algorithm`: Further formalises the work to an almost done version
    - `nbs_40_run_all_camels`: Final version of the code used to generate results. This folder stores output results too. The repo [run_experiment](https://github.com/Daafip/run_experiment) also contains the same code. This stores mean, max, min at every timestep. 
    - `nbs_41_callibration`: Final version of the calibration of used to generate results. The repo [run_calibration](https://github.com/Daafip/run_calibration) also contains this code. The repo has the results. 
    - `nbs_42_also_store_Q_best`: adjusted version of the 'nbs_40_run_all_camels': this also stores the best value at every timestep. The repo [run_experiment_best](https://github.com/Daafip/run_experiment_best) also contains this code. The repo has the results. 
    - `nbs_43_analyse_diff_best_callibrate`: analyses the results from the 'nbs_41_callibration' compared to 'nbs_42_also_store_Q_best'