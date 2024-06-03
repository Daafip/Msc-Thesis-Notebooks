import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
import xarray as xr
from tqdm import tqdm
import gc

from ewatercycle.forcing import sources
from ewatercycle_DA import DA

def H(Z):
    """Operator function extracts observable state from the state vector"""
    len_Z = 15
    if len(Z) == len_Z:
        return Z[-1] 
    else: 
        raise SyntaxWarning(f"Length of statevector should be {len_Z} but is {len(Z)}")

def calc_NSE(Qo, Qm):
    QoAv  = np.mean(Qo)
    ErrUp = np.sum((Qm - Qo)**2)
    ErrDo = np.sum((Qo - QoAv)**2)
    return 1 - (ErrUp / ErrDo)
    
def calc_log_NSE(Qo, Qm):
    QoAv  = np.mean(Qo)
    ErrUp = np.sum((np.log(Qm) - np.log(Qo))**2)
    ErrDo = np.sum((np.log(Qo) - np.log(QoAv))**2)
    return 1 - (ErrUp / ErrDo)

param_names = ["Imax", "Ce", "Sumax", "Beta", "Pmax", "Tlag", "Kf", "Ks", "FM"]
stor_names = ["Si", "Su", "Sf", "Ss", "Sp"]

path = Path.cwd()
forcing_path = path / "Forcing"
observations_path = path / "Observations"
output_path = path / "Output"
paths = forcing_path, output_path, observations_path
for path_i in list(paths):
    path_i.mkdir(exist_ok=True)


def experiment_run(n_particles, p_max_initial, p_min_initial, s_0, model_name, camels_forcing, 
                   paths, HRU_id, sigma_tuple, H, assimilate_window):

    forcing_path, output_path, observations_path = paths
    # set up ensemble
    ensemble = DA.Ensemble(N=n_particles)
    ensemble.setup()
    
    # initial values
    array_random_num = np.array([[np.random.random() for i in range(len(p_max_initial))] for i in range(n_particles)])
    p_intial = p_min_initial + array_random_num * (p_max_initial-p_min_initial)
    
    # values wihch you 
    setup_kwargs_lst = []
    for index in range(n_particles):
        setup_kwargs_lst.append({'parameters':','.join([str(p) for p in p_intial[index]]), 
                                'initial_storage':','.join([str(s) for s in s_0]),
                                 })
    
    
    # not required as of ewatercycle-HBV==1.8.2
    # ensemble.loaded_models.update({'HBVLocal': HBVLocal})
    print(f'init', end=" ")
    # this initializes the models for all ensemble members. 
    ensemble.initialize(model_name=[model_name]*n_particles,
                        forcing=[camels_forcing]*n_particles,
                        setup_kwargs=setup_kwargs_lst) 
    
    # create a reference model
    ref_model = ensemble.ensemble_list[0].model
    
    # load observations
    ds_obs_dir = observations_path / f'{HRU_id}_streamflow_qc.nc'
    if not ds_obs_dir.exists():

        ds = xr.open_dataset(forcing_path / ref_model.forcing.pr)
        basin_area = ds.attrs['area basin(m^2)']
        ds.close()

        observations = observations_path / f'{HRU_id}_streamflow_qc.txt'
        cubic_ft_to_cubic_m = 0.0283168466

        new_header = ['GAGEID','Year','Month', 'Day', 'Streamflow(cubic feet per second)','QC_flag']
        new_header_dict = dict(list(zip(range(len(new_header)),new_header)))
        df_Q = pd.read_fwf(observations,delimiter=' ',encoding='utf-8',header=None)
        df_Q = df_Q.rename(columns=new_header_dict)
        df_Q['Streamflow(cubic feet per second)'] = df_Q['Streamflow(cubic feet per second)'].apply(lambda x: np.nan if x==-999.00 else x)
        df_Q['Q (m3/s)'] = df_Q['Streamflow(cubic feet per second)'] * cubic_ft_to_cubic_m
        df_Q['Q'] = df_Q['Q (m3/s)'] / basin_area * 3600 * 24 * 1000 # m3/s -> m/s ->m/d -> mm/d
        df_Q.index = df_Q.apply(lambda x: pd.Timestamp(f'{int(x.Year)}-{int(x.Month)}-{int(x.Day)}'),axis=1)
        df_Q.index.name = "time"
        df_Q.drop(columns=['Year','Month', 'Day','Streamflow(cubic feet per second)'],inplace=True)
        df_Q = df_Q.dropna(axis=0)

        ds_obs = xr.Dataset(data_vars=df_Q[['Q']])
        ds_obs.to_netcdf(ds_obs_dir)
    else:
        ds_obs = xr.open_dataset(ds_obs_dir)

    # set up hyperparameters
    sigma_pp , sigma_ps, sigma_w, sigma_p_Sf = sigma_tuple
    # "Imax", "Ce", "Sumax", "Beta", "Pmax", "Tlag", "Kf", "Ks", "FM" "Si", "Su", "Sf", "Ss", "Sp + Q
    lst_like_sigma = [sigma_pp] * 9 + [sigma_ps, sigma_ps, sigma_p_Sf, sigma_ps, sigma_ps] + [0]
    hyper_parameters = {'like_sigma_weights' : sigma_w,
                        'like_sigma_state_vector' : lst_like_sigma,
                       }
    print(f'init_da',end=" ")
    
    ensemble.initialize_da_method(ensemble_method_name = "PF", 
                                  hyper_parameters=hyper_parameters,                           
                                  state_vector_variables = "all", # the next three are keyword arguments but are needed. 
                                  observation_path = ds_obs_dir,
                                  observed_variable_name = "Q",
                                  measurement_operator = H, 
                               
                                )
    # extract units for later
    state_vector_variables = ensemble.ensemble_list[0].variable_names
    units = {}
    for var in state_vector_variables:
        units.update({var : ref_model.bmi.get_var_units(var)})
        

    ## run!
    n_timesteps = int((ref_model.end_time - ref_model.start_time) /  ref_model.time_step)
    time = []
    lst_state_vector = []
    lst_N_eff = []
    lst_n_resample_indexes = []

    try:
        for i in tqdm(range(n_timesteps)):
            time.append(pd.Timestamp(ref_model.time_as_datetime.date()))
            # update every 3 steps
            if i % assimilate_window == 0:
                assimilate = True
            else:
                assimilate = False
            ensemble.update(assimilate=assimilate)

            state_vector = ensemble.get_state_vector()
            min = state_vector.T.min(axis=1)
            max = state_vector.T.max(axis=1)
            mean = state_vector.T.mean(axis=1)
            summarised_state_vector = np.array([min, max, mean])
            lst_state_vector.append(summarised_state_vector)
            del state_vector, min, max, mean, summarised_state_vector
            gc.collect()

            lst_N_eff.append(ensemble.ensemble_method.N_eff)
            if ensemble.ensemble_method.resample:
                lst_n_resample_indexes.append(
                    len(set(ensemble.ensemble_method.resample_indices)))

            else:
                lst_n_resample_indexes.append(np.nan)
    except KeyboardInterrupt: # saves deleting N folders if quit manually
        ensemble.finalize()

    ensemble.finalize()

    # post process
    state_vector_arr = np.array(lst_state_vector)
    del lst_state_vector

    data_vars = {}
    for i, name in enumerate(param_names + stor_names + ["Q"]):
        storage_terms_i = xr.DataArray(state_vector_arr[:, :, i].T,
                                       name=name,
                                       dims=["summary_stat", "time"],
                                       coords=[['min','max','mean'],
                                               time],
                                       attrs={
                                           "title": f"HBV storage terms data over time for {n_particles} particles ",
                                           "history": f"Storage term results from ewatercycle_HBV.model",
                                           "description": "Moddeled values",
                                           "units": f"{units[name]}"})
        data_vars[name] = storage_terms_i

    ds_combined = xr.Dataset(data_vars,
                             attrs={
                                 "title": f"HBV storage & parameter terms data over time for {n_particles} particles ",
                                 "history": f"Storage term results from ewatercycle_HBV.model",
                                 "sigma_pp": sigma_pp,
                                 "sigma_ps": sigma_ps,
                                 "sigma_w": sigma_w,
                                 "sigma_p_Sf": sigma_p_Sf,
                                 "assimilate_window": assimilate_window,
                                 "n_particles": n_particles,
                                 "HRU_id": HRU_id,
                                  }
                             )

    ds_observations = ds_obs['Q'].sel(time=time)
    ds_obs.close()
    ds_combined['Q_obs'] = ds_observations
    ds_combined['Q_obs'].attrs.update({
        'history': 'USGS streamflow data obtained from CAMELS dataset',
        'url':'https://dx.doi.org/10.5065/D6MW2F4D'})

    df_n_eff = pd.DataFrame(index=time,
                            data=lst_N_eff,
                            columns=['Neff'])
    df_n_eff.index.name = 'time'
    ds_combined['Neff'] = df_n_eff['Neff']
    ds_combined['Neff'].attrs.update({
        'info': 'DA debug: 1/sum(weights^2): measure for effective ensemble size'})

    df_n_eff = pd.DataFrame(index=time,
                            data=lst_n_resample_indexes,
                            columns=['n_resample'])
    df_n_eff.index.name = 'time'
    ds_combined['n_resample'] = df_n_eff['n_resample']
    ds_combined['n_resample'].attrs.update({
        'info': 'DA debug: number of uniquely resampled particles'})

    del time, ds_obs, lst_n_resample_indexes, lst_N_eff

    gc.collect()
    return ds_combined


def run(camels_forcing, HRU_id, sigma_p_Sf, n_particles):
    assimilate_window = 3  # after how many time steps to run the assimilate steps

    sigma_w = 0.45
    sigma_pp = 0.003
    sigma_ps = 2
    sigma_tuple = sigma_pp, sigma_ps, sigma_w, sigma_p_Sf

    model_name = "HBVLocal"

    save = True

    s_0 = np.array([0, 100, 0, 5, 0])
    p_min_initial = np.array([0, 0.2, 40, .5, .001, 1, .01, .0001, 6])
    p_max_initial = np.array([8, 1, 800, 4, .3, 10, .1, .01, 0.1])

    ds_summary = experiment_run(n_particles, p_max_initial, p_min_initial, s_0, model_name, camels_forcing,
                                                paths, HRU_id, sigma_tuple, H, assimilate_window)


    
    current_time = str(datetime.now())[:-10].replace(":","_")
    if save:
        file_dir = output_path / (f'{HRU_id}_psf-{sigma_p_Sf}_pp-'f'{sigma_pp}_'
                                  f'ps-{sigma_ps}_w-{sigma_w}_N-{n_particles}_'
                                  f'{current_time}.nc')
        ds_summary.to_netcdf(file_dir)
    

"""
Check list for a new experiment:
    - All values passed correctly from main -> run ->  experiment_run
        preferably don't change the actual values passed experiment_run:
        if needed refactor to list/tuple
    - preferably keep iterable set in main
    - add iterable in attrs 
    - Meaningful file path

"""
def main():
    experiment_start_date = "1997-08-01T00:00:00Z"
    experiment_end_date = "2007-09-01T00:00:00Z"

    HRU_ids = [path.name[1:8] for path in
               forcing_path.glob("*_lump_cida_forcing_leap.txt")]
    # for index, HRU_id_int in enumerate(HRU_ids[15:]):
    #     if index == 0:
    #         lst_sig_p = [0.0005]
        # else:
        #     lst_sig_p = [2, 1, 0.2, 0.05, 0.000   5]
    lst_n = [700, 800, 900, 1000]
    for n_particles in lst_n:
        for index, HRU_id_int in enumerate(["01181000"]):
            # lst_sig_p = [2, 1, 0.2, 0.05, 0.01, 0.005, 0.001 ,0.00075,0.0005,0.00025, 0.0001, 0.00005]
            lst_sig_p = [0.00005]
            # lst_sig_p = [0.00025, 0.0001, 0.00005]
            # lst_sig_p = [2]
            for sigma_p_Sf in lst_sig_p:

                HRU_id = f'{HRU_id_int}'
                if len(HRU_id) < 8:
                    HRU_id = '0' + HRU_id

                alpha = 1.26

                camels_forcing = sources.HBVForcing(start_time=experiment_start_date,
                                                    end_time=experiment_end_date,
                                                    directory=forcing_path,
                                                    camels_file=f'{HRU_id}_lump_cida_forcing_leap.txt',
                                                    alpha=alpha,
                                                    )

                current_time = str(datetime.now())[:-10].replace(":", "_")
                print(f'for {HRU_id_int} starting {sigma_p_Sf} at {current_time}')
                run(camels_forcing, HRU_id, sigma_p_Sf, n_particles)

                # remove temp file once run - in case of camels just one file
                forcing_file = forcing_path / camels_forcing.pr
                forcing_file.unlink(missing_ok=True)

   
if __name__ == "__main__":
    main()
   
