# general python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import os
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import scipy
import xarray as xr
from tqdm import tqdm
import glob
# from devtools import pprint
# from rich import print

# general eWC
import ewatercycle
import ewatercycle.forcing
import ewatercycle.models
from tqdm import tqdm


path = Path.cwd()
forcing_path = path / "Forcing"
observations_path = path / "Observations"
figure_path = path / "Figures"
output_path = path / "Output"

from ewatercycle.forcing import sources
from ewatercycle_DA import DA


def calc_NSE(Qo, Qm):
    QoAv  = np.mean(Qo)
    ErrUp = np.sum((Qm - Qo)**2)
    ErrDo = np.sum((Qo - QoAv)**2)
    return 1 - (ErrUp / ErrDo)

## Array of initial storage terms - we keep these constant for now 
##              Si,  Su, Sf, Ss
s_0 = np.array([0,  100,  0,  5, 0])

## Array of parameters min/max bounds as a reference
##                      Imax,  Ce,  Sumax, beta,  Pmax,  T_lag,   Kf,   Ks, FM
p_min_initial= np.array([0,   0.2,  40,    .5,   .001,   1,     .01,  .0001, 6])
p_max_initial = np.array([8,    1,  800,   4,    .3,     10,    .1,   .01, 0.1])
p_names = ["$I_{max}$",  "$C_e$",  "$Su_{max}$", "β",  "$P_{max}$",  "$T_{lag}$",   "$K_f$",   "$K_s$", "FM"]
S_names = ["Interception storage", "Unsaturated Rootzone Storage", "Fastflow storage", "Groundwater storage", "Snowpack storage"]
param_names = ["Imax","Ce",  "Sumax", "Beta",  "Pmax",  "Tlag",   "Kf",   "Ks", "FM"]
stor_names = ["Si", "Su", "Sf", "Ss", "Sp"]

from ewatercycle_DA.local_models.HBV import HBVLocal
experiment_start_date = "1997-08-01T00:00:00Z"
experiment_end_date = "2000-08-01T00:00:00Z"
# HRU_id = 1411300
alpha = 1.26

HRU_ids = [path.name[1:8] for path in  forcing_path.glob("*.txt")]

def run_callibration(HRU_id):
    camels_forcing_callibration = sources.HBVForcing(start_time = experiment_start_date,
                              end_time = experiment_end_date,
                              directory = forcing_path,
                              camels_file = f'0{HRU_id}_lump_cida_forcing_leap.txt',
                              alpha = alpha,
                              )
    
    ensemble = DA.Ensemble(N=n_particles)
    ensemble.setup()
    
    array_random_num = np.array([[np.random.random() for i in range(len(p_max_initial))] for i in range(n_particles)])
    p_intial = p_min_initial + array_random_num * (p_max_initial-p_min_initial)
    # values wihch you 
    setup_kwargs_lst = []
    for index in range(n_particles):
        setup_kwargs_lst.append({'parameters':','.join([str(p) for p in p_intial[index]]), 
                                'initial_storage':','.join([str(s) for s in s_0]),
                                 })
    ensemble.loaded_models.update({'HBVLocal': HBVLocal})
    # this initializes the models for all ensemble members. 
    ensemble.initialize(model_name=["HBVLocal"]*n_particles,
                        forcing=[camels_forcing_callibration]*n_particles,
                        setup_kwargs=setup_kwargs_lst) 
    ensemble.set_state_vector_variables('all')
    ref_model = ensemble.ensemble_list[0].model
    n_timesteps = int((ref_model.end_time - ref_model.start_time) /  ref_model.time_step)
    
    time_cal = []
    lst_state_vector = []
    lst_Q_cal = [] 
    
    lst_state_vector.append(ensemble.get_state_vector())
    for i in tqdm(range(n_timesteps)):    
        time_cal.append(pd.Timestamp(ref_model.time_as_datetime.date()))
        ensemble.update(assimilate=False)
        lst_Q_cal.append(ensemble.get_value("Q").flatten()) 
        
    lst_state_vector.append(ensemble.get_state_vector())
    # end model - IMPORTANT! when working with dockers
    ensemble.finalize()
    
    Q_m_arr = np.array(lst_Q_cal).T
    # state_vector_arr = np.array(lst_state_vector)
    df_ensemble = pd.DataFrame(data=Q_m_arr[:,:len(time_cal)].T,index=time_cal,columns=[f'particle {n}' for n in range(n_particles)])
    
    ds = xr.open_dataset(forcing_path / ref_model.forcing.pr)
    observations = observations_path / f'0{HRU_id}_streamflow_qc.txt'
    cubic_ft_to_cubic_m = 0.0283168466 
    new_header = ['GAGEID','Year','Month', 'Day', 'Streamflow(cubic feet per second)','QC_flag']
    new_header_dict = dict(list(zip(range(len(new_header)),new_header)))
    
    df_Q = pd.read_fwf(observations,delimiter=' ',encoding='utf-8',header=None)
    df_Q = df_Q.rename(columns=new_header_dict)
    df_Q['Streamflow(cubic feet per second)'] = df_Q['Streamflow(cubic feet per second)'].apply(lambda x: np.nan if x==-999.00 else x)
    df_Q['Q (m3/s)'] = df_Q['Streamflow(cubic feet per second)'] * cubic_ft_to_cubic_m
    df_Q['Q'] = df_Q['Q (m3/s)'] / ds.attrs['area basin(m^2)'] * 3600 * 24 * 1000 # m3/s -> m/s ->m/d -> mm/d
    df_Q.index = df_Q.apply(lambda x: pd.Timestamp(f'{int(x.Year)}-{int(x.Month)}-{int(x.Day)}'),axis=1)
    df_Q.index.name = "time"
    df_Q.drop(columns=['Year','Month', 'Day','Streamflow(cubic feet per second)'],inplace=True)
    df_Q = df_Q.dropna(axis=0)
    df_Q_Cal = df_Q.loc[time_cal]
    
    lst_nse = []
    for i in range(n_particles):
        lst_nse.append(calc_NSE(df_Q['Q'],df_ensemble[f'particle {i}']))
    
    ensmble_best_run = df_ensemble[f'particle {np.array(lst_nse).argmax()}']
    
    state_vector = lst_state_vector[-1][np.array(lst_nse).argmax()].copy()

    del df_ensemble, lst_Q_cal, ensemble, lst_state_vector
    return ensmble_best_run, state_vector


def main():
    lst_ensmble_best_run = []
    lst_state_vector = []
    for hruid in HRU_ids:
        ensmble_best_run, state_vector = run_callibration(hruid)
        lst_ensmble_best_run.append(ensmble_best_run)
        lst_state_vector.append(state_vector)

    current_time = str(datetime.now())[:-10].replace(":","_")
    np.savetxt(ouput_path / f"lst_nse_{current_time}.txt" ,np.array(lst_ensmble_best_run))
    np.savetxt(ouput_path / f"lst_state_vectors_{current_time}.txt" ,np.array(lst_state_vector))

if __name__ == "__main__":
    main()



