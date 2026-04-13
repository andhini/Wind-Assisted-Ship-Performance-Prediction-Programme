## --------------Using Aframax ship-------------------##

## -------Parallel version of the multisolve Aframax--##
## This is an example case of calling 'multi_solve' for arrays
## of inputs Us, TWA, & TWS. The re-initialization is automatic included
## in the solver. The results are BHP csv files (manual & percentage difference)
## of no sail & sail conditions.
## Lines to change: Us_input, rps, PD. Then, uncomment lines 52-54 for saving.
## ---------------------------------------------------##
import os
import numpy as np
import pandas as pd
from VPP_WAShip import stationary_VPP as VPP
from Aframax import *
import multiprocessing as mp
from math import radians

def main():
    filepath= os.path.abspath(__file__)
    maindir=os.path.dirname(filepath)+'/'
    ## ---- new initialization & inputs ------##
    Us_input = 11./1.9438  ## m/s, ship speed
    
    ## for multisolve, set 'indata=True' then input the TWAs & TWs instead of the single ones.
    ## Data below is for 35 TWA and 7 TWS. Pay attention to the units
    TWA_array = np.linspace(-175,180,72) ## deg,
    TWS_array = np.linspace(1,18,18) ## m/s, 
    # print(TWA_array)

    n_cores = 8  # core number for parallel is manually set, should be divided by TWAxTWS

    cases = [(Us_input, twa, tws) for twa in TWA_array for tws in TWS_array]

    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(run_case, cases, chunksize=1)

    # -------- BUILD MATRICES -------- #
    bhp_pct_mat     = np.full((len(TWS_array), len(TWA_array)), np.nan)
    bhp_nosail_mat  = np.full_like(bhp_pct_mat, np.nan)
    bhp_sail_mat    = np.full_like(bhp_pct_mat, np.nan)

    for twa, tws, res in results:
        if res is not None:
            i = np.argmin(np.abs(TWS_array - tws))
            j = np.argmin(np.abs(TWA_array - twa))

            bhp_pct_mat[i, j]    = res[0]
            bhp_nosail_mat[i, j] = res[1]
            bhp_sail_mat[i, j]   = res[2]

    prename="EPS_BHP_fullyload_9kts_"
    # save_csv(bhp_pct_mat,    maindir+prename+"percentsaving.csv", TWS_array, TWA_array)
    # save_csv(bhp_nosail_mat, maindir+prename+"nosail.csv", TWS_array, TWA_array)
    # save_csv(bhp_sail_mat,   maindir+prename+"sail.csv", TWS_array, TWA_array)

    print("Saved: BHP_percent.csv, BHP_nosail.csv, BHP_sail.csv")

# -------- SAVE CSV -------- #
def save_csv(matrix, filename,TWS_array, TWA_array):
    df = pd.DataFrame(matrix,
            index=np.round(TWS_array, 3),
            columns=np.round(TWA_array, 3))
    df.index.name = "TWS (m/s)"
    df.columns.name = "TWA (deg)"
    df.to_csv(filename)
   

## ---------------- PARALLEL WRAPPER ---------------- ##
def run_case(args):
    Us_input, twa, tws = args
    res = VPP_4DOF_solver(Us_input, twa, tws)
    return (twa, tws, res)

## ------calling VPP for parallel "single-solve"-----##
def VPP_4DOF_solver(Us_input,TWA,TWS):
    ## ---- Initialization & inputs ------##
    rps = 80./60. ## rps, prop rotation
    PD = 0.67 ## non dimensional, P/D
    D = 7.3 ## metre, prop diameter
    pitch = PD*D ## metre pitch

    # ## for angle initalizations
    init_beta = -1.## deg, leeway angle
    init_heel = -0.1 ## deg, heel angle
    init_delta = .7 ## deg, rudder angle

    n_opt=8
    init = np.array([[pitch]+[i for i in sorted(np.random.uniform(0.39,1.12,n_opt)*D,reverse=False)],\
            [init_beta]+[radians(i) for i in sorted(np.random.uniform(-6,6,n_opt),reverse=False)],\
            [init_heel]+[radians(i) for i in sorted(np.random.uniform(-3,3,n_opt),reverse=True)],\
            [init_delta]+[radians(i) for i in sorted(np.random.uniform(-35,35,n_opt),reverse=True)]])

    ## ----------- end here ---------##

    ## calling VPP single solve
    aerodyn_data=['no_sail.csv','int_para.csv']

    for n_it in range(len(init[0])):
        outputs = []
        for k in range(2):
            try:
                ##  init the  prop = False--> using Wageningen
                newprop=Aframax_prop(inUs=Us_input,inn=rps,inp=init[0,n_it] )

                output=VPP(aerofile=aerodyn_data[k],aero_data='Fujiwara',\
                algorithm_option="single_solve",variable_option=3,\
                vesseltype=Aframax_ship(),vesselsail=Aframax_sail(),vesselprop = newprop,\
                vesselrudder=Aframax_rudd(),vesselresistance=hull_resistance(),\
                twin_screw=False, prop_data=False,\
                singleTWA=TWA,singleTWS=TWS,\
                init_beta=init[1,n_it],init_phi=init[2,n_it],init_delta=init[3,n_it],\
                savegraph=False,printcsv=False,par=True)
            except ValueError:
                output = None

            outputs.append(output)
        # Check convergence safely
        # Safe convergence check
        if (outputs[0] is None or outputs[1] is None):
            print("Not converging at iteration", n_it)
        else:
            break

    if (outputs[0] is None or outputs[1] is None or
        any(v is None for v in outputs[0]) or
        any(v is None for v in outputs[1])):
        return None

    eps = 1e-12

    # Absolute BHPs
    bhp_nosail = outputs[0][0]
    bhp_sail   = outputs[1][0]

    if abs(bhp_nosail) < eps:
        return None

    # Percentage change
    bhp_pct = (bhp_sail - bhp_nosail) / bhp_nosail * 100

    return (bhp_pct, bhp_nosail, bhp_sail)

    
if __name__ == "__main__":
    main()
    print ('Finished!')
