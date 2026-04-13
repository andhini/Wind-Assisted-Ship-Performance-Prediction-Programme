# ============================================================
#   PARALLEL CODE for MONTE CARLO + SHAP + SOBOL SENSITIVITIES
#
# Inputs to watch for, and what they mean. Ctrl+F to find. 
# TOTAL_SAMPLES(34): training number for monte carlo
# Training(39) : True or False. Monte Carlo runs if "True" and Read provided data if "False" 
# TrainingDat(40) : filename for saving a trained during Monte-Carlo, or reading a file
# Us_input(66) : ship speed in m/s
# rps(102) : propeller's rotation/sec
# PD(103) : propeller's pitch/diameter
# idx(242-243) : indices where SHAP case is picked. First index is picked by case
# ============================================================


import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from math import radians
from sklearn.ensemble import RandomForestRegressor
from multiprocessing import Pool, cpu_count
from VPP_WAShip import stationary_VPP as VPP, plot3D
from Aframax import *
from wind_probability import *
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.metrics import r2_score
import joblib, os

# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------

TOTAL_SAMPLES = 8000
KTS_TO_MS = 1/1.9438452
Filepath= os.path.abspath(__file__)
Trainingdir=os.path.dirname(Filepath)+'/'

Training= False # False will skip training with RandomForrest, and will read TrainingDat below
TrainingDat="13kts_fullyload" # or "9kts_ballast2"

# LOAD WIND PROBABILITY GRID
def load_wind_probability():
    # Assume rows = wind speed (3–25 kts), cols = angle (5–355 deg)
    TWS_bins = distribution().WS#np.arange(3, 3 + prob_matrix.shape[0])
    TWA_bins = distribution().WA# np.linspace(5, 355, prob_matrix.shape[1])

    probs = distribution().P.ravel()
    probs /= probs.sum()

    TWAs = np.repeat(TWA_bins, len(TWS_bins))
    TWSs = np.tile(TWS_bins, len(TWA_bins))

    return TWAs, TWSs, probs

# MONTE CARLO WORKER
def monte_carlo_worker(args):
    TWAs, TWSs, probs, N_local = args

    idx = np.random.choice(len(probs), size=N_local, replace=True, p=probs)

    X_list = []
    y_list = []
    n_failed = 0

    Us_input = 13.0 / 1.9438452 # m/s

    for i in range(N_local):

        twa = TWAs[idx[i]]
        tws = TWSs[idx[i]]

        result = VPP_4DOF_solver(Us_input, twa, tws)

        if result is None:
            continue

        bhp, dCxs,dCys,dCMxs,dCMzs,dCxh,dCyh,dCMxh,dCMzh,\
        dCxp,dCxr,dCyr,dCMxr,dCMzr = result

        X_list.append([
            dCxs,dCys,dCMxs,dCMzs,
            dCxh,dCyh,dCMxh,dCMzh,
            # dCxp,
            dCxr,dCyr,dCMxr,dCMzr
        ])

        y_list.append(bhp)
    return np.array(X_list), np.array(y_list)

# FIRST ORDER SENSITIVITY (APPROX)
def first_order_sensitivity(xi, y, nbins=20):
    bins = np.linspace(xi.min(), xi.max(), nbins)
    digitized = np.digitize(xi, bins)
    means = np.array([y[digitized==i].mean()
                      for i in range(1, nbins)
                      if np.any(digitized==i)])
    return np.var(means)/np.var(y)

def VPP_4DOF_solver(Us_input,TWA,TWS):
    ## ---- Initialization & inputs ------##
    rps = 82./60. ## rps, prop rotation
    PD = 0.645 ## non dimensional, P/D
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

    # print('outputs',outputs)
    if (outputs[0] is None or outputs[1] is None or
        any(v is None for v in outputs[0]) or
        any(v is None for v in outputs[1])): return None
    else:
        eps = 1e-12
        result = []

        for x in range(14):
            if abs(outputs[0][x]) < eps:
                return None  # avoid division instability

            value = (outputs[1][x] - outputs[0][x]) / outputs[0][x] * 100
            result.append(value)
        return result

def main():
    if(Training == True ):
        print("Training starts. \nLoading wind probability...")
        TWAs, TWSs, probs = load_wind_probability()

        num_workers = cpu_count()
        N_per_worker = TOTAL_SAMPLES // num_workers

        print(f"Running Monte Carlo in parallel on {num_workers} CPUs...")
        tasks = [(TWAs, TWSs, probs, N_per_worker)
                 for _ in range(num_workers)]

        with Pool(num_workers) as pool:
            results = pool.map(monte_carlo_worker, tasks)


        X = np.vstack([r[0] for r in results])
        y = np.hstack([r[1] for r in results])

        print("\n========== ROUTE-WEIGHTED RESULTS ==========")

        mean_bhp = np.nanmean(y)
        std_bhp = np.nanstd(y, ddof=1)

        print(f"Mean BHP reduction: {mean_bhp:.3f}%")
        print(f"Std deviation:      {std_bhp:.3f}%")

        # --------------------------------------------------------
        #Sensitivity analysis
        # --------------------------------------------------------

        # Bootstrap CI
        nboots = 2000
        boots = np.zeros(nboots)
        rng = np.random.default_rng(42)

        for b in range(nboots):
            draw = rng.integers(0, len(y), len(y))
            boots[b] = y[draw].mean()

        ci_lower, ci_upper = np.percentile(boots, [2.5, 97.5])
        print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]%")

        prob_above_8 = np.mean(np.abs(y) <= 8.0) #presenting BHP reduction bigger that 8%
        print(f"P(BHP ≥ 8%): {prob_above_8:.3f}")

        # --------------------------------------------------------
        # RANDOM FOREST SURROGATE
        # --------------------------------------------------------

        print("\nTraining Random Forest surrogate...")

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=0,
            n_jobs=-1
        )

        model.fit(X, y)

        ##saving the model here:
        save_model = {"model": model,"X": X,"y": y}
        joblib.dump(save_model,Trainingdir+"vpp_surrogate_"+TrainingDat+".pkl")
        print("Model and dataset saved.")

    else:
        print("Loading trained model...")
        model_data = joblib.load(Trainingdir+"vpp_surrogate_"+TrainingDat+".pkl")
        model = model_data["model"]
        X = model_data["X"]
        y = model_data["y"]

    y_pred = model.predict(X)
    print("\nSurrogate R2:", r2_score(y, y_pred))

    # pick a specific case for waterfall plot
    idx = np.where(np.logical_and(y_pred < -34,y_pred > -40))[0]
    case = idx[0]
    print('list of y:',y_pred[idx[0]:], ' got:',y_pred[case])


    print("\nFeature Importances:")
    for name, imp in zip(
            [r"$\varepsilon C_{X,s}$",r"$\varepsilon C_{Y,s}$",\
            r"$\varepsilon C_{MX,s}$",r"$\varepsilon C_{MZ,s}$",\
            r"$\varepsilon C_{X,h}$",r"$\varepsilon C_{Y,h}$",\
            r"$\varepsilon C_{MX,h}$",r"$\varepsilon C_{MZ,h}$",\
            r"$\varepsilon C_{X,r}$",r"$\varepsilon C_{Y,r}$",\
            r"$\varepsilon C_{MX,r}$",r"$\varepsilon C_{MZ,r}$"],
            model.feature_importances_):
        print(f"{name}: {imp:.3f}")

    # --------------------------------------------------------
    # SHAP GLOBAL ANALYSIS
    # --------------------------------------------------------

    print("\nGenerating SHAP summary plot...")
    feature_names=[r"$\varepsilon C_{X,s}$",r"$\varepsilon C_{Y,s}$",\
                    r"$\varepsilon C_{MX,s}$",r"$\varepsilon C_{MZ,s}$",\
                    r"$\varepsilon C_{X,h}$",r"$\varepsilon C_{Y,h}$",\
                    r"$\varepsilon C_{MX,h}$",r"$\varepsilon C_{MZ,h}$",\
                    r"$\varepsilon C_{X,r}$",r"$\varepsilon C_{Y,r}$",\
                    r"$\varepsilon C_{MX,r}$",r"$\varepsilon C_{MZ,r}$"]
    explainer = shap.Explainer(model, X,feature_names=feature_names)
    shap_values = explainer(X)

    shap.summary_plot(shap_values,X,feature_names=feature_names,show=False)

    global_importance = np.mean(np.abs(shap_values.values), axis=0)
    print("\nSHAP Global Importance:")
    for name, val in zip(feature_names, global_importance):
        print(f"{name}: {val:.4f}")

    # --------------------------------------------------------
    # FIRST ORDER SENSITIVITY
    # --------------------------------------------------------

    print("\nApprox First-Order Sensitivity:")
    for i,name in enumerate([r"$\varepsilon C_{X,s}$",r"$\varepsilon C_{Y,s}$",\
                    r"$\varepsilon C_{MX,s}$",r"$\varepsilon C_{MZ,s}$",\
                    r"$\varepsilon C_{X,h}$",r"$\varepsilon C_{Y,h}$",\
                    r"$\varepsilon C_{MX,h}$",r"$\varepsilon C_{MZ,h}$",\
                    r"$\varepsilon C_{X,r}$",r"$\varepsilon C_{Y,r}$",\
                    r"$\varepsilon C_{MX,r}$",r"$\varepsilon C_{MZ,r}$"]):
        S = first_order_sensitivity(X[:,i], y)
        print(f"{name}: {S:.3f}")

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------
    #
    # plt.figure(figsize=(7,4))
    # plt.hist(y, bins=40)
    # plt.axvline(mean_bhp, linestyle='--', label='Mean')
    # plt.axvline(8.0, linestyle='-', label='8% reference')
    # plt.legend()
    # plt.xlabel("BHP Reduction (%)")
    # plt.ylabel("Count")
    # plt.title("Monte Carlo Route-Weighted Distribution")
    plt.show()

    # Create a SHAP Explainer waterfall plot
    shap_values = explainer(X)
    shap.plots.waterfall(shap_values[case], max_display=20)
    plt.show()
    print("\nInput perturbations for selected case:")
    for name, val in zip(feature_names, X[case]):
        print(f"{name}: {val:.3f}")

    # --------------------------------------------------------
    # SOBOL GLOBAL SENSITIVITY (SURROGATE-BASED)
    # --------------------------------------------------------

    print("\n\nRunning Sobol sensitivity analysis (using RF surrogate)...")

    problem = {
        'num_vars': X.shape[1],
        'names': feature_names,
        'bounds': [[X[:,i].min(), X[:,i].max()] for i in range(X.shape[1])]
    }

    # Generate Sobol samples
    param_values = saltelli.sample(problem, 2000, calc_second_order=False)

    # Evaluate surrogate model
    Y_sobol = model.predict(param_values)
    Si = sobol.analyze(problem, Y_sobol, calc_second_order=False)

    print("\n\nSobol First-Order Indices:")
    for name, val in zip(problem['names'], Si['S1']):
        print(f"{name}: {val:.3f}")

    print("\nSobol Total Indices:")
    for name, val in zip(problem['names'], Si['ST']):
        print(f"{name}: {val:.3f}")

    plt.figure(figsize=(8,4))
    plt.bar(problem['names'], Si['S1'])
    plt.xticks(rotation=45)
    plt.ylabel("Sobol First-Order Index")
    plt.title("Global Sensitivity (Sobol)")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------

if __name__ == "__main__":
    main()
    print ('Finished!')
