## ---------------Using Aframax ship-------------------##

## This is an example case of calling 'multi_solve' for arrays
## of inputs Us, TWA, & TWS. The re-initialization is automatic included
## in the solver. The results are the png and csv files of 6 outputs
## if 'savegraph' & 'printcsv' are True. The screen will print the progress &
## re-iteration. 'printcsv' will save 4 variables (Us/RPM/P/D/,leeway,heel,rudder angles)
## and two outputs (BHP & fuel saving).
## This still use serial mode (non parallel)
##-----------------------------------------------------##

import numpy as np
from VPP_WAShip import stationary_VPP as VPP
from Aframax import *

## ---- new initialization & inputs ------##
Us_input = 11./1.9438  ## m/s, ship speed
rps = 80/60. ## rps, prop rotation
PD = 0.67 ## non dimensional, P/D
D = 7.3 ## metre, prop diameter
pitch = PD*D ## metre

## for multisolve, set 'indata=True' then input the TWAs & TWs instead of the single ones.
## Data below is for 35 TWA and 7 TWS. Pay attention to the units
TWA_array = np.linspace(-175,180,25) ## deg, or ;np.array([-90,-40, 0, 60, 156])
TWS_array = np.linspace(1,18,7) ## m/s,  or np.array([2,6,10,14,18])/1.9438452 
# print(TWA_array)

## No need initalization

##  init the prop
newprop=Aframax_prop(inUs=Us_input,inn=rps,inp=pitch )
manual_prop = False # using an OW prop data if True, Wageningen if False


## ----------- end here ---------##

## calling VPP multi solve
VPP(aerofile='int_para.csv',aero_data='Fujiwara',
    algorithm_option="multi_solve",variable_option=3,
    vesseltype=Aframax_ship(),vesselsail=Aframax_sail(),vesselprop = newprop,
    vesselrudder=Aframax_rudd(),vesselresistance=hull_resistance(),
    indata=True,twin_screw=False, prop_data=manual_prop,
    TWAs=TWA_array,TWSs=TWS_array,
    savegraph=False,printcsv=False,outpolar=False) 
# note: set savegraph & printcsv to True for saving
