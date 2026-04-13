## --------------Using Aframax ship-------------------##
## This is an example case of calling 'single_solve' once for one pair
## of inputs: Us, TWA, & TWS. There's no re-initialization
## constructed. If the solver doesn't find solution, try to modify
## the angles in initialiations.
##----------------------------------------------------##
import numpy as np
from VPP_WAShip import stationary_VPP as VPP
from Aframax import *

## ----------- Initialization & inputs ---------##
Us_input = 9./1.9438  ## m/s, ship speed
rps = 81/60. ## rps, prop rotation
PD = 0.745 ## non dimensional, P/D
D = 7.3 ## metre, prop diameter
pitch = PD*D ## metre

## for single_solve & comparation with real ship's BHP, set TWA=0 and TWS=0:
singleTWA = 0. ## deg, True Wind Angle 
singleTWS = 0./1.9438 ## m/s, True Wind Speed 
#-----------------------------------------------##

## for angle initalizations
init_beta = -1.## deg, leeway angles
init_heel = -0.1 ## deg, heel angle
init_delta = .7 ## deg, rudder angle


##  init the prop
newprop=Aframax_prop(inUs=Us_input,inn=rps,inp=pitch )
manual_prop = False # using Wageningen


## ----------- end here ---------##

## calling VPP single solve
VPP(aerofile='no_sail.csv',aero_data='Fujiwara',
    algorithm_option="single_solve",variable_option=3,
    vesseltype=Aframax_ship(),vesselsail=Aframax_sail(),vesselprop = newprop,
    vesselrudder=Aframax_rudd(),vesselresistance=hull_resistance(),
    twin_screw=False, prop_data=manual_prop,
    singleTWA=singleTWA,singleTWS=singleTWS,
    init_beta=init_beta,init_phi=init_heel,init_delta=init_delta,
    savegraph=False,printcsv=False,par=False)
