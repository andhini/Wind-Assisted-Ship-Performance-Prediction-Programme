## -----------Aframax ship properties --------- ##
## All convention are based on Fujiwara's ##
## Note: For fully loaded cases, uncomment line 19-26, and change Tp and Wp in 94-95
##----------------------------------------------##
import numpy as np

def Aframax_ship():
    ## from Neda Aframax 'Seriana' by Sumitomo
    shipname = "Aframax"
    ## ballasted
    W_s = 58723.*1000 #kg, displ in mass term, Ballasted
    LPP = 234.34 #m, Length between perpendicular (WL length from bow to mid of rudder)
    COG = np.array([LPP/2,0.0,7.5-2.88]) ##UNKNOWN; m,distances: [longitudinal from stern,mid,vertical from keel]
    LBT = np.array([ 238.30,42,7.5]) #m,[LOA,Breadth,Draught from keel] LOA is L_WL
    LBT_WL= np.array([LBT[0],LBT[1],LBT[2]]) ## LBT at waterline
    COB = np.array([-2.02/100*LPP+0.5*abs(LPP-LBT_WL[0]),0.0,7.5-3.6]) #m, center of buoyancy from 1/2*L_WL ,midship,COG-Z
    GM = 17.59# UNKNOWN,m, upright from COG

    # ## in fully loaded condition, uncomment corresponded parameters above, and use the following!
    # W_s = 120000.*1000 #kg, displ in mass term, fully loaded
    # LPP = 234.34 #m, Length between perpendicular (WL length from bow to mid of rudder)
    # COG = np.array([LPP/2,0.0,13.6-2.88]) ##UNKNOWN; m,distances: [longitudinal from stern,mid,vertical from keel]
    # LBT = np.array([ 238.30,42,13.6]) #m,[LOA,Breadth,Draught from keel] LOA is L_WL
    # LBT_WL= np.array([LBT[0],LBT[1],LBT[2]]) ## LBT at waterline
    # COB = np.array([-2.02/100*LPP+0.5*abs(LPP-LBT_WL[0]),0.0,13.6-3.6]) #m, center of buoyancy from 1/2*L_WL ,midship,COG-Z
    # GM = 17.59*(0.82)# UNKNOWN,m, upright from COG, reduced by 18%

    ## Froude scale
    Lambda_Fr = 32.5##self.LBT[0]/ 2.64 #geometry scale, based on Froude
    return shipname, W_s, COG, COB, LBT, GM, LBT_WL, LPP, Lambda_Fr

class hull_resistance():
    def __init__(self):
        self.name = 'Fujiwara hull resistance'

        ## Calm water resistance regression Coefficients in the form of a list of 4
        ## X0= c0 + c1*Fn + c2*Fn**2 + c3*Fn**3 into matrix [c0,c1,c2,c3]
        ## If self.X0s=None, use HoltropMennen by default. If 'Fujiwara' for Fujiwara coeffs

        self.X0s = None# 'Fujiwara' # if None, use HoltropMennen by default

        ## Hull resistance coefficients with the effect of yaw & roll angles
        ## in the form of (see paper by Fujiwara et. al. page 135 equation 4a):
        ##    Xh = -X0 + (ci[0,0]*beta**2 +ci[0,1]*beta*phi_h+ ci[0,2]*phi_h**2+ ci[0,3]*beta**4)
        ##    Yh = ci[1,0]*beta + ci[1,1]*phi_h + ci[1,2]*beta**3 + ci[1,3]*(beta**2)*phi_h + \
        ##            ci[1,4]*beta*(phi_h**2) + ci[1,5]*phi_h**3
        ##    Nh = ci[2,0]*beta + ci[2,1]*phi_h + ci[2,2]*beta**3 + ci[2,3]*(beta**2)*phi_h + \
        ##            ci[2,4]*beta*(phi_h**2) + ci[2,5]*phi_h**3
        ##    Kh = ci[3,0]*beta + ci[3,1]*phi_h + ci[3,2]*beta**3 + ci[3,3]*(beta**2)*phi_h + \
        ##            ci[3,4]*beta*(phi_h**2) + ci[3,5]*phi_h**3
        ## Input example:
        ##     self.ci = [[-4.57534910e-01, -2.59403308e-03, -4.70276480e-02,  9.38358765e+00, 0.,0.],\
        ##     [0.18011349, -0.02443035, 12.86616537,  5.20279989,  0.21490457,  0.96260525],\
        ##     [0.10686391, -0.00880339, -3.2544272,  -0.56043158,  2.50407179,  1.1917856],\
        ##     [0.23275386,  0.07297612, 10.12644594, 17.39707893, 17.81728813,  0.84021688]]
        ## if self.ci=None, use fujiwara by default

        self.ci = None  # if None, use fujiwara by default

class Aframax_rudd():
    def __init__(self):
        ## rudder 1 & 2
        self.name = 'Aframax rudder'
        self.h = 9.6##m, rudder height or span
        self.area = 57.6 ##m^2
        self.AR = (self.h**2)/self.area
        self.COP1 = np.array([-117.24,0.,Aframax_ship()[2][2]-4.]) # rudder's Center of Pressure from CoG
        self.COP2 = np.array([0.,0.,0.])# rudder's Center of Pressure from CoG, zeros if only 1 prop propvided
        self.aH =  54  ## rudder force coefficient, non-dimensional
        self.aH_K = -42 # if different than SideForce aH, put 'None' if unused
        self.xH_prime =  -0.464 ##   moment coefficient xH=xH_prime*LPP
        self.kappa = -.5 ## kx/epsilon adjusting Fujiwara, straightening coefficient, non-dimensional
        self.C_1mintR = 19.5  ## None if using Fujiwara, non-dimensional
        self.epsilon = 1.09 ## None if using Fujiwara, non-dimensional

class Aframax_prop():
    def __init__(self,inUs,inn,inp):
        self.name = 'Aframax propeller'
        self.D = 7.3 ##m, diameter prop 1 & 2
        self.COP1 = np.array([-113.5 ,0.,Aframax_ship()[2][2]-4.]) # Prop's Center of Pressure from COG
        self.COP2 = np.array([0. ,0. , 0.]) # Prop's Center of Pressure from COG, zeros if only 1 prop propvided
        self.n =  inn ##rps, fixed shaft speed rotation at working, MCR=90.7
        self.p = inp ##m, propeller pitch for P/D=0.7

        ## For Wageningen B4
        self.Z = 4 #num of blade --->change here
        self.AE = 40 # AE/A0 is expanded ratio. AE entrance diameter
        self.A0 = 100 # outlet diameter, same unit with AE

        ## Others, for ballast, uncommented for fully loaded
        self.Tp = 0.24 ##0.24## thrust deduction factor: None if unknown
        self.Wp0 = 0.12 ##0.12 ## wake fraction: None if unknown
        ## Others, for fully loaded, uncommented for ballast
        # self.Tp = 0.## thrust deduction factor: None if unknown
        # self.Wp0 = 0.393 ## wake fraction: None if unknown

        ## ship initial speed if fixed
        self.Us = inUs # m/s ## 11./1.94384

class Aframax_sail():
    def __init__(self):
        ## sail wing 1 data using Fujiwara's
        self.name = 'Fujiwara sail'
        self.area = 553 ##sail surface planform in m**2, transverse or general area
        self.area_longt = 1997 ## #sail surface planform in m**2, longitudinal
        self.chord = None #m ##----->change here
        self.COP = np.array([0,0.0,0.]) # wing's Center of Pressure from ship's pivot ##----->change here
