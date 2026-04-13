## -----------Holtrop-Mennen Ship properties --------- ##
import numpy as np

def HM_ship():
    ## from Holtrop Mennen paper
    shipname = "HoltropMennen_ship"
    W_s = 37500*phys().rho_w #kg, displ in mass term
    LPP =200 #m, Length between perpendicular (WL length from bow to mid of rudder)
    COG = np.array([LPP/2,0.0,6.91]) ##UNKNOWN; m,distances: [longitudinal from stern,mid,vertical from keel]
    LBT = np.array([205,32,10]) #m,[LOA,Breadth,Draught from keel] LOA is L_WL
    LBT_WL= np.array([LBT[0],LBT[1],LBT[2]]) ## LBT at waterline
    COB = np.array([-2.02/100*LPP+0.5*abs(LPP-LBT_WL[0]),0.0,3.43]) #m, center of buoyancy from 1/2*L_WL ,midship,COG-Z
    GM = 0.69# UNKNOWN,m, upright from COG

    ## Froude scale
    Lambda_Fr = 32.5##self.LBT[0]/ 2.64 #geometry scale, based on Froude
    return shipname, W_s, COG, COB, LBT, GM, LBT_WL, LPP, Lambda_Fr

def Calm_water_X0():
    ## Calm water resistance regression Coefficients in the form of
    ## X0= c0 + c1*Fn + c2*Fn**2 + c3*Fn**3 into matrix [c0,c1,c2,c3]
    X0_Cs = [0.,0.,0.,0.] # if all zeros, use HoltropMennen
    return X0_Cs

class HM_rudd():
    def __init__(self):
        ## rudder 1 & 2
        self.name = 'HM rudder'
        self.h = 9.6##m, rudder height or span
        self.area = 57.6 ##m^2
        self.AR = (self.h**2)/self.area
        self.COP1 = np.array([-117.24,0.,Alframax_ship()[2][2]-4.]) # rudder's Center of Pressure from CoG
        self.COP2 = np.array([0.,0.,0.])# rudder's Center of Pressure from CoG, zeros if only 1 prop propvided
        self.aH =  54  ## rudder force coefficient, non-dimensional
        self.aH_K = -42 # if different than SideForce aH, put 'None' if unused
        self.xH_prime =  -0.464 ##   moment coefficient xH=xH_prime*LPP
        self.kappa = -.5 ## kx/epsilon adjusting Fujiwara, straightening coefficient, non-dimensional
        self.C_1mintR = 19.5  ## None if using Fujiwara, non-dimensional
        self.epsilon = 1.09 ## None if using Fujiwara, non-dimensional

class HM_prop():
    def __init__(self,inUs,inn,inp):
        self.name = 'HM propeller'
        self.D = 7.3 ##m, diameter prop 1 & 2
        self.COP1 = np.array([-113.5 ,0.,Alframax_ship()[2][2]-4.]) # Prop's Center of Pressure from COG
        self.COP2 = np.array([0. ,0. , 0.]) # Prop's Center of Pressure from COG, zeros if only 1 prop propvided
        self.n = inn ##rps, fixed shaft speed rotation at working, MCR=90.7
        self.p = inp ##m, propeller pitch for P/D=0.7

        ## For Wageningen B4
        self.Z = 4 #num of blade --->change here
        self.AE = 40 # AE/A0 is expanded ratio. AE entrance diameter
        self.A0 = 100 # outlet diameter, same unit with AE

        ## Others
        self.Tp = None ## thrust deduction factor: None if unknown
        self.Wp0 = None ## wake fraction: None if unknown

        ## ship initial speed if fixed
        self.Us = inUs # m/s ## 11./1.94384

class HM_sail():
    def __init__(self):
        ## sail wing 1 data using Fujiwara's
        self.name = 'Fujiwara sail'
        self.area = 553 ##sail surface planform in m**2, transverse or general area
        self.area_longt = 1997 ## #sail surface planform in m**2, longitudinal
        self.chord = None #m ##----->change here
        self.COP = np.array([0,0.0,0.]) # wing's Center of Pressure from ship's pivot ##----->change here
