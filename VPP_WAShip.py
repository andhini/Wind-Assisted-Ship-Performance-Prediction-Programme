##  Wind-Assisted Ship (W-A-Ship) Performance Pediction Programme ##
## ----Routines of the 6-DOF (currently set to 4-DOF)-------------##
## ----of equations for the balance of forces and moments---------##
## --Conventions: x(+):forward, y(+):starboard, z(+):down---------##
## --Rotation: clockwise (+) away from orgin (Right-hand rules)---##
## --For steady condition, no acceleration------------------------##
import os,sys,signal
import csv,re,random
import pandas as pd
import numpy as np
from numpy import linalg as LA
from math import tan,sin,cos,atan,atan2,acos,asin,pi,sqrt,log10,degrees,radians
from math import pow,exp,log,remainder,tau
from scipy.optimize import minimize,newton,basinhopping,approx_fprime,fsolve,root
from scipy.interpolate import CubicSpline,interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from mpl_toolkits import mplot3d
import warnings
from Example_power_fuel import *
from Aframax import *

def stationary_VPP(aerofile,aero_data,algorithm_option,variable_option,
    vesseltype,vesselsail,vesselprop,vesselrudder,vesselresistance,
    indata=False,twin_screw=False,prop_data=False,TWAs=0,TWSs=0,
    singleTWA=0.,singleTWS=0.,init_beta=0.15,init_phi=0.11,init_delta = 10.111,
    savegraph=True,printcsv=True,outpolar=False,par=False):

    filepath= os.path.abspath(__file__)
    maindir=os.path.dirname(filepath)+'/'
    figSize = (10,15)

    ## -------general inputs to call from outside or to fill start here-------##
    ## read WingSail aerodynamic data, currently aerodata= 'G' or 'Fujiwara, then see aerofile in the directory.
    WS = Read_aero(aero_data,filename=aerofile,mirror=True)
    # WS.Plot_wingsail(polar=True,save=False) ## uncomment to plot wingsail data
    # print("[q,Cx,Cy,CMx,CMz]:", WS.Coefficients(AWAi=radians(-90))) ## uncomment to check values

    # prop_data = True if Propeller curve is provided, False if Wageningen in use
    global_properties = vesseltype,vesselsail,vesselprop,vesselrudder,vesselresistance
    algorithm_option = algorithm_option ## options for solving: "checking","single_solve","multi_solve". Putting "" will turn off
    variable_option = variable_option ## options for first variable in initials: "1" ship speed, "2" prop's rps, "3" prop's pitch
    if ((algorithm_option=="checking") or (algorithm_option=="single_solve")):
        True_wind = angle_speed(radians(singleTWA),singleTWS) ## input:true wind angle & speed as vector towards origin cwise(+)
        myShip = angle_speed(radians(init_beta), vesselprop.Us) ## ship direction & motion speed as vector from origin cwise(+)
        Ship_speed=myShip.speed
        Leeway_angle = myShip.angle
        Heel_angle = radians(init_phi) ##rads
        Rudder_angle = radians(init_delta)
        Propeller_speed = vesselprop.n ##rps, propeller shaft rotation
        Propeller_pitch = vesselprop.p ##m, propeller pitch
    ## ----------------------------------------------------------------------##

    ##  Algorithm for single case without solving
    if (algorithm_option=="checking"):
        print(" scipy solver is not in use ")
        if (variable_option==1):
            fixed_variables = Propeller_speed,Propeller_pitch
            initials = [Ship_speed,Leeway_angle,Heel_angle,Rudder_angle]
        elif (variable_option==2):
            fixed_variables = Ship_speed,Propeller_pitch
            initials = [Propeller_speed,Leeway_angle,Heel_angle,Rudder_angle]
        elif (variable_option==3):
            fixed_variables = Ship_speed,Propeller_speed
            initials = [Propeller_pitch,Leeway_angle,Heel_angle,Rudder_angle]
        else :
            print("Fill variable_option : 1, 2, or 3")
            exit_program()
        properties = global_properties,fixed_variables
        total_FM=Force_moment(variable_option,properties,True_wind,WS,initials,twin_screw=twin_screw,printval=True,PropType=prop_data)
        print('Total forces & moments initial guess, kN or kNm', total_FM[0]*1E-3)

    ##  Algorithm for single case
    elif(algorithm_option=="single_solve"):
        if (variable_option==1):
            fixed_variables = Propeller_speed,Propeller_pitch
            initials = [Ship_speed,Leeway_angle,Heel_angle,Rudder_angle]
        elif (variable_option==2):
            fixed_variables = Ship_speed,Propeller_pitch
            initials = [Propeller_speed,Leeway_angle,Heel_angle,Rudder_angle]
        elif (variable_option==3):
            fixed_variables = Ship_speed,Propeller_speed
            initials = [Propeller_pitch,Leeway_angle,Heel_angle,Rudder_angle]
        else :
            print("Fill variable_option : 1, 2, or 3")
            exit_program()
        properties = global_properties,fixed_variables

        # solving multi-variate equations with root-finding algo (N-X in & N-Y out, N = parameter numbers):
        func = lambda x : Force_moment(variable_option,properties,True_wind,WS,x,twin_screw=twin_screw,PropType=prop_data)[2]
        froot = root(func, x0=initials, method='krylov') #  best method in order: krylov,anderson,diagbroyden
        if(all(abs(i) >= 1 for i in func(froot.x))==True):print('not converging')
        if(par==False):
            print('x-force after root-finding algo:',froot.x,'F&M:',Force_moment(variable_option,\
                properties,True_wind,WS,froot.x,twin_screw=twin_screw,printval=True,PropType=prop_data)[2])
        if(par): return Force_moment(variable_option,\
            properties,True_wind,WS,froot.x,twin_screw=twin_screw,printval=False,PropType=prop_data)[5]
    ## Algorithm for multiple solutions of wind conditions
    elif(algorithm_option=="multi_solve"):
        if (outpolar==True):
            fig, ((ax,ax1),(ax2,ax3),(ax4,ax5)) = plt.subplots(3,2,constrained_layout=True,figsize=figSize,subplot_kw={'projection': 'polar'})
        else:
            fig, ((ax,ax1),(ax2,ax3),(ax4,ax5)) = plt.subplots(3,2,constrained_layout=True,figsize=figSize)#,subplot_kw={'projection': '3d'})#,subplot_kw={'projection': 'polar'})
        if (indata==False):
            wind_dir = np.linspace(-180,88,25) ## deg 190,358,100 for fujiwara
            wind_speed = np.array([5,11,18,25])/1.9438452 ##  m/s (kts in bracket) np.linspace(5.,25.,4)
        else :
            wind_dir = TWAs
            wind_speed = TWSs
        colors =cm.Blues(np.linspace(0.4, 1.,num=int(6)))
        data_arr = np.empty((len(wind_dir), len(wind_speed),6), float) # data buffer

        ## 4-pair different inital conditions to try if not converging, calling init[][]
        ## [0]:Ship_speed or rps or pitch ,  [1]:Leeway_angle,  [2]:Heel_angle,  [3]:Rudder_angle
        if (variable_option==1):
            intx = [i/1.94384 for i in [6.13,14.27]] ## m/s, brackets = kts
        elif (variable_option==2):
            intx = [i/60 for i in [80,142]] ## rps, brackets = rpm
        elif (variable_option==3):
            intx = [i*vesselprop.D for i in [0.39,1.12]] ## m, brackets = P/D
        else :
            print("Fill variable_option : 1, 2, or 3")
            exit_program()

        n_opt=8  # numbers of iteration option
        init = [sorted(np.random.uniform(intx[0],intx[1],n_opt),reverse=False),\
                [radians(i) for i in sorted(np.random.uniform(-6,6,n_opt),reverse=False)],\
                [radians(i) for i in sorted(np.random.uniform(-3,3,n_opt),reverse=True)],\
                [radians(i) for i in sorted(np.random.uniform(-35,35,n_opt),reverse=True)]]

        for i in range(len(wind_speed)):
            for j in range(len(wind_dir)):
                ## loop for initial conditions
                converge=True
                for n in range(len(init[0])):
                    ## given parameters
                    True_wind = angle_speed(radians(wind_dir[j]),wind_speed[i]) ## true wind angle & speed as vector to origin cwise(+)
                    Heel_angle = init[2][n] ##rads
                    Rudder_angle = init[3][n] ##rads

                    # initial assumptions for optimised parameters
                    if (variable_option==1):
                        myShip=angle_speed(init[1][n],init[0][n]) ## ship direction & motion speed as vector from origin ccwise(-)
                        Ship_speed=myShip.speed
                        Leeway_angle = myShip.angle
                        Propeller_speed = vesselprop.n ##rps, propeller shaft rotation
                        Propeller_pitch = vesselprop.p ##m, propeller pitch
                        fixed_variables = Propeller_speed,Propeller_pitch
                        initials = [Ship_speed,Leeway_angle,Heel_angle,Rudder_angle]
                    elif (variable_option==2):
                        myShip=angle_speed(init[1][n],vesselprop.Us) ## ship direction & motion speed as vector from origin ccwise(-)
                        Ship_speed=myShip.speed
                        Leeway_angle = myShip.angle
                        Propeller_speed = init[0][n] ##rps, propeller shaft rotation
                        Propeller_pitch = vesselprop.p ##m, propeller pitch
                        fixed_variables = Ship_speed,Propeller_pitch
                        initials = [Propeller_speed,Leeway_angle,Heel_angle,Rudder_angle]
                    elif (variable_option==3):
                        myShip=angle_speed(init[1][n],vesselprop.Us) ## ship direction & motion speed as vector from origin ccwise(-)
                        Ship_speed=myShip.speed
                        Leeway_angle = myShip.angle
                        Propeller_speed = vesselprop.n ##rps, propeller shaft rotation
                        Propeller_pitch = init[0][n] ##m, propeller pitch
                        fixed_variables = Ship_speed,Propeller_speed
                        initials = [Propeller_pitch,Leeway_angle,Heel_angle,Rudder_angle]
                    else :
                        print("Fill variable_option : 1, 2, or 3")
                        exit_program()
                    properties = global_properties,fixed_variables

                    # solving multi-variate equations with root-finding algo (N-X in & N-Y out, N = parameter numbers)
                    func = lambda x : Force_moment(variable_option,properties,True_wind,WS,x,twin_screw=twin_screw,PropType=prop_data)[2]

                    try:
                        ## Using first method of multi dimensions-multi variable algorithm
                        froot = root(func, x0=initials, method='krylov') #  other methods: krylov,anderson,diagbroyden
                        #print('x-force after root-finding algo:',froot.x,'F&M:',func(froot.x))
                        # if(all(abs(i) >= 1 for i in func(froot.x))==True):print('not converging')
                        outdata=list(Force_moment(variable_option,properties,True_wind,WS,froot.x,twin_screw=twin_screw,PropType=prop_data)[3:5])

                        ## if any force & moment above 1e-1: the solver doesn't converge, store zeros in data_arr & re-iterate!
                        if(all(abs(i) >= 1e-1 for i in func(froot.x))==True):
                            data_arr[j,i,:]=np.empty((6,1,)).fill(np.nan)
                            print('not converging F&M > 0.1: '+str(n+1)+' wind_speed='+str(wind_speed[i]*1.9438452)+' dir='+str(wind_dir[j]))
                            converge=False

                        ## if rudder angle (most sensitive)>60 deg: incorrect solution, store zeros in data_arr & re-iterate!
                        elif(abs(degrees(froot.x[3])) > 60):
                            data_arr[j,i,:]=np.empty((6,1,)).fill(np.nan)
                            print('incorrect rudder angle: '+str(n+1)+' wind_speed='+str(wind_speed[i]*1.9438452)+' dir='+str(wind_dir[j]))
                            converge=False

                        ## if first variable falls to negative: incorrect solution, store zeros in data_arr & re-iterate!
                        elif(froot.x[0] < 0):
                            data_arr[j,i,:]=np.empty((6,1,)).fill(np.nan)
                            print('incorrect first variable: '+str(n+1)+' wind_speed='+str(wind_speed[i]*1.9438452)+' dir='+str(wind_dir[j]))
                            converge=False

                        ## if converging, postprocess!
                        else:
                            if(variable_option==1): froot.x[0]=froot.x[0]*1.94384 # m/s to kts
                            if(variable_option==2): froot.x[0]=froot.x[0]*60 # rps to rpm
                            if(variable_option==3): froot.x[0] /= vesselprop.D# p to P/D
                            froot.x[1] *= (180/pi); froot.x[2] *= (180/pi); froot.x[3] *= (180/pi) ## in degrees
                            ## fill 3rd axes of data_arr with: Ship_speed (kts),leeway,phi_h,delta_r,out1,out2
                            data_arr[j,i,:]=np.asarray([*froot.x[:],*outdata[:]])
                            converge=True

                    ## if any ValueError is returned causing Scipy stops (e.g. sqrt(negative)), re-iterate!
                    except ValueError:
                        data_arr[j,i,:]=np.empty((6,1,)).fill(np.nan)
                        print('ValueError: '+str(n+1)+' wind_speed='+str(wind_speed[i]*1.9438452)+' dir='+str(wind_dir[j]))
                        converge=False

                    if (converge==True): break ## break from the init loop

                ## if error or non-convergence persists, continue to the next j-iteration
                continue

        ## Graph elements
        Ztitles =['Ship speed (knot)','Leeway_angle (deg)','Heel_angle (deg)','Rudder_angle (deg)','BHP (kW)','Consumption(kg/hr)']
        ntitles =['Ship_speed','Leeway_angle','Heel_angle','Rudder_angle','BHP','Consumption']
        if(variable_option==2):
            Ztitles[0]= 'Propeller speed (RPM)';ntitles[0]='Propeller_speed'
        if(variable_option==3):
            Ztitles[0]= 'Propeller P/D';ntitles[0]='Propeller_PD'

        if(printcsv==True):
            for i in range(len(ntitles)):
                # if(i<4):continue ## only write BHP and consumption data
                df_out=pd.DataFrame(np.transpose(data_arr[:,:,i]),index=wind_speed,columns=wind_dir)
                # df_out = df_out.interpolate(method='spline', order=2) ## interpolate nan values
                df_out.to_csv(maindir+ntitles[i]+"_{:.0f}".format(vesselprop.Us*1.9438452)+\
                    "kts_"+aerofile.split('.')[0]+".csv")

        axs=[ax,ax1,ax2,ax3,ax4,ax5]
        # rangeYs =[[0.,0.75],[-8.,8.],[-.2,.2],[-22.,22.],[40,4000],[0,750]] #manual range
        ## Plot grap 2D
        for k in range(len(axs)):
            for i in range(len(wind_speed)):
                rangeY = [np.nanmin(data_arr[:,i,k]),np.nanmax(data_arr[:,i,k])]
                # rangeY = [rangeYs[k][0],rangeYs[k][1]] ## manual max & min ranges
                if (outpolar==False):
                    plotgraph(axs[k],wind_dir,data_arr[:,i,k],rangeY[0],rangeY[1],Ztitles[k],colors[i],markers='',\
                        linestyles='-',labels="{:.1f}".format(wind_speed[i]*1.9438452)+"kts",xtitle="TWA(degs),to origin");
                else:
                    plotpolar(axs[k],wind_dir/180*pi,data_arr[:,i,k],rangeY[0],rangeY[1],Ztitles[k],colors[i],markers='',\
                        linestyles='-',labels="{:.1f}".format(wind_speed[i]*1.9438452)+"kts",xtitle="TWA(degs),to origin");

        plt.legend(bbox_to_anchor=(1.,1), loc='upper left');

        # ## Plot 3D grap
        # # fig = plt.figure(figsize=plt.figaspect(0.3)) #3D plot
        # for k in range(6):
        #     for i in range(len(wind_speed)):
        #         plot3D(axs[k],wind_speed,wind_dir,data_arr[:,:,k],\
        #             xtitle="TWS(m/s)",ytitle ="TWA(degs),to origin",ztitle=Ztitles[k])

        if(savegraph==True):
            plt.savefig(maindir+"{:.0f}".format(vesselprop.Us*1.9438452)+\
            'kts_'+aerofile.split('.')[0]+'.png',dpi=300, transparent=True) #format='pdf'

        plt.show()
    else:
        print("Fill algorithm_option : 'checking', 'single_solve' or 'multi_solve'")

def plotpolar(ax,X,Y,rmin,rmax,Ytitle="",colors='b',labels=None,xtitle=None,ytickrange=5,linestyles='-',markers='o',thetamin=0,thetamax=360):
    ## Dont forget to activate "subplot_kw={'projection': 'polar'}" in main()
    ax.plot(X,Y,linestyle=linestyles,color=colors,label=labels,marker=markers)

    ## polar diagram properties
    ax.set_theta_direction(-1) # change direction to CCW
    ax.set_thetamin(thetamin);ax.set_thetamax(thetamax); # set the radial limits for visualization
    ax.set_theta_offset(.5*np.pi) # point the origin towards the top
    ax.set_rmin(rmin);ax.set_rmax(rmax);#for polar diagram
    ax.set_rticks(np.linspace(rmin, rmax, 4))  # set number of radial ticks:set_rticks()

    ticks_x = ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x/pi*180))
    ax.xaxis.set_major_formatter(ticks_x);
    if(xtitle !=None):ax.set_xlabel(xtitle);

    ax.set_ylabel(Ytitle)
    ax.grid(True)

def plotgraph(ax,X,Y,ymin,ymax,Ytitle="",colors='b',labels=None,xtitle=None,ytickrange=5,linestyles='-',markers='o',xmin=-180,xmax=180):
    ax.plot(X,Y ,linestyle=linestyles,color=colors,label=labels,marker=markers,mfc='none')

    ##  lateral/normal diagram properties
    ax.set_ylim((ymin,ymax)); # for lateral diagram
    ax.set_yticks(np.linspace(ymin, ymax, ytickrange))  # set number of ticks:set_yticks

    ax.set_xticks(np.linspace(xmin, xmax, 10)) # set number of x-ticks range
    # ticks_x = ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x/pi*180))
    # ax.xaxis.set_major_formatter(ticks_x);
    if(xtitle !=None):ax.set_xlabel(xtitle);

    ax.set_ylabel(Ytitle)
    ax.grid(True)

def plot3D(ax,Xin,Yin,Z,rmin=None,rmax=None,labels=None,xtitle=None,ytitle=None,ztitle=None):
    X, Y = np.meshgrid(Xin, Yin);
    surf = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, alpha=0.3, colors='k',rasterized=True)#, cmap=cm.Blues,
                           # linewidth=0, antialiased=False)
    if(rmin != None): ax.set_zlim(rmin, rmax)

    cset = ax.contourf(X, Y, Z, zdir='z', offset=np.nanmin(Z), cmap=cm.coolwarm)
    cset = ax.contourf(X, Y, Z, zdir='x', offset=Xin[0]-1, cmap=cm.coolwarm)
    cset = ax.contourf(X, Y, Z, zdir='y', offset=Yin[-1]+1, cmap=cm.coolwarm)

    ax.set_xlabel(xtitle, rotation=150)
    ax.set_ylabel(ytitle, rotation=0)
    ax.set_zlabel(ztitle, rotation=60)

def Force_moment(variable_option,properties,True_wind,WS,initial_inputs,twin_screw=True,printval=False,PropType=False):
    if (variable_option==2):
        ## solving Propeller_speed
        Propeller_speed,leeway,phi_h,delta_r = initial_inputs
        [vess,sail,prop,rudd,hull],[Ship_speed,Propeller_pitch]=properties
        if(printval==True): print('Prop speed(RPM):',Propeller_speed*60,'\n leeway(deg):',degrees(leeway),'\n heel(deg):',degrees(phi_h),'\n rudder(deg):',degrees(delta_r),'\n')
    elif (variable_option==3):
        ## solving Propeller_pitch
        Propeller_pitch,leeway,phi_h,delta_r = initial_inputs
        [vess,sail,prop,rudd,hull],[Ship_speed,Propeller_speed]=properties
        if(printval==True): print('P/D:',Propeller_pitch/prop.D,'\n leeway(deg):',degrees(leeway),'\n heel(deg):',degrees(phi_h),'\n rudder(deg):',degrees(delta_r),'\n')
    else:
        ## variable_option==1, solving Ship_speed
        Ship_speed,leeway,phi_h,delta_r = initial_inputs
        [vess,sail,prop,rudd,hull],[Propeller_speed,Propeller_pitch]=properties
        if(printval==True): print('Us(kts):',Ship_speed*1.94384,'\n leeway(deg):',degrees(leeway),'\n heel(deg):',degrees(phi_h),'\n rudder(deg):',degrees(delta_r),'\n')

    ## Convention from Fujiwara:
    ## Ship_v vec pointing away +cw, True_wind_v vec pointing into +ccw, in radians, Apparent_wind_v vec = True_wind_v+Ship_v pointing into +ccw
    # Apparent_wind_angle = atan2((True_wind.speed*sin(True_wind.angle)-Ship_speed*sin(leeway)),\
    #     (True_wind.speed*cos(True_wind.angle)+Ship_speed*cos(leeway)))
    # Apparent_wind_speed = sqrt(pow(True_wind.speed,2)+pow(Ship_speed,2)+ \
    #     2*True_wind.speed*Ship_speed*cos(True_wind.angle+leeway))
    # # Apparent_wind_angle %= 2*pi ##change into 0-2*pi range
    # # Apparent_wind_angle = -Apparent_wind_angle ## Fujiwara change convention to positive cc
    # Apparent_wind = angle_speed(Apparent_wind_angle,Apparent_wind_speed)
    # if(printval==True): print('apparent wind angle & speed  in deg & m/s:',degrees(Apparent_wind.angle),Apparent_wind.speed,\
    #     Apparent_wind.speed*1.94384,"(kts)")

    ## Another method for checking AWA & AWS, same convention as above
    Ship_v = np.array([[Ship_speed*cos(leeway)],[-Ship_speed*sin(leeway)],[0]])
    True_wind_v = np.array([[True_wind.speed*cos(True_wind.angle)],[True_wind.speed*sin(True_wind.angle)],[0]])
    Apparent_wind_v = True_wind_v+Ship_v
    Apparent_wind = angle_speed(atan2(Apparent_wind_v[1],Apparent_wind_v[0]),LA.norm(Apparent_wind_v))
    if(printval==True): print('2st method \n AWA(deg):',degrees(Apparent_wind.angle),'\n AWS(m/s):',Apparent_wind.speed,"\n AWS(kts)",\
         Apparent_wind.speed*1.94384,"\n")

    ## Pick ship type and store in parameter called 'ship':
    shipname, W_s, COG, COB, LBT, GM, LBT_WL, LPP, Lambda_Fr = vess
    ship = vessel(shipname, W_s, COG, COB, LBT, GM, LBT_WL, LPP, Lambda_Fr)
    if(printval==True):print('Currently using vessel: ',ship.shipname,'\n twin screw:',twin_screw,'\n')

    ## aerodynamic of 1st sail
    ## --- turn off Fz & My manually before proceeding for 4DOF!
    q_sail,Cx_sail,Cy_sail,CMx_sail,CMz_sail = WS.Coefficients(Apparent_wind.angle)

    F_M_wingsail1 = Wingsail(ship,sail,Apparent_wind.angle,Apparent_wind.speed,Cx=Cx_sail,\
        Cy=Cy_sail,CMx=CMx_sail,CMz=CMz_sail)
    if(printval==True):
        print('Currently using Wingsail: ',sail.name,'\n')
        print('Wingsail, Force(N) & Moment(Nm):\n',F_M_wingsail1,'\n')

    ## hull resistance, currently using Holtrop Mennen, no effect of beta
    # F_M_hull = Hull_HoltropMennen(ship,leeway,Ship_speed)
    # if(printval==True):print('Hull Holtrop Mennen, Force(N) & Moment(Nm):\n',F_M_hull)

    ## hull resistance, currently using Hollenbach method, no effect of beta
    # F_M_hull = Hull_Hollenbach(ship,prop,leeway,Ship_speed,twin_screw=False)
    # if(printval==True):print('hull Hollenbach,Force(N) & Moment(Nm):\n',F_M_hull)

    ## side force based on Prandtl's Lifting line theory
    # SF = SF_LiftingLine(ship,leeway,Ship_speed)
    # if(printval==True): print('Sideforce,Force(N) & Moment(Nm):\n',SF)

    ## hull resistance, currently using Fujiwara's method
    F_M_hull = Hull_Fujiwara(ship,hull,leeway,Ship_speed,phi_h,printval)
    if(printval==True):print('Hull Fujiwara, Force(N) & Moment(Nm):\n',F_M_hull[0],"\n")

    ## hydrodynamics of propeller
    F_M_propellers = Propeller(ship,prop,leeway,Ship_speed,-F_M_hull[0][0],\
        ni = Propeller_speed,p_in=Propeller_pitch,twin_screw=twin_screw,PropType=PropType)
    if(printval==True):
        print('Wageningen-B series prop :',not(PropType))
        print('Propeller,Force(N) & Moment(Nm):\n',F_M_propellers[0])
        print("eta_0:",F_M_propellers[1],"\n KQ: ",F_M_propellers[4],"\n KT:",F_M_propellers[5],\
            "\n J:",F_M_propellers[6],"\n Wp:",F_M_propellers[7],"\n Tp:",F_M_propellers[8],"\n")

    ## hydrodynamics of rudders
    F_M_rudders = Rudder(ship,rudd,prop,leeway,Ship_speed,delta_r,ni=Propeller_speed,\
        p_in=Propeller_pitch,twin_screw=twin_screw,PropType=PropType)
    if(printval==True): print('Rudder, Force(N) & Moment(Nm):\n',F_M_rudders[0])

    ## upright stability moment
    up_hull = np.array([0,0,0,-ship.GM*phys().rho_w*phys().ga*ship.nabla*sin(phi_h),0,0])
    M_hull =  np.array([up_hull]).T
    if(printval==True): print('HullMoment, Force(N) & Moment(Nm):\n',M_hull)

    final_array = F_M_wingsail1+F_M_hull[0]+F_M_propellers[0]+F_M_rudders[0]+M_hull#+SF
    final_array[4]=0 ## no zero-balance for pitch

    ## Fuel
    currentBHP = F_M_propellers[3] ## currently using BHP [3]Fujiwara or [2]G plot
    consumption = power_fuel(currentBHP)
    if(printval==True): print('BHP(kW):',currentBHP/1000,'\n Load(%):',consumption[1],'\n fuel(g/hr):',consumption[0],'\n SFOC:',consumption[2] )

    ## errors & flattened final array (in 4-DOF)
    RMSLE = np.square(np.log10(np.abs(final_array+np.ones_like(final_array)))-np.log10(np.ones_like(final_array)))
    RMSLE = sqrt(np.sum(RMSLE)/6)
    flat4DOF = final_array.flatten()
    flat4DOF = [flat4DOF[0],flat4DOF[1],flat4DOF[3],flat4DOF[5]]
    out1 = currentBHP/1000 ##sqrt(F_M_wingsail1[0]**2+F_M_wingsail1[1]**2)*cos(Apparent_wind.angle)*(Apparent_wind.speed)##F_M_propellers[2] ##flat4DOF[0] ##
    out2 = consumption[0]/1000 #sqrt(F_M_propellers[0][0]**2+F_M_propellers[0][1]**2)*cos(leeway)*(Ship_speed)##flat4DOF[1]##
    # out3 = [Cx_sail,Cy_sail,CMx_sail,CMz_sail]

    qsF = 0.5*phys().rho_w*(Ship_speed**2)*ship.LPP*ship.LBT_WL[2] #force scale
    qsK = 0.5*phys().rho_w*(Ship_speed**2)*ship.LPP*(ship.LBT_WL[2]**2) #Roll scale
    qsN = 0.5*phys().rho_w*(Ship_speed**2)*(ship.LPP**2)*ship.LBT_WL[2] #Yaw scale
    ## 4-DoF scaled hull Aerodynamics
    sail_sc = [F_M_wingsail1[0]/qsF,F_M_wingsail1[1]/qsF,F_M_wingsail1[3]/qsK,F_M_wingsail1[5]/qsN]
    sail_sc = np.concatenate(sail_sc, axis=0).tolist()
    hull_sc = [F_M_hull[0][0]/qsF,F_M_hull[0][1]/qsF,F_M_hull[0][3]/qsK,F_M_hull[0][5]/qsN]
    hull_sc = np.concatenate(hull_sc, axis=0).tolist()
    prop_sc = [F_M_propellers[0][0]/qsF]
    prop_sc = np.concatenate(prop_sc, axis=0).tolist()
    rudd_sc = [F_M_rudders[0][0]/qsF,F_M_rudders[0][1]/qsF,F_M_rudders[0][3]/qsK,F_M_rudders[0][5]/qsN]
    rudd_sc = np.concatenate(rudd_sc, axis=0).tolist()

    return final_array,RMSLE,flat4DOF,out1,out2,[out1]+sail_sc+hull_sc+prop_sc+rudd_sc
    # return final_array,RMSLE,flat4DOF,[out1]+out3,out2

def Hull_HoltropMennen(ship,beta,Us):
    Re = Us*ship.LBT_WL[0]/phys().nu_w
    Cf = 0.075/((log10(Re)-2)**2)
    q =  0.5*phys().rho_w*Us**2 # dynamic pressure
    S_s_HM = ship.LBT_WL[0]*(2*ship.LBT_WL[2]+ship.LBT_WL[1])*sqrt(ship.C_M)* \
        (0.453+0.4425*ship.CB-0.2862*ship.C_M-0.003467* \
        ship.LBT_WL[1]/ship.LBT_WL[2]+0.3696*ship.C_WP)+ \
        2.38*ship.A_BT/ship.CB  ## wetted surface calculation based on Holtrop-Mennen instead of using S_s

    ## Friction resistance
    R_F = Cf*q*S_s_HM #N, friction force or resistance

    ## form factor calculation
    lcb = ship.COB[0]/(ship.LBT_WL[0])*100 #in percent, move the reference to 1/2LWL then divided by L_WL
    L_R = ship.LBT_WL[0]*(1-ship.C_PL+0.06*ship.C_PL*lcb/(4*ship.C_PL-1))
    k1=-1+0.93+0.487118*(1+0.011*ship.C_stern)*(ship.LBT_WL[1]/ship.LBT_WL[0])**1.06806*\
        (ship.LBT_WL[2]/ship.LBT_WL[0])**0.46106*(ship.LBT_WL[0]/L_R)**0.121563*\
        ((ship.LBT_WL[0]**3)/ship.nabla)**0.36486*(1-ship.C_PL)**(-0.604247)

    ## Appendage resistance
    R_App =0.5*phys().rho_w*Us**2*ship.S_App*(1+ship.k2)*Cf

    ## wave resistance
    B_of_L = ship.LBT_WL[1]/ship.LBT_WL[0]
    if(B_of_L < 0.11):
        c7 = 0.229577*B_of_L**0.33333
    elif((B_of_L > 0.11) and (B_of_L < 0.25)):
        c7 = B_of_L
    else:
        c7 = 0.5-0.0625*(1/B_of_L)

    iE = 1+89*exp(-1*(1/B_of_L)**0.80856*(1-ship.C_WP)**0.30484* \
        (1-ship.C_PL-0.0225*lcb)**0.6367*(L_R/ship.LBT_WL[1])**0.34574* \
        (100*ship.nabla/ship.LBT_WL[0]**3)**0.16302)
    # print('lcb',lcb,'problematic value:',1-ship.C_PL-0.0225*lcb, ' C_PL:',ship.C_PL)
    c1 = 2223105*c7**3.78613*(ship.LBT_WL[2]/ship.LBT_WL[1])**1.07961* \
        (90-iE)**(-1.37565)
    c3 = 0.56*ship.A_BT**1.5/(ship.LBT_WL[1]*ship.LBT_WL[2]* \
        (0.31*sqrt(ship.A_BT)+ship.T_F-ship.h_B))
    c2 = exp(-1.89*sqrt(c3))
    c5 = 1-0.8*ship.A_T/(ship.LBT_WL[1]*ship.LBT_WL[2]*ship.C_M)

    if(ship.C_PL<0.8):
        c16 = 8.07981*ship.C_PL-13.8673*ship.C_PL**2+6.984388*ship.C_PL**3
    else :
        c16 = 1.73014-0.7067*ship.C_PL
    m1 = 0.0140407* ship.LBT_WL[0]/ship.LBT_WL[2]-1.75254*ship.nabla**(1/3)/ \
        ship.LBT_WL[0]-4.79323*ship.LBT_WL[1]/ship.LBT_WL[0]-c16

    Fn = Us/sqrt(phys().ga*ship.LBT_WL[0])#Froude number, based on waterline length

    if((ship.LBT_WL[0]/ship.LBT_WL[1])<12):
        lambda1 = 1.446*ship.C_PL-0.03*(ship.LBT_WL[0]/ship.LBT_WL[1])
    else:
        lambda1 =  1.446*ship.C_PL-0.36

    if((ship.LBT_WL[0]**3/ship.nabla)<512):
        c15 = -1.69385
    elif((ship.LBT_WL[0]**3/ship.nabla)>1727.91):
        c15 = 0.
    else :
        c15 = -1.69385+((ship.LBT_WL[0]**3/ship.nabla)**(1/3)-8)/2.36

    m2 = c15*ship.C_PL**2*exp(-0.1*Fn**(-2.))
    R_W = c1*c2*c5*ship.nabla*phys().rho_w*phys().ga* \
            exp(m1*Fn**ship.d_tunnel+m2*cos(lambda1*Fn**(-2)))
    # print('c1',c1,'c2',c2,'c5',c5,'m1',m1,'m2',m2,'dtunnel',ship.d_tunnel,'lambda1',lambda1,'Fn',Fn,'c7',c7,'iE',iE)
    ## bulbous bow resistance
    P_B = 0.56*sqrt(ship.A_BT)/(ship.T_F-1.5*ship.h_B)
    F_ni = Us/sqrt(phys().ga*(ship.T_F-ship.h_B-0.25*sqrt(ship.A_BT))+0.15*Us**2)
    R_B = 0.11*exp(-3*P_B**(-2))*F_ni**3*ship.A_BT**1.5*phys().rho_w*phys().ga/ \
        (1+F_ni**2)

    ## immersed transom resistance
    F_nT = Us/sqrt(2.*phys().ga*ship.A_T/(ship.LBT_WL[1]+ship.LBT_WL[1]*ship.C_WP))
    if(F_nT<5):
        c6 = 0.2*(1-0.2*F_nT)
    else:
        c6 = 0
    R_Tr = 0.5*phys().rho_w*Us**2*ship.A_T*c6

    ## model-ship correlation resistance
    if((ship.T_F/ship.LBT_WL[0]) <= 0.04):
        c4 = ship.T_F/ship.LBT_WL[0]
    else:
        c4 = 0.04
    C_A = 0.006*(ship.LBT_WL[0]+100)**(-0.16)-0.00205+0.003* \
        sqrt(ship.LBT_WL[0]/7.5)*ship.CB**4*c2*(0.04-c4)
    R_A = 0.5*phys().rho_w*Us**2*S_s_HM*C_A

    R0 = -1*(R_F*(1+k1)+R_App+R_W+R_B+R_Tr+R_A)
    R_total = np.array([R0,0.,0.,0.,0.,0.],dtype=float)
    F_M = np.array([R_total]).T
    return F_M,k1,Cf,C_A,F_M[0][0]*Us,-R0

def Hull_Hollenbach(ship,prop_properties,beta,Us,twin_screw=False):
    prop = prop_properties
    Re = Us*ship.LBT_WL[0]/phys().nu_w
    Cf = 0.075/((log10(Re)-2)**2)
    q =  0.5*phys().rho_w*Us**2 # dynamic pressure

    ## all coefficients based on mean residuary resistance for design draft condition,
    ## with bulb
    if(twin_screw==False):
        s0=-0.6837 ; s1=0.2771 ; s2=0.6542 ; s3=0.6422 ; s4=0.0075 ;
        s5=0.0275 ; s6=-0.0045 ; s7=-0.4798 ; s8=0.0376 ;
        k_rudd = 0 ; k_bracket = 0 ; k_bossing = 0;
        b11=-0.57424; b12=13.3893; b13=90.596;
        b21=4.6614 ; b22=-39.721 ; b23=-351.483;
        b31=-1.14215 ; b32=-12.3296 ; b33=459.254;
        d1=0.854; d2=-1.228 ; d3=0.497 ;
        e1=2.1701 ; e2=-0.1602 ;
        a1=0.3382 ; a2=-0.8086 ; a3=-6.0258 ; a4=-3.5632 ; a5=9.4405 ; a6=0.0146 ;
        a7=0. ; a8=0. ; a9=0. ; a10=0. ;
        N_rudd = 1 ; N_bracket = 1 ; N_bossing = 1;
    else:
        s0=-0.4319 ; s1=0.1685 ; s2=0.5637 ; s3=0.5891 ; s4=0.0033 ;
        s5=0.0134 ; s6=-0.0005 ; s7=-2.7932 ; s8=0.0072 ;
        k_rudd =0.0131 ; k_bracket =-0.003 ; k_bossing =0.0061;
        b11=-5.3475 ; b12=55.6532 ; b13=-114.905;
        b21=19.2714 ; b22=-192.388 ; b23=388.333;
        b31=-14.3571 ; b32=142.738 ; b33=-254.762;
        d1=0.897 ; d2=-1.457 ; d3=0.767 ;
        e1=1.8319 ; e2=-0.1237 ;
        a1=0.2748 ; a2=-0.5747 ; a3=-6.761 ; a4=-4.3834 ; a5=8.8158 ; a6=-0.1418 ;
        a7=-0.1258 ; a8=0.0481 ; a9=0.1699 ; a10=0.0728 ;
        N_rudd = 2 ; N_bracket = 2 ; N_bossing = 2;

    L_OS = ship.L_OS ## length over wetted surface, in case different than L_WL ----> change here
    T_A = ship.T_F # draft at forward, in case different than T_F --->change here
    k =  s0+s1*(L_OS/ship.LBT_WL[0])+s2*(ship.LBT_WL[0]/ship.LPP)+s3*ship.CB+\
        s4*(ship.LPP/ship.LBT_WL[1])+s5*(ship.LBT_WL[1]/ship.LBT_WL[2])+\
        s6*(ship.LPP/ship.LBT_WL[2])+s7*(T_A-ship.T_F)/ship.LPP+\
        s8*(prop.D/ship.LBT_WL[2])+k_rudd*N_rudd+k_bracket*N_bracket+\
        k_bossing*N_bossing ##shape factor based on Hollenbach
    S_Hl = k*ship.LPP*(ship.LBT_WL[1]+2*ship.LBT_WL[2]) ## wetted surface calculation based on Hollenbach instead of using S_s

    ## residuary component
    Fn = Us/sqrt(phys().ga*ship.LBT_WL[0])#Froude number, based on waterline length
    C_R_Std = b11+b12*Fn+b13*Fn**2+(b21+b22*Fn+b23*Fn**2)*ship.CB+\
        (b31+b32*Fn+b33*Fn**2)*ship.CB**2
    Fr_crit = d1+d2*ship.CB+d3*ship.CB**2
    c1 =Fn/Fr_crit # twist screw= single screw
    if(Fn<Fr_crit):
        k_Fr = 1
    else:
        k_Fr = (Fn/Fr_crit)**c1
    k_L = e1*(ship.LPP)**e2
    if((ship.LBT_WL[1]/ship.LBT_WL[2])<1.99):
        k_BT = 1.99**a1
    else:
        k_BT = (ship.LBT_WL[1]/ship.LBT_WL[2])**a1
    if((ship.LPP/ship.LBT_WL[1])>7.11):
        k_LB = 7.11**a2
    else:
        k_LB = (ship.LPP/ship.LBT_WL[1])**a2

    if((L_OS/ship.LBT_WL[0])>1.05):
        k_LL = 1.05**a3
    else:
        k_LL = (L_OS/ship.LBT_WL[0])**a3
    if((ship.LBT_WL[0]/ship.LPP)>1.06):
        k_AO = 1.06**a4
    else:
        k_AO = (ship.LBT_WL[0]/ship.LPP)**a4
    k_Tr = (1+(T_A-ship.T_F)/ship.LPP)**a5
    if((prop.D/T_A)<0.43):
        k_Pr = 0.43**a6
    elif((prop.D/T_A)>0.84):
        k_Pr = 0.84**a6
    else:
        k_Pr = (prop.D/T_A)**a6

    N_thruster= 3 # number of side thrusters
    C_R_BT = C_R_Std*k_Fr*k_L*k_BT*k_LB*k_LL*k_AO*k_Tr*k_Pr*(N_rudd**a7)*\
        (N_bracket**a8)*(N_bossing**a9)*(N_thruster**a10)
    C_R = C_R_BT*ship.LBT_WL[1]*ship.LBT_WL[2]/(10*S_Hl)

    ## allowance component
    if(ship.LPP<175):
        C_A = (0.35-0.002*ship.LPP)/1000
    else:
        C_A = 0

    ## Appendage component
    R_App = 0.5*phys().rho_w*Us**2*ship.S_App*(1+ship.k2)*Cf
    d_TH =  prop.D #m, thruster opening diameter, approximated from prop diameter
    C_D_TH = 0.003+0.003*(10*d_TH/ship.LBT_WL[2]-1)
    R_TH = phys().rho_w*Us**2*pi*d_TH**2*C_D_TH
    C_App = (R_App +R_TH)/(0.5*phys().rho_w*Us**2*S_Hl)

    ## environmental component
    C_DA = 0.8
    A_VS = 383.76 ##m^2 transverse  vertical area above waterline --->change here
    C_AAS = C_DA * phys().rho_a*A_VS/(phys().rho_w*S_Hl)
    C_wind = 1 # --->change here
    C_wave = 1 # --->change here
    C_Env = C_AAS*(C_wind+C_wave)

    C_total = Cf + C_R + C_A + C_App + C_Env
    # print('CF:',Cf,'C_R:',C_R,'C_A:',C_A,'C_App:',C_App,'C_Env',C_Env)
    R_total = R_total = np.array([-1*C_total*q*S_Hl,0,0,0,0,0])

    return np.array([R_total]).T

def Hull_Fujiwara(ship,hull_properties,beta,Us,phi_h,printval=False):
    hull = hull_properties
    ## Notes: Fujiwara limitation: L/B~5.8, B/T ~2.6, CB ~0.8, beta ~[-30deg,30deg]
    ##         phi_h #[0,90deg], Us ~ 15m/s, usng right-hand Conventions
    ## Fujiwara beta positive counterclockwise against RH,range -pi to pi
    Fn = Us/sqrt(phys().ga*ship.LBT_WL[0])#Froude number, based on waterline length
    qS = 0.5*phys().rho_w*(Us**2)*ship.LPP*ship.LBT_WL[2] # dynamic pressure
    if(type(hull.X0s) == list):
        ## Tank-test calm water resistance
        if(printval==True):print('Manual input for calm water resistance')
        X0s= np.array(hull.X0s)
        X0 = X0s[0]+X0s[1]*Fn+X0s[2]*Fn**2+X0s[3]*Fn**3;
    else :
        if(hull.X0s=='Fujiwara'):
            if(printval==True):print("Fujiwara calm water resistance")
            ## Fujiwara calm water resistance
            X0 = 1.16E-2 - 1.51E-2*Fn -1.58E-1*Fn**2 +1.14*Fn**3;

        elif(hull.X0s==None):
            if(printval==True):print("Holtrop-Mennen default for calm water resistance")
            ## calm water resistance based on Holtrop-Mennen
            X0 = Hull_HoltropMennen(ship,beta,Us)[5]/qS
        else:
            print("Only accept input of a list of 4, or None, or 'Fujiwara'")

    if(type(hull.ci) == list):
        if(printval==True):print('Manual input for hull coefficients')
        ci =np.array(hull.ci)
    elif(hull.ci==None):
        ## by defaul, use Fujiwara's coefficients
        if(printval==True):print('Fujiwara hull coefficients')
        ci = np.array([[+0.0046, -0.0277, +0.0176, +0.1616, 0.,0.],\
            [0.28360, +0.0237, +0.6724, +0.3467, +1.5391, -0.6382],\
            [0.12640, -0.0225, -0.0085, -0.0379, -0.0454, -0.0775],\
            [-0.0312, -0.0582, -1.1221, +2.4186, +1.5020, +2.5521]])
    else:
        print("Only accept input of a list or None")

    Xh = -X0 + (ci[0,0]*beta**2 +ci[0,1]*beta*phi_h+ ci[0,2]*phi_h**2+ ci[0,3]*beta**4)
    Yh = ci[1,0]*beta + ci[1,1]*phi_h + ci[1,2]*beta**3 + ci[1,3]*(beta**2)*phi_h + \
            ci[1,4]*beta*(phi_h**2) + ci[1,5]*phi_h**3
    Nh = ci[2,0]*beta + ci[2,1]*phi_h + ci[2,2]*beta**3 + ci[2,3]*(beta**2)*phi_h + \
            ci[2,4]*beta*(phi_h**2) + ci[2,5]*phi_h**3
    Kh = ci[3,0]*beta + ci[3,1]*phi_h + ci[3,2]*beta**3 + ci[3,3]*(beta**2)*phi_h + \
            ci[3,4]*beta*(phi_h**2) + ci[3,5]*phi_h**3

    R_total = np.array([Xh*qS,Yh*qS,0.,Kh*qS*ship.LBT_WL[2],0.,Nh*qS*ship.LPP],dtype=float)

    ## transfer to left-hand Conventions:Fz -> -Fz if needed, but it is now just identity
    to_R= np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],\
        [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],dtype=float)

    F_M = np.matmul(to_R,np.array([R_total]).T)
    return F_M,F_M[0][0]*Us

def Propeller(ship,prop_properties,beta,Us,hull_thrust,ni = 2.,p_in=2.5,twin_screw=True,PropType=False):
    prop = prop_properties

    # wake fraction, manual input or based on SRP book
    if (prop.Wp0 == None):
        # wake fraction based on SRP book
        k1,Cf,C_A = Hull_HoltropMennen(ship,beta,Us)[1:4]
        Cv = (1+k1)*Cf+C_A
        if(twin_screw == True):
            ## wake fraction based on SRP book p.164 for twin screw
            Wp = (0.3095*ship.CB+10*Cv**ship.CB-0.23*prop.D/sqrt(ship.LBT_WL[1]*ship.LBT_WL[2]))
        else:
            ## wake fraction based on SRP book p.161 for single screw
            Wp = 1.095-3.4*ship.CB+3.3*(ship.CB**2)+0.5*(ship.CB**2)*(6.5-ship.LBT_WL[0]/(ship.LBT_WL[1]))/(ship.LBT_WL[0]/(ship.LBT_WL[1]))
    else:
        Wp = prop.Wp0 ## wake factor or wake fraction, manual input
    ## Wp = Wp*exp(-4*beta**2) ##take into account Leeway_angle, based on Fujiwara
    # print('Wp:',Wp)

    small_u = abs(Us*cos(beta))
    Va = (1-Wp)*small_u##m/s inflow velocity
    J = small_u*(1-Wp)/(ni*prop.D)## advance coefficient
    cx = 2.208*(prop.D/prop.Z)*(prop.AE/prop.A0) ##c75 in Birk book
    Rex = cx*sqrt(Va**2+(0.75*pi*ni*prop.D)**2)/phys().nu_w

    if(PropType == False):
        ## using WageningenB
        PropType = WageningenB(prop,J,Rex,p_in)
        KQ = PropType.KQ()
        KT = PropType.KT()
        n0 = KT/KQ*J/(2*pi)  ## openwater efficiency
    else :
        ## using tank-test data provided in ship python script
        PropType = OW_prop(J,p_in/prop.D)#Tanktest_prop(J)
        KQ = PropType.KQ()
        KT = PropType.KT()
        n0 = PropType.eff() ## openwater efficiency

    # Prop's thrust deduction factor
    if (prop.Tp == None):
        if(twin_screw == True):
            Tp = 0.325*ship.CB-0.1885*prop.D/ \
                sqrt(ship.LBT_WL[1]*ship.LBT_WL[2])## Holtrop, SRP book p.166
        else:
            lcb = ship.COB[0]/(ship.LBT_WL[0])*100 #in percent, move the reference to 1/2LWL then divided by L_WL
            Tp = 0.25014*(ship.LBT_WL[1]/ship.LBT_WL[0])* \
                (sqrt(ship.LBT_WL[1]*ship.LBT_WL[2])/prop.D)**0.2624/ \
                (1-ship.C_PL+0.0225*(lcb-0.5*ship.LBT_WL[0])/ship.LBT_WL[0])**0.01762+ \
                0.0015*ship.C_stern ## Holtrop, SRP book p.163
    else :
        Tp = prop.Tp ##if using tank-test

    # print('Wp:',Wp,' J:',J,' KT:',KT, ' KQ:',KQ, ' n0:', n0,' Tp:',Tp)
    ## Force total, see Fujiwara
    F_prop1 = (1-Tp)*(ni**2)*(prop.D**4)*KT*phys().rho_w
    # F_prop1 = (1-Tp)*KT*phys().rho_w*(prop.D**2)*Va**2/(J**2)# same result as above
    F_prop1 = np.array([F_prop1,0.,0.],dtype=float)
    if(twin_screw==True):
         F_prop2 = F_prop1
    else:
         F_prop2 = np.zeros_like(F_prop1)
    F_prop = F_prop1 + F_prop2

    ## Moment total = vector(r) x vector(F)
    M_prop1 = np.cross(prop.COP1,F_prop1)
    M_prop2 = np.cross(prop.COP2,F_prop2)
    M_prop = M_prop1 + M_prop2

    ## Transpose
    F_prop = np.array([F_prop]).T
    M_prop = np.array([M_prop]).T

    ## Using G's prop data
    PD = p_in/prop.D
    if (PD<0.2):
        PD = 0.2 ## limit PD minimum for the power factor
    elif (PD>1.15):
        PD = 1.15 ## limit PD max
    else:
        PD = PD
    G_power_factor = 185.14*exp(2.8002*PD) ##kW
    BHP = G_power_factor*1000 ## watt

    ## Delivered horsepower,Fujiwara for 1 prop
    eta_QCPC = 1.03 ## ratio of propulsive coefficients
    BHP1 = 2*pi*KQ*phys().rho_w*(ni**3)*(prop.D**5)*eta_QCPC ##watt, if HP /735.5

    # eta_d = n0*((1-Tp)/(1-Wp))*prop.eta_R #quasi propulsive coefficient
    # BHP1=(F_prop[0][0]*small_u*(1-Tp))/eta_d*1/prop.eta_T

    # ## Delivered horsepower,Birk
    # BHP2 = 2*pi*KQ*phys().rho_w*(Va**5)/(ni**2*J**5) #/eta_QCPC

    return np.concatenate((F_prop,M_prop),axis=0),n0,BHP,BHP1,KQ,KT,J,Wp,Tp

def Rudder(ship,rudder_properties,prop_properties,beta,Us,rudder_delta,ni,p_in,twin_screw=True,PropType=False):
    rudd=rudder_properties
    prop=prop_properties

    f_A = 6.13*rudd.AR/(2.25+rudd.AR)
    eta_p = prop.D/rudd.h
    d = ship.LBT_WL[2] ## Draught
    B = ship.LBT_WL[1] ## Breadth
    Wr0 = -7.44*d*ship.CB/ship.LPP - 2.39*ship.CB*B/ship.LPP*ship.SigmaA+0.851
    Wr = Wr0*exp(-4*beta**2) ## wake factor of rudder

    # wake fraction, manual input or based on SRP book
    if (prop.Wp0 == None):
        if(twin_screw == True):
            k1,Cf,C_A = Hull_HoltropMennen(ship,beta,Us)[1:4]
            Cv = (1+k1)*Cf+C_A
            ## wake fraction based on SRP book p.164 for twin screw
            Wp = (0.3095*ship.CB+10*Cv**ship.CB-0.23*prop.D/sqrt(ship.LBT_WL[1]*ship.LBT_WL[2]))
        else:
            ## wake fraction based on SRP book p.161 for single screw
            Wp = 1.095-3.4*ship.CB+3.3*(ship.CB**2)+0.5*(ship.CB**2)*(6.5-ship.LBT_WL[0]/(ship.LBT_WL[1]))/(ship.LBT_WL[0]/(ship.LBT_WL[1]))
    else:
        Wp = prop.Wp0#  ## wake fraction manual input
    # Wp = Wp*exp(-4*beta**2) ##take into account Leeway_angle, from Fujiwara

    # k = 0.6*(1-Wp)/(1-Wr)
    small_u = Us*cos(beta)
    # s = 1-small_u*(1-Wp)/(ni*p_in)
    # UR2 = (1+eta_p*k*(2-(2-k)*s)*s/(1-s)**2)*(1-Wr)**2
    ea = ship.LPP/B*(1-ship.C_PL)
    ea_prime = ea/sqrt(0.25+1/(B/d)**2)
    gamma_R = 4.02*d*(1-ship.CB)/B+1.98*(d*(1-ship.CB)/B*ea_prime)**2 \
            -1.54*(d*(1-ship.CB)/B*ea_prime)+0.22
    # gamma_R = -22.2*((ship.CB*B/ship.LPP)**2)+0.02*((ship.CB*B/ship.LPP)**2)+0.68 ##using Kijima 1990
    alpha_R = (rudder_delta) - gamma_R*(beta) ## rad

    # ## F_normal rudder, from Fujiwara
    # qS = 0.5*phys().rho_w*(Us**2)#*(ship.LBT_WL[2]*1.3)#*ship.LPP # dynamic pressure ## adjustment for Fy & Mz, side force is wrong!!
    # FN_prime = f_A*rudd.AR*UR2*sin(alpha_R)
    # ## print('FNprime',FN_prime,' rudder_delta(deg):',degrees(rudder_delta),' gamma_R(deg):',(gamma_R),' alpha_R(deg):',(alpha_R),' UR2',UR2)
    # FN = np.array([0.,FN_prime*qS,0.],dtype=float)

    ## if using MMG from Yasukawa & Yoshimura method,commented if unused
    if(rudd.epsilon == None):
        epsilon = (1-Wr)/(1-Wp)
    else:
        epsilon = rudd.epsilon
    Va = (1-Wp)*small_u ##m/s inflow velocity
    J = small_u*(1-Wp)/(ni*prop.D)
    cx = 2.208*(prop.D/prop.Z)*(prop.AE/prop.A0) ##c75 in Birk book
    Rex = cx*sqrt(Va**2+(0.75*pi*ni*prop.D)**2)/phys().nu_w

    if(PropType == False):
        ## using WageningenB
        PropType = WageningenB(prop,J,Rex,p_in)
    else :
        ## using tank-test data provided in ship python script
        PropType = OW_prop(J,p_in/prop.D)#Tanktest_prop(J)
    KT = PropType.KT()
    kappa = rudd.kappa
    if (KT<0):
        # print('math domain error cause:',1+8*KT/(pi*J**2),' J=',J,' KT=',KT,' Us=',Us)
        KT = 0.
    u_R = epsilon*(1-Wp)*small_u*sqrt(eta_p*(1+kappa*(sqrt(1+8*KT/(pi*J**2))-1))**2+(1-eta_p))
    # u_R = (0.7*pi*ni*prop.D)*1. ## another version  from okuda et al.
    v_R = Us*gamma_R*beta ## with yaw_rate=0 because it's steady
    alpha_R = rudder_delta-atan(-v_R/u_R)
    UR = sqrt(v_R**2+u_R**2)
    FN = np.array([0.,0.5*phys().rho_w*rudd.AR*(UR**2)*f_A*sin(alpha_R),0.],dtype=float)
    # print('current FN=',FN[1]/1000,'kNm','UR=',UR, 'alpha_R=',alpha_R,'f_A=',f_A,'Us=',Us)
    ## rotate the frame of reference pi rads

    if (rudd.C_1mintR==None):
        C_1mintR =  0.28*ship.CB+0.55
    else:
        C_1mintR =  rudd.C_1mintR

    rudder_to_ship= Rotate(0.,0.,rudder_delta+pi)
    rudd_coeffs = np.array([C_1mintR,(1+rudd.aH),0])
    F_rudd = np.matmul(rudder_to_ship.RH_yaw(),FN) # rotate to ship ref frame = F.(R_yaw)
    F_rudd1 = np.multiply(F_rudd, rudd_coeffs) # multiply by coefficients of reduction

    if(twin_screw==True):
        F_rudd2 = F_rudd1
    else:
        F_rudd2 = np.zeros_like(F_rudd1)
    F_rudder = F_rudd1 + F_rudd2

    if (rudd.aH_K != None):
        rudd_M_coeffs = np.array([(1.+rudd.aH_K),0,1]) # 0 at y is zeroing the My, *(-1) for reverse rudder rotation
    else:
        rudd_M_coeffs = np.array([(1.+rudd.aH),0,1]) # 0 at y is zeroing the My, *(-1) for reverse rudder rotation
    M_rudd1 = np.cross(rudd.COP1,F_rudd)
    M_rudd1 = np.multiply(M_rudd1,rudd_M_coeffs)
    M_rudd1[2] += rudd.aH*rudd.xH_prime*ship.LPP*F_rudd[1] ## MMG interactive force between rudder and hull -a_H*x_H*FN*cos(rudder_delta)

    if(twin_screw==True):
        M_rudd2 = np.cross(rudd.COP2,F_rudd)
        M_rudd2 = np.multiply(M_rudd2,rudd_M_coeffs)
        M_rudd2[2] += rudd.aH*rudd.xH_prime*ship.LPP*F_rudd[1]
    else:
        M_rudd2 = np.zeros_like(M_rudd1)
    M_rudder = M_rudd1+M_rudd2
    # # # rudder checkpoints:
    # print('F_Rudd1                   ',F_rudd1, ' FN:',FN[1])
    # print('Fujiwara Matrix of Forces',np.array([-FN[1]*C_1mintR*sin(rudder_delta),\
    # -FN[1]*(1+rudd.aH)*cos(rudder_delta),0.],dtype=float))
    # print('M_Rudd1                   ',M_rudd1)
    # print('Fujiwara Matrix of Moments',np.array([-FN[1]*(1.+rudd.aH)*0.68*ship.LBT_WL[2]*cos(rudder_delta),0.,\
    # -FN[1]*(-0.5-0.4*(rudd.aH))*ship.LPP*cos(rudder_delta)],dtype=float))

    ## rotate
    F_rudder = np.array([F_rudder]).T
    M_rudder = np.array([M_rudder]).T
    return np.concatenate((F_rudder,M_rudder),axis=0),M_rudd1,M_rudd2

def SF_LiftingLine(ship,beta,Us):
    ## This is Prandtl's lifting line theory for Resistance based on side-force evolution
    ## (only applicable below 5 deg drift angle, the rest is very non-linear)
    AR = ship.LBT_WL[2]/ship.LPP
    qS = 0.5*phys().rho_w*(Us**2)*(ship.LBT_WL[2]*ship.LPP) ## dyn pressure*Area
    Cy = beta*AR*pi
    Fy = np.array([0.,Cy*qS,0.,0.,0.,0.],dtype=float)
    return np.array([Fy]).T

def power_fuel(BHP):
    ## This is power to fuel relation, interpolation is linear
    ## BHP_in should be in kWatt, SFOC in g/kWh
    load = np.interp(BHP/1000,G_power_fuel().BHP_in, G_power_fuel().load_in)
    if(load<25.0):
        percent_load = 25.
    elif(load>110.0):
        percent_load = 110.
    else:
        percent_load =load
    SFOC = np.interp(percent_load,G_power_fuel().load_out,G_power_fuel().fuel_out)
    return SFOC*(BHP/1000),percent_load,SFOC

## --- aerodynamic properties, cuurently for 1 sail:
## outline of algorithm
## 1. read WOLFSON data based on apparent wind angle & speed (AWA, AWS) to the wingsail
## 2. find corresponding CL & CD based on TWA & TWS
## 3. calculate forces
## 4. rotate frame of reference for forces
## 5. calculate moments from force in ship frame of reference
def Wingsail(ship,sail_properties,AWA,AWS,Cx=0.,Cy=0.,CMx=0.,CMz=0.):
    sail = sail_properties
    wind = angle_speed(AWA,AWS)
    # print('apparent wing angle & speed TO THE WING SAIL=',degrees(wind.angle), 'degs &', wind.speed, 'm/s')

    # ## Random Data for trial: flower function: r= a+bcos(cx)
    # ax=1.5 ;bx=1. ;cx=2; dx=1.
    # CL = (ax+bx*cos(cx*wind.angle+dx*pi))
    # CD = -(ax+bx*cos(cx*wind.angle+dx*pi)) ##----->the one generated thrust is here!

    Uw=abs(wind.speed) #should be magnitude
    qw = 0.5*phys().rho_a*(Uw**2) #dynamic pressure= 0.5 * air_density * square(u)
    F_thrust = qw*Cx*sail.area
    if (sail.area_longt==None):
        area_longt= sail.area
    else:
        area_longt= sail.area_longt
    F_lift = qw*Cy*area_longt

    ## rotational matrix from local wing to ship direction in yaw
    wing_on_ship = Rotate(radians(0.),radians(0.),radians(wind.angle))

    ## based on apparent wind on the wing sail,find force & moments. Negatives reverse directions.
    F_wing =  np.array([[F_thrust],\
                        [F_lift], \
                        [0.]])

    F_aero = F_wing.T ## np.matmul(F_wing.T,wing_on_ship.RH_yaw().T) # rotate to ship ref frame = F.Inv(R_yaw)

    if(CMz != 0):
        # heeling moment rotation of WOlFSON Unit is at waterline, then adjusted to COG
        M_aero =np.array([CMx*qw*sail.area+F_aero[0][1]*(ship.LBT[2]-ship.COG[2]),0.,CMz*qw*sail.area]) ## Wolfson Unit data
    else:
        M_aero = np.cross(sail.COP,F_aero[0]) # M= r x F, in ship ref frame

    # Fujiwara's way of calculating sail moments
    if (sail.name=='Fujiwara sail'):
        M_aero =np.array([(CMx*qw*area_longt**2./ship.LBT_WL[0])+F_aero[0][1]*ship.COG[2],0.,CMz*qw*area_longt*ship.LBT_WL[0]]) ##Fujiwara method

    M_aero = np.array([M_aero]).T # turn into column matrix
    F_aero = np.array([F_aero][0]).T # turn into column matrix

    return np.concatenate((F_aero,M_aero),axis=0)

def exit_program():
    print("Exiting the WASp algorithm...")
    sys.exit(0)

class Read_aero():
    ## This is class for reading Aero CFD-data
    def __init__(self,aero_data,filename,mirror=False):
        self.filepath = os.path.abspath(__file__)
        self.filedir = os.path.dirname(self.filepath)+'/'
        self.aero_data = aero_data
        self.filename = filename
        self.Fujiwara_aero = False; self.G_aero = False
        if (aero_data=='Fujiwara'):
            self.Fujiwara_aero = True
        elif(aero_data=='G'):
            self.G_aero = True
        else:
            print("call 'Fujiwara' or 'G' only")
            exit_program()
        self.mirror = mirror

        if(self.G_aero==False and self.Fujiwara_aero==False):
            print("No aero data found! Define the aero data as in G_aero or Fujiwara_aero.")
            exit_program()

        if(self.G_aero==True):
            ## delim_whitespace = False if reading CSV, but True if textfile
            self.df=pd.read_csv(self.filedir+'G_aero/'+self.filename,sep='\\s+',\
                    names=["AWA","Theta","Cx_rig","Cy_rig","CMx_rig","CMz_rig","q","Cx","Cy",\
                    "CMx","CMz","Drag","Lift"]) ## sep=None. for csv, sep='\\s+' for *.txt
            self.df.drop(["Theta","Cx_rig","Cy_rig","CMx_rig","CMz_rig","Drag","Lift"],axis=1, inplace=True) ## drop unnecessary
            self.df.drop(self.df.index[0], inplace=True) ## drop first row
            self.df=self.df.apply(pd.to_numeric, errors = 'coerce') ## all to numeric

            ## reverse all convention in Wolfson's AERO data: +x aft, +y starboard, +z up
            self.df["Cx"] = -self.df["Cx"];self.df["CMx"] = -self.df["CMx"];
            self.df["CMz"] = -self.df["CMz"];

            ## AWA adjustment
            self.df["AWA"] =-self.df["AWA"]# Wolfson's AWA: 0deg headwind, +90 wind from port
            # self.df["AWA"] %= 360 # turn AWA into 0-360deg range
            self.df = self.df.sort_values(["AWA"]) ## sort all based on AWA (ascending)

            # ## mirror empty data from 90 to 180 degrees for Wolfon dat, then *-1 (for AWA AND OTHERS, check first before *-1)
            if(self.mirror==True):
                self.df = pd.concat([self.df, self.df.iloc[:4]], ignore_index=True)
                self.df.loc[:3,["AWA","Cy","CMx","CMz"]] *=-1
            self.df1 = self.Refine()

        ## using fujiwara aero
        if(self.Fujiwara_aero==True):
            ## delim_whitespace = False if reading CSV, but True if textfile
            self.df=pd.read_csv(self.filedir+'Fujiwara_aero/'+self.filename,sep=None, \
                names=["AWA","Theta","Cx_rig","Cy_rig","CMx_rig","CMz_rig","q","Cx","Cy",\
                        "CMx","CMz","Drag","Lift"],engine='python')
            self.df.drop(["Theta","Cx_rig","Cy_rig","CMx_rig","CMz_rig","Drag","Lift"],axis=1, inplace=True) ## drop unnecessary
            self.df.drop(self.df.index[0], inplace=True) ## drop first row
            self.df=self.df.apply(pd.to_numeric, errors = 'coerce') ## all to numeric

            ## turn AWA into 0-360deg range
            self.df["AWA"] %= 360;
            self.df = self.df.sort_values(["AWA"]) ## sort all based on AWA (ascending)

            if(self.mirror==True):
                ## mirror empty data from 90 to 180 degrees, then *-1 (for AWA AND OTHERS, check first before *-1)
                self.df= self.df[self.df['AWA'] <= 180] ## drop all zero data >180 degs
                num_i=len(self.df.index)
                self.df = pd.concat([self.df, self.df.iloc[1:-1]], ignore_index=True)
                self.df.loc[num_i:,["AWA","Cy","CMx","CMz"]] *=-1
                self.df = self.df.sort_values(["AWA"]) ## sort all based on AWA (ascending)
            self.df1 = self.Refine()

    def Refine(self):
        ## refining data to 1000 points
        df1 = self.df.copy()
        df1.set_index("AWA",inplace =True)
        upsample = np.linspace(df1.index.min(),df1.index.max(),50)
        df1= df1.reindex(df1.index.union(upsample)).interpolate(method="piecewise_polynomial").loc[upsample]
        # Apply zero values to all nan's
        df1 = df1.fillna(0)
        df1 = df1.reset_index() ## AWA is no longer index
        return df1

    def Coefficients(self,AWAi):
        AWAi = remainder(AWAi, tau) ## change into [-pi,pi] range
        AWAi =degrees(AWAi)## change new AWA data from input (in degrees)
        df = self.df1
        df.loc[-1, "AWA"] = AWAi
        df = df.sort_values(["AWA"]) ## sort all based on AWA (ascending)
        df.set_index("AWA",inplace =True) ## set AWA variable as index

        # Apply zero values to all nan's, or interpolate all nan values
        df = df.interpolate(method="piecewise_polynomial")

        df = df.reset_index() ## AWA is no longer index

        ## find corresponding index after sorting:
        sorted_id = df.iloc[(df["AWA"]-AWAi).abs().argsort()[:1]]
        id = sorted_id.index.tolist()[0]

        checknan=df.iloc[id][1:].isnull().sum().sum()
        if (checknan>0): return [0.,0.,0.,0.,0.]
        else: return df.iloc[id][1:].tolist()

    def Plot_wingsail(self,polar=False,save=False):
        ## plor original data (in scatter) to the interpolated data (line)
        df1 = self.df.copy()

        ## refining data
        df2 = self.df1.copy()
        df2=self.Refine()

        ys = ["q","Cx","Cy","CMx","CMz"]
        ylims=[[0,180],[-3,4],[-8,8],[-80,80],[-150,150]]
        if (self.Fujiwara_aero==True): ylims=[[0,180],[-2,8],[-2,3],[-4,8],[-0.2,0.2]]

        if(polar==False):
            fig, ((ax,ax1),(ax2,ax3),(ax4,ax5)) = plt.subplots(3,2,\
                constrained_layout=True,figsize=(8,15))
        else:
            fig, ((ax,ax1),(ax2,ax3),(ax4,ax5)) = plt.subplots(3,2,\
                constrained_layout=True,figsize=(8,15),\
                subplot_kw={'projection': 'polar'})
        axs = [ax,ax1,ax2,ax3,ax4,ax5]

        for i in range(len(ys)):
            axi=axs[i]
            if(polar==False):
                df1.plot.scatter(ax=axi, x="AWA", y=ys[i],marker='o',color='r',legend=False)#,label="CFD_ship_only")
                df2.plot(ax=axi, x="AWA", y=ys[i],color='k',legend=False)#,label="extrapolated")
                axi.set_ylim((ylims[i][0],ylims[i][1]));
            else:
                # ylims = [np.nanmin(df2.loc[:,ys[i]]),np.nanmax(df2.loc[:,ys[i]])]
                plotpolar(axi,df2["AWA"]/180*pi,df2.loc[:,ys[i]],ylims[i][0],ylims[i][1],\
                    Ytitle=ys[i],linestyles="-",markers="",thetamin=0,thetamax=360)# title=ys[i]
        plt.show();
        if(save==True):
            fig.savefig(self.filedir+"Aero_ship_only_CFD_nolegend.png", dpi=300, transparent=True)

class Rotate():
    ## -- Euler rotational matrix based on angles made on axes: x,y,z (shape 3x3)
    ## -- using Right-hand (RH) coordinate system, unless stated otherwise (LH)
    ## -- Inverse(R)=Transpose(R)
    ## -- Not for converting between RH and LH
    def __init__(self,phi,theta,psi):
        self.phi = phi      ## angle around x
        self.theta = theta  ## ..... around y
        self.psi = psi      ## ..... around z

    def LH_roll(self):
        phi = self.phi
        R1 = np.array([[1,0,0],[0,cos(phi),sin(phi)],[0,-sin(phi),cos(phi)]])
        return R1

    def LH_pitch(self):
        theta = self.theta
        R2 = np.array([[cos(theta),0,-sin(theta)],[0,1,0],[sin(theta),0,cos(theta)]])
        return R2

    def LH_yaw(self):
        psi = self.psi
        R3 = np.array([[cos(psi),sin(psi),0],[-sin(psi),cos(psi),0],[0,0,1]])
        return R3

    def LH_body_rpy(self): #roll_pitch_yaw: inertial frame -->body-fixed frame
        R1 = self.LH_roll()
        R2 = self.LH_pitch()
        R3 = self.LH_yaw()
        return np.matmul(np.matmul(R1,R2),R3)

    def LH_inertial_rpy(self): #yaw-pitch-roll: body-fixed frame --> inertial frame
        R1 = -self.LH_roll()
        R2 = -self.LH_pitch()
        R3 = -self.LH_yaw()
        return np.matmul(np.matmul(R3,R2),R1)

    ## Right-Hand (RH) rule rotation matrices
    def RH_roll(self):
        phi = self.phi
        R1 = np.array([[1,0,0],[0,cos(phi),-sin(phi)],[0,sin(phi),cos(phi)]])
        return R1

    def RH_pitch(self):
        theta = self.theta
        R2 = np.array([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]])
        return R2

    def RH_yaw(self):
        psi = self.psi
        R3 = np.array([[cos(psi),-sin(psi),0],[sin(psi),cos(psi),0],[0,0,1]])
        return R3

    def RH_body_rpy(self): #roll_pitch_yaw: inertial frame -->body-fixed frame
        R1 = self.RH_roll()
        R2 = self.RH_pitch()
        R3 = self.RH_yaw()
        return np.matmul(np.matmul(R1,R2),R3)

    def RH_inertial_rpy(self): #yaw-pitch-roll: body-fixed frame --> inertial frame
        R1 = -self.RH_roll()
        R2 = -self.RH_pitch()
        R3 = -self.RH_yaw()
        return np.matmul(np.matmul(R3,R2),R1)

class angle_speed():
    def __init__(self,angle,speed):    ## true wind direction & speed , ask WOlFSON for reference of wind direction
        self.angle = angle ##----->change here
        self.speed = speed ## m/s ##----->change here

class phys():
    def __init__(self):
            ## fluid & physic properties
            self.rho_w =1026.021 #kg/m^3,seawater density at 15C
            self.rho_a =1.225 #kg/m^3, air density at sea level
            self.nu_w = 1.1892E-6#m^2/s, kinematic viscosity of saltwater at 10C
            self.ga = 9.81 #m/s^2, gravity acceleration

class vessel():
    def __init__(self,shipname,W_s,COG,COB,LBT,GM,LBT_WL,LPP,Lambda_Fr):
        self.shipname = shipname
        self.W_s = W_s #kg, displ in mass term
        self.COG = COG #m,distances: [longitudinal from stern,mid,vertical from keel]
        self.COB = COB #m, center of buoyancy
        self.LBT = LBT #m,[LOA,Breadth,Draught from keel]
        # self.LBTm = np.array([2.64,0.438,0.067])#m dimensions of model
        self.GM = GM # m, upright-->something not right here
        self.LBT_WL= LBT_WL ## LBT at waterline
        self.LPP = LPP # perpendicular length
        self.Lambda_Fr = Lambda_Fr #geometry scale, based on Froude

        # ## additional data for HM_ship
        # self.A_BT = 20 #m^2
        # self.h_B = 4 #m
        # self.C_M = 0.98
        # self.C_WP = 0.75
        # self.A_T = 16 #m^2
        # self.S_App = 50 #m^2
        # self.C_stern = 10
        # self.C_PL = 0.5833
        # self.k2 = 0.5
        # self.T_F =10 #m, draft at aft

        # ## additional data for KCS_fullscale
        # self.CB = 0.6505
        # self.C_M = 0.9849
        # self.C_stern = 10
        # self.k2 = 0.5
        # self.S_App = 115 #m^2, wetted surface area of appendages

        # ## additional data for Birk's Basic ship
        # self.A_BT = 14 #m^2, bulbous bow cross section area at FP
        # self.h_B = 4 #UNKNOWN,m
        # self.C_M = 0.98 #UNKNOWN,
        # self.C_WP = 0.75 #UNKNOWN,
        # self.A_T = 16 #UNKNOWN,m^2
        # self.S_App = 52 #m^2
        # self.C_stern = 10 #UNKNOWN,
        # self.C_PL = 0.56783
        # self.k2 = 0.4
        # self.T_F =8.2 #m, draft at aft
        # self.L_OS = 151 #m, length over wetted surface
        # self.CB = 0.6613

        ## derived data of ship
        self.Tau_Fr = (1.3063E-6/1.1512E-6)*self.Lambda_Fr**2  #time scale = nu_model/nu_Ship*lambda^2, nu:kinematic visc at 10deg
        self.nabla = self.W_s/phys().rho_w #m^3, displacement vol
        self.delta = self.nabla*phys().rho_w #kg, displacement force
        self.CB = self.nabla/(self.LBT_WL[0]*self.LBT_WL[1]*self.LBT_WL[2]) #block coefficient
        # self.B_T = self.LBT[1]/self.LBT[2]
        # self.CS = 6.554-1.226*self.B_T+0.216*(self.B_T**2)-15.409*self.CB+ \
        #     4.468*self.B_T*self.CB-0.694*(self.B_T**2)*self.CB+ \
        #     15.404*(self.CB**2)-4.527*self.B_T*(self.CB**2)+0.655*(self.B_T**2)*self.CB**2
        # self.S_s = self.CS*sqrt(self.nabla*self.LBT[0]) # m^2, wetted surface, eq.10.105 Molland et al.

        # ## general data for G, to comment if necessary
        self.BM = (self.LBT[1]* self.LBT[0]**3)/(12*self.nabla) # pg.135 Ship Stability book
        self.A_m = 0.98555*self.LBT_WL[1]*self.LBT_WL[2] # m^2, midship sectional area (vertical cross-section)## ----->measured from CAD
        self.C_PL = self.nabla/(self.A_m*self.LPP) ## longitudinal prismatic coefficient ## ----->measured & change here, affecting iE and c1!
        self.A_w = 0.9192278*self.LBT_WL[0]*self.LBT_WL[1] #m^2, waterplane sectional area (horizontal cross-section)## ----->measured from CAD
        self.C_PV = self.nabla/(self.A_w*self.LBT_WL[2]) ##vertical prismatic coefficient
        self.C_WP = self.A_w/(self.LBT_WL[0]*self.LBT_WL[1]) ##fineness waterplane coefficient
        self.C_M = self.A_m/(self.LBT_WL[1]*self.LBT_WL[2]) ##midship sectional coefficient
        self.C_stern = 0. # Stern shape coefficient, zero is normal section shape## ----> change here
        self.k2 = 1-2.8 # twin screw balance rudders coefficients
        self.S_App = 50 #m^2, wetted surface area of appendages
        self.A_BT = 20. #m^2 transverse area ## ----> change here
        self.h_B = 3.5 #m, position of A_BT center above keel ## ----> change here
        self.A_T = 10 ##m^2, transom area ##----- change here
        self.T_F = self.LBT_WL[2] #m, forward draught of ship ## ----> change here, if Aft & Fwd are different

        self.SigmaA = (1-self.C_WP)/(1-self.C_PV)
        self.d_tunnel = -0.9# tunnel diameter for appendage resistance of Holtrop Mennen

class WageningenB():
    def __init__(self,prop_properties,J,Rex,p_in,twin_screw=False):
        ## propeller 1 & 2 Data
        self.prop = prop_properties
        self.J = J # advance ratio
        self.Rex = Rex
        self.twin_screw = twin_screw
        self.P_D = p_in/self.prop.D #P/D, pitch to diameter ratio
        self.AE_A0 = self.prop.AE/self.prop.A0 # AE/A0, expanded area ratio
        self.Z = self.prop.Z #number of blades
        ## 𝑖, 𝑎𝑖, 𝑏𝑖, 𝑐𝑖, 𝑑𝑖, 𝑒𝑖
        self.C_KT=[[0,0.00880496,0,0,0,0],[1,-0.204554,1,0,0,0],[2,0.166351,0,1,0,0],\
            [3,0.158114,0,2,0,0],[4,-0.147581,2,0,1,0],[5,-0.481497,1,1,1,0],\
            [6,0.415437,0,2,1,0],[7,0.0144043,0,0,0,1],[8,-0.0530054,2,0,0,1],
            [9,0.0143481,0,1,0,1],[10,0.0606826,1,1,0,1],[11,-0.0125894,0,0,1,1],\
            [12,0.0109689,1,0,1,1],[13,-0.133698,0,3,0,0],[14,0.00638407,0,6,0,0],\
            [15,-0.00132718,2,6,0,0],[16,0.168496,3,0,1,0],[17,-0.0507214,0,0,2,0],\
            [18,0.0854559,2,0,2,0],[19,-0.0504475,3,0,2,0],[20,0.010465,1,6,2,0],\
            [21,-0.00648272,2,6,2,0],[22,-0.00841728,0,3,0,1],[23,0.0168424,1,3,0,1],\
            [24,-0.00102296,3,3,0,1],[25,-0.0317791,0,3,1,1],[26,0.018604,1,0,2,1],\
            [27,-0.00410798,0,2,2,1],[28,-0.000606848,0,0,0,2],[29,-0.0049819,1,0,0,2],\
            [30,0.0025983,2,0,0,2],[31,-0.000560528,3,0,0,2],[32,-0.00163652,1,2,0,2],\
            [33,-0.000328787,1,6,0,2],[34,0.000116502,2,6,0,2],[35,0.000690904,0,0,1,2],\
            [36,0.00421749,0,3,1,2],[37,0.0000565229,3,6,1,2],[38,-0.00146564,0,3,2,2]];
        self.C_KQ=[[0,0.00379368,0,0,0,0],[1,0.00886523,2,0,0,0],[2,-0.032241,1,1,0,0],\
            [3,0.00344778,0,2,0,0],[4,-0.0408811,0,1,1,0],[5,-0.108009,1,1,1,0],\
            [6,-0.0885381,2,1,1,0],[7,0.188561,0,2,1,0],[8,-0.00370871,1,0,0,1],\
            [9,0.00513696,0,1,0,1],[10,0.0209449,1,1,0,1],[11,0.00474319,2,1,0,1],\
            [12,-0.00723408,2,0,1,1],[13,0.00438388,1,1,1,1],[14,-0.0269403,0,2,1,1],\
            [15,0.0558082,3,0,1,0],[16,0.0161886,0,3,1,0],[17,0.00318086,1,3,1,0],\
            [18,0.015896,0,0,2,0],[19,0.0471729,1,0,2,0],[20,0.0196283,3,0,2,0],\
            [21,-0.0502782,0,1,2,0],[22,-0.030055,3,1,2,0],[23,0.0417122,2,2,2,0],\
            [24,-0.0397722,0,3,2,0],[25,-0.00350024,0,6,2,0],[26,-0.0106854,3,0,0,1],\
            [27,0.00110903,3,3,0,1],[28,-0.000313912,0,6,0,1],[29,0.0035985,3,0,1,1],\
            [30,-0.00142121,0,6,1,1],[31,-0.00383637,1,0,2,1],[32,0.0126803,0,2,2,1],\
            [33,-0.00318278,2,3,2,1],[34,0.00334268,0,6,2,1],[35,-0.00183491,1,1,0,2],\
            [36,0.000112451,3,2,0,2],[37,-0.0000297228,3,6,0,2],[38,0.000269551,1,0,1,2],\
            [39,0.00083265,2,0,1,2],[40,0.00155334,0,2,1,2],[41,0.000302683,0,6,1,2],\
            [42,-0.0001843,0,0,2,2],[43,-0.000425399,0,3,2,2],[44,0.0000869243,3,3,2,2],\
            [45,-0.0004659,0,6,2,2],[46,0.0000554194,1,6,2,2]];
        self.C_dKT = [[0,0.000353485,0,0,0,0,0],[1,-0.00333758,2,0,1,0,0],\
            [2,-0.00478125,1,1,1,0,0],[3,0.000257792,2,0,1,0,2],\
            [4,0.0000643192,2,6,0,0,1],[5,-0.0000110636,2,6,0,0,2],\
            [6,-0.0000276305,2,0,1,1,2],[7,0.0000954,1,1,1,1,1],\
            [8,0.0000032049,1,3,1,2,1]]
        self.C_dKQ = [[0,-0.000591412,0,0,0,0,0],[1,0.00696898,0,1,0,0,0],\
            [2,-0.0000666654,0,6,0,1,0],[3,0.0160818,0,0,2,0,0],\
            [4,-0.000938091,0,1,0,0,1],[5,-0.00059593,0,2,0,0,1],\
            [6,0.0000782099,0,2,0,0,2],[7,0.0000052199,2,0,1,1,1],\
            [8,-0.00000088538,1,1,1,1,2],[9,0.0000230171,0,6,0,1,1],\
            [10,-0.00000184341,0,6,0,1,2],[11,-0.00400252,0,0,2,0,1],\
            [12,0.000220915,0,0,2,0,2]]
    def KT(self):
        self.KT = 0;
        for i in range(39):
            self.KT = self.KT+self.C_KT[i][1]*self.J**(self.C_KT[i][2])*\
            (self.P_D)**(self.C_KT[i][3])*(self.AE_A0)**(self.C_KT[i][4])*\
            self.Z**(self.C_KT[i][5]);
        self.deltaKT = 0
        for i in range(9):
            self.deltaKT = self.deltaKT+self.C_dKT[i][1]*self.J**(self.C_dKT[i][2])*\
            (self.P_D)**(self.C_dKT[i][3])*(self.AE_A0)**(self.C_dKT[i][4])*\
            self.Z**(self.C_dKT[i][5])*log10(self.Rex-0.301)**(self.C_dKT[i][6]);
        return self.KT+self.deltaKT
    def KQ(self):
        self.KQ = 0 ;
        for i in range(47):
            self.KQ = self.KQ+self.C_KQ[i][1]*self.J**(self.C_KQ[i][2])*\
            (self.P_D)**(self.C_KQ[i][3])*(self.AE_A0)**(self.C_KQ[i][4])*\
            self.Z**(self.C_KQ[i][5]);
        self.deltaKQ = 0
        for i in range(13):
            self.deltaKQ = self.deltaKQ+self.C_dKQ[i][1]*self.J**(self.C_dKQ[i][2])*\
            (self.P_D)**(self.C_dKQ[i][3])*(self.AE_A0)**(self.C_dKQ[i][4])*\
            self.Z**(self.C_dKQ[i][5])*log10(self.Rex-0.301)**(self.C_dKQ[i][6]);
        return self.KQ+self.deltaKQ

## --- python first line reads here, if compiled from this code
if __name__=='__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print ('Finished!')
