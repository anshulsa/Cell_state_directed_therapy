# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:18:25 2022

@author: anshulsa
"""

from numpy import genfromtxt
import numpy as np
import math
import matplotlib.pyplot as plt
from tmz_model_annabelle_oral_tmz import Oral_TMZ
from basal_tmz_model_ode import basal_model_ode
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.integrate as spi
import time

start = time.time()       ## Code computation timer

def basal_tmz_model():
    
    ##  TMZ Dosing ##
    
    TMZ_Dose = 334.6;#334.6;   #%in microM  %334.6;
    TimeOfDoses = [24, 48, 72, 96, 120];#[40, 64, 88, 112, 136] #[24, 48, 72, 96, 120];    #%Time in hours of doses of TMZ
    th = 150;        # time of simulation(s) in hours
    div = 30;        # time steps in seconds at which function is calculated 
    hr = 3600        #  factor used in the program for second to hour conversion
    
    meth_output = Oral_TMZ(th, div, TMZ_Dose, TimeOfDoses);    #   Methylating cation in Micro molar
    simulation_time_hr = np.linspace(0, th, int(th*hr/div)); # simulation time array to create a function
    meth_fun = InterpolatedUnivariateSpline(simulation_time_hr, meth_output, k = 1); # Methylating cation function which will be feed into DDR module
      
    ## Array intilized to store protein values ##
    
    p53combine_total = []
    ATMP_total = []
    ATRac_total = []
    MGMT_total = [];
    DSB_total = [];
    SSB_total = [];
    O6MG_total = [];
    N3MAG_total = [];
    GT_total = [];
    Chk1_total = [];
    Chk1ac_total = [];
    Chk2_total = [];
    Chk2ac_total = [];
    Cdc25_total = [];
    Nbs1_total = [];
    CyA_total  = [];
    Aa_total = [];
    C9_total = [];
    X_total = [];
    C3_total =[];
    C3a_total =[];
    C9a_total =[];
    BAXa_total =[];
    SMACa_total =[];
    parp_total =[];
    C3_parp_total =[];
    cparp_total =[];
    t_total = [];
 
    #Species initial values 
    
    s0 = np.zeros(64)
    s0[0] = 0.2966386                   # p53inac
    s0[1] = 0.006245528                 # p53ac
    s0[2] = 0.2056385                   # Mdm2
    s0[3] = 0.002230736                 # Wip2
    s0[4] = 0.0                         # ATMP
    s0[5] = 0.0                         # ATRac
    s0[6] = 0.006245528                 # Mdm2product1
    s0[7] = 0.006245528                 # Mdm2product2
    s0[8] = 0.006245528                 # Mdm2product3
    s0[9] = 0.006245528                 # Mdm2product4
    s0[10] = 0.006245528                # Mdm2product5
    s0[11] = 0.006245528                # Mdm2product6
    s0[12] = 0.006245528                # Mdm2product7
    s0[13] = 0.006245528                # Mdm2product8
    s0[14] = 0.006245528                # Mdm2product9
    s0[15] = 0.006245528                # Mdm2pro
    s0[16] = 0.006245528                # Wip1product1
    s0[17] = 0.006245528                # Wip1product2
    s0[18] = 0.006245528                # Wip1product3
    s0[19] = 0.006245528                # Wip1product4
    s0[20] = 0.006245528                # Wip1product5
    s0[21] = 0.006245528                # Wip1product6
    s0[22] = 0.006245528                # Wip1product7
    s0[23] = 0.006245528                # Wip1product8
    s0[24] = 0.006245528                # Wip1product9
    s0[25] = 0.006245528                # Wip1pro
    s0[26] = 0.00386                    # MGMT
    s0[27] = 0                          # damageDSB
    s0[28] = 0                          # damageSSb
    s0[29] = 0.0                        # ppAKT_Mdm2
    s0[30] = 0.0                        # pMdm2
    s0[31] = 0.0                        # ARF
    s0[32] = 2.721609*(10**-5)          # MDM4
    s0[33] = 0.0                        # p53ac_MDM4
    s0[34] = 0.001503781                # ATMinac
    s0[35] = 0.002400357                # ATRinac   
    s0[36] = 0.0                        # O6MG
    s0[37] = 0.00                       # N3MAG
    s0[38] = 0.00                       # GT
    s0[39] = 0.1                        # Chk1
    s0[40] = 0.00                       # Chk1ac
    s0[41] = 0.1                        # Chk2
    s0[42] = 0.00                       # Chk2ac
    s0[43] = 0.1                        # Cdc25
    s0[44] = 0.00                       # Nbs1
    s0[45] = 0                          # A*
    s0[46] = 20                         # C9
    s0[47] = 0                          # C9X
    s0[48] = 40                         # X
    s0[49] = 0                          # A*C9X
    s0[50] = 0                          # A*C9
    s0[51] = 200                        # C3
    s0[52] = 0                          # C3*
    s0[53] = 0                          # C3*X
    s0[54] = 0                          # C9*X
    s0[55] = 0                          # C9*
    s0[56] = 0                          # A*C9*
    s0[57] = 0                          # A*C9*X
    s0[58] = 0                          # BAXa
    s0[59] = 0                          # SMACa   
    s0[60] = 0                          # SMACaX   
    s0[61] = 6.40*10**(2)               # parp
    s0[62] = 2.75*10**(-6)              # C3_parp   
    s0[63] = 1.1                        # cparp  
    
    
    t = np.linspace(0, 80, 1000);                                 # Simulation time 
    y = spi.odeint(basal_model_ode, s0, t, args=(meth_fun,));      # ODE solver with Methlylating cation as argument
    
    
    ## Storing values in the array ##
        
    for j in range(1, len(t)):
         
         p53combine_total.append(y[j, 1])
         ATMP_total.append (  y[j, 4])        
         ATRac_total.append( y[j, 5])
         MGMT_total.append(  y[j, 26])
         DSB_total.append(   y[j, 27])
         SSB_total.append(   y[j, 28])
         O6MG_total.append(  y[j, 36])      
         N3MAG_total.append( y[j, 37])
         GT_total.append(    y[j, 38])
         Chk1_total.append(  y[j, 39])
         Chk1ac_total.append(y[j, 40])
         Chk2_total.append(  y[j, 41])
         Chk2ac_total.append(y[j, 42])
         Cdc25_total.append( y[j, 43])
         Nbs1_total.append(  y[j, 44])
         Aa_total.append(    y[j, 45]);
         C9_total.append(    y[j, 46]);
         X_total.append(     y[j, 48]);
         C3_total.append(    y[j, 51]);
         C3a_total.append(   y[j, 52]);
         C9a_total.append(   y[j, 55]);
         BAXa_total.append(  y[j, 58]);
         SMACa_total.append( y[j, 59])
         parp_total.append(  y[j, 61]);
         C3_parp_total.append(y[j, 62]);
         cparp_total.append( y[j, 63])
         t_total.append(t[j])
       
        
         ## CyA  paramters and expression ## Need to be evaluated separately
         omega=2*math.pi/24;
         M=1;
         A=1;
         phi=15;
         K_cdc25 = 0.001
         K_nbs1 =  0.01    
         n_cya = 20
         CyA = (M + A*math.cos(omega*(t[j]- phi)))*((y[j, 43])**n_cya/(K_cdc25**n_cya + (y[j, 43])**n_cya))*(K_nbs1**n_cya/(K_nbs1**n_cya + (y[j, 44])**n_cya))
         CyA_total.append(CyA)
        
    ## Coverting All quantities in Nanomolar for plotting ##
        
    
    t_total_hr = [i for i in t_total];     
    O6MG_total_nm = [i*1000 for i in O6MG_total]            #Convert to nM
    GT_total_nm = [i*1000 for i in GT_total]                #Convert to nM
    MGMT_total_nm = [i*1000 for i in MGMT_total]            #Convert to nM
    SSB_total_nm =  [i*1000 for i in SSB_total]             #Convert to nM
    DSB_total_nm =  [i*1000 for i in DSB_total]             #Convert to nM
    ATMP_total_nm = [i*1000 for i in ATMP_total]            #Convert to nM
    ATRac_total_nm = [i*1000 for i in ATRac_total]          #Convert to nM
    Cdc25_total_nm =[i*1000 for i in Cdc25_total]           #Convert to nM
    Nbs1_total_nm = [i*1000 for i in Nbs1_total]            #Convert to nM
    Chk1_total_nm = [i*1000 for i in Chk1_total]            #Convert to nM
    Chk1ac_total_nm=[i*1000 for i in Chk1ac_total]          #Convert to nM
    Chk2_total_nm = [i*1000 for i in Chk2_total]            #Convert to nM
    Chk2ac_total_nm = [i*1000 for i in Chk2ac_total]        #Convert to nM
    p53combine_total_nm = [i*1000 for i in p53combine_total]#Convert to nM
    CyA_total_nm =  [i for i in CyA_total]                  #Convert to nM
    SMACa_total_nm = [i for i in SMACa_total]               #Convert to nM
    BAXa_total_nm = [i for i in BAXa_total]                 #Convert to nM
    Aa_total_nm = [i for i in Aa_total]                     #Convert to nM
    C9_total_nm = [i for i in C9_total]                     #Convert to nM
    X_total_nm = [i for i in X_total]                       #Convert to nM
    C3_total_nm = [i for i in C3_total]                     #Convert to nM
    C3a_total_nm = [i for i in C3a_total]                   #Convert to nM
    parp_total_nm = [i for i in parp_total]                 #Convert to nM
    C3_parp_total_nm = [i for i in C3_parp_total]           #Convert to nM
    cparp_total_nm = [i for i in cparp_total]               #Convert to nM
    N3MAG_total_nm = [i*1000 for i in N3MAG_total]               #Convert to nM
    
    
    ## Plot ##
    
    
    fig, axs = plt.subplots(6, 4, figsize=(11, 10))
    axs[0, 0].plot(t_total_hr, O6MG_total_nm)
    axs[0, 0].set_title('O6mG (nm)')
    axs[0, 1].plot(t_total_hr, MGMT_total_nm)
    axs[0, 1].set_title('MGMT protein (nm)')
    axs[0, 2].plot(t_total_hr, SSB_total_nm)
    axs[0, 2].set_title('SSB (nm)')
    axs[0, 3].plot(t_total_hr, DSB_total_nm)
    axs[0, 3].set_title('DSB (nm)')
    axs[1, 0].plot(t_total_hr, ATMP_total_nm)
    axs[1, 0].set_title('ATMP (nm)')
    axs[1, 1].plot(t_total_hr, GT_total_nm)
   # axs[1, 1].plot(t_total_hr, O6MG_total_nm)
    axs[1, 1].set_title('GT (nm)')
    axs[1, 2].plot(t_total_hr, ATRac_total_nm)
    axs[1, 2].set_title('ATRac (nm)')
    axs[1, 3].plot(t_total_hr, CyA_total_nm)
    axs[1, 3].set_title('CyA (nm)')
    axs[2, 0].plot(t_total_hr, Chk1_total_nm)
    axs[2, 0].set_title('Chk1 (nm)')
    axs[2, 1].plot(t_total_hr, Chk1ac_total_nm)
    axs[2, 1].set_title('Chk1 ac (nm)')
    axs[2, 2].plot(t_total_hr, Chk2_total_nm)
    axs[2, 2].set_title('Chk2 (nm)')
    axs[2, 3].plot(t_total_hr, Chk2ac_total_nm)
    axs[2, 3].set_title('Chk2 ac (nm)')
    axs[3, 0].plot(t_total_hr, p53combine_total_nm)
    axs[3, 0].set_title('P53 total (nm)')
    axs[3, 1].plot(t_total_hr, SMACa_total_nm)
    axs[3, 1].set_title('SMAC* (nm)')
    axs[3, 2].plot(t_total_hr, BAXa_total_nm)
    axs[3, 2].set_title('BAX* (nm)')
    axs[3, 3].plot(t_total_hr, Aa_total_nm)
    axs[3, 3].set_title('A* (nm)')
    axs[4, 0].plot(t_total_hr, C9_total_nm)
    axs[4, 0].set_title('C9 (nm)')
    axs[4, 1].plot(t_total_hr, X_total_nm)
    axs[4, 1].set_title('X (nm)')
    axs[4, 2].plot(t_total_hr, C3_total_nm)
    axs[4, 2].set_title('C3 (nm)')
    axs[4, 3].plot(t_total_hr, C3a_total_nm)
    axs[4, 3].set_title('C3* (nm)')
    axs[5, 0].plot(t_total_hr, parp_total_nm)
    axs[5, 0].set_title('parp (nm)')
    axs[5, 1].plot(t_total_hr, C3_parp_total_nm)
    axs[5, 1].set_title('C3_parp (nm)')
    axs[5, 2].plot(t_total_hr, cparp_total_nm)
    axs[5, 2].set_title('cparp (nm)')
    axs[5, 3].plot(t_total_hr, N3MAG_total_nm)
    axs[5, 3].set_title('   N3 adduct (nm)')
    fig.suptitle('TMZ = 150')
    plt.rcParams['font.size'] = 8

    fig.tight_layout()
    
    return y   


stoichometric_mat_basal = genfromtxt('SM_basal_model.csv', delimiter=',') ## Stoichometric matrix for printing

## Running the model ##

test = basal_tmz_model()    

end = time.time()
print(end - start)