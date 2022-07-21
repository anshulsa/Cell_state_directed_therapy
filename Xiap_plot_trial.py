# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 21:28:28 2022

@author: anshulsa
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from tmz_model_annabelle_oral_tmz import Oral_TMZ
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.integrate as spi
from epigenetic_module_file import epigenetic_module
import time
from numpy import genfromtxt

start = time.time()


stoichometric_mat = genfromtxt('SM_epigenetic_tmz_model_new.csv', delimiter=',')



def TMZ_xiap_epigen_ode(s, t, epi_state, tmz_fun, sw):
    
    epi_value1 = epi_state[0]
    epi_value2 = epi_state[1]
    epi_value3 = epi_state[2]
 
    cf = 3600
 
    #dS = np.zeros(38)
    vD = np.zeros((126, 1)) 
    kA = np.zeros(52)
    kD = np.zeros(60)
    kC = np.zeros(10)
    kM = np.zeros(27)
    dS = np.zeros(67)
    
    n = 20
    
    
        ### Rate constants ###
    
    # All units are in Micro molar and Hrs except apoptosis module where units are Nanomolar and Hrs
    
    kD[1]=0.9;                # bp
    kD[2]=5;                  # ampi
    kD[3]=2;                  # api
    kD[4]=10;                 # bsp
    kD[5]=4;                  # ns 
    kD[6]=1;                  # Ts
    kD[7]=1.4;                # ampa
    kD[8]=0.14;               # awpa
    kD[9]=0.9;                # bm
    kD[10]=0.2;               # bmi
    kD[11]=1;                 # am
    kD[12]=0.5;               # asm
    kD[13]=0.25;              # bw
    kD[14]=0.7;               # aw
    kD[15]=0.1;               # asm2
    kD[16]=50;                # aws
    kD[17]=4;                 # nw
    kD[18]=0.2;               # Tw
    kD[19]=7.5;               # as1
    kD[20]=10;                # bs
    kD[21]=4.5;               # bs2    
    kD[22]=0.033948;          # tau1
    kD[23]=0.058614;          # tau2
    kD[24]=0.0035778;         # tau3
    kD[25]=0.33705;           # tau4
    kD[26]=0.013496;          # tau5
    kD[27]=0.030939;          # tau6
    kD[28]=0.0042462;         # tau7
    kD[29]=0.081109;          # tau8
    kD[30]=0.10093;           # tau9
    kD[31]=0.13295;           # tau10
    kD[32]=0.080377;          # tau11
    kD[33]=0.12247;           # tau12
    kD[34]=0.16522;           # tau13
    kD[35]=0.099235;          # tau14
    kD[36]=0.13629;           # tau15
    kD[37]=0.28505;           # tau16
    kD[38]=0.10575;           # tau17
    kD[39]=0.066299;          # tau18
    kD[40]=0.11092;           # tau19
    kD[41]=0.042434;          # tau20
    kD[42]=0;
    kD[43]=0;
    kD[44]=0;
    kD[45]=0;
    kD[46]=0.0005;              # basalp53act    
    kD[47]=20;                  # kDnSS
    kD[48]=20;                  # kDnDS
    kD[49]=2*10**(-6);          # kDkmSS
    kD[50]=2*10**(-6);          # kDkmDS    
    kD[51]= 1;                  # k_gt
    kD[52]= 0.01;               # k_mmr
    kD[53]= 0.1;                # k_dsb
    kD[54]= 0.01                # k_hr
    kD[55]= 0.09*1.81;          # k_N3add
    kD[56]= 10**6               # k_ber
    kD[57]= 23.4*10**3                           # k_o6_mgmt   MGMT o6 rate constant
    kD[58] = 0.05*1.81;                          # k_O6add     hr-1, DNA adduct formation rate constant
    kD[59] = 0.0041;                             # l_O6add     hr-1, DNA adduct degradation rate constant
       
    kC[1] = 0.1              # k_chk1
    kC[2] = 0.02             # K_chk1
    kC[3] = 0.1              # k_chk2
    kC[4] = 0.02             # K_chk2
    kC[5] = 50               # k_cdc25
    kC[6] = 0.001            # K_cdc25
    kC[7] = 0.05             # k_nbs1
    kC[8] = 0.01             # K_nbs1
    
    kA[1] =  cf*2*10**(-3)       #k1      # In Legewie model
    kA[2] =  cf*0.1              #k1r     # In Legewie model
    kA[3] =  cf*5*10**(-6)       #k2      # In Legewie model
    kA[4] =  cf*3.5*10**(-4)     #k3      # In Legewie model
    kA[5] =  cf*2*10**(-4)       #k4      # In Legewie model
    kA[6] =  cf*2*10**(-4)       #k5      # In Legewie model
    kA[7] =  cf*5*10**(-5)       #k6      # In Legewie model
    kA[8] =  cf*3.5*10**(-3)     #k7      # In Legewie model
    kA[9] =  cf*2*10**(-3)       #k8      # In Legewie model
    kA[10] = cf*0.1              #k8r     # In Legewie model
    kA[11] = cf*10**(-3)         #k9      # In Legewie model
    kA[12] = cf*10**(-3)         #k9r     # In Legewie model     
    kA[13] = cf*10**(-3)         #k10     # In Legewie model
    kA[14] = cf*10**(-3)         #k10r    # In Legewie model
    kA[15] = cf*10**(-3)         #k11     # In Legewie model
    kA[16] = cf*10**(-3)         #k11r    # In Legewie model
    kA[17] = cf*10**(-3)         #k12     # In Legewie model
    kA[18] = cf*10**(-3)         #k12r    # In Legewie model
    kA[19] = cf*2*10**(-3)       #k13     # In Legewie model
    kA[20] = cf*0.1              #k13r    # In Legewie model
    kA[21] = cf*2*10**(-3)       #k14     # In Legewie model
    kA[22] = cf*0.1              #k14r    # In Legewie model
    kA[23] = cf*3*10**(-3)       #k15     # In Legewie model
    kA[24] = cf*10**(-3)         #k15r    # In Legewie model
    kA[25] = cf*10**(-3)         #k16     # In Legewie model
    kA[26] = cf*0*0.02           #k16r    # In Legewie model
    kA[27] = cf*10**(-3)         #k17     # In Legewie model
    kA[28] = cf*0.02             #k17r    # In Legewie model
    kA[29] = cf*10**(-3)         #k18     # In Legewie model
    kA[30] = cf*0.04             #k18r    # In Legewie model
    kA[31] = cf*10**(-3)         #k19     # In Legewie model
    kA[32] = cf*10**(-3)         #k20     # In Legewie model
    kA[33] = cf*10**(-3)         #k21     # In Legewie model
    kA[34] = cf*10**(-3)         #k22     # In Legewie model
    kA[35] = cf*0.2              #k22r    # In Legewie model
    kA[36] = cf*10**(-3)         #k23     # In Legewie model
    kA[37] = cf*10**(-3)         #k24     # In Legewie model
    kA[38] = cf*10**(-3)         #k25     # In Legewie model
    kA[39] = cf*10**(-3)         #k26     # In Legewie model
    kA[40] = cf*10**(-3)         #k27     # In Legewie model
    kA[41] = cf*10**(-3)         #k28     # In Leewie model
    kA[42] = cf*10**(-3)         #k1      # In Fey model
    kA[43] = cf*5*10**(-3)       #k2      # In Fey model
    kA[44] = cf*10**(-3)         #k3      # In Fey model 
    kA[45] = cf*10**(-1)         #k19     # In Fey model  
    kA[46] = cf*10**(-1)         #k20     # In Fey model
    kA[47] = cf*10**(-1)         #k21     # In Fey model
    kA[48] = cf*10**(-3)         #k22     # In Fey model
    kA[49] = 0.0031620749        #k_c3_parp   
    kA[50] = 10**(-2)            #k_c3_parpr
    kA[51] = 1                   #k_cparp
    
 
   
    
   # All units are in Micro molar
    #k_tc_mpc =   0.63   
    #k_tc_um = k_tc_mpc*0.00031624803333254277/1000    
    
    kM[1]= 1                                            # k           constant in the hill function    
    ## MGMT ##
    k_mgmt_max = 100
    kM[2]= k_mgmt_max*(0.00031624803333254277/1000);         # k_p0        Max protein value
    kM[3]= 1.878513317995304e-07 #k_tc_um               # k_tcr       transcription constant
    kM[4]= 0.05402550121277828 #math.log(2)/(12.83);    # l_mr        mRNA Degradation rate 
    kM[5]= 120.08;                                      # k_tln       translation constant   
    kM[6]= 0.020173084416762087#math.log(2)/(34.36);    # l_mp        protein degradation rate
    
 
    ## MDM2 ##
    k_MDM2_max = 100
    kM[7]= k_MDM2_max*(0.00031624803333254277/1000);           # k_p0        Max protein value
    kM[8]= 0.00019307574931018399   #k_tc_um                 # k_tcr       transcription constant
    kM[9]= 0.21660849392498288      #math.log(2)/(3.2);     # l_mr         mRNA Degradation rate 
    kM[10]= 11.51;                                             # k_tln       translation constant   
    kM[11]= 0.9366853791350612    #math.log(2)/(0.74);    # l_mp          protein degradation rate   
   
    ## XIAP ##
    k_X_max = 100
    kM[12]= k_X_max*(0.00031624803333254277);         # k_p0        Max protein value
    kM[13]= 8000*8.334849742653165e-05 #k_tc_um               # k_tcr       transcription constant
    kM[14]= 10*0.0333243836807666 #math.log(2)/(2.08);    # l_mr        P53 mRNA Degradation rate 
    kM[15]= 10*7.61;                                      # k_tln       translation constant   
    kM[16]= 4#3.6#math.log(2)/(6.74);    # l_mp        P53 protein degradation rate
    
    ##   Assigning rate constant values stored in the arrays (kD, kA, kC, kM)
    ##   to the actual values to be used in ODEs
    
    
    bp=      kD[1];
    ampi=    kD[2];
    api=     kD[3];
    bsp=     kD[4];
    ns=      kD[5];
    Ts=      kD[6];
    ampa=    kD[7];
    awpa=    kD[8];
    bm=      kD[9];
    bmi=     kD[10];
    am=      kD[11];
    asm=     kD[12];
    bw=      kD[13];
    aw=      kD[14];
    asm2=    kD[15];
    aws=     kD[16];
    nw=      kD[17];
    Tw=      kD[18];
    as1=     kD[19];
    bs=      kD[20];
    bs2=     kD[21];
    tau1=    kD[22];
    tau2=    kD[23];
    tau3=    kD[24];
    tau4=    kD[25];
    tau5=    kD[26];
    tau6=    kD[27];
    tau7=    kD[28];
    tau8=    kD[29];
    tau9=    kD[30];
    tau10=   kD[31];
    tau11=   kD[32];
    tau12=   kD[33];
    tau13=   kD[34];
    tau14=   kD[35];
    tau15=   kD[36];
    tau16=   kD[37];
    tau17=   kD[38];
    tau18=   kD[39];
    tau19=   kD[40];
    tau20=   kD[41];
    kD[42]=  kD[42];
    kD[43]=  kD[43];
    kD[44]=  kD[44];
    kD[45]=  kD[45];
    basalp53act=kD[46];
    kDnSS=   kD[47]; 
    kDnDS=   kD[48]; 
    kDkmSS=  kD[49]; 
    kDkmDS=  kD[50];     
    k_gt  =  kD[51]
    k_mmr =  kD[52]
    k_dsb =  kD[53]
    k_hr =   kD[54]
    k_N3add= kD[55]
    k_ber=   kD[56]         
    k_O6_mgmt = kD[57];
    k_O6add   = kD[58]
    l_O6off   = kD[59]   
       
    k_chk1 =  kC[1]
    K_chk1 =  kC[2]
    k_chk2 =  kC[3]
    K_chk2 =  kC[4]
    k_cdc25 = kC[5]
    K_cdc25 = kC[6]
    k_nbs1 =  kC[7] 
    K_nbs1 =  kC[8]    
    
    k1  = kA[1]
    k1r = kA[2]
    k2  = kA[3]
    k3  = kA[4]
    k4  = kA[5]
    k5  = kA[6]
    k6  = kA[7]
    k7  = kA[8]
    k8  = kA[9]
    k8r = kA[10]
    k9  = kA[11]
    k9r = kA[12]
    k10 = kA[13]
    k10r= kA[14]
    k11 = kA[15]
    k11r= kA[16]
    k12 = kA[17]
    k12r= kA[18]
    k13 = kA[19]
    k13r= kA[20]
    k14 = kA[21]
    k14r= kA[22]
    k15 = kA[23]
    k15r= kA[24]
    k16 = kA[25]
    k16r= kA[26]
    k17 = kA[27]
    k17r= kA[28]
    k18 = kA[29]
    k18r= kA[30]
    k19 = kA[31]
    k20 = kA[32]
    k21 = kA[33]
    k22 = kA[34]
    k22r= kA[35]
    k23 = kA[36]
    k24 = kA[37]
    k25 = kA[38]
    k26 = kA[39]
    k27 = kA[40]
    k28 = kA[41]
    k29  =  kA[42]
    k30 =   kA[43]
    k31  =  kA[44]  
    k32=   kA[45]
    k33 =  kA[46]
    k34=   kA[47]
    k35 =  kA[48]        
    k_c3_parp = kA[49]
    k_c3_parpr = kA[50]
    k_cparp = kA[51]
        
    
    kh =     kM[1];
    k_p01 =  kM[2];  
    k_tcr1 = kM[3];
    l_mr1 =  kM[4];
    k_tln1 = kM[5];
    l_mp1 =  kM[6];   
    k_p02 =  kM[7];  
    k_tcr2 = kM[8];
    l_mr2 =  kM[9];
    k_tln2 = kM[10];
    l_mp2 =  kM[11];    
    k_p03 =  kM[12];  
    k_tcr3 = kM[13];
    l_mr3 =  kM[14];
    k_tln3 = kM[15];
    l_mp3 =  kM[16];  
    

    #### Species (Proteins, DNA adducts and protein complexes)  ###
    
    p53inac =    s[0];
    p53ac =      s[1];
    Mdm2 =       s[2];
    Wip1 =       s[3];
    ATMP =       s[4];
    ATRac =      s[5];
    Mdm2product1=s[6];
    Mdm2product2=s[7];
    Mdm2product3=s[8];
    Mdm2product4=s[9];
    Mdm2product5=s[10];
    Mdm2product6=s[11];
    Mdm2product7=s[12];
    Mdm2product8=s[13];
    Mdm2product9=s[14];
    Mdm2pro =    s[15];
    Wip1product1=s[16];
    Wip1product2=s[17];
    Wip1product3=s[18];
    Wip1product4=s[19];
    Wip1product5=s[20];
    Wip1product6=s[21];
    Wip1product7=s[22];
    Wip1product8=s[23];
    Wip1product9=s[24];
    Wip1pro =  s[25];
    MGMT=      s[26];
    damageDSB= s[27];
    damageSSB= s[28];
    ppAKT_Mdm2=s[29];
    pMdm2=     s[30];
    ARF=       s[31];
    MDM4=      s[32];
    p53ac_MDM4=s[33]; 
    ATMinac=  s[34];
    ATRinac=  s[35];
    O6MG =    s[36];   
    N3MAG =   s[37];    
    GT =      s[38];
    Chk1 =    s[39];
    Chk1ac =  s[40];
    Chk2 =    s[41];
    Chk2ac =  s[42];
    Cdc25 =   s[43];
    Nbs1 =    s[44];   
    Aa =      s[45];
    C9 =      s[46];
    C9X =     s[47];
    X =       s[48];
    AaC9X =   s[49];
    AaC9 =    s[50];
    C3=       s[51];
    C3a=      s[52];
    C3aX=     s[53];
    C9aX=     s[54];
    C9a=      s[55];
    AaC9a=    s[56];
    AaC9aX=   s[57];
    BAXa =    s[58]
    SMACa =   s[59]
    SMACaX =  s[60]    
    parp =    s[61]
    C3_parp = s[62]
    cparp =   s[63]    
    MGMT_r =    s[64];
    Mdm2_r =    s[65];
    X_r =    s[66];
  
    sw1 = sw[0]
    sw2 = sw[1]  
    sw3 = sw[2]
 
    
     
    ## BAX and Apaf total defined ##
    
    BAXtot = 0.5
    Atot = 20
    BAX = BAXtot - BAXa 
    A = Atot - Aa - AaC9 - AaC9X- AaC9a - AaC9aX
    
    ## CyA and its paramaters ## CyA is not part of ODEs has to be evaluated separately
    
    omega=2*math.pi/24;
    M=1;
    A_CyA=1;
    phi=15;
    K_cya = 0.3*A_CyA+M;
    n_cya = 20
    K_cdc25 = 0.001
    K_nbs1 =  0.01    
    CyA = (M + A_CyA*math.cos(omega*(t-phi)))*(Cdc25**n_cya/(K_cdc25**n_cya + Cdc25**n_cya))*(K_nbs1**n_cya/(K_nbs1**n_cya + Nbs1**n_cya))
    
    
    ### Reactions ###
    
    # DNA Damage 
    
    vD[0] = bp;
    vD[1] = ampi*Mdm2*p53inac; 
    vD[2] = basalp53act + bsp*p53inac*(ATMP**ns/(ATMP**ns+Ts**ns)+ATRac**ns/(ATRac**ns+Ts**ns)); 
    vD[3] = awpa*Wip1*p53ac;
    vD[4] = api*p53inac; 
    vD[5] = ampa*Mdm2*p53ac; 
    vD[6] = bm*Mdm2pro;
    vD[7] = bmi;
    vD[8] = asm*Mdm2*ATMP; 
    vD[9] = asm2*Mdm2*ATRac; 
    vD[10] = am*Mdm2;
    vD[11] = bw*Wip1pro;
    vD[12] = aw*Wip1; 
    vD[13] = bs*((damageDSB**kDnDS)/((kDkmDS**kDnDS)+(damageDSB**kDnDS))); 
    vD[14] = aws*ATMP*Wip1**nw/(Wip1**nw+Tw**nw); 
    vD[15] = as1*ATMP; 
    vD[16] = bs2*((damageSSB**kDnSS)/((kDkmSS**kDnSS)+(damageSSB**kDnSS)));
    vD[17] = as1*ATRac; 
    vD[18] = Mdm2product1/tau1;
    vD[19] = p53ac/tau1;
    vD[20] = Mdm2product2/tau2;
    vD[21] = Mdm2product1/tau2;
    vD[22] = Mdm2product3/tau3;
    vD[23] = Mdm2product2/tau3;
    vD[24] = Mdm2product4/tau4;
    vD[25] = Mdm2product3/tau4;
    vD[26] = Mdm2product5/tau5;
    vD[27] = Mdm2product4/tau5;
    vD[28] = Mdm2product6/tau6;
    vD[29] = Mdm2product5/tau6;
    vD[30] = Mdm2product7/tau7;
    vD[31] = Mdm2product6/tau7;
    vD[32] = Mdm2product8/tau8;
    vD[33] = Mdm2product7/tau8;
    vD[34] = Mdm2product9/tau9;
    vD[35] = Mdm2product8/tau9;
    vD[36] = Mdm2pro/tau10;
    vD[37] = Mdm2product9/tau10;
    vD[38] = Wip1product1/tau11;
    vD[39] = p53ac/tau11;
    vD[40] = Wip1product2/tau12;
    vD[41] = Wip1product1/tau12;
    vD[42] = Wip1product3/tau13;
    vD[43] = Wip1product2/tau13;
    vD[44] = Wip1product4/tau14;
    vD[45] = Wip1product3/tau14;
    vD[46] = Wip1product5/tau15;
    vD[47] = Wip1product4/tau15;
    vD[48] = Wip1product6/tau16;
    vD[49] = Wip1product5/tau16;
    vD[50] = Wip1product7/tau17;
    vD[51] = Wip1product6/tau17;
    vD[52] = Wip1product8/tau18;
    vD[53] = Wip1product7/tau18;
    vD[54] = Wip1product9/tau19;
    vD[55] = Wip1product8/tau19;
    vD[56] = Wip1pro/tau20;
    vD[57] = Wip1product9/tau20;
    vD[58] = kD[42]*ARF*Mdm2;
    vD[59] = kD[43]*ARF*pMdm2;
    vD[60] = kD[44]*MDM4*p53ac;
    vD[61] = kD[45]*p53ac_MDM4;
    vD[62] = k_O6_mgmt*MGMT*O6MG                            
    vD[63] = k_O6add*tmz_fun(t)
    vD[64] = l_O6off*O6MG        
    vD[65] = k_gt*O6MG*(CyA**n/(K_cya**n + CyA**n))
    vD[66] = k_mmr*GT
    vD[67] = k_dsb*damageSSB*(CyA**n/(K_cya**n + CyA**n))
    vD[68] = k_hr*damageDSB
    vD[69] = k_N3add*tmz_fun(t)                             
    vD[70] = k_ber*N3MAG
    vD[71] = k_dsb*N3MAG*(CyA**n/(K_cya**n + CyA**n))

    #   Cell cycle 
  
    vD[72] = k_chk1*Chk1*(ATRac**n/(K_chk1**n + ATRac**n))
    vD[73] = k_chk2*Chk2*(ATMP**n/(K_chk2**n + ATMP**n))
    vD[74] = k_cdc25*Chk1ac*Cdc25
    vD[75] = k_nbs1*Chk2ac
    
    #   Apoptosis 
      
    vD[76] = k1*Aa*C9 - k1r*AaC9
    vD[77] = k2*C3*C9
    vD[78] = k3*C3*AaC9
    vD[79] = k4*C9*C3a
    vD[80] = k5*AaC9*C3a
    vD[81] = k6*C3*C9a
    vD[82] = k7*C3*AaC9a
    vD[83] = k8*C9a*Aa - k8r*AaC9a
    vD[84] = k9*C9*X - k9r*C9X
    vD[85] = k10*AaC9*X - k10r*AaC9X
    vD[86] = k11*C9a*X - k11r*C9aX
    vD[87] = k12*AaC9a*X - k12r*AaC9aX
    vD[88] = k13*C9X*Aa - k13r*AaC9X
    vD[89] = k14*C9aX*Aa - k14r*AaC9aX
    vD[90] = k15*C3a*X - k15r*C3aX
    vD[91] = k16r - k16*Aa
    vD[92] = k17r - k17*C9
    vD[93] = 0*(k18r - k18*X)
    vD[94] = k19*C9X
    vD[95] = k20*AaC9X
    vD[96] = k21*AaC9
    vD[97] = k22r - k22*C3
    vD[98] = k23*C3a
    vD[99] = k24*C3aX
    vD[100] = k25*C9aX
    vD[101] = k26*C9a
    vD[102] = k27*AaC9a
    vD[103] = k28*AaC9aX
    vD[104] = k29*p53ac*BAX
    vD[105] = k30*BAXa
    vD[106] = k31*A*BAXa
    vD[107] = k32*BAXa
    vD[108] = k33*SMACa*X
    vD[109] = k34*SMACaX
    vD[110] = k35*SMACaX       
    vD[111] = k_c3_parp*C3a*parp;
    vD[112] = k_c3_parpr*C3_parp;
    vD[113] = k_cparp*C3_parp;
    
    # protein degradation
    
    ## MGMT transcription translation model
    vD[114] = sw1*(k_tcr1*(kh*MGMT**n/(k_p01**n + kh*MGMT**n))*epi_value1(t));                  # protein feedback on trinscription of active gene 
    vD[115] = sw1*l_mr1*MGMT_r;                                           # mRNA degradation 
    vD[116] = sw1*k_tln1*MGMT_r;                                           # translation      
    vD[117] = sw1*l_mp1*MGMT;     
    ## MDM2 transcription translation model 
    vD[118] = sw2*(k_tcr2*(kh*Mdm2**n/(k_p02**n + kh*Mdm2**n))*epi_value2(t));    # protein feedback on trinscription of active gene 
    vD[119] = sw2*l_mr2*Mdm2_r;                                           # mRNA degradation 
    vD[120] = sw2*k_tln2*Mdm2_r;                                           # translation      
    vD[121] = sw2*l_mp2*Mdm2; 
    ## XIAP transcription translation model
    vD[122] = sw3*k_tcr3*(kh*X**n/(k_p02**n + kh*X**n))*epi_value3(t);                  # protein feedback on trinscription of active gene 
    vD[123] = sw3*l_mr3*X_r;                                           # mRNA degradation 
    vD[124] = sw3*k_tln3*X_r;                                           # translation      
    vD[125] = sw3*l_mp3*X; 

    
    dS1 = np.dot(stoichometric_mat, vD) 
    for i in range(0, 67):
         dS[i] = dS1[i]
         
    return dS

    
    
  
def xiap_plot_fun(tmz_val, xiap_multiplier):
    
    TMZ_Dose = tmz_val;   #%in microM  %334.6;
    TimeOfDoses = [24, 48, 72, 96, 120];    #%Time in hours of doses of TMZ
    th = 150;        #%time of simulation(s) in hours
    div = 30;
    hr = 3600
    
    TMZ_input = Oral_TMZ(th, div, TMZ_Dose, TimeOfDoses);  #nano molar
    simulation_time_hr = np.linspace(0, th, int(th*hr/div));
    tmz_fun = InterpolatedUnivariateSpline(simulation_time_hr, TMZ_input, k = 1);
        
    p53combine_total = [];
    MDM2_total =  [];
    ATMP_total =  [];
    ATRac_total = [];
    MGMT_total =  [];
    DSB_total =   [];
    SSB_total =   [];
    O6MG_total =  [];
    N3MAG_total = [];     
    X_total =     [];
    Xrna_total =  [];
    C3a_total =   []; 
    cparp_total = [];
    CyA_total  =  []; 
    t_total =     [];

    
    
    #Species initial values 
    
    s0 = np.zeros(67)
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
    s0[64] = 0.00000139                 # MGMT mRNA
    s0[65] = 3.478180945913966e-06      # MDM2 mRNA
    s0[66] = 2#0.047389767794881535       # XIap mRNA


        
    k_cdc25 = 50
    K_cdc25 = 0.001
    k_nbs1 =  0.05
    K_nbs1 =  0.01    
        
    y0 = s0;
        
    omega=2*math.pi/24;
    M=1;
    A=1;
    phi=15;
    K_cya = 0.3*A+M;
    
   
    t = np.linspace(0, 150, 1000);
                    
    zero_mat  = np.full((1000), 0)
    time_array = np.linspace(0, 150, 1000)
    zero_fun  = InterpolatedUnivariateSpline(time_array, zero_mat, k = 1) 
    trans_state = [0, 0, 1]   # [MGMT, MDM2, X]
    epi_state =   [zero_fun, zero_fun,  xiap_multiplier]   # 
    
    y = spi.odeint(TMZ_xiap_epigen_ode, y0, t, args=(epi_state, tmz_fun, trans_state,));
            
    for j in range(0, len(t)):
         
         p53combine_total.append(y[j, 1])
         MDM2_total.append(y[j, 2])
         ATMP_total.append (  y[j, 4])        
         ATRac_total.append( y[j, 5])
         MGMT_total.append(  y[j, 26])
         DSB_total.append(   y[j, 27])
         SSB_total.append(   y[j, 28])
         O6MG_total.append(  y[j, 36])      
         N3MAG_total.append( y[j, 37])   
         X_total.append(     y[j, 48]);
         C3a_total.append(   y[j, 52]);  
         cparp_total.append( y[j, 63])
         Xrna_total.append(y[j, 66]);

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
   
        
        
    t_total_hr = [i for i in t_total];     
    O6MG_total_nm = [i*1000 for i in O6MG_total]              # Convert to nM
    MGMT_total_nm = [i*1000 for i in MGMT_total]              # Convert to nM   
    SSB_total_nm = [i*1000 for i in SSB_total]                # Convert to nM
    DSB_total_nm = [i*1000 for i in DSB_total]                # Convert to nM
    MDM2_total_nm = [i*1000 for i in MDM2_total]              # Convert to nM
    ATMP_total_nm = [i*1000 for i in ATMP_total]              # Convert to nM
    ATRac_total_nm = [i*1000 for i in ATRac_total]            # Convert to nM  
    p53combine_total_nm = [i*1000 for i in p53combine_total]  # Convert to nM
    CyA_total_nm = [i for i in CyA_total]                     # Convert to nM 
    X_total_nm = [i for i in X_total]                         # Convert to nM
    C3a_total_nm = [i for i in C3a_total]                     # Convert to nM
    cparp_total_nm = [i for i in cparp_total]                 # Convert to nM
    Xrna_total_nm = [i for i in Xrna_total] #%Convert to nM

  
    fig, axs = plt.subplots(4, 3, figsize=(11, 10))
    axs[0, 0].plot(t_total_hr, O6MG_total_nm)
    axs[0, 0].set_title('O6mG (nm)')
    axs[0, 1].plot(t_total_hr, MGMT_total_nm)
    axs[0, 1].set_title('MGMT protein (nm)')
    axs[0, 2].plot(t_total_hr, SSB_total_nm)
    axs[0, 2].set_title('SSB (nm)')
    axs[1, 0].plot(t_total_hr, DSB_total_nm)
    axs[1, 0].set_title('DSB (nm)')
    axs[1, 1].plot(t_total_hr, ATMP_total_nm)
    axs[1, 1].set_title('ATMP (nm)')
    axs[1, 2].plot(t_total_hr, ATRac_total_nm)
    axs[1, 2].set_title('ATRac (nm)')
    axs[2, 0].plot(t_total_hr, CyA_total_nm)
    axs[2, 0].set_title('CyA (nm)')
    axs[2, 1].plot(t_total_hr, p53combine_total_nm)
    axs[2, 1].set_title('p53 (hrs)')
    axs[2, 2].plot(t_total_hr, X_total_nm)
    axs[2, 2].set_title('X (nm)') 
    axs[2, 2].set_ylim(0, 45) 

    axs[3, 0].plot(t_total_hr, C3a_total_nm)
    axs[3, 0].set_title('C3* (nm)')
    axs[3, 1].plot(t_total_hr, cparp_total_nm)
    axs[3, 1].set_ylim(0, 700)
    axs[3, 1].set_title('cparp (nm)') 
    axs[3, 2].plot(t_total_hr,Xrna_total_nm)
    axs[3, 2].set_title('X rna (nm)')
        
    fig.suptitle('TMZ = 150 mg/m2')
    fig.tight_layout()
    
    
    return [t_total_hr, O6MG_total_nm,
          SSB_total_nm, 
          X_total_nm, 
          Xrna_total_nm, 
          p53combine_total_nm,  
          cparp_total_nm]  
    
epigen_mat = [i*1.5 for i in epigenetic_module(150)]  
resgen_mat = np.full((100), 1)
pergen_mat  = np.full((100), 1.5)
time_array = np.linspace(0, 150, 100)

epigen_fun  = InterpolatedUnivariateSpline(time_array, epigen_mat, k = 1)   
resgen_fun  = InterpolatedUnivariateSpline(time_array, resgen_mat, k = 1) 
pergen_fun  = InterpolatedUnivariateSpline(time_array, pergen_mat, k = 1)   
 
## TMZ QSP figure #######
mdm2_multi_array = [resgen_fun, pergen_fun, epigen_fun]

#mdm2_multi_array = [1, 2, 3, 3]
tmz_array = [334.6]
model_output_array = []
for j in mdm2_multi_array:
    for i in tmz_array:
        test = xiap_plot_fun(i, j)
        model_output_array.append(test)


fig, axs = plt.subplots(7, 3, figsize=(11, 10))
axs[0, 0].plot(model_output_array[0][0], model_output_array[0][1])
axs[0, 0].set_ylabel('O6mG (nm)')
axs[0, 0].set_title('XIAP transcription (Restrictive)')
axs[0, 1].plot(model_output_array[0][0], model_output_array[1][1])
axs[0, 1].set_title('XIAP transcription (Permissive)')
axs[0, 2].plot(model_output_array[0][0], model_output_array[2][1])
axs[0, 2].set_title('XIAP transcription with EI')

axs[1, 0].plot(model_output_array[0][0], model_output_array[0][2])
axs[1, 0].set_ylabel('SSB (nm)')
axs[1, 1].plot(model_output_array[0][0], model_output_array[1][2])
axs[1, 2].plot(model_output_array[0][0], model_output_array[2][2])

axs[2, 0].plot(time_array,resgen_mat)
axs[2, 0].set_ylim(0, 2.1)
axs[2, 0].set_ylabel('Epigenetic value')
axs[2, 1].plot(time_array,pergen_mat)
axs[2, 1].set_ylim(0, 2.1)
axs[2, 2].plot(time_array,epigen_mat)
axs[2, 2].set_ylim(0, 2.1)

axs[3, 0].plot(model_output_array[0][0], model_output_array[0][5])
#axs[3, 0].set_ylim(0, 45)
axs[3, 0].set_ylabel('P53 active (nm)')
axs[3, 1].plot(model_output_array[0][0], model_output_array[1][5])
axs[3, 2].plot(model_output_array[0][0], model_output_array[2][5])

axs[4, 0].plot(model_output_array[0][0], model_output_array[0][4])
axs[4, 0].set_ylabel('XIAP mRNA (nm)')
axs[4, 1].plot(model_output_array[0][0], model_output_array[1][4])
axs[4, 2].plot(model_output_array[0][0], model_output_array[2][4])


axs[5, 0].plot(model_output_array[0][0], model_output_array[0][3])
axs[5, 0].set_ylabel('XIAP (nm)')
axs[5, 1].plot(model_output_array[0][0], model_output_array[1][3])
axs[5, 2].plot(model_output_array[0][0], model_output_array[2][3])

axs[6, 0].plot(model_output_array[0][0], model_output_array[0][6])
axs[6, 0].set_ylim(0, 700)
axs[6, 0].set_ylabel('cPARP (nm)')
axs[6, 0].set_xlabel('Time (Hrs)')

axs[6, 1].plot(model_output_array[0][0], model_output_array[1][6])
axs[6, 1].set_ylim(0, 700)
axs[6, 1].set_xlabel('Time (Hrs)')

axs[6, 2].plot(model_output_array[0][0], model_output_array[2][6])
axs[6, 2].set_ylim(0, 700)
axs[6, 2].set_xlabel('Time (Hrs)')


#fig.suptitle('TMZ = 20')
fig.tight_layout()

end = time.time()
print(end - start)


