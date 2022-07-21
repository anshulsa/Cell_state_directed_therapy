# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:38:43 2022

@author: anshulsa
"""
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import trapz
import scipy.integrate as spi
from scipy.interpolate import InterpolatedUnivariateSpline
import time
start = time.time()

def cp_fun(t):
    
    dose = 10      #mg
    ka = 0.66
    v_f = 56.1
    cl_f = 7.54    
    t_hl = math.log(2)*v_f/cl_f
    k = math.log(2)/t_hl
    
    cp_val = dose*ka*(math.exp(-k*t)- math.exp(-ka*t))/(v_f*(ka-k)) 
    
    return cp_val

def gene_expression(s, t, cbt):
    
    stoi = [[1, -1, 0, 0],
            [0, 0, 1, -1]]
    
    #dS = np.zeros(38)
    vD = np.zeros((4, 1)) 
    kD = np.zeros(6)
    dS = np.zeros(2)
    
   # mpc2um = 0.00031624803333254277/1000
    #kD **
    kD[1]= 100*0.00031624803333254277/1000;                                         #Max MGMT protein value
    kD[2]= (0.000165*3600)*0.00031624803333254277/1000  #1.50071E-05;                  #transcription constant
    kD[3]= math.log(2)/(12.83);                      #MGMT mRNA Degradation rate 
    kD[4]= 0.03335555*3600#5.60363E-06;                #translation constant   
    kD[5]= math.log(2)/(34.36);                 #MGMT protein degradation rate



    m_r1 = s[0];    #MGMT mRNa
    m_p1 = s[1];    #MGMT protein

    k_p01 = kD[1];      
    k_tc1 = kD[2];
    l_mr1 = kD[3];
    k_tl1 = kD[4];
    l_mp1 = kD[5];
 
    n = 2
    kcbt_50 = 0.001
    vD[0] = k_tc1*(m_p1**n/(k_p01**n + m_p1**n))*(1 - (cbt(t)**n/(kcbt_50**n + cbt(t)**n)));                  # protein feedback on trinscription of active gene 
    vD[1] = l_mr1*m_r1;                                           # mRNA degradation 
    vD[2] = k_tl1*m_r1;                                           # translation      
    vD[3] = l_mp1*m_p1;                                           # protein degradation
    
    dS1 = np.dot(stoi, vD) 

    for i in range(0, 2):
         dS[i] = dS1[i]
    return  dS 



def brd_fun(dose_schedule, c_bt_val):

    t1 = dose_schedule[0]
    t2 = dose_schedule[1]
    t3 = dose_schedule[2]
    t4 = dose_schedule[3]
    t5 = dose_schedule[4]
    
    
    t_brd = np.linspace(0, 150, 1000);
      
    cp_array = []
        
    #print(t1, t2, t3, t4, t5)
    
    for i in t_brd:
        
        if i <= t1:
            cp_temp = cp_fun(i) 
        elif t1 < i < t2:
            cp_temp = cp_fun(i) + cp_fun(i - t1)
        elif t2 < i < t3:
            cp_temp = cp_fun(i) + cp_fun(i - t1) + cp_fun(i - t2)
        elif t3 < i < t4:
            cp_temp = cp_fun(i) + cp_fun(i - t1) + cp_fun(i - t2) + cp_fun(i - t3)
        else: #if 96 < i < 120:
            cp_temp = cp_fun(i) + cp_fun(i - t1) + cp_fun(i - t2) + cp_fun(i - t3) + cp_fun(i - t4)
            
        cp_array.append(cp_temp)
        
    cpd_fun = InterpolatedUnivariateSpline(t_brd, cp_array, k = 1);
    
    cbt_array = [i*c_bt_val for i in cp_array]
    cbt_fun = InterpolatedUnivariateSpline(t_brd, cbt_array, k = 1);
    cbt_fun_array = []
    for i in t_brd:
        cbt_fun_array.append(cbt_fun(i))
    
    AUC_cbt = trapz(cbt_fun_array, t_brd)                   ## AUC  calculation
    #print(AUC_cbt)    
    #plt.plot(t_brd, cbt_fun_array)    
    # plt.plot(t_brd, cbt_array)
    # plt.ylabel('C_bt (ug)')
    # plt.xlabel('t (hrs)')
    # plt.show()    
    
    
    
    
 #   y0 = [0.00000139, 0.00386]
    y0 = [0.0000035, 0.016]

    #y0 = [5, 126.752]
    MGMT_mrna = [];
    MGMT_protein = [];
    cp_out = []
    cbt_out = []
                    
    t = np.linspace(0, 150, 1000)
       
    y = spi.odeint(gene_expression, y0, t, args = (cbt_fun, ));
    
    for j in range(0, len(t)):
        
        MGMT_mrna.append(y[j, 0])
        MGMT_protein.append(y[j, 1])
        cp_out.append(1000*cpd_fun(t[j]))
        cbt_out.append(1000*cbt_fun(t[j]))
    
    AUC_mgmt = trapz(MGMT_protein, t)                   ## AUC  calculation
    #print(AUC_mgmt) 
    eff_ratio = AUC_mgmt/AUC_cbt
    #print(eff_ratio)
    
    # fig, axs = plt.subplots(3, figsize=(11, 10))
    # axs[0].plot(t, cbt_out, label="MGMT")
    # axs[0].set_title('cbt')
    # axs[1].plot(t, MGMT_mrna, label="MGMT")
    # axs[1].set_title('MGMT mRNA (nm)')
    # axs[2].plot(t, MGMT_protein, label="SSB_total")
    # axs[2].set_title('MGMT protein (nm)')
    # fig.tight_layout()
    
    return [eff_ratio, AUC_mgmt, t, cp_out, cbt_out, MGMT_mrna, MGMT_protein]
 
# result_array = [[], []]
   
# for i in range(0, 200):
    
#     x1 = random.uniform(12, 36)
#     x2 = random.uniform(36, 60)
#     x3 = random.uniform(48, 84)
#     x4 = random.uniform(84, 108)
#     x5 = random.uniform(108, 132)
#     dose_random = [x1,x2,x3,x4,x5]   
#     trial_dose =  sorted(dose_random)
    
#     test_run = brd_fun(trial_dose, 0.04)[0]
#     print(i, test_run)
#     result_array[0].append(test_run)
#     result_array[1].append(trial_dose)


# minIndex = min(result_array[0])
# opt_index = 0 

# for i in range(0, len(result_array[0])):
#     if result_array[0][i]== minIndex:
#         opt_index = i
#         print(i)

# print(result_array[1][opt_index])
# #best_run = brd_fun(result_array[1][opt_index])




c_bt_standard = 0.04
c_bt_noval = 0.0

best_run = brd_fun([27, 32, 68, 80, 115], c_bt_standard)
best_run_control = brd_fun([27, 32, 68, 80, 115], c_bt_noval)
#flat_cbt_arr = flatten(best_run[3])

fig, axs = plt.subplots(2, 2, figsize=(14, 9))
axs[0, 0].plot(best_run[2], best_run[3])
axs[0, 0].set_ylabel('$C_{p}$ (µg/L)')
axs[0, 0].set_xlabel('Time (Hrs)')

axs[0, 1].plot(best_run[2], best_run[4], label="MGMT")
axs[0, 1].set_ylabel('$C_{AS}$ (µg/L)')
axs[0, 1].set_xlabel('Time (Hrs)')

axs[1, 0].plot(best_run[2], best_run[5], label="With EI modulation")
axs[1, 0].plot(best_run_control[2], best_run_control[5], label="Control")
axs[1, 0].set_ylabel('MGMT mRNA (µM)')
axs[1, 0].set_xlabel('Time (Hrs)')


axs[1, 1].plot(best_run[2], best_run[6], label="With EI modulation")
axs[1, 1].plot(best_run_control[2], best_run_control[6], label="Control")
axs[1, 1].set_ylabel('MGMT protein (µM)')
axs[1, 1].set_xlabel('Time (Hrs)')
axs[1, 1].legend(loc="upper left")

fig.tight_layout()


# random_run1 = brd_fun(result_array[1][15])
# random_run2 = brd_fun(result_array[1][79])
# random_run3 = brd_fun(result_array[1][132])


# # fig, axs = plt.subplots(4, 3, figsize=(14, 9))
# # axs[0, 0].plot(random_run1[2], random_run1[3])
# # axs[0, 0].set_ylabel('cp (µg/L)')
# # axs[0, 0].set_xlabel('Time (hrs)')
# # axs[0, 1].plot(random_run2[2], random_run2[3])
# # axs[0, 1].set_ylabel('cp (µg/L)')
# # axs[0, 1].set_xlabel('Time (hrs)')
# # axs[0, 2].plot(random_run3[2], random_run3[3])
# # axs[0, 2].set_ylabel('cp (µg/L)')
# # axs[0, 2].set_xlabel('Time (hrs)')


# fig, axs = plt.subplots(4, 3, figsize=(14, 11))
# axs[0, 0].plot(random_run1[2], random_run1[3], label="MGMT")
# axs[0, 0].set_ylabel('Cp (µg/L)')
# axs[0, 0].set_xlabel('Time (hrs)')
# axs[0, 0].set_title('Dosing Schedule 1')
# axs[0, 1].plot(random_run2[2], random_run2[3], label="MGMT")
# axs[0, 1].set_ylabel('Cp (µg/L)')
# axs[0, 1].set_xlabel('Time (hrs)')
# axs[0, 1].set_title('Dosing Schedule 2')
# axs[0, 2].plot(random_run3[2], random_run3[3], label="MGMT")
# axs[0, 2].set_ylabel('Cp (µg/L)')
# axs[0, 2].set_xlabel('Time (hrs)')
# axs[0, 2].set_title('Dosing Schedule 3')

# axs[1, 0].plot(random_run1[2], random_run1[4], label="MGMT")
# axs[1, 0].set_ylabel('Cbt (µg/L)')
# axs[1, 0].set_xlabel('Time (hrs)')
# axs[1, 1].plot(random_run2[2], random_run2[4], label="MGMT")
# axs[1, 1].set_ylabel('Cbt (µg/L)')
# axs[1, 1].set_xlabel('Time (hrs)')
# axs[1, 2].plot(random_run3[2], random_run3[4], label="MGMT")
# axs[1, 2].set_ylabel('cBt (µg/L)')
# axs[1, 2].set_xlabel('Time (hrs)')

# axs[2, 0].plot(random_run1[2], random_run1[5], label="MGMT")
# axs[2, 0].set_ylabel('MGMT mRNA (µM)')
# axs[2, 0].set_xlabel('Time (hrs)')
# axs[2, 1].plot(random_run2[2], random_run2[5], label="MGMT")
# axs[2, 1].set_ylabel('MGMT mRNA (µM)')
# axs[2, 1].set_xlabel('Time (hrs)')
# axs[2, 2].plot(random_run3[2], random_run3[5], label="MGMT")
# axs[2, 2].set_ylabel('MGMT mRNA (µM)')
# axs[2, 2].set_xlabel('Time (hrs)')

# axs[3, 0].plot(random_run1[2], random_run1[6], label="SSB_total")
# axs[3, 0].set_ylabel('MGMT protein (µM)')
# axs[3, 0].set_xlabel('Time (hrs)')
# axs[3, 1].plot(random_run2[2], random_run2[6], label="SSB_total")
# axs[3, 1].set_ylabel('MGMT protein (µM)')
# axs[3, 1].set_xlabel('Time (hrs)')
# axs[3, 2].plot(random_run3[2], random_run3[6], label="SSB_total")
# axs[3, 2].set_ylabel('MGMT protein (µM)')
# axs[3, 2].set_xlabel('Time (hrs)')

# fig.tight_layout()







end = time.time()
print(end - start)