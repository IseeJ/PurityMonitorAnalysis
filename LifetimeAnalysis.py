import pandas as pd
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import datetime
import matplotlib.dates as mdates


#%matplotlib notebook
%matplotlib inline
def calc110(CV1,CV2,AV1,AV2,t1,t2,t3, t4):
    #delt = t3-t1
    T1 = t2-t1 #time from photocathode to cathode grid
    T2 = t3-t2 #time from cathode grid to anode grid
    T3 = t4-t3 #time from anode grid to anode
    driftt = T2+((T1+T3)/2)
    VC = np.abs(CV2-CV1)
    VA = np.abs(AV2-AV1)
    #convert to Q with linear fit from calibration change this based on pre-amp used
    #QC = VC*0.00061+0.01193 
    #QA = VA*0.00064+0.00470
    QC = VC*0.00063+0.00066
    QA = VA*0.00066+0.00009

    ratio = QA/QC
    tau = -round(driftt/(np.log(ratio)),1)
    n = 1/((4e12)*(tau*1e-6))
    print(f"ΔV_A = {VA:.2f} mV, ΔV_C = {VC:.2f} mV, ΔV_A/ΔV_C = {VA/VC:.4f}")
    print(f"QA = {QA:.4f} pC, QC = {QC:.4f} pC, QA/QC = {ratio:.4f}")
    #print(f"t1 = {T1:.2f} us, t2 = {T2:.2f} us, t3 = {T3:.2f} us, delt_t = {driftt} us, tau = {tau} us")
    #print(f"CV1 = {CV1}, CV2 = {CV2}, ΔV_C = {VC:.2f} mV, AV1 = {AV1}, AV2 = {AV2}, ΔV_A = {VA:.2f} mV")
    #print(f"T1 = {T1:.2f}, T2 = {T2:.2f}, T3 = {T3:.2f}, tdrift = {round(driftt,1)} μs, \nQA = {QA} pC, QC = {QC} pC, QC/QA = {ratio}, \n𝜏 = {tau} μs, n = {round(n*1e9,2)} ppb")

#plt.figure()
plt.figure(figsize=(10,5),dpi=200)
