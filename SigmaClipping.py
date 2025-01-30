# for 01/10/25 100 raw waveforms from Rigol_DHO4804_Scope

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
import os
import pandas as pd

def get_data2(file_path):
    df = pd.read_csv(file_path)
    time = np.array(df["Time(s)"]) * 1e6
    CH2 = np.array(df["CH2V"]) * 1e3
    CH3 = np.array(df["CH3V"]) * 1e3
    return time, CH2, CH3

folder = "Data_aft_purification"
time_list = []
CH2_list = []
CH3_list = []

min_length = float('inf') 
for file_name in os.listdir(folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder, file_name)
        time, CH2, CH3 = get_data2(file_path)
        time_list.append(time)
        CH2_list.append(CH2)
        CH3_list.append(CH3)
        min_length = min(min_length, len(time), len(CH2), len(CH3)) #one of the files have different # of data points

time_list = [time[:min_length] for time in time_list]
CH2_list = [CH2[:min_length] for CH2 in CH2_list]
CH3_list = [CH3[:min_length] for CH3 in CH3_list]

time_avg = time_list[0]

#do sigma clipping
# https://docs.astropy.org/en/latest/api/astropy.stats.sigma_clip.html
# cenfunc = center value of the clipping (median or mean)
sigma_val = 1
CH2_clipped = sigma_clip(CH2_list, sigma=sigma_val, axis=0, cenfunc=np.mean) 
CH3_clipped = sigma_clip(CH3_list, sigma=sigma_val, axis=0, cenfunc=np.mean)

CH2_avg = np.mean(CH2_clipped, axis=0)
CH3_avg = np.mean(CH3_clipped, axis=0)


#need to fix this baseline

CH3_dy = np.diff(CH3_avg) 
rise_index = np.where(CH3_dy>0)[0][0]+1 #first inc
CH3_baseline = np.mean(CH3_avg[:rise_index])

CH2_dy = np.diff(CH2_avg) 
fall_index = np.where(CH2_dy<0)[0][0]+1 #first dec
CH2_baseline = np.mean(CH2_avg[:fall_index])


CH2_baseline = np.mean(CH2_avg[:200])
CH3_baseline = np.mean(CH3_avg[:200])
CH2_norm = CH2_avg - CH2_baseline
CH3_norm = CH3_avg - CH3_baseline


plt.figure(figsize=(15, 6), dpi=200)
plt.plot(time_list[0],CH2_list[0]-CH2_baseline,label='Raw data (wf1)', color='gray', alpha=0.3)
plt.plot(time_avg, CH2_norm, label='Normalized Data (all)', color=[0/235,141/235,235/235],linewidth=1)
plt.fill_between(time_avg, CH2_norm, color='b', alpha=0.3)
plt.title(f"01/10/25: 1 atm Argon Gas after purification: CH2 Cathode")
plt.xlabel("Time (µs)")
plt.ylabel("Voltage (mV)")
plt.legend()
plt.show()


area_CH2 = np.trapz(CH2_norm, time_avg)
print("Cathode area =", area_CH2)


plt.figure(figsize=(15, 6),dpi=200)
plt.plot(time_list[0],CH3_list[0]-CH3_baseline,label='Raw data (wf1)', color='gray', alpha=0.3)
plt.plot(time_avg, CH3_norm, label='Normalized Data (all)', color=[235/235, 80/235, 79/235],linewidth=1)
plt.fill_between(time_avg, CH3_norm, color='r', alpha=0.3)
plt.title(f"01/10/25: 1 atm Argon Gas after purification: CH3 Anode")
plt.xlabel("Time (µs)")
plt.ylabel("Voltage (mV)")
plt.legend()
plt.show()


area_CH3 = np.trapz(CH3_norm, time_avg)
print("Anode area =", area_CH3)



plt.figure(figsize=(15, 6),dpi=200)
plt.plot(time_avg, CH2_norm, label='CH2 Cathode', color=[0/235,141/235,235/235],linewidth=1)
plt.plot(time_avg, CH3_norm, label='CH3 Anode', color=[235/235, 80/235, 79/235],linewidth=1)
plt.title(f"01/10/25: 1 atm Argon Gas after purification")
plt.xlabel("Time (µs)")
plt.ylabel("Voltage (mV)")
plt.legend()
plt.show()
