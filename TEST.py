import csv
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import ExpressionModel
import pandas as pd
import Purity as pur
import RigolTools as rigol

from scipy.signal import savgol_filter


import Process_data as pro
import os

path = '4th_PurityvsTime'
save_path = '4th_PurityvsTime/Vmax_result'
Edrift = 160 #V/cm

fnames_list = []
results = []

for fname in os.listdir(path):
    if fname.endswith('.csv'):
        df = rigol.read_csv_2(f"{path}/{fname}", ch3='anode', ch4='cathode', tunit='us', vunit='mV')
        rigol.subtract_baseline(df, chans=['cathode', 'anode'])
        cathode, anode = df['cathode'].values, df['anode'].values
        time = df['time'].values
        cathode_smooth, anode_smooth = savgol_filter(cathode, window_length=20, polyorder=2),savgol_filter(anode, window_length=20, polyorder=2)

        VA_max,VC_max,T_1, T_2, T_3 = pro.get_Vmax(path, fname, cathode_smooth, anode_smooth, time, save_path)

        
        #VA_max,VC_max,T_1, T_2, T_3 = pro.get_Vmax(path, fname, C_thres,A_thres, save_path)
        QC,QC_err,QA,QA_err,tdrift, lifetime,O2, O2_err = pro.calc_correct(VC_max,VA_max, T_1, T_2, T_3, Edrift)

        results.append({
            'Time': fname[:-4],
            'VA_max (mV)': VA_max,
            'VC_max (mV)': VC_max,
            'T_1 (us)': T_1,
            'T_2 (us)': T_2,
            'T_3 (us)': T_3,
            'QC (fC)': QC,
            'QC_err': QC_err,
            'QA (fC)': QA,
            'QA_err': QA_err,
            'tdrift (us)': tdrift,
            'lifetime (us)': lifetime,
            'O2 (ppb)': O2,
            'O2_err': O2_err

        })

        fnames_list.append(fname)

df = pd.DataFrame(results)
output_filename = save_path+'/'+'Vmax_Results.csv'
df.to_csv(output_filename, index=False)






#plot result
df = pd.read_csv(output_filename)
df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%dT%H%M%S')
df = df.sort_values('Time')
df['Time_Minutes'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds() / 3600


plt.figure(dpi=200, figsize = (15,5))
plt.errorbar(df['Time_Minutes'], df['QA (fC)'], yerr=df['QA_err'], fmt='o',color='r', label='QA', capsize=2)
plt.errorbar(df['Time_Minutes'], df['QC (fC)'], yerr=df['QC_err'], fmt='o',color = 'b', label='QC', capsize=2)

plt.xlabel('Time [hrs]')
plt.ylabel('Charge [fC]')
plt.title('Charge over Time')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_path}/result/QvsTime.png')
plt.clf()




plt.figure(dpi=200, figsize = (15,5))
plt.scatter(df['Time_Minutes'], df['O2 (ppb)'], color='k')


plt.xlabel('Time (hrs)')
plt.ylabel('Charge (fC)')
plt.title('O2 concentration over Time')
#plt.legend()
plt.tight_layout()
plt.savefig(f'{save_path}/result/O2vsTime.png')
plt.clf()




#save gif


import os
from PIL import Image
from datetime import datetime

folder_path = '4th_PurityvsTime/Vmax_result'

filenames = os.listdir(folder_path)
image_files = [f for f in filenames if f.endswith(('.png'))]

sorted_files = sorted(image_files, key=lambda x: datetime.strptime(x.split('.')[0], '%Y%m%dT%H%M%S'))

images = [Image.open(os.path.join(folder_path, file)) for file in sorted_files]

output_gif_path = f'{folder_path}/animated.gif'
images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)



import matplotlib.pyplot as plt

plt.figure(dpi=200, figsize=(15, 5))

plt.scatter(df['Time_Minutes'], df['O2 (ppb)'], color='k')

plt.xlabel('Time (hrs)')
plt.ylabel('Charge (fC)')
plt.title('O2 concentration over Time')

plt.subplot(1, 2, 2)
plt.scatter(df['Time_Minutes'], df['O2 (ppb)'], color='k')
plt.xlim(0, 30)
plt.ylim(min(df['O2 (ppb)']), max(df['O2 (ppb)']))
plt.title('Zoomed in O2 concentration (First 30 minutes)')
plt.xlabel('Time (hrs)')
plt.ylabel('Charge (fC)')

plt.tight_layout()

plt.savefig(f'{save_path}/result/O2vsTime_with_zoom.png')
plt.clf()




path = 'Amplitude_vs_field'
fname = '20250402T152907.csv'
pro.Fit_cathode(path,fname)
