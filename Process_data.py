import csv
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import ExpressionModel

import Purity as pur
import RigolTools as rigol

from scipy.signal import savgol_filter


def calculate_attachment_rate(field):
    
    p = 11
    a_1 = 39.4
    a_2 = 1.20062
    a_3 = 0
    a_4 = 0
    b_1 = 0.925794
    b_2 = 1.63816
    b_3 = 0
    b_4 = 0 

    field_voltage = field / 1000  # Convert field
    field_voltage =field 
    numerator = (a_1 / b_1) + a_1 * field_voltage + a_2 * (field_voltage ** 2)+a_3 * (field_voltage ** 3)+a_4 * (field_voltage ** 4)
    denominator = 1 + b_1* field_voltage + b_2 * (field_voltage ** 2) + b_3 * (field_voltage ** 3) + b_4 * (field_voltage ** 4)
    attachment_rate = (10 ** p) * (numerator / denominator)
    return attachment_rate #/s

def Vmax_cathode(path, fname, title):
    df = rigol.read_csv_2(path+'/'+fname, ch3='anode', ch4='cathode', tunit='us', vunit='mV')
    rigol.subtract_baseline(df, chans=['cathode', 'anode'])

    
    cathode = df['cathode'].values
    time = df['time'].values
    
    cathode_smooth = savgol_filter(cathode, window_length=20, polyorder=2)
    slope = np.diff(cathode_smooth)
    base = -0.4*np.max(np.abs(slope))
    start = np.argmax(slope < base)
    peak = np.argmin(cathode_smooth)

    V_max = -np.min(cathode_smooth)
    #print(V_max)
    t1 = time[start]
    t2 = time[peak]
    td = t2-t1

    vgain = 2
    Cf = 1.4 #pF
    Rf = 100 #MOhm
    tauf = Rf*Cf
    F = (1/Cf)*(tauf/td)*(1-np.exp(-td/tauf))

    Q0 =(V_max/vgain)/F
    td = td

    Q_err1 = 1/F
    xx = np.exp(td/tauf)-1
    Q_err2 = (V_max*Cf/tauf)*(((tauf*xx-td)*np.exp(td/tauf))/(tauf*xx**2))
    Q_err = np.sqrt(Q_err1**2+Q_err2**2)

    
    plt.axvline(time[start],  linewidth=0.5, color='b', linestyle='--', label='signal start')
    plt.axvline(time[peak],  linewidth=0.5, color='r', linestyle='--', label='signal peak')
    plt.plot(time,cathode_smooth*1e3, linewidth=0.5, color='black', label = 'data')
    
    plt.title(title)

    plt.xlabel("Time [µs]")
    plt.ylabel('Voltage [mV]')
    #plt.legend()
    plt.savefig(f'{path}/Vmax/{fname}.png')
    plt.clf()

    
    return V_max, F, Q0, Q_err, t1, t2

def calc_correct(VC_max,VA_max, T_1, T_2, T_3, Edrift):
    # E in V/cm
    E = Edrift
    T = 87 #K                                                                   
    mu =( (551.6 + 7158.3*(E/1000) + 4440.43*((E/1000)**(3/2)) + 4.29*((E/1000)**(5/2))) /
    (1 + (7158.3/551.6)*(E/1000) + 43.63*((E/1000)**2) + 0.2053*((E/1000)**3))) * ((T/89)**(-3/2))
    v_d = mu*E/1000000

    a1 = 39.4
    a2 = 1.20062
    a3 = 0
    a4 = 0

    b1 = 0.925794
    b2 = 1.63816
    b3 = 0
    b4 = 0
    k = (1e11) * ( ( (a1 / b1) + a1 * (E/1000) + a2 * ((E/1000)**2) + a3 * ((E/1000)**3) + a4 * ((E/1000)**4) ) /
              ( 1 + b1 * (E/1000) + b2 * ((E/1000)**2) + b3 *((E/1000)**3) +b4 * ((E/1000)**4) ) )
    #k = (1e11) * ( ( (76.2749 / 1.88083) + 76.2749 * (E/1000) + 4.24596 * ((E/1000)**2) + 0 * ((E/1000)**3) + 0 * ((E/1000)**4) ) /
    #          ( 1 + 1.88083 * (E/1000) + 2.62643 * ((E/1000)**2) + 0.0632332 *((E/1000)**3) - 0.000211009 * ((E/1000)**4) ) )
    
    #k = calculate_attachment_rate(E)
    Cf = 1.4 # pF
    Rf = 100 # Mohm
    vgain = 2
    
    tauf = Rf*Cf

    tdrift = T_2+(T_1+T_3)/2
    FC = (1/Cf)*(tauf/T_1)*(1-np.exp(-T_1/tauf))
    QC = (VC_max)/FC
    QC = (VC_max/vgain)/FC
    FA = (1/Cf)*(tauf/T_3)*(1-np.exp(-T_3/tauf))
    QA = (VA_max)/FA
    QA = (VA_max/vgain)/FA

    QC_err1, QA_err1 = 1/FC, 1/FA
    CC, AA = np.exp(T_1/tauf)-1, np.exp(T_3/tauf)-1
    QC_err2 = (VC_max*Cf/tauf)*(((tauf*CC-T_1)*np.exp(T_1/tauf))/(tauf*CC**2))
    QA_err2 = (VA_max*Cf/tauf)*(((tauf*AA-T_3)*np.exp(T_3/tauf))/(tauf*AA**2))

    QC_err, QA_err = np.sqrt(QC_err1**2+QC_err2**2),np.sqrt(QA_err1**2+QA_err2**2)


    lifetime = -tdrift/np.log(QA/QC) #us
    O2 = 1/(k*lifetime*0.000001)*1000000000 #ppb



    n_1 = -1/(k*tdrift*QA)
    n_2 = 1/(k*tdrift*QC)
    n_3 = np.log(QA/QC)/(k*(tdrift**2))    
    O2_err = np.sqrt((n_1)**2+(n_2)**2+(n_3)**2)
    
    return QC,QC_err,QA,QA_err,tdrift, lifetime,O2, O2_err

    
    

def get_Vmax(path, fname, cathode_smooth, anode_smooth,time,save_path,tcut=None):
    
    df = rigol.read_csv_2(f"{path}/{fname}", ch3='anode', ch4='cathode', tunit='us', vunit='mV')
    rigol.subtract_baseline(df, chans=['cathode', 'anode'])
    cathode, anode = df['cathode'].values, df['anode'].values
    time = df['time'].values
    if tcut:
        anode[time<=tcut]=0
    else:
        anode_new = anode
    
    cathode_smooth, anode_smooth = savgol_filter(cathode, window_length=20, polyorder=2),savgol_filter(anode, window_length=20, polyorder=2)

    #cathode_smooth, anode_smooth = savgol_filter(cathode, window_length=20, polyorder=2),savgol_filter(anode, window_length=20, polyorder=2)
    
    cathode_slope = np.diff(cathode_smooth)
    cathode_base = -0.2*np.max(np.abs(cathode_slope))
    cathode_start = np.argmax(cathode_slope < cathode_base)
    cathode_peak = np.argmin(cathode_smooth)

    anode_slope = np.diff(anode_smooth)
    #anode_base = A_thres*np.max(np.abs(anode_slope))
    #anode_start = np.argmax(anode_slope > anode_base)
    anode_peak = np.argmax(anode_smooth)


    valid_rise_time = False
    threshold_factors = np.linspace(0.2, 0.8, 60)

    for factor in threshold_factors:
        anode_base = factor * np.max(anode_slope)
        possible_indices = np.where(anode_slope > anode_base)[0]

        if len(possible_indices) == 0:
            continue

        anode_start = possible_indices[0]
        rise_time = time[anode_peak] - time[anode_start]

        if 10e-6 < rise_time < 30e-6: # set the valid rise time range here. In our case, the anode waveform rise time is usually between 16-22 µs.Here I set 10 to 30 us to be more conservative.
            valid_rise_time = True
            break


    
    t1,C1 = time[cathode_start], cathode[cathode_start]
    t2,C2 = time[cathode_peak], cathode[cathode_peak]
    t3,A1 = time[anode_start], anode[anode_start]
    t4,A2 = time[anode_peak], anode[anode_peak]


    T_1 = t2-t1
    T_2 = t3-t2
    T_3 = t4-t3
    
    VA_max = np.abs(A2-A1)
    VC_max = np.abs(C2-C1)

    plt.figure(dpi=200)
    plt.title(f"{fname[:-4]}")
    plt.plot(time, anode_smooth, linewidth=0.5, color = 'r', label='Anode')
    plt.plot(time, cathode_smooth, linewidth=0.5, color = 'b', label='Cathode')

    plt.plot(time, anode, linewidth=1, color = 'r', alpha = 0.3)
    plt.plot(time, cathode, linewidth=1, color = 'b',alpha = 0.3)
    plt.axvline(t1, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(t2, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(t3, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(t4, color='k', linestyle='--', linewidth=0.5)

    plt.xlim(-350,650)
    plt.ylim(-55,40)
    plt.xlabel("Time (μs)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.savefig(f'{save_path}/{fname[:-4]}.png')
    plt.clf()

    print(t1,t2,t3,t4)

    return VA_max,VC_max,T_1, T_2, T_3






    
def Fit_cathode(path,fname):
    df = rigol.read_csv_2(f'{path}/{fname}', ch3='anode', ch4='cathode', tunit='us', vunit='mV')
    rigol.subtract_baseline(df, chans=['cathode', 'anode'])
    cathode = df['cathode'].values
    time = df['time'].values

    Cf = 1.4
    vgain = 2
    vmod = ExpressionModel(" (-2*(1e3*Q0*140)/(td*1.4)) * ( (1-exp(-x/140))*(x<td) + (exp(td/140)-1)*(exp(-x/140))*(x>=td) )")

    cathode_smooth = savgol_filter(cathode, window_length=20, polyorder=2)
    slope = np.diff(cathode_smooth)
    base = -0.4*np.max(np.abs(slope))
    start = np.argmax(slope < base)
    peak = np.argmin(cathode_smooth)

    V_max = -np.min(cathode_smooth)

    #print(V_max)                                                                                                                          
    t1 = time[start]
    t2 = time[peak]
    td = t2-t1


    
    time_new = df['time'].values[start:]
    cathode_new = cathode_smooth[start:]

    Q0 = np.max(np.abs(cathode_new)) * Cf * td / vgain
    result = vmod.fit(cathode_new, x=time_new, Q0=Q0, td=td)

    plt.figure(dpi=200)

    plt.axvline(time[start], color='gray', linestyle='--')
    plt.axvline(time[peak],  color='gray', linestyle='--')
    plt.plot(time,cathode_smooth, color='k', label = 'data')

    plt.xlabel("Time [µs]")
    plt.ylabel('Voltage [mV]')
#    plt.legend()                                                                                                                           
    plt.savefig(f'{path}/results/comp/vmax.png')
    plt.clf()



    
    Q0_val = result.params['Q0'].value
    Q0_err = result.params['Q0'].stderr
    td_val = result.params['td'].value
    td_err = result.params['td'].stderr

    print(result.fit_report())

    plt.figure(dpi=200)

    plt.plot(df['time'].values,cathode_smooth, color='k', label = 'data')
    plt.axvline(time[peak],  color='gray', linestyle='--')
    plt.axvline(time[start],  color='gray', linestyle='--', label='$t_1$')

    
    plt.plot(time_new, cathode_new, linewidth=0.5, color='black')                                                                              
    plt.plot(time_new, result.best_fit,color='red', linestyle='--', label='fit')


    plt.xlabel("Time [µs]")
    plt.ylabel('Voltage [mV]')
    plt.legend()
    plt.savefig(f'{path}/results/comp/fit.png')
    plt.clf()



"""    
# process
folder = 'Amplitude_vs_field'
filelist = '4thrun_data - Amplitude_vs_field.csv'

fnames_list = []
E2_list = []
results = []
    

#process
with open(filelist, mode='r', newline='') as file:
    reader = csv.DictReader(file)

    for row in reader:
        filename = row['Filename']
        CV = int(row['C (V)'])
        AG = int(row['AG (V)'])
        E2 = AG/5.98
        A = int(row['A (V)'])
        ratio = float(row['E Ratio'])

        fnames_list.append(filename)
        E2_list.append(E2)


for i in range(len(fnames_list)):
    V_max,F, Q0, Q_err, t1, t2 = Vmax_cathode(folder,fnames_list[i],'E2 = '+str(f"{E2_list[i]:.2f}")+' V/cm')
    results.append([E2_list[i],V_max, F, Q0, Q_err, t1, t2, t2-t1])
    print(f"QC = {Q0}+/-{Q_err} \n T1 = {t2-t1}")
    
with open(folder+'/Vmax_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['E2 (V/cm)','V_max','F', 'Q0', 'Q_err', 't1', 't2', 'T1']) 
    writer.writerows(results)




#plot all                                                                                                                                                     
dfs = []
for ii, fname in enumerate(fnames_list):
    dfs.append(rigol.read_csv_2(folder+'/'+fname, ch3='anode', ch4='cathode', tunit='us', vunit='mV'))
    dfs[-1].attrs['E2'] = E2_list[ii]
for df in dfs:
    rigol.subtract_baseline(df, chans=['cathode', 'anode'])




plt.figure(dpi=200, figsize=(15,10))
for df in dfs:
    l = f"{df.attrs['E2']:.2f}" 
    plt.plot(df['time'], df['cathode'], linewidth=0.5,label=l)
    ##plt.plot(df['time'], df['anode'], rigol.acol)                                                                                                           

plt.legend()
#plt.title(f"Cathode Signal vs. Cathode Field (with E2=E1)")                                                                                                  
plt.xlabel("Time [us]")
plt.ylabel("Voltage [mV]")
plt.savefig(f'{folder}/cathode_only_purity.pdf')
plt.savefig(f'{folder}/cathode_only_purity.png')
plt.clf()


# smooth with Savitsky-Golay                                                                                                      
wl = 20
po = 2
for df in dfs:
    df['cathode_filt'] = savgol_filter(df['cathode'].values, window_length=wl, polyorder=po)

plt.figure(dpi=200,figsize=(15,10))

for df in dfs:
    l = f"{df.attrs['E2']:.2f}"
    plt.plot(df['time'], df['cathode_filt'], linewidth=0.5, label=l)
#plt.title(f"Cathode: Smoothed Savitsky-Golay window={wl}, poly={po}")                                                            
plt.xlabel("Time [us]")
plt.ylabel("Voltage [mV]")
plt.legend()
plt.savefig(f'{folder}/cathode_only_purity_filtered.pdf')
plt.savefig(f'{folder}/cathode_only_purity_filtered.png')
plt.clf()


"""
