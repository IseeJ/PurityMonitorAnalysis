def process(folder):
    time_list = []
    CH2_list = []
    CH3_list = []

    for file_name in os.listdir(folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder, file_name)
            time, CH2, CH3 = get_data2(file_path)
            time_list.append(time)
            CH2_list.append(CH2)
            CH3_list.append(CH3)

    time_avg = time_list[0]

    #do sigma clipping
    # https://docs.astropy.org/en/latest/api/astropy.stats.sigma_clip.html
    # cenfunc = center value of the clipping (median or mean)
    sigma_val = 1
    CH2_clipped = sigma_clip(CH2_list, sigma=sigma_val, axis=0, cenfunc=np.mean) 
    CH3_clipped = sigma_clip(CH3_list, sigma=sigma_val, axis=0, cenfunc=np.mean)

    CH2_avg = np.mean(CH2_clipped, axis=0)
    CH3_avg = np.mean(CH3_clipped, axis=0)

    n =200

    CH3_dy = np.diff(CH3_avg) 
    CH3_baseline = np.mean(CH3_avg[:n])

    CH2_dy = np.diff(CH2_avg) 
    CH2_baseline = np.mean(CH2_avg[:n])

    CH2_norm = CH2_avg - CH2_baseline
    CH3_norm = CH3_avg - CH3_baseline

    plt.figure(figsize=(15, 6), dpi=200)
    plt.plot(time_list[0],CH2_list[0]-CH2_baseline,label='Raw data wf1', color='gray', alpha=0.3)
    plt.plot(time_avg, CH2_norm, label='Normalized Data', color=[0/235,141/235,235/235],linewidth=1)
    plt.fill_between(time_avg, CH2_norm, color='b', alpha=0.3)
    plt.title(f"{folder}: Cathode")
    plt.xlabel("Time (µs)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.show()


    area_CH2 = np.trapz(CH2_norm, time_avg)
    print("Cathode area =", area_CH2)


    plt.figure(figsize=(15, 6),dpi=200)
    plt.plot(time_list[0],CH3_list[0]-CH3_baseline,label='Raw data wf1', color='gray', alpha=0.3)
    plt.plot(time_avg, CH3_norm, label='Normalized Data', color=[235/235, 80/235, 79/235],linewidth=1)
    plt.fill_between(time_avg, CH3_norm, color='r', alpha=0.3)
    plt.title(f"{folder}: Anode")
    plt.xlabel("Time (µs)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.show()


    area_CH3 = np.trapz(CH3_norm, time_avg)
    print("Anode area =", area_CH3)


    plt.figure(figsize=(10, 6),dpi=200)
    plt.plot(time_avg, CH2_norm, label='Cathode', color=[0/235,141/235,235/235],linewidth=1)
    plt.plot(time_avg, CH3_norm, label='Anode', color=[235/235, 80/235, 79/235],linewidth=1)
    plt.title(f"{folder}")
    plt.xlabel("Time (µs)")
    plt.ylabel("Voltage (mV)")
    plt.ylim(-20,15)
    plt.xlim(-250,850)
    plt.legend()
    plt.show()
