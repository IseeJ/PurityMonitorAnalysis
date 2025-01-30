# final 01/30/25
# for the 01/14/25 purity vs time data


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import gaussian_filter

def get_data3(file_path):
    df = pd.read_csv(file_path)

    time = np.array(df["Time(s)"]) * 1e6  
    CH1 = np.array(df["CH1V"]) * 1e3     
    CH1_new = gaussian_filter(CH1, sigma=3) 
    
    CH3 = np.array(df["CH3V"]) * 1e3  
    CH3_new = gaussian_filter(CH3, sigma=3) 
    
    #return time, CH1_new, CH3_new
    return time, CH1, CH3

def getarea(file_path):
    time, CH1, CH3 = get_data3(file_path)
    
    #Cathode
    dy_cathode = np.diff(CH1)
    fall_index_cathode = np.where(dy_cathode < 0)[0][0] + 1
    baseline_cathode = np.mean(CH1[:fall_index_cathode])
    CH1_norm = CH1 - baseline_cathode
    area_cathode = np.trapz(CH1_norm, time)
    
    #Anode
    dy_anode = np.diff(CH3)
    fall_index_anode = np.where(dy_anode < 0)[0][0] + 1
    baseline_anode = np.mean(CH3[:fall_index_anode])
    CH3_norm = CH3 - baseline_anode
    area_anode = np.trapz(CH3_norm, time)

    return CH1_norm,  CH3_norm, baseline_cathode, area_cathode, baseline_anode, area_anode


def make_plot(file_path, timestamp):
    time, CH1, CH3 = get_data3(file_path)
    CH1_norm,  CH3_norm, baseline_cathode, area_cathode, baseline_anode, area_anode = getarea(file_path)
    
    plt.figure(figsize=(15, 6), dpi=300)
    title = file_path[15:]
    
    plt.plot(time, CH1_norm, color=[0/235,141/235,235/235])
    plt.title(f"{title}")
    plt.plot(time, CH3_norm, color=[235/235, 80/235, 79/235])
    plt.ylim(-15,12.5)
    
    plt.annotate(f"Time:{timestamp}\n"
                 f"Area Cathode: {abs(area_cathode):.2f}\n"
                 f"Area Anode: {area_anode:.2f}", 
                 xy=(0.02, 0.98), xycoords='axes fraction', fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.5), verticalalignment='top')
    plt.savefig(f'plots/{title[:-4]}.jpg')

    plt.show()
    

folder = "purity_vs_time"
results = []

files = os.listdir(folder)
file_paths = [os.path.join(folder, file) for file in files]

sorted_files = sorted(file_paths, key=lambda x: os.path.getmtime(x))

for filename in sorted_files:
    if filename.endswith(".csv"):
        filetime = os.path.getmtime(filename)
        filetime_str = datetime.fromtimestamp(filetime).strftime("%D %H:%M:%S")
        
        CH1_norm,  CH3_norm, baseline_cathode, area_cathode, baseline_anode, area_anode = getarea(filename)
        results.append({"Time": filetime_str,
            "Filename": filename,
            "Cathode Area": abs(area_cathode),
            "Anode Area": area_anode})

results_df = pd.DataFrame(results)
results_df["Time"] = pd.to_datetime(results_df["Time"])
results_df = results_df.sort_values(by="Time")

results_df.to_csv("purity_vs_time_results.csv", index=False)
#print(results_df)

for i in range(len(results_df["Filename"])):
    make_plot(results_df["Filename"][i],results_df["Time"][i])


# make into gif
# https://medium.com/@theriyasharma24/creating-gifs-from-images-using-python-88946aa47881
from PIL import Image
def make_gif(folder_path, output_gif_path, duration=500):
    image_paths = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg'))]
    )
    images = [Image.open(image_path) for image_path in image_paths]
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  
    )
    print("done")

if __name__ == "__main__":
    folder_path = "./plots"  
    output_gif_path = "./plots/output.gif"
    make_gif(folder_path, output_gif_path)
