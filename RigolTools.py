import pandas as pd

acol = 'red'
ccol = 'blue'

def find_baseline(series, npts=50):
    return series[0:npts].mean()

    
def subtract_baseline(df, chans=['anode','cathode']):
    for chan in chans:
        df[chan] -= find_baseline(df[chan]) 

def df_from_csv(fname):
    df = pd.read_csv(fname)
    return df

def read_csv(fname, ch3=None, ch4=None, ch1=None, ch2=None, vunit='mV', tunit='us'):
    # FIXME: add check that file exists

    #e.g. rigol.readCsv(filename, ch3='cathode', ch4='anode', vunit='mV', tunit='us')
    #returns
    #      time   cathode     anode
    #0   -337.0  64.73467  75.52400
    #1   -336.0  64.40639  75.42400
    
    # if doing this manually...
    # first line is a header
    # e.g.
    #    Time(s),CH3V,CH4V
    #    -3.370000e-04,6.473467e-02,7.552400e-02
    #    -3.360000e-04,6.440639e-02,7.542400e-02
    # expect to have time and then channel data
    #with open(fname) as fh:
    #    header = fh.readline()
    #    print(header)

    # or just use pandas...
    df = df_from_csv(fname)

    # rename the time column
    df.rename(columns={"Time(s)":"time"}, inplace=True)
    voltage_labels = [] # track which channels are "in play"
    if ch1:
        df.rename(columns={"CH1V":ch1}, inplace=True)
        voltage_labels.append(ch1)
    if ch2:
        df.rename(columns={"CH2V":ch2}, inplace=True)
        voltage_labels.append(ch2)
    if ch3:
        df.rename(columns={"CH3V":ch3}, inplace=True)
        voltage_labels.append(ch3)
    if ch4:
        df.rename(columns={"CH4V":ch4}, inplace=True)
        voltage_labels.append(ch4)

    # If requested, then:
    # scale time from seconds to microseconds
    if tunit == 'us':
        df['time'] *= 1e6
    # and voltages to mV
    if vunit == 'mV':
        for vl in voltage_labels:
            df[vl] *= 1e3

    return df



def read_csv_2(fname, ch3=None, ch4=None, ch1=None, ch2=None, vunit='mV', tunit='us'):
    df = df_from_csv(fname)

    # rename the time column                                                                                  
    df.rename(columns={"Time(s)":"time"}, inplace=True)
    voltage_labels = [] # track which channels are "in play"                                                  
    if ch1:
        df.rename(columns={"CH1(V)":ch1}, inplace=True)
        voltage_labels.append(ch1)
    if ch2:
        df.rename(columns={"CH2(V)":ch2}, inplace=True)
        voltage_labels.append(ch2)
    if ch3:
        df.rename(columns={"CH3(V)":ch3}, inplace=True)
        voltage_labels.append(ch3)
    if ch4:
        df.rename(columns={"CH4(V)":ch4}, inplace=True)
        voltage_labels.append(ch4)

    # If requested, then:                                                                                     
    # scale time from seconds to microseconds                                                                 
    if tunit == 'us':
        df['time'] *= 1e6
    # and voltages to mV                                                                                      
    if vunit == 'mV':
        for vl in voltage_labels:
            df[vl] *= 1e3

    return df
