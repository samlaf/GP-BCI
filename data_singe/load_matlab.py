import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, date, time
import pandas as pd

# mat is a dict that contains this_resp
# this_resp is a 3x7x9 cell (3 blocks, 7 muscle emgs, 9 conditions)
# where block 0: A ch25, B ch1
#       block 1: A ch25, B ch6
#       block 2: A ch6, B: ch1
# conditions are
# 0 : seulement channel A
# 1: channel A, delay 40ms, channel B
# 2: channel A, delay 20ms, channel B
# 3: channel A, delay 10ms, channel B
# 4: channel A et channel B simultaneous
# 5: channel B, delay 10ms, channel A
# 6: channel B, delay 20ms, channel A
# 7: channel B, delay 40ms, channel A
# 8:  seulement channel B
# each cell contains a 20x733 matrix (20 stimulations, 733 time series
# emg response)
rawmat = loadmat('RawMonkeyEmgResponse.mat')
rawdata = rawmat['raw_resp']
filtmat = loadmat('FilteredMonkeyEmgResponse.mat')
filtdata = filtmat['filt_resp']

# Let's build a proper datastructure
# dct[chi][chj][deltat] will contain all timeseries for pair chi, chj
# with time delay delta_t
# dct[ch_i][ch_i] will contain single channel pulses
EMG = 0
dct = {1:{1:{},6:{},25:{}},
       6:{1:{},6:{},25:{}},
       25:{1:{},6:{},25:{}}}
dct[1][1] = {'data': filtdata[0,EMG,8]}
dct[6][6] = {'data': filtdata[1,EMG,8]}
dct[25][25] = {'data': filtdata[0,EMG,0]}
for cond,delay in enumerate([40,20,10,0],1):
    dct[25][1][delay] = {'data': filtdata[0,EMG,cond]}
    dct[25][6][delay] = {'data': filtdata[1,EMG,cond]}
    dct[6][1][delay] = {'data': filtdata[2,EMG,cond]}
for cond,delay in enumerate([0,10,20,40],4):
    dct[1][25][delay] = {'data': filtdata[0,EMG,cond]}
    dct[6][25][delay] = {'data': filtdata[1,EMG,cond]}
    dct[1][6][delay] = {'data': filtdata[2,EMG,cond]}

# Now gather meanmax statistics
# store them in dct[ch_i][ch_j][deltat]['meanmax']
for ch1 in [1,6,25]:
    for ch2 in [1,6,25]:
        if ch1 == ch2:
            ch = dct[ch1][ch2]['data']
            meanmax = ch.max(axis=1).mean()
            dct[ch1][ch2]['meanmax'] = meanmax
        elif ch1 != ch2:
            for dt in [0,10,20,40]:
                ch = dct[ch1][ch2][dt]['data']
                meanmax = ch.max(axis=1).mean()
                dct[ch1][ch2][dt]['meanmax'] = meanmax


def analyze1d(data):
    # Let's use emg 0 and build a "prior" datastructure (single pulses)
    EMG = 0

    ch1 = data[1][1]
    ch6 = data[6][6]
    ch25 = data[25][25]

    meanmaxs_single = []
    for ch in [ch1,ch6,ch25]:
        mm = ch.max(axis=1).mean()
        meanmaxs_single.append(mm)

    plt.figure()
    plt.plot(meanmaxs_single)



    #Let's print the different curves and the mean
    plt.figure()
    for i,ts in enumerate(ch1, 1):
        plt.subplot(4,5,i)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.plot(ts)

    plt.figure()
    ch1avg = ch1.mean(axis=0)
    plt.plot(ch1avg)

# analyze1d(dct)
# analyze1d(rawdata)
# plt.show()



# Tester resultat en 2d et essayer de trouver un prior qui va bien fit
# Regarder pour chaque delta_t

def build_f_grid(dt=40, f='meanmax'):
    z_grid = np.zeros((3,3))
    for i,ch1 in enumerate([1,6,25]):
        for j,ch2 in enumerate([1,6,25]):
            if ch1 == ch2:
                z_grid[i][j] = dct[ch1][ch2][f]
            elif ch1 != ch2:
                z_grid[i][j] = dct[ch1][ch2][dt][f]
    return z_grid

# Let's look at 40ms time delay
def plot_2d(dt,f='meanmax'):
    x = [0,1,2]
    y = [0,1,2]
    x_grid, y_grid = np.meshgrid(x,y)
    z_grid = build_f_grid(dt,f)
    
    ax = plt.subplot(1,1,1,projection='3d')
    surf = ax.plot_surface(x_grid, y_grid, z_grid,
                       #cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

def plot_all_2d(f='meanmax'):
    x = [0,1,2]
    y = [0,1,2]
    x_grid, y_grid = np.meshgrid(x,y)
    
    for i,dt in enumerate([0,10,20,40],1):
        z_grid = build_f_grid(dt,f)
        ax = plt.subplot(4,1,i,projection='3d')
        surf = ax.plot_surface(x_grid, y_grid, z_grid,
                       #cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        ax.set_title('delta_t = {}'.format(dt))
        ax.axes.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

plot_all_2d('meanmax')
plt.show()





