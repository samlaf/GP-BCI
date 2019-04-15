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
            maxs = ch.max(axis=1)
            dct[ch1][ch2]['meanmax'] = maxs.mean()
            dct[ch1][ch2]['stdmax'] = maxs.std()
        elif ch1 != ch2:
            for dt in [0,10,20,40]:
                ch = dct[ch1][ch2][dt]['data']
                maxs = ch.max(axis=1)
                dct[ch1][ch2][dt]['meanmax'] = maxs.mean()
                dct[ch1][ch2][dt]['stdmax'] = maxs.std()


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

# Plot the response graph for a given delta_t (0,10,20,40)
def plot_2d(dt=40,f='meanmax', usestd=True, ax=None, title=None):
    """ f can be either a str (which will be used to call build_f_grid)
or a grid already built, for eg. grid of diff between response and prior"""
    x = [0,1,2]
    y = [0,1,2]
    x_grid, y_grid = np.meshgrid(x,y)
    if type(f) == str:
        z_grid = build_f_grid(dt,f)
    else:
        z_grid = f

    if ax is None:
        ax = plt.subplot(1,1,1,projection='3d')
    ax.get_xaxis().set_ticks([0,1,2])
    ax.get_yaxis().set_ticks([0,1,2])
    if title is None:
        ax.set_title('delta_t = {}'.format(dt))
    else:
        ax.set_title(title)
    surf = ax.plot_surface(x_grid, y_grid, z_grid,
                       #cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    if usestd:
        std_grid = build_f_grid(dt,f='stdmax')
        # We plot 1 std deviation above and below the mean
        std_min = z_grid - std_grid
        std_max = z_grid + std_grid
        ax.scatter(x_grid, y_grid, std_min, c='red')
        ax.scatter(x_grid, y_grid, std_max, c='red')
        # We plotted the scatter for aesthetic reasons,
        # now we plot 2d lines to show the std_dev more clearly
        for x,y,smin,smax in zip(x_grid.flatten(), y_grid.flatten(),
                                 std_min.flatten(), std_max.flatten()):
            xs = np.ones(100) * x
            ys = np.ones(100) * y
            zs = np.linspace(smin,smax,100)
            ax.plot(xs, ys, zs, c='red')

    
def plot_all_2d(f='meanmax'):
    plt.figure()
    for i,dt in enumerate([0,10,20,40],1):
        ax = plt.subplot(2,2,i,projection='3d')
        plot_2d(dt,f,ax=ax)

#plot_2d(0)
#plot_all_2d('meanmax')

resp2d = build_f_grid()
prior1d = resp2d.diagonal().reshape((3,1))
prior2d_add = (prior1d + prior1d.T) * 1/2
prior2d_mult = (prior1d * prior1d.T)
diff_add = resp2d - prior2d_add
diff_mult = resp2d - prior2d_mult
plt.figure()
for i,(z_grid,title) in enumerate([(prior2d_add, "prior2d_add"),
                                   (diff_add, "diff_add"),
                                   (prior2d_mult, "prior2d_mult"),
                                   (diff_mult, "diff_mult")],
                                  1):
    ax = plt.subplot(2,2,i,projection='3d')
    plot_2d(f=z_grid, usestd=False, title=title, ax=ax)

plt.show()
