import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, date, time
import matplotlib.gridspec as gridspec
import random

"""
# TODO #
Currently dt=0 responses aren't symmetric
(for eg trains.build_f_grid(dt=40)[:3,:3] is not symmetric)
technically this should be symmetric, so best might be to combine
[ch1,ch2] respones with those of [ch2,ch1] and average over these 40 responses for both
"""

CHS = [13,17,21,18,22,2,6,10,14,9]
# Channels are organized on the following grid
"""
13---17---21---18---22
|    |    |    |    |
2 ---6 ---10---14---9
"""
# which we give the following (x,y) coordinates
"""
(0,1)---(1,1)---(2,1)---(3,1)---(4,1)
  |       |       |       |       |
(0,0)---(1,0)---(2,0)---(3,0)---(4,0)
"""
# and use the following array to map coordinates to channel
xy2ch = [[2,6,10,14,9],
         [13,17,21,18,22]]
ch2xy = {}
for ch in CHS:
    x,y = np.where(np.array(xy2ch)==ch)
    ch2xy[ch] = [x[0],y[0]]
#CHS = [2, 6, 9, 10, 13, 14, 17, 18, 21, 22]
DTS = [0, 10, 20, 40, 60, 80, 100]
EMG = 4
N_EMGS = 7

class Trains:

    def __init__(self, emg = EMG, N_EMGS = N_EMGS, path_to_data=None):
        self.chs = CHS
        self.n_ch = len(self.chs)
        self.dts = DTS

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
        if path_to_data:
            filtmat = loadmat(path_to_data)
        else:
            filtmat = loadmat('FilteredPairedTrains.mat')
        filtdata = filtmat['gfilt_resp']

        # Let's build a proper datastructure
        # trains[chi][chj][deltat] will contain all timeseries for pair chi, chj
        # with time delay delta_t
        # trains[ch_i][ch_i] will contain single channel pulses
        self.N_EMGS = N_EMGS
        self.emgdct = {}
        for emg in range(N_EMGS):
            # Pairs
            dct = {ch1:{ch2:{} for ch2 in self.chs} for ch1 in self.chs}
            for i,ch1 in enumerate(self.chs):
                for j,ch2 in enumerate(self.chs):
                    # We deal with dt=0 separately, since it should be
                    # symmetric! (ch1,ch2 = ch2,ch1 since stimulations
                    # are done simultaneously (dt=0))
                    if i == j:
                        dct[ch1][ch1][0] = {'data': filtdata[emg,i,j,0]}
                    else:
                        dct[ch1][ch2][0] = dct[ch2][ch1][0] = {'data': np.vstack((filtdata[emg,i,j,0], filtdata[emg,j,i,0]))}
                    ch = dct[ch1][ch2][0]['data']
                    maxs = ch.max(axis=1)
                    dct[ch1][ch2][0]['meanmax'] = dct[ch2][ch1][0]['meanmax'] = maxs.mean()
                    dct[ch1][ch2][0]['stdmax'] = dct[ch2][ch1][0]['stdmax'] = maxs.std()
                    for k,dt in enumerate(self.dts[1:]):
                        dct[ch1][ch2][dt] = {'data': filtdata[emg,i,j,k]}
                        # We also precompute the meanmax and stdmax
                        # statistics
                        ch = dct[ch1][ch2][dt]['data']
                        maxs = ch.max(axis=1)
                        dct[ch1][ch2][dt]['meanmax'] = maxs.mean()
                        dct[ch1][ch2][dt]['stdmax'] = maxs.std()
            self.emgdct[emg] = dct
            ## Note: trains[chi][chi][10] and [20] shouldn't have anything
            ## But they contain same as trains[chi][chi][0] for some reason.
            ## Not super important for now but make sure not to take this as real data.

        self.trains = self.emgdct[emg]

    ######### GETTERS ############
    def get_emgdct(self, emg):
        return self.emgdct[emg]

    def build_f_grid(self, emg=2, dt=40, f='meanmax'):
        z_grid = np.zeros((self.n_ch,self.n_ch))
        for i,ch1 in enumerate(self.chs):
            for j,ch2 in enumerate(self.chs):
                z_grid[i][j] = self.emgdct[emg][ch1][ch2][dt][f]
        return z_grid

    def build_f_grid_1d(self, f='meanmax'):
        #only dt=0 makes sense in 1d so we don't need dt as argument
        grid = np.zeros((2,5))
        for ch in self.chs:
            x,y = ch2xy[ch]
            grid[x][y] = self.trains[ch][ch][0][f]
        return grid

    def max_ch_2d(self, emg=2, dt=40):
        grid = self.build_f_grid(emg, dt)
        x,y = np.unravel_index(grid.argmax(), grid.shape)
        return [self.chs[x], self.chs[y]]

    def max_ch_1d(self):
        grid = self.build_f_grid_1d()
        x,y = np.unravel_index(grid.argmax(), grid.shape)
        return xy2ch[x][y]

    def build_1d_prior(self, f='meanmax'):
        """ This function returns the function f evaluated at each point of the 2x5 grid.
We call it 1d to contrast with 2d trains (2 channels simulated at the same time)"""
        prior = np.zeros_like(self.n_ch)
        for i,ch in enumerate(self.chs):
            prior[i] = self.trains[ch][ch][0][f]
        return prior

    # Plot the response graph for a given delta_t (0,10,20,40)
    def plot_2d(self, dt=40,f='meanmax', usestd=True, ax=None, title=None, zmax=None):
        """ f can be either a str (which will be used to call build_f_grid)
        or a grid already built, for eg. grid of diff between response and prior"""
        x = list(range(self.n_ch))
        y = list(range(self.n_ch))
        x_grid, y_grid = np.meshgrid(x,y)
        if type(f) == str:
            z_grid = self.build_f_grid(dt,f)
        else:
            z_grid = f

        if ax is None:
            ax = plt.subplot(1,1,1,projection='3d')
        ax.get_xaxis().set_ticks(x)
        ax.get_yaxis().set_ticks(y)
        if title is None:
            ax.set_title('delta_t = {}'.format(dt))
        else:
            ax.set_title(title)
        if zmax is not None:
            ax.set_zlim([0,zmax])
        surf = ax.plot_surface(x_grid, y_grid, z_grid,
                           #cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        if usestd:
            std_grid = self.build_f_grid(dt,f='stdmax')
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
                ax.plot(xs, ys, zs, c='red', alpha=0.3)

    
    def plot_all_2d(self, f='meanmax', zmax=None, usestd=True):
        plt.figure()
        for i,dt in enumerate(self.dts,1):
            ax = plt.subplot(2,4,i,projection='3d')
            self.plot_2d(dt,f,ax=ax, zmax=zmax, usestd=usestd)

    def plot_2d_priors(self):
        resp2d = self.build_f_grid()
        prior1d = self.build_1d_prior().reshape((self.n_ch,1))
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
            self.plot_2d(f=z_grid, usestd=False, title=title, ax=ax)

    # Following 2 functions are used in GP.py
    def sampleResp(self,x,y):
        ch = xy2ch[x][y]
        ts = random.choice(self.trains[ch][ch][0]['data'])
        resp = max(ts)
        return resp

    def xy2resp(self,x,y):
        ch = xy2ch[x][y]
        resp = self.trains[ch][ch][0]['meanmax']
        return resp

    def xy2ch(self,x,y):
        return xy2ch[x][y]

    def plot_single_means(self):
        # Plot single pulse response means
        # means = self.build_1d_prior()
        # stds = self.build_1d_prior(f='stdmax')
        # plt.errorbar(list(range(len(means))),means,stds, linestyle='None', marker='.')
        plt.figure()
        plt.suptitle("Means of maxs for 1d responses")

        _x = np.arange(0,2)
        _y = np.arange(0,5)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        z = np.vectorize(self.xy2resp)(x,y)

        ax = plt.subplot(1,1,1,projection='3d')
        surf = ax.bar3d(x,y,np.zeros_like(z),1,1,z)

    def plot_single_responses(self):
        # Plot single channel responses (why is ch 17 the biggest by far?)
        fig = plt.figure()
        fig.suptitle("Responses for all combinations of EMG and channels in 1d")
        outer = gridspec.GridSpec(2, 4, wspace=0.2, hspace=0.2)

        for emg in range(self.N_EMGS):
            inner = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=outer[emg])
            for i,ch in enumerate(self.chs):
                ax = plt.Subplot(fig, inner[i])
                ax.set_ylim([0,0.05])
                ax.set_title("{}".format(ch))
                ax.plot(self.emgdct[emg][ch][ch][0]['data'].T)
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                fig.add_subplot(ax)


    def plot_response_matrix(self, emg=2, dt=40):
        fig = plt.figure()
        plt.suptitle("Matrix of responses for EMG {} with dt={}".format(emg, dt))
        gs = gridspec.GridSpec(12,12)
        for i,ch in enumerate(self.chs):
            ax = plt.subplot(gs[0,i+2])
            ax.set_ylim([0,0.05])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(ch)
            ax.plot(self.emgdct[emg][ch][ch][0]['data'].T)
            ax.text(0,0,"{:.2}".format(self.emgdct[emg][ch][ch][0]['meanmax']))

            ax = plt.subplot(gs[i+2,0])
            ax.set_ylim([0,0.05])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(ch)
            ax.plot(self.emgdct[emg][ch][ch][0]['data'].T)

        for i,ch1 in enumerate(self.chs):
            for j,ch2 in enumerate(self.chs):
                ax = plt.subplot(gs[i+2,j+2])
                ax.set_ylim([0,0.05])
                ax.set_xticks([])
                ax.set_yticks([])
                bbox = None
                if ch1!=ch2:
                    plt.plot(self.emgdct[emg][ch1][ch2][dt]['data'].T)
                    mm = self.emgdct[emg][ch1][ch2][dt]['meanmax']
                    if mm > 0.01:
                        bbox = dict(facecolor='red', alpha=0.5)
                    plt.text(0,0,"{:.2}".format(mm), bbox=bbox)
                else:
                    plt.plot(self.emgdct[emg][ch1][ch1][dt]['data'].T)
                    mm = self.emgdct[emg][ch1][ch1][dt]['meanmax']
                    if mm > 0.01:
                        bbox = dict(facecolor='red', alpha=0.5)
                    plt.text(0,0,"{:.2}".format(mm), bbox=bbox)

if __name__ == "__main__":
    # trainsC = Trains(emg=EMG)
    # trainsC.build_f_grid_1d()
    # print(trains.build_f_grid())
    # print(trains.build_1d_prior())
    # trains.plot_2d(dt=40, usestd=False)
    # trains.plot_all_2d('meanmax', zmax=0.04, usestd=False)
    # trains.plot_2d_priors()
    # trains.plot_single_means()
    # trains.plot_single_responses()
    for dt in DTS:
        trainsC.plot_response_matrix(emg=4, dt=dt)
    plt.show()
