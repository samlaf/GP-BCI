import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, date, time
import matplotlib.gridspec as gridspec
import random
import itertools

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
                    dct[ch1][ch2][0]['maxs'] = dct[ch2][ch1][0]['maxs'] = maxs
                    dct[ch1][ch2][0]['meanmax'] = dct[ch2][ch1][0]['meanmax'] = maxs.mean()
                    dct[ch1][ch2][0]['stdmax'] = dct[ch2][ch1][0]['stdmax'] = maxs.std()
                    # Then we deal with dt!=0
                    for k,dt in enumerate(self.dts[1:],1):
                        dct[ch1][ch2][dt] = {'data': filtdata[emg,i,j,k]}
                        # We also precompute the meanmax and stdmax
                        # statistics
                        ch = dct[ch1][ch2][dt]['data']
                        maxs = ch.max(axis=1)
                        dct[ch1][ch2][dt]['maxs'] = maxs
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

    def build_f_grid(self, emg=2, syn=None, dt=40, f='meanmax'):
        if syn is None:
            syn = (emg, None)
        z_grid = np.zeros((self.n_ch,self.n_ch))
        for i,ch1 in enumerate(self.chs):
            for j,ch2 in enumerate(self.chs):
                #TODO: now with synergies we can only do f='meanmax'
                #      maybe change this later if needed
                z_grid[i][j] = self.synergy(*syn,ch1,ch2,dt).max(axis=1).mean()
                #self.emgdct[emg][ch1][ch2][dt][f]
        return z_grid

    def build_f_grid_1d(self, f='meanmax'):
        #only dt=0 makes sense in 1d so we don't need dt as argument
        grid = np.zeros((2,5))
        for ch in self.chs:
            x,y = ch2xy[ch]
            grid[x][y] = self.trains[ch][ch][0][f]
        return grid

    def synergy(self, emg1, emg2, ch1, ch2, dt=0, tau=40, a=1, b=1):
        # synergy is just a linear combination of emgs
        # We use these to define a new cost function (max synergy
        # instead of max of a particular channel)
        # dt is dt between stim pulses
        # tau is how much we shift resp2
        if emg2 is None:
            emg2=0
            b=0
        resps1 = self.emgdct[emg1][ch1][ch2][dt]['data']
        resps2 = self.emgdct[emg2][ch1][ch2][dt]['data']
        # resps are 1466 (resps1.shape[1]) ticks ts, which last 300ms
        dtidx = int(tau/300*resps1.shape[1])+1
        resps2_shifted = np.zeros_like(resps2)
        resps2_shifted[:,:-dtidx] = resps2[:,dtidx:]
        return a*resps1 + b*resps2_shifted

    def max_ch_2d(self, emg=2, dt=40, syn=None):
        if syn is None:
            syn = (emg,None)
        grid = self.build_f_grid(syn=syn, dt=dt)
        x,y = np.unravel_index(grid.argmax(), grid.shape)
        return [self.chs[x], self.chs[y]]

    def max_ch_1d(self):
        grid = self.build_f_grid_1d()
        x,y = np.unravel_index(grid.argmax(), grid.shape)
        return xy2ch[x][y]

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

    #################  PLOTTING ###########
    
    def plot_emg_resps(self, ch1, ch2, emg=4, dt=0, n=None, ax=None):
        if ax is None:
            fig,ax = plt.subplots(1)
        resps = self.emgdct[emg][ch1][ch2][dt]['data']
        if n:
            resps = np.array(random.choices(resps, k=n))
        ax.plot(resps.T)
        ax.set_title('emg={}, dt={}, ch1={}, ch2={}'.format(emg,dt,ch1,ch2))

    def plot_ch2emg_resps(self, chs=None, n=None, avgResp=True, ylim=0.025, lw_avg=3, emgs=[0,1,2,4,5,6,(0,4,40),(4,0,40)], stimLine=True):
        if chs is None:
            chs = self.chs; n_ch = self.n_ch
        else:
            n_ch = len(chs)
        fig,axes = plt.subplots(n_ch, len(emgs), sharex=True, sharey=True)
        axes[0][0].set_ylim(0,ylim) #note this will change all axes
        for i,ch in enumerate(chs):
            for j,emg in enumerate(emgs):
                if not hasattr(emg, "__iter__"):
                    resps = self.emgdct[emg][ch][ch][0]['data']
                else: #synergy
                    emg1,emg2,tau = emg
                    resps = self.synergy(emg1,emg2,dt=0, tau=tau, ch1=ch, ch2=ch)
                if avgResp:
                    avgresp = resps.mean(axis=0)
                if n:
                    resps = np.array(random.choices(resps, k=n))
                axes[i][j].plot(resps.T)
                if avgResp:
                    axes[i][j].plot(avgresp, color='k', linewidth=lw_avg, label='avg resp')
                if stimLine:
                    lenresps = resps.shape[1]
                    axes[i][j].axvline(lenresps/2, color='r')
        #add labels
        for i,ch in enumerate(chs):
            axes[i][0].set_ylabel('ch {}'.format(ch))
        for j,emg in enumerate(emgs):
            axes[-1][j].set_xlabel('emg {}'.format(emg))
        #add title
        if avgResp:
            fig.suptitle("Showing avg emg resps for each ch")
        elif n:
            fig.suptitle("Showing {} rnd emg resps for each ch".format(n))
        else:
            fig.suptitle("Showing emg resps for each ch")
        axes[0][0].legend()

    def plot_chs2emg_resps(self, chs1=13, chs2=[], dts=0, emgs=[0,1,2,4,5,6,(0,4,40),(4,0,40)], n=None, avgResp=True, ylim=0.025, stimLine=True):
        # emg can be either an emg or a synergy
        if not hasattr(chs1, "__iter__"): chs1=[chs1]
        if not hasattr(chs2, "__iter__"): chs2=[chs2]
        if not hasattr(dts, "__iter__"): dts=[dts];
        if chs1==[]: chs1=self.chs
        if chs2==[]: chs2=self.chs
        if dts==[]: dts=self.dts
        if len(chs1)!= 1 or len(chs2) != 1:
            assert len(dts)==1
        # what variable to loop y axis on
        if len(dts)==1:
            fig,axes = plt.subplots(len(chs1)*len(chs2), len(emgs), sharex=True, sharey=True)
            fig.suptitle("Showing emg resps for dt={}".format(dts[0]))
            axes[0][0].set_ylim(0,ylim)
            for i,(ch1,ch2) in enumerate(itertools.product(chs1,chs2)):
                for j,emg in enumerate(emgs):
                    if not hasattr(emg, "__iter__"):
                        resps = self.emgdct[emg][ch1][ch2][dts[0]]['data']
                    else: #synergy
                        emg1,emg2,tau = emg
                        resps = self.synergy(emg1,emg2,dt=dts[0], tau=tau, ch1=ch1, ch2=ch2)
                    lenresps = resps.shape[1]
                    if avgResp:
                        avgresp = resps.mean(axis=0)
                    if n:
                        resps = np.array(random.choices(resps, k=n))
                    axes[i][j].plot(resps.T)
                    if avgResp:
                        axes[i][j].plot(avgresp, color='k', linewidth=3, label='avg resp')
                    if stimLine:
                        axes[i][j].axvline(lenresps/2, color='r')
                        axes[i][j].axvline(lenresps/2 + dts[0]/300*lenresps, color='r')
                axes[i][0].set_ylabel('chs1={}, ch2={}'.format(ch1,ch2))
        else:
            # chs1 and ch2 are fixed, we vary dt
            fig,axes = plt.subplots(len(dts), len(emgs), sharex=True, sharey=True)
            fig.suptitle("Showing emg resps for ch1={}, ch2={}".format(chs1[0],chs2[0]))
            axes[0][0].set_ylim(0,ylim)
            for i,dt in enumerate(dts):
                for j,emg in enumerate(emgs):
                    if not hasattr(emg, "__iter__"):
                        resps = self.emgdct[emg][chs1[0]][chs2[0]][dt]['data']
                    else: #synergy
                        emg1,emg2,tau = emg
                        resps = self.synergy(emg1,emg2,dt=dt, tau=tau, ch1=chs1[0], ch2=chs2[0])
                    lenresps = resps.shape[1]
                    if avgResp:
                        avgresp = resps.mean(axis=0)
                    if n:
                        resps = np.array(random.choices(resps, k=n))
                    axes[i][j].plot(resps.T)
                    if avgResp:
                        axes[i][j].plot(avgresp, color='k', linewidth=3, label='avg resp')
                    if stimLine:
                        axes[i][j].axvline(lenresps/2, color='r')
                        axes[i][j].axvline(lenresps/2 + dt/300*lenresps, color='r')
                axes[i][0].set_ylabel('dt={}'.format(dt))
                
        for j,emg in enumerate(emgs):
            axes[-1][j].set_xlabel('emg {}'.format(emg))
        axes[0][0].legend()

    def plot_single_means(self):
        plt.figure()
        plt.suptitle("Means of maxs for 1d responses")

        _x = np.arange(0,2)
        _y = np.arange(0,5)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        z = np.vectorize(self.xy2resp)(x,y)

        ax = plt.subplot(1,1,1,projection='3d')
        surf = ax.bar3d(x,y,np.zeros_like(z),1,1,z)

    def plot_all_pair_responses(self, emg=2, dt=0, n=None):
        fig, axes = plt.subplots(self.n_ch, self.n_ch, sharex=True, sharey=True)
        for i,ch1 in enumerate(self.chs):
            for j,ch2 in enumerate(self.chs):
                resps = self.emgdct[emg][ch1][ch2][dt]['data']
                if n:
                    resps = np.array(random.choices(resps, k=n))
                axes[i][j].plot(resps.T)

    def plot_all_single_responses(self):
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


    def plot_response_matrix(self, emg=2, syn=None, dt=40):
        if syn is None:
            syn = (emg, None)
        fig = plt.figure()
        plt.suptitle("Matrix of responses for EMG1={},EMG2={} with dt={} (1stch left, 2ndch top)".format(*syn, dt))
        gs = gridspec.GridSpec(12,12)
        # We first plot the 1d responses on left and top
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

        
        maxch1,maxch2 = self.max_ch_2d(syn=syn,dt=dt)
        maxr = self.synergy(*syn,maxch1,maxch2,dt).max(axis=1).mean()
        for i,ch1 in enumerate(self.chs):
            for j,ch2 in enumerate(self.chs):
                ax = plt.subplot(gs[i+2,j+2])
                ax.set_ylim([0,0.05])
                ax.set_xticks([])
                ax.set_yticks([])
                bbox = None
                data = self.synergy(*syn,ch1,ch2,dt)
                plt.plot(data.T)
                mm = data.max(axis=1).mean()
                if ch1==maxch1 and ch2==maxch2:
                    bbox = dict(facecolor='green', alpha=0.5)
                elif mm > maxr - 0.005:
                    bbox = dict(facecolor='red', alpha=0.5)
                plt.text(0,0,"{:.2}".format(mm), bbox=bbox)

if __name__ == "__main__":
    trainsC = Trains(emg=EMG)
    # trainsC.synergy(0,4,13,13,dt=0)
    for dt in [0]:
        trainsC.plot_response_matrix(emg=4, dt=dt)
    trainsC.plot_response_matrix(syn=(0,4),dt=0)
    # trainsC.plot_all_pair_responses(dt=40,n=5)
    # trainsC.plot_ch2emg_resps(avgResp=True)
    # trainsC.plot_chs2emg_resps(chs1=17,chs2=17,dts=[])
    # trainsC.plot_chs2emg_resps(chs1=[13,17],chs2=[13,17],dts=40)
    plt.show()
