"""
We fit GPs to the full dataset, testing different models and kernels
"""

# Idea:
# Query new points twice + fit heteroskedastic noise to empirical data
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from load_matlab import *
import numpy as np
import GPy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numpy import linalg as LA
import argparse
import os
from os import path
import itertools
import pickle
import operator
import matplotlib.cm as cm

parser = argparse.ArgumentParser()
parser.add_argument('--uid', type=int, default=2, help='uid for job number')
parser.add_argument('--emg', type=int, default=2, choices=range(7), help='emg. between 0-6')

# DEFAULT
# there is no dt in 1d. But we need this to access the trains dct
dt=0

def make_dataset_1d(trainsC, emg=2, mean=False, n=20):
    # n means number of datapt per channel
    # Used to test/debug certain models
    trains = trainsC.get_emgdct(emg=emg)
    X = []
    Y = []
    Xmean = []
    Ymean = []
    Yvars = []
    for ch in CHS:
        ys = random.sample(trains[ch][ch][0]['data'].max(axis=1).tolist(),n)
        Y.extend(ys)
        var = trains[ch][ch][0]['stdmax'] ** 2
        Yvars.extend([var] * len(ys))
        xy = ch2xy[ch]
        X.extend([xy]*len(ys))
        Ymean.append(trains[ch][ch][0]['meanmax'])
        Xmean.extend([xy])
    if mean:
        Xmean = np.array(Xmean)
        Ymean = np.array(Ymean).reshape((-1,1))
        return Xmean, Ymean
    X = np.array(X)
    Y = np.array(Y).reshape((-1,1))
    return X,Y

def train_model_1d(X,Y, num_restarts=3, ARD=True, constrain=[0.3,3.0], verbose=True):

    matk = GPy.kern.Matern52(input_dim=2, ARD=ARD)
    if constrain:
        matk.lengthscale.constrain_bounded(*constrain, warning=verbose)
    m = GPy.models.GPRegression(X,Y,matk)
    m.optimize_restarts(num_restarts=num_restarts, verbose=verbose)

    return m

### We try the co-kriging (multi output GP) approach to learn all EMGS
### at the same time
def make_mo_dataset(trainsC, emgs=[0,1,2,4,5,6]):
    X = []
    Y = []
    for emg in emgs:
        trains = trainsC.get_emgdct(emg=emg)
        for ch in CHS:
            ys = trains[ch][ch][0]['maxs']
            Y.extend(ys)
            xy = ch2xy[ch]
            xyemg = xy + [emg]
            X.extend([xyemg]*len(ys))
    X = np.array(X)
    Y = np.array(Y).reshape((-1,1))
    return X,Y

def train_mo_model(X,Y, ARD=True, num_restarts=1):
    #first make separable kernel
    # kx = GPy.kern.Matern52(input_dim=2, active_dims=[0,1], ARD=ARD)
    # ky = GPy.kern.Matern52(input_dim=1, active_dims=[2])
    # k = kx + ky
    k = GPy.kern.Matern52(input_dim=3, ARD=True)
    
    m = GPy.models.GPRegression(X,Y,k)
    m.optimize_restarts(num_restarts=num_restarts)
    return m

def plot_mo_model(m, plot_data=True):
    emgs = np.unique(m.X[:,2]).tolist()
    fig, axes = plt.subplots(2,len(emgs),sharex=True,sharey=True)
    fig.suptitle("Co-kriging model")
    for j,emg in enumerate(emgs):
        for i in [0,1]:
            ax=axes[i][j]
            m.plot(ax=ax, fixed_inputs=[(0,i), (2,emg)],
                   plot_data=False, lower=17, upper=83,
                   legend=False)
        axes[0][j].set_title("emg {}".format(emg))
    if plot_data:
        t=1
        norm = colors.Normalize(vmin=-50, vmax=len(m.X))
        for (i,j,emg),y in zip(m.X, m.Y):
            i,j = int(i), int(j)
            k = emgs.index(emg)
            axes[i][k].plot(j, y, 'x', color='C{}'.format(j))
            t+=1

##### end of co-kriging section ######

def train_model_seq(trainsC, emg=0, n_random_pts=10, n_total_pts=25, ARD=True, num_restarts=3, continue_opt=False, k=2, constrain=[0.3,3.0], verbose=True):
    X = []
    Y = []
    for _ in range(n_random_pts):
        ch = random.choice(CHS)
        X.append(ch2xy[ch])
        resp = random.choice(trainsC.emgdct[emg][ch][ch][dt]['data'].max(axis=1))
        Y.append(resp)
    matk = GPy.kern.Matern52(input_dim=2, ARD=ARD)
    if constrain:
        matk.lengthscale.constrain_bounded(*constrain, warning=verbose)
    #Make model
    models = []
    m = GPy.models.GPRegression(np.array(X),np.array(Y)[:,None],matk)
    m.optimize_restarts(num_restarts=num_restarts, verbose=verbose)
    # We optimize this kernel once and then use it for all future models
    optim_params = m[:]
    models.append(m)
    for _ in range(n_total_pts-n_random_pts):
        nextx = get_next_x(m, k=k)
        X.append(nextx)
        ch = xy2ch[nextx[0]][nextx[1]]
        resp = random.choice(trainsC.emgdct[emg][ch][ch][dt]['data'].max(axis=1))
        Y.append(resp)
        m = GPy.models.GPRegression(np.array(X), np.array(Y)[:,None],matk.copy())
        m[:] = optim_params
        ## TODO: also set gp's noise variance to be same as previous!
        if continue_opt:
            m.optimize_restarts(num_restarts=num_restarts, verbose=verbose)
        models.append(m)
    return models

def plot_seq_values(m, n_random_pts, trainsC=None, ax=None, legend=False):
    if ax is None:
        ax = plt.figure()
    ax.plot(m.Y[:n_random_pts+1], c='b', label="{} random init pts".format(n_random_pts))
    ax.plot(range(n_random_pts, len(m.Y)), m.Y[n_random_pts:,:], c='r', label="Sequential pts")
    ax.set_title("Value of selected channel")
    if trainsC:
        maxch = trainsC.max_ch_1d()
        for i,resp in enumerate(m.Y):
            x,y = m.X[i]
            ch = xy2ch[int(x)][int(y)]
            if ch == maxch:
                ax.plot(i, resp, 'x', c='k')
    if legend:
        ax.legend()

def plot_conseq_dists(m, n_random_pts, trainsC = None, ax=None, legend=False):
    if ax is None:
        ax = plt.figure()
    dists = [LA.norm(m.X[i]-m.X[i+1]) for i in range(len(m.X)-1)]
    ax.plot(dists[:n_random_pts+1], c='b', label="{} random init pts".format(n_random_pts))
    ax.plot(range(n_random_pts, len(dists)), dists[n_random_pts:], c='r', label="Sequential pts")
    ax.set_title("Distance between consecutive channels")
    if trainsC:
        maxch = trainsC.max_ch_1d()
        label="max channel ({})".format(maxch)
        for i,dist in enumerate(dists):
            x,y = m.X[i]
            ch = xy2ch[int(x)][int(y)]
            if ch == maxch:
                ax.plot(i, dist, 'x', c='k', label=label)
                label=None
    if legend:
        ax.legend()

def plot_seq_stats(m, n_random_pts, trainsC=None, title=None, plot_acq=False, plot_gp=True):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,2)
    ax2 = fig.add_subplot(2,2,4)
    # We first plot sequential values
    plot_seq_values(m, n_random_pts, trainsC=trainsC, ax=ax1)
    # Then distance between consecutive x's
    plot_conseq_dists(m, n_random_pts, trainsC=trainsC, ax=ax2, legend=True)
    # Finally we plot the GP with the data points
    ax3 = fig.add_subplot(2,2,1)
    ax4 = fig.add_subplot(2,2,3,sharex=ax3, sharey=ax3)
    axes = [ax3,ax4]
    lengthscales = [m.Mat52.lengthscale[i] for i in range(len(m.Mat52.lengthscale))]
    fig.suptitle(("{}: ls="+" {:.2} "*len(lengthscales)).format(title,*lengthscales))
    for i,ax in zip([0,1],axes):
        m.plot(ax=ax, fixed_inputs=[(0,i)],
               plot_data=False, title='Channels {}'.format(xy2ch[i]),
               lower=17, upper=83, legend=False)
    # Plot data (m.plot plots all of the data in every slice, which is
    # wrong)
    axes[int(m.X[0][0])].plot(int(m.X[0][1]), m.Y[0][0], '+', label="{} random init pts".format(n_random_pts))
    for (i,j),y in zip(m.X[1:n_random_pts,:], m.Y[1:n_random_pts,:]):
        i,j = int(i), int(j)
        axes[i].plot(j, y, '+', c='b')
    t=1
    norm = colors.Normalize(vmin=0, vmax=len(m.X)-n_random_pts)
    for (i,j),y in zip(m.X[n_random_pts:,:], m.Y[n_random_pts:,:]):
        i,j = int(i), int(j)
        axes[i].plot(j, y, 'x', color=plt.cm.Reds(norm(t)))
        t+=1
    if plot_acq:
        acqmap = get_acq_map(m)
        axes[1].plot(acqmap[:5], c='y', label='acq fct')
        axes[0].plot(acqmap[5:], c='y', label='acq fct')
    axes[0].legend()

def get_acq_map(m, k=2):
    # We use UCB, k is the "exploration" parameter
    mean,var = m.predict(np.array(list(ch2xy.values())))
    std = np.sqrt(var)
    acq = mean + k*std
    return acq

def get_next_x(m, k=2):
    acq = get_acq_map(m,k)
    maxidx = acq.argmax()
    maxch = CHS[maxidx]
    xy = ch2xy[maxch]
    return xy
    
def plot_model_1d(m, title=None, plot_acq=False, plot_data=True, verbose=True):
    if verbose:
        print(m)
        print(m.kern)
        print(m.kern.lengthscale)

    fig, axes = plt.subplots(2,1,
                             sharex=False,
                             sharey=True)
    #lengthscales = [m.Mat52.lengthscale[i] for i in range(len(m.Mat52.lengthscale))]
    fig.suptitle(title)
    #"*len(lengthscales)).format(title,*lengthscales))
    for i,ax in zip([0,1],axes):
        m.plot(ax=ax, fixed_inputs=[(0,i)],
               plot_data=False, lower=17, upper=83, legend=False)
        ax.set_xticks([0,1,2,3,4])
        ax.set_xticklabels([[i,j] for j in range(5)])
    if plot_data:
        t=1
        norm = colors.Normalize(vmin=-50, vmax=len(m.X))
        for (i,j),y in zip(m.X, m.Y):
            i,j = int(i), int(j)
            axes[i].plot(j, y, 'x', color='C{}'.format(j))
            t+=1
    if plot_acq:
        acqmap = get_acq_map(m)
        axes[1].plot(acqmap[:5], c='y', label='acq fct')
        axes[0].plot(acqmap[5:], c='y', label='acq fct')
    axes[0].legend()

def plot_model_surface(m, ax=None, plot_data=True, zlim=None, extra_xlim=1, plot_colorbar=True):
    x = np.linspace(0-extra_xlim,1+extra_xlim,50)
    y = np.linspace(0-extra_xlim,4+extra_xlim,50)
    x,y = np.meshgrid(x,y)
    x_,y_ = x.ravel()[:,None], y.ravel()[:,None]
    z = np.hstack((x_,y_))
    mean,var = m.predict(z)
    std = np.sqrt(var)
    mean,std = mean.reshape(50,50), std.reshape(50,50)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    norm = plt.Normalize()
    surf = ax.plot_surface(x, y, mean, linewidth=0, antialiased=False, facecolors=cm.jet(norm(std)))
    if zlim:
        ax.set_zlim([0,0.014])
    if plot_data:
        ax.scatter(m.X[:,0], m.X[:,1], m.Y, c='g')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('V')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1,2,3,4])
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    if plot_colorbar:
        cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])  # This is the position for the colorbar
        clb = plt.colorbar(m, cax = cbaxes)
        clb.ax.yaxis.set_ticks_position('left')
        clb.ax.set_title('std')


def l2dist(m1, m2):
    X = np.array(list(itertools.product(range(2), range(5))))
    pred1 = m1.predict(X)[0]
    pred2 = m2.predict(X)[0]
    return LA.norm(pred1-pred2)
def linfdist(m1, m2):
    X = np.array(list(itertools.product(range(2), range(5))))
    pred1 = m1.predict(X)[0]
    pred2 = m2.predict(X)[0]
    return abs((pred1.max() - pred2.max())/pred2.max())

def run_dist_exps(args):
    exppath = path.join('exps', '1d', 'emg{}'.format(args.emg), 'exp{}'.format(args.uid))
    if not path.isdir(exppath):
        os.makedirs(exppath)
    trainsC = Trains(emg=args.emg)
    trains = trainsC.get_emgdct(emg=args.emg)
    
    # We train all models with n rnd start pts and m sequential pts
    # And compare them to the model trained with all datapts
    # Then compute statistics and plot them
    X,Y = make_dataset_1d(trainsC, emg=args.emg)
    mfull = train_model_1d(X,Y, ARD=False)
    nrnd = range(5,50,5)
    nseq = range(0,50,5)
    N = 50
    l2s = np.zeros((N, len(nrnd),len(nseq)))
    linfs = np.zeros((N, len(nrnd),len(nseq)))
    for k in range(N):
        print("Starting loop", k)
        for i,n1 in enumerate(nrnd):
            for j,n2 in enumerate(nseq):
                print(n1,n2)
                models = train_model_seq(trainsC,emg=args.emg, n_random_pts=n1, n_total_pts=n1+n2, ARD=False)
                m = models[-1]
                l2 = l2dist(m, mfull)
                linf = linfdist(m, mfull)
                l2s[k][i][j] = l2
                linfs[k][i][j] = linf
        np.save(os.path.join(exppath,"l2s"), l2s)
        np.save(os.path.join(exppath, "linfs"), linfs)

    plt.imshow(l2s.mean(axis=0), extent=[0,50,50,5])
    plt.title("1d l2 dist to true gp mean")
    plt.ylabel("N random pts")
    plt.xlabel("N sequential")
    plt.colorbar()
    plt.savefig(os.path.join(exppath, "1d_l2.pdf"))
    plt.close()

    plt.figure()
    plt.imshow(linfs.mean(axis=0), extent=[0,50,50,5])
    plt.title("1d linf dist to true gp mean")
    plt.ylabel("N random pts")
    plt.xlabel("N sequential")
    plt.colorbar()
    plt.savefig(os.path.join(exppath, "1d_linf.pdf"))
    plt.close()

def get_ch(xy):
    x,y = xy
    x = int(x)
    y = int(y)
    return xy2ch[x][y]

def get_maxch(m):
    X = np.array(list(itertools.product(range(2),range(5))))
    means,_ = m.predict(X)
    maxidx = means.argmax()
    maxxy = np.unravel_index(maxidx, (2,5))
    maxch = get_ch(maxxy)
    return maxch

def get_nmaxch(m, n=3):
    if n==0:
        return []
    X = np.array(list(itertools.product(range(2),range(5))))
    means,_ = m.predict(X)
    indexed = list(zip(X.tolist(), means.flatten()))
    top_3 = sorted(indexed, key=operator.itemgetter(1))[-n:]
    xys = list(reversed([xy for xy, v in top_3]))
    return xys

def run_ch_stats_exps(trainsC, emgs=[0,2,4], repeat=25, uid=None, jobid=None, continue_opt=True, k=2, ntotal=100, nrnd=[5,75,10], ARD=True):
    # here we run a bunch of runs, gather all statistics and save as
    # npy array, to later plot in jupyter notebook
    if uid is None:
        uid = random.randrange(99999)
    exppath = path.join('exps', '1d', 'exp{}'.format(uid), 'k{}'.format(k), 'ARD{}'.format(ARD))
    print("Will save in path", exppath)
    if not path.isdir(exppath):
        os.makedirs(exppath)
    if jobid:
        # We save the name of the jobid in a file, so that if the
        # experiment fails, we can ask sacct for info
        filename = os.path.join(exppath, 'sbatchjobid={}'.format(jobid))
        with open(filename, 'w') as f:
            f.write('sbatcjobid = {}'.format(jobid))
    nrnd = range(*nrnd)
    dct = {}
    for emg in emgs:
        print("Starting emg {}".format(emg))
        queriedchs = np.zeros((repeat, len(nrnd), ntotal))
        maxchs = np.zeros((repeat, len(nrnd), ntotal))
        # we save all values the model predicted (for 10 chs)
        # from this we can compute l2 and linf distances later
        vals = np.zeros((repeat, len(nrnd), ntotal, 10))
        for r in range(repeat):
            print("Repeat", r)
            for i,n1 in enumerate(nrnd):
                print("nrnd: {}".format(n1))
                models = train_model_seq(trainsC, emg=emg, n_random_pts=n1,
                                         n_total_pts=ntotal, ARD=ARD,
                                         continue_opt=continue_opt, num_restarts=1, k=k)
                queriedchs[r][i] = [get_ch(xy) for xy in models[-1].X]
                for midx,m in enumerate(models,n1-1):
                    maxchs[r][i][midx] = get_maxch(m)
                X = np.array(list(itertools.product(range(2),range(5))))
                vals[r,i,n1-1:] = np.hstack([m.predict(X)[0] for m in models]).T
        dct[emg] = {
            'queriedchs': queriedchs,
            'maxchs': maxchs,
            'vals': vals,
            'nrnd': nrnd,
            'ntotal': ntotal,
            'true_ch': trainsC.max_ch_1d(emg=emg),
            'k': k
        }
    with open(os.path.join(exppath, 'chruns_dct.pkl'), 'wb') as f:
        print("Saving in path", exppath)
        pickle.dump(dct, f)
    return dct

if __name__ == '__main__':
    args = parser.parse_args()
    trainsC = Trains(emg=args.emg)

    for n in [25,20,15,10,5,1]:
        m, = train_model_seq(trainsC, emg=0, n_random_pts=n, n_total_pts=n,k=6)
        plot_model_surface(m, plot_data=False, zlim=[0,0.014], plot_colorbar=False)
    plt.show()

    emg=0
    X,Y = make_dataset_1d(trainsC, emg=emg)
    m = train_model_1d(X,Y)
    print(m.kern.lengthscale)
    plot_model_surface(m, plot_data=False, plot_colorbar=False)
    plt.show()
    
