"""
We fit GPs to the full dataset, testing different models and kernels
"""

# Idea:
# Query new points twice + fit heteroskedastic noise to empirical data

from load_matlab import *
import numpy as np
import GPy
import matplotlib.pyplot as plt
from numpy import linalg as LA
import argparse
import os
from os import path
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--uid', type=int, default=0, help='uid for job number')
parser.add_argument('--emg', type=int, default=2, choices=range(7), help='emg. between 0-6')

# DEFAULT
# there is no dt in 1d. But we need this to access the trains dct
dt=0

def make_dataset_1d(trains, mean=False, n=20):
    # n means number of datapt per channel
    # Used to test/debug certain models
    X = []
    Y = []
    Xmean = []
    Ymean = []
    Yvars = []
    for ch in CHS:
        ys = random.sample(trains[ch][ch][dt]['data'].max(axis=1).tolist(),n)
        Y.extend(ys)
        var = trains[ch][ch][dt]['stdmax'] ** 2
        Yvars.extend([var] * len(ys))
        xy = ch2xy[ch]
        X.extend([xy]*len(ys))
        Ymean.append(trains[ch][ch][dt]['meanmax'])
        Xmean.extend([xy])
    if mean:
        Xmean = np.array(Xmean)
        Ymean = np.array(Ymean).reshape((-1,1))
        return Xmean, Ymean
    X = np.array(X)
    Y = np.array(Y).reshape((-1,1))
    return X,Y

# Heteroskedastic model
# Yvars = np.array(Yvars).reshape((-1,1))
# matk = GPy.kern.Matern52(input_dim=2, ARD=True)
# mhetefix = GPy.models.GPHeteroscedasticRegression(X,Y,matk)
# # We use the real noise
# mhetefix.het_Gauss.variance = Yvars
# mhetefix.het_Gauss.fix()
# mhetefix.optimize(messages=True,max_f_eval = 1000)

def train_models_1d(X,Y, model_names=['homo'], num_restarts=5, ARD=True):
    if model_names == 'all':
        model_names = ['homo', 'hete']
    models = []
    if 'hete' in model_names:
        matk = GPy.kern.Matern52(input_dim=2, ARD=ARD)
        mhete = GPy.models.GPHeteroscedasticRegression(X,Y,matk)
        mhete.optimize(messages=True,max_f_eval = 1000)
        models.append(mhete)

    if 'homo' in model_names:
        matk = GPy.kern.Matern52(input_dim=2, ARD=ARD)
        mhomo = GPy.models.GPRegression(X,Y,matk)
        mhomo.optimize_restarts(num_restarts=num_restarts)
        models.append(mhomo)

    return models

def train_model_seq(trains, n_random_pts=10, n_total_pts=15, ARD=True):
    X = []
    Y = []
    for _ in range(n_random_pts):
        ch = random.choice(CHS)
        X.append(ch2xy[ch])
        resp = random.choice(trains[ch][ch][dt]['data'].max(axis=1))
        Y.append(resp)
    matk = GPy.kern.Matern52(input_dim=2, ARD=ARD)
    #Make model
    models = []
    m = GPy.models.GPRegression(np.array(X),np.array(Y)[:,None],matk)
    m.optimize()
    # We optimize this kernel once and then use it for all future models
    matk = m.kern
    gnoise = m.Gaussian_noise.variance
    models.append(m)
    for _ in range(n_total_pts-n_random_pts):
        nextx = get_next_x(m)
        X.append(nextx)
        ch = xy2ch[nextx[0]][nextx[1]]
        resp = random.choice(trains[ch][ch][dt]['data'].max(axis=1))
        Y.append(resp)
        m = GPy.models.GPRegression(np.array(X), np.array(Y)[:,None],matk.copy())
        m.Gaussian_noise.variance = gnoise.copy()
        ## TODO: also set gp's noise variance to be same as previous!
        models.append(m)
    return models

def get_acq_map(m, k=1):
    # We use UCB, k is the "exploration" parameter
    mean,var = m.predict(np.array(list(ch2xy.values())))
    std = np.sqrt(var)
    acq = mean + k*std
    return acq

def get_next_x(m, k=1):
    acq = get_acq_map(m,k)
    maxidx = acq.argmax()
    maxch = CHS[maxidx]
    xy = ch2xy[maxch]
    return xy
    
def plot_model_1d(m, title=None, plot_acq=False, plot_data=True):
    print(m)
    print(m.kern)
    print(m.kern.lengthscale)

    fig, axes = plt.subplots(2,1,
                             sharex=True,
                             sharey=True)
    lengthscales = [m.Mat52.lengthscale[i] for i in range(len(m.Mat52.lengthscale))]
    fig.suptitle(("{}: ls="+" {:.2} "*len(lengthscales)).format(title,*lengthscales))
    for i,ax in zip([0,1],axes):
        m.plot(ax=ax, fixed_inputs=[(0,i)],
               plot_data=False, title='Channels {}'.format(xy2ch[i]),
               lower=17, upper=83)
        # Plot data (m.plot plots all of the data in every slice, which is
        # wrong)
    if plot_data:
        for (i,j),y in zip(m.X, m.Y):
            i,j = int(i), int(j)
            axes[i].plot(j, y, 'x', color='C{}'.format(j))
    if plot_acq:
        acqmap = get_acq_map(m)
        axes[1].plot(acqmap[:5])
        axes[0].plot(acqmap[5:])
        # for j,ch in enumerate(xy2ch[i]):
        #     ax.plot(np.ones(20)*j,trains[ch][ch][dt]['data'].max(axis=1),'x')
        #     ax.plot(j, trains[ch][ch][dt]['meanmax'], 'r+')
        #     ax.errorbar(j, trains[ch][ch][dt]['meanmax'], yerr=2*trains[ch][ch][dt]['stdmax'],ecolor='r')

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

def run_dists_exps(args):
    exppath = path.join('exps', '1d', 'emg{}'.format(args.emg), 'exp{}'.format(args.uid))
    if not path.isdir(exppath):
        os.makedirs(exppath)
    trainsC = Trains(emg=args.emg)
    trains = trainsC.trains

    # We train all models with n rnd start pts and m sequential pts
    # And compare them to the model trained with all datapts
    # Then compute statistics and plot them
    X,Y = make_dataset_1d(trains)
    mfull, = train_models_1d(X,Y, ARD=False)
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
                models = train_model_seq(trains,n_random_pts=n1, n_total_pts=n1+n2, ARD=False)
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
    plt.savefig(os.path.join(exppath, "2d_l2{}_{}.png".format(name,k)))
    plt.close()

    plt.figure()
    plt.imshow(linfs.mean(axis=0), extent=[0,50,50,5])
    plt.title("1d linf dist to true gp mean")
    plt.ylabel("N random pts")
    plt.xlabel("N sequential")
    plt.colorbar()
    plt.savefig(os.path.join(exppath, "2d_l2{}_{}.png".format(name,k)))
    plt.close()

if __name__ == '__main__':
    # trainsC = Trains(emg=args.emg)
    # trains = trainsC.trains
    # X,Y = make_dataset_1d(trains)
    # m1, = train_models_1d(X,Y)
    # m2, = train_models_1d(X,Y, ARD=False)
    # plot_model_1d(m1, title="homoskedastic")
    # plot_model_1d(m2, title="homoskedastic")
    # plt.show()

    # test seq model
    # models = train_model_seq(trains, n_random_pts=10, n_total_pts=20, ARD=False)
    # plot_model_1d(models[-1], plot_acq=True)
    # plt.show()

    args = parser.parse_args()
    run_dists_exps(args)
