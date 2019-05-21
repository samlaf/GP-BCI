"""
We fit GPs to the full dataset, testing different models and kernels
"""

from load_matlab import *
from gp_full_1d import *
from gp_full_2d import make_dataset_2d, build_prior
import numpy as np
import GPy
import matplotlib.pyplot as plt
import pickle
import os, os.path as path
import h5py
import itertools
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--uid', type=str, default=1, help='uid for job number')
parser.add_argument('--dts', type=int, default=[40,60], help='list of dts. choose among (0,10,20,40,60,80,100)')
parser.add_argument('--emg', type=int, default=4, choices=range(7), help='emg. between 0-6')

# DEFAULT
dts=(40,60)
emg=2

def make_dataset_dt2d(trainsC, emg=emg, syn=None, dts=dts, means=False, n=None):
    X,Y = make_dataset_2d(trainsC=trainsC,emg=emg,syn=syn,dt=dts[0],means=means,n=n)
    X = np.hstack((X, np.ones((len(X),1))*dts[0]))
    for dt in dts[1:]:
        X_,Y_ = make_dataset_2d(trainsC=trainsC,emg=emg,syn=syn,dt=dt,means=means,n=n)
        X_ = np.hstack((X_, np.ones((len(X_),1))*dt))
        X = np.vstack((X,X_))
        Y = np.vstack((Y,Y_))
    return X,Y

def train_models_dt2d(X,Y, kerneltype='add', symkern=False, num_restarts=1, prior1d=None, optimize=True, ARD=True, dtprior=False, constrain=False):
    k1 = GPy.kern.Matern52(input_dim=2, active_dims=[0,1], ARD=ARD)
    k2 = GPy.kern.Matern52(input_dim=2, active_dims=[2,3], ARD=ARD)
    kdt = GPy.kern.Matern52(input_dim=1, active_dims=[4], lengthscale=20)
    if prior1d:
        k1.lengthscale = k2.lengthscale = prior1d.Mat52.lengthscale
        k1.variance = k2.variance = prior1d.Mat52.variance
    if constrain:
        k1.lengthscale.constrain_bounded(1,2)
        k2.lengthscale.constrain_bounded(1,2)
        k1.variance.constrain_bounded(5e-4, 1e-3)
        k2.variance.constrain_bounded(5e-4, 1e-3)
        # do we need constraints on kdt??
    if kerneltype == 'add':
        k = (k1 + k2) * kdt
    elif kerneltype == 'mult':
        k = (k1 * k2) * kdt
    else:
        raise Exception("kerneltype should be add or mult")
    if symkern:
        symM = np.zeros((5,5))
        symM[4][4] = 1
        symM[:4,:4] = np.block([[np.zeros((2,2)),np.eye(2)],[np.eye(2),np.zeros((2,2))]])
        k = GPy.kern.Symmetric(k, symM)

    if prior1d:
        m = GPy.models.GPRegression(X,Y,k, mean_function= build_prior(prior1d, dtprior=dtprior, input_dim=5))
        m.Gaussian_noise.variance = prior1d.Gaussian_noise.variance
    else:
        m = GPy.models.GPRegression(X,Y,k)
    if optimize:
        m.optimize_restarts(num_restarts=num_restarts)

    return m

def plot_model_dt2d(m, k=-1, fulldatam=None, title="", plot_acq=False):
    # We make a different plot per dt
    dts = np.unique(m.X[:,-1])

    axesdct = {}
    for dt in dts:
        fig, axes = plt.subplots(4,5, sharex=True, sharey=True)
        axesdct[dt] = axes
        fig.suptitle("dt = {}".format(dt))
        for i in [0,1]:
            for j in range(5):
                ch1 = xy2ch[i][j]
                title='ch1 = {}'.format(ch1)
                axes[2*i][j].set_title(title)
                for x2i in [0,1]:
                    ax = axes[2*i+x2i][j]
                    m.plot(ax=ax, fixed_inputs=[(0,i),(1,j),(2,x2i),(4,dt)], plot_data=False, legend=False)
    # We also plot the data
    for (x1i,x1j,x2i,x2j,dt),y in zip(m.X, m.Y):
        axes = axesdct[dt]
        x1i,x1j,x2i,x2j = int(x1i), int(x1j), int(x2i), int(x2j)
        ax = axes[2*x1i+x2i][x1j]
        ax.plot(x2j, y, 'x', color='C{}'.format(x2j))
# if plot_acq:
    #     plt.figure()
    #     _,acq = get_acq_map(m)
    #     sm = softmax(acq).reshape((10,10))
    #     plt.imshow(sm)
    #     plt.colorbar()

if __name__ == "__main__":
    args = parser.parse_args()
    dts = args.dts
    emg = args.emg
    trainsC = Trains(emg=args.emg)

    X1d,Y1d = make_dataset_1d(trainsC, emg=4)
    m1d, = train_models_1d(X1d,Y1d, ARD=True)

    X,Y = make_dataset_dt2d(trainsC)
    m = train_models_dt2d(X,Y,prior1d=m1d, kerneltype='mult')

    plot_model_dt2d(m)
    plt.show()
