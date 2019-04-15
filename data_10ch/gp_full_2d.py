"""
We fit GPs to the full dataset, testing different models and kernels
"""

from load_matlab import *
from gp_full_1d import *
import numpy as np
import GPy
import matplotlib.pyplot as plt
import pickle
import os
import h5py
import itertools

emg=2
dt=0

def make_dataset_2d(trains, means=False, dt=dt, n=None):
    X = []
    Y = []
    Xmean = []
    Ymean = []
    for i,ch1 in enumerate(CHS):
        for j,ch2 in enumerate(CHS):
            xych1 = ch2xy[ch1]
            xych2 = ch2xy[ch2]
            # Note that here we are only taking the pairs (ch1,ch2)
            # But will miss (ch2,ch1). These are the same data (for
            # dt=0),but the GP should still have this info
            # However, with 2000 pts its already slow enough so we
            # don't include them for now (this is prob bad though)
            # train time: 4000 pts = 10 mins
            #             2000 pts = 2 mins
            ys = trains[ch1][ch2][dt]['data'].max(axis=1)
            if n:
                ys = random.sample(trains[ch1][ch2][dt]['data'].max(axis=1).tolist(),n)
            Y.extend(ys)
            X.extend([xych1 + xych2]*len(ys))
            # we also make a small dataset with means
            Ymean.append(trains[ch1][ch2][dt]['meanmax'])
            Xmean.extend([xych1 + xych2])
    if means:
        Xmean = np.array(Xmean)
        Ymean = np.array(Ymean).reshape((-1,1))
        return Xmean, Ymean
    # if n:
    #     # We take n random elements only
    #     X = random.sample(X, n)
    #     Y = random.sample(Y, n)
    X = np.array(X)
    Y = np.array(Y).reshape((-1,1))
    return X,Y

def build_prior(m1d):
    mf_1 = GPy.core.Mapping(4,1)
    def f1(x):
        return m1d.predict(x[:,0:2])[0]
    mf_1.f = f1
    mf_1.update_gradients = lambda a,b: None
    mf_2 = GPy.core.Mapping(4,1)
    def f2(x):
        return m1d.predict(x[:,2:4])[0]
    mf_2.f = f2
    mf_2.update_gradients = lambda a,b: None
    mf = GPy.mappings.Additive(GPy.mappings.Compound(mf_1, GPy.mappings.Linear(1,1)),
                               GPy.mappings.Compound(mf_2, GPy.mappings.Linear(1,1)))
    return mf

def train_models_2d(X,Y, models=['add'], num_restarts=1, prior1d=None, optimize=True, ARD=False):
    kernels = []
    # Additive kernel
    if 'add' in models:
        k1 = GPy.kern.Matern52(input_dim=2, active_dims=[0,1], ARD=ARD)
        k2 = GPy.kern.Matern52(input_dim=2, active_dims=[2,3], ARD=ARD)
        k = k1 + k2
        kernels.append(k)

    # Symmetric SE
    if 'sym' in models:
        matk = GPy.kern.Matern52(input_dim=4, ARD=ARD)
        symM = np.block([[np.zeros((2,2)),np.eye(2)],[np.eye(2),np.zeros((2,2))]])
        symk = GPy.kern.Symmetric(matk, symM)
        kernels.append(symk)

    # Full SE
    if 'ard' in models:
       matk = GPy.kern.Matern52(input_dim=4, ARD=ARD)
       kernels.append(matk)

    models = []
    for k in kernels:
        if prior1d:
            m = GPy.models.GPRegression(X,Y,k, mean_function= build_prior(prior1d))
            m.sum.Mat52.lengthscale = m.sum.Mat52_1.lengthscale = prior1d.Mat52.lengthscale
            m.sum.Mat52.variance = m.sum.Mat52_1.variance = prior1d.Mat52.variance
            m.Gaussian_noise.variance = prior1d.Gaussian_noise.variance
        else:
            m = GPy.models.GPRegression(X,Y,k)
        if optimize:
            m.optimize_restarts(num_restarts=num_restarts)
        models.append(m)
    return models

def make_add_model(X,Y,prior1d=None, prevmodel=None, ARD=False):
    k1 = GPy.kern.Matern52(input_dim=2, active_dims=[0,1], ARD=ARD)
    k2 = GPy.kern.Matern52(input_dim=2, active_dims=[2,3], ARD=ARD)
    k = k1 + k2
    if prevmodel and prior1d:
        m = GPy.models.GPRegression(X,Y,kernel=prevmodel.sum.copy(), mean_function=prevmodel.mapping.copy())
        m.Gaussian_noise.variance = prevmodel.Gaussian_noise.variance
    elif prevmodel:
        #There is a previous model but it doesn't use prior (so no
        #mean mapping)
        m = GPy.models.GPRegression(X,Y,kernel=prevmodel.sum.copy())
        m.Gaussian_noise.variance = prevmodel.Gaussian_noise.variance
    elif prior1d:
        #We are building a model for first time, but with a 1d prior
        m = GPy.models.GPRegression(X,Y,k, mean_function= build_prior(prior1d))
        m.sum.Mat52.lengthscale = m.sum.Mat52_1.lengthscale = prior1d.Mat52.lengthscale
        m.sum.Mat52.variance = m.sum.Mat52_1.variance = prior1d.Mat52.variance
        m.Gaussian_noise.variance = prior1d.Gaussian_noise.variance
    else:
        m = GPy.models.GPRegression(X,Y,k)
    return m

def train_model_seq_2d(trains, n_random_pts=10, n_total_pts=15, num_restarts=1, ARD=False, prior1d=None, fix=False):
    X = []
    Y = []
    for _ in range(n_random_pts):
        ch1 = random.choice(CHS)
        ch2 = random.choice(CHS)
        X.append(ch2xy[ch1] + ch2xy[ch2])
        resp = random.choice(trains[ch1][ch2][dt]['data'].max(axis=1))
        Y.append(resp)
    #We save every model after each query
    models = []
    m = make_add_model(np.array(X),np.array(Y)[:,None], prior1d=prior1d, ARD=ARD)
    if fix:
        # fix all kernel parameters and only optimize for mean (prior) mapping
        m.sum.fix()
        m.Gaussian_noise.fix()
    m.optimize_restarts(num_restarts=num_restarts)
    models.append(m)
    for _ in range(n_total_pts-n_random_pts):
        nextx = get_next_x(m)
        X.append(nextx)
        ch1 = xy2ch[nextx[0]][nextx[1]]
        ch2 = xy2ch[nextx[2]][nextx[3]]
        resp = random.choice(trains[ch1][ch2][dt]['data'].max(axis=1))
        Y.append(resp)
        m = make_add_model(np.array(X), np.array(Y)[:,None], prior1d=prior1d, prevmodel=models[-1], ARD=ARD)
        # We only optimize hyperparameters with random points now
        #m.optimize_restarts(num_restarts=num_restarts)
        models.append(m)
    return models

def get_next_x(m, k=2):
    X,acq = get_acq_map(m,k)
    maxidx = acq.argmax()
    nextx = X[maxidx]
    return nextx

def get_acq_map(m, k=2):
    # We use UCB, k is the "exploration" parameter
    X,_ = make_dataset_2d(trains,means=True)
    mean,var = m.predict(X)
    std = np.sqrt(var)
    acq = mean + k*std
    return X,acq

# Code for generating plots
def plot_model_2d(m, fulldatam=None, title=""):
    fig, axes = plt.subplots(4,5, sharex=True, sharey=True)
    fig.suptitle(title)
    for i in [0,1]:
        for j in range(5):
            for x2i in [0,1]:
                ax = axes[2*i+x2i][j]
                ch1 = xy2ch[i][j]
                m.plot(ax=ax, fixed_inputs=[(0,i),(1,j),(2,x2i)], plot_data=False, legend=False,
                       title='ch1 = {}'.format(ch1))
                
                # We also plot the max found
                maxx = m.predict(make_dataset_2d(trains,means=True)[0])[0].max()
                x = np.arange(0,4,0.1)
                ax.plot(x,np.ones(len(x))*maxx, c='r')
                # And the mean of the full-data-gp, if present
                if fulldatam:
                    fulldatam.plot_mean(ax=ax, fixed_inputs=[(0,i),(1,j),(2,x2i)], color='y')
    for (x1i,x1j,x2i,x2j),y in zip(m.X, m.Y):
        x1i,x1j,x2i,x2j = int(x1i), int(x1j), int(x2i), int(x2j)
        ax = axes[2*x1i+x2i][x1j]
        ax.plot(x2j, y, 'x', color='C{}'.format(x2j))

def l2dist(m1, m2):
    X = np.array(list(itertools.product(range(5),range(2), range(5), range(2))))
    pred1 = m1.predict(X)[0]
    pred2 = m2.predict(X)[0]
    return LA.norm(pred1-pred2)
def linfdist(m1, m2):
    X = np.array(list(itertools.product(range(5),range(2), range(5), range(2))))
    pred1 = m1.predict(X)[0]
    pred2 = m2.predict(X)[0]
    return abs(pred1.max() - pred2.max())

if __name__ == "__main__":
    trainsC = Trains(emg=2)
    trains = trainsC.trains

    X1d,Y1d = make_dataset_1d(trains)
    m1d, = train_models_1d(X1d,Y1d, ARD=False)

    X,Y = make_dataset_2d(trains)
    if os.path.exists("data/maddprior.h5"):
        with h5py.File('data/maddprior.h5') as f:
            maddprior, = train_models_2d(X,Y, prior1d=m1d, optimize=False)
            maddprior[:] = f['param_array']
    else:
        maddprior, = train_models_2d(X,Y, prior1d=m1d)
        maddprior.save('data/maddprior.h5')

    if os.path.exists("data/madd.h5"):
        with h5py.File('data/madd.h5') as f:
            madd, = train_models_2d(X,Y, optimize=False)
            madd[:] = f['param_array']
    else:
        madd, = train_models_2d(X,Y)
        madd.save('data/madd.h5')
    
    # We train all models with n rnd start pts and m sequential pts
    # And compare them to the model trained with all datapts
    # Then compute statistics and plot them
    nrnd = range(10,100,10)
    nseq = range(0,100,10)
    N = 50
    l2s = np.zeros((3, N, len(nrnd),len(nseq)))
    linfs = np.zeros((3, N, len(nrnd),len(nseq)))
    for k in range(N):
        print("Starting loop", k)
        for i,n1 in enumerate(nrnd):
            for j,n2 in enumerate(nseq):
                print(n1,n2)
                models = train_model_seq_2d(trains,n_random_pts=n1, n_total_pts=n1+n2)
                modelsprior = train_model_seq_2d(trains,n_random_pts=n1, n_total_pts=n1+n2, prior1d=m1d)
                modelspriorfix = train_model_seq_2d(trains,n_random_pts=n1, n_total_pts=n1+n2, prior1d=m1d, fix=True)
                mnoprior, mprior, mpriorfix = models[-1], modelsprior[-1], modelspriorfix[-1]
                for midx,m in enumerate([mnoprior, mprior, mpriorfix]):
                    l2 = l2dist(m, madd)
                    linf = linfdist(m, madd)
                    l2s[midx][k][i][j] = l2
                    linfs[midx][k][i][j] = linf
    np.save("data/l2s", l2s)
    np.save("data/linfs", linfs)

    for i,name in enumerate(["","_prior","_priorfix"]):
        plt.figure()
        plt.imshow(l2s[i].mean(axis=0), extent=[0,100,200,10])
        plt.title("2d l2 dist to full model{}".format(name))
        plt.ylabel("N random pts")
        plt.xlabel("N sequential")
        plt.colorbar()
        plt.savefig("images/2d_l2{}.png".format(name))

        plt.figure()
        plt.imshow(linfs[i].mean(axis=0), extent=[0,50,100,10])
        plt.title("2d linf dist to full model{}".format(name))
        plt.ylabel("N random pts")
        plt.xlabel("N sequential")
        plt.colorbar()
        plt.savefig("images/2d_linf{}.png".format(name))


