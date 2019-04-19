"""
We fit GPs to the full dataset, testing different models and kernels
"""

from load_matlab import *
from gp_full_1d import *
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
parser.add_argument('--uid', type=int, default=1, help='uid for job number')
parser.add_argument('--dt', type=int, default=0, choices=(0,10,20,40,60,80,100), help='dt. one of (0,10,20,40,60,80,100)')
parser.add_argument('--emg', type=int, default=2, choices=range(7), help='emg. between 0-6')

# DEFAULT
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

class Abs(Mapping):
    def __init__(self, mapping):
        input_dim, output_dim = mapping.input_dim, mapping.output_dim
        assert(output_dim == 1)
        super(Abs, self).__init__(input_dim=input_dim, output_dim=output_dim)
        self.mapping = mapping

    def f(self, X):
        return np.abs(self.mapping.f(X))

    def update_gradients(self, dL_dF, X):
        None

    def gradients_X(self, dL_dF, X):
        if X >= 0:
            return X
        else:
            return -X


def build_prior(m1d, complicated=False):
    f1 = GPy.core.Mapping(4,1)
    def f_1(x):
        return m1d.predict(x[:,0:2])[0]
    f1.f = f_1
    f1.update_gradients = lambda a,b: None
    
    f2 = GPy.core.Mapping(4,1)
    def f_2(x):
        return m1d.predict(x[:,2:4])[0]
    f2.f = f_2
    f2.update_gradients = lambda a,b: None

    if not complicated:
        # prior = a*f1 + b*f2
        mf = GPy.mappings.Additive(GPy.mappings.Compound(f1, GPy.mappings.Linear(1,1)),
                                   GPy.mappings.Compound(f2, GPy.mappings.Linear(1,1)))
        mf.mapping.linmap.A = mf.mapping_1.linmap.A = 1/2
    else:
        # prior = c*(a*f1+b*f2) + d*|a*f1-b*f2| (where c and d could
        # be exponentials related to dt in the future)
        negf = GPy.mappings.Linear(1,1)
        negf.A = -1
        negf.fix()
        a = GPy.mappings.Linear(1,1)
        b = GPy.mappings.Linear(1,1)
        # a*f1 + b*f2
        mfadd = GPy.mappings.Additive(GPy.mappings.Compound(f1, a),
                                      GPy.mappings.Compound(f2, b))
        # |a*f1 - b*f2|
        mfsub = Abs(GPy.mappings.Additive(GPy.mappings.Compound(f1, a),
                                          GPy.mappings.Compound(GPy.mappings.Compound(f2, b), negf)))
        mf = GPy.mappings.Additive(GPy.mappings.Compound(mfadd, GPy.mappings.Linear(1,1)),
                                   GPy.mappings.Compound(mfsub, GPy.mappings.Linear(1,1)))
    return mf

def train_models_2d(X,Y, models=['add'], num_restarts=1, prior1d=None, optimize=True, ARD=False, complicated=False):
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
            m = GPy.models.GPRegression(X,Y,k, mean_function= build_prior(prior1d, complicated=complicated))
            m.sum.Mat52.lengthscale = m.sum.Mat52_1.lengthscale = prior1d.Mat52.lengthscale
            m.sum.Mat52.variance = m.sum.Mat52_1.variance = prior1d.Mat52.variance
            m.Gaussian_noise.variance = prior1d.Gaussian_noise.variance
        else:
            m = GPy.models.GPRegression(X,Y,k)
        if optimize:
            m.optimize_restarts(num_restarts=num_restarts)
        models.append(m)
    return models

def make_add_model(X,Y,prior1d=None, prevmodel=None, ARD=False, complicated=False):
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
        m = GPy.models.GPRegression(X,Y,k, mean_function=build_prior(prior1d, complicated=complicated))
        m.sum.Mat52.lengthscale = m.sum.Mat52_1.lengthscale = prior1d.Mat52.lengthscale
        m.sum.Mat52.variance = m.sum.Mat52_1.variance = prior1d.Mat52.variance
        m.Gaussian_noise.variance = prior1d.Gaussian_noise.variance
    else:
        m = GPy.models.GPRegression(X,Y,k)
    return m

def train_model_seq_2d(trains, n_random_pts=10, n_total_pts=15, num_restarts=1, ARD=False, prior1d=None, fix=False, continue_opt=False, dt=dt, complicated=False):
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
    m = make_add_model(np.array(X),np.array(Y)[:,None], prior1d=prior1d, ARD=ARD, complicated=complicated)
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
        m = make_add_model(np.array(X), np.array(Y)[:,None], prior1d=prior1d, prevmodel=models[-1], ARD=ARD, complicated=complicated)
        # If continue optimize, we optimize params after every query
        if continue_opt:
            m.optimize_restarts(num_restarts=num_restarts)
        models.append(m)
    return models

def get_next_x(m, k=2):
    X,acq = get_acq_map(m,k)
    maxidx = acq.argmax()
    nextx = X[maxidx]
    return nextx

def get_acq_map(m, k=2):
    # We use UCB, k is the "exploration" parameter
    X = np.array(list(itertools.product(range(2),range(5), range(2), range(5))))
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
            ch1 = xy2ch[i][j]
            title='ch1 = {}'.format(ch1)
            axes[2*i][j].set_title(title)
            for x2i in [0,1]:
                ax = axes[2*i+x2i][j]
                m.plot(ax=ax, fixed_inputs=[(0,i),(1,j),(2,x2i)], plot_data=False, legend=False)
                
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
    X = np.array(list(itertools.product(range(2),range(5), range(2), range(5))))
    pred1 = m1.predict(X)[0]
    pred2 = m2.predict(X)[0]
    return LA.norm(pred1-pred2)
def linfdist(m1, m2):
    X = np.array(list(itertools.product(range(2),range(5), range(2), range(5))))
    pred1 = m1.predict(X)[0]
    pred2 = m2.predict(X)[0]
    return abs(pred1.max() - pred2.max())

def get_ch_pair(wxyz):
    w,x,y,z = wxyz
    return [get_ch([w,x]), get_ch([y,z])]

def get_maxchpair(m):
    X = np.array(list(itertools.product(range(2),range(5), range(2), range(5))))
    means,_ = m.predict(X)
    maxidx = means.argmax()
    maxwxyz = np.unravel_index(maxidx, (2,5,2,5))
    maxchpair = get_ch_pair(maxwxyz)
    return maxchpair

def run_ch_stats_exps(trains, args, repeat=25, continue_opt=True, k=2):

    exppath = path.join('exps', '2d', 'chruns', 'emg{}'.format(args.emg),
                        'dt{}'.format(args.dt), 'exp{}'.format(args.uid))
    if not path.isdir(exppath):
        os.makedirs(exppath)
    ntotal=100
    n_ch=2
    n_models=2
    nrnd = range(15,76,10)
    # Build 1d model for modelsprior
    X1d,Y1d = make_dataset_1d(trains)
    m1d, = train_models_1d(X1d,Y1d, ARD=False)
    # queriedchs contains <n_ch> queried channels for all <repeat> runs of <ntotal>
    # queries with <nrnd> initial random pts for each of <n_models> models
    queriedchs = np.zeros((n_models, repeat, len(nrnd), ntotal, n_ch))
    maxchs = np.zeros((n_models, repeat, len(nrnd), ntotal, n_ch))
    for repeat in range(repeat):
        print("Repeat", repeat)
        for i,n1 in enumerate(nrnd):
            print(n1, "random init pts")
            models = train_model_seq_2d(trains,n_random_pts=n1, n_total_pts=ntotal,
                                        num_restarts=1, continue_opt=continue_opt, dt=args.dt)
            modelsprior = train_model_seq_2d(trains,n_random_pts=n1, n_total_pts=ntotal,
                                             num_restarts=1, continue_opt=continue_opt,
                                             prior1d=m1d, dt=args.dt)
            queriedchs[0][repeat][i] = [get_ch_pair(xy) for xy in models[-1].X]
            queriedchs[1][repeat][i] = [get_ch_pair(xy) for xy in modelsprior[-1].X]
            for r,m in enumerate(models,n1-1):
                maxchs[0][repeat][i][r] = get_maxchpair(m)
            for r,m in enumerate(modelsprior,n1-1):
                maxchs[1][repeat][i][r] = get_maxchpair(m)
    dct = {
        'queriedchs': queriedchs,
        'maxchs': maxchs,
        'nrnd': nrnd,
        'ntotal': ntotal,
        'true_chpair': [17,17]
    }
    with open(os.path.join(exppath, 'chruns2d_dct.pkl'), 'wb') as f:
        pickle.dump(dct, f)
    return queriedchs, maxchs

def run_dist_exps(args):
    emgdtpath = path.join('exps', 'emg{}'.format(args.emg), 'dt{}'.format(args.dt))
    exppath = path.join(emgdtpath, 'exp{}'.format(args.uid))
    if not path.isdir(exppath):
        os.makedirs(exppath)

    trainsC = Trains(emg=args.emg)
    trains = trainsC.trains

    X1d,Y1d = make_dataset_1d(trains)
    m1d, = train_models_1d(X1d,Y1d, ARD=False)

    X,Y = make_dataset_2d(trains, dt=args.dt)
    
    # Note that the full-data models can be shared for all exps (with
    # same emg and dt).
    # Hence we save them in emgdtpath instead of exppath
    addpriorpath = path.join(emgdtpath, 'maddprior.h5')
    if os.path.exists(addpriorpath):
        with h5py.File(addpriorpath) as f:
            maddprior, = train_models_2d(X,Y, prior1d=m1d, optimize=False)
            maddprior[:] = f['param_array']
    else:
        maddprior, = train_models_2d(X,Y, prior1d=m1d)
        maddprior.save(addpriorpath)

    addpath = path.join(emgdtpath, 'madd.h5')
    if path.exists(addpath):
        with h5py.File(addpath) as f:
            madd, = train_models_2d(X,Y, optimize=False)
            madd[:] = f['param_array']
    else:
        madd, = train_models_2d(X,Y)
        madd.save(addpath)
    
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
                models = train_model_seq_2d(trains,n_random_pts=n1, n_total_pts=n1+n2, dt=args.dt)
                modelsprior = train_model_seq_2d(trains,n_random_pts=n1, n_total_pts=n1+n2, prior1d=m1d, dt=args.dt)
                modelspriorfix = train_model_seq_2d(trains,n_random_pts=n1, n_total_pts=n1+n2, prior1d=m1d, fix=True, dt=args.dt)
                mnoprior, mprior, mpriorfix = models[-1], modelsprior[-1], modelspriorfix[-1]
                for midx,m in enumerate([mnoprior, mprior, mpriorfix]):
                    l2 = l2dist(m, madd)
                    linf = linfdist(m, madd)
                    l2s[midx][k][i][j] = l2
                    linfs[midx][k][i][j] = linf
        np.save(os.path.join(exppath,"l2s"), l2s)
        np.save(os.path.join(exppath, "linfs"), linfs)

        for i,name in enumerate(["","_prior","_priorfix"]):
            plt.figure()
            plt.imshow(l2s[i][:k+1].mean(axis=0), extent=[0,100,100,10])
            plt.title("2d l2 dist to full model{}".format(name))
            plt.ylabel("N random pts")
            plt.xlabel("N sequential")
            plt.colorbar()
            plt.savefig(os.path.join(exppath, "2d_l2{}_{}.png".format(name,k)))
            plt.close()

            plt.figure()
            plt.imshow(linfs[i][:k+1].mean(axis=0), extent=[0,100,100,10])
            plt.title("2d linf dist to full model{}".format(name))
            plt.ylabel("N random pts")
            plt.xlabel("N sequential")
            plt.colorbar()
            plt.savefig(os.path.join(exppath, "2d_linf{}_{}.png".format(name,k)))
            plt.close()

def test_modif_max(trains, args):
    X1d,Y1d = make_dataset_1d(trains)
    m1d, = train_models_1d(X1d,Y1d, ARD=False)

    # We test changing the max,
    # and changing the dataset
    new_trains = copy.deepcopy(trains)
    for ch in [6,13,21,17]:
        new_trains[17][ch][dt]['data'] /= 2
        new_trains[17][17][dt]['meanmax'] = new_trains[17][ch][dt]['data'].max(axis=1).mean()
        new_trains[ch][17][dt]['data'] /= 2
        new_trains[ch][17][dt]['meanmax'] = new_trains[ch][17][dt]['data'].max(axis=1).mean()

    Xmean,Ymean = make_dataset_2d(new_trains, dt=args.dt, means=True)
    madd, = train_models_2d(Xmean,Ymean, ARD=False)
    plot_model_2d(madd)

    modelsprior = train_model_seq_2d(new_trains,n_random_pts=50, n_total_pts=100, prior1d=m1d, continue_opt=True, fix=True)
    m = modelsprior[-1]
    plot_model_2d(m)



if __name__ == "__main__":
    args = parser.parse_args()
    dt = args.dt
    trainsC = Trains(emg=args.emg)
    trains = trainsC.trains
    
    # queriedchs, maxchs = run_ch_stats_exps(trains, args)
    X,Y = make_dataset_2d(trains, dt=40)
    train_models_2d(X,Y, complicated=True)

    plt.show()
