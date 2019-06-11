"""
We fit GPs to the full dataset, testing different models and kernels
"""

from load_matlab import *
from gp_full_1d import *
from gp_full_2d import make_dataset_2d, build_prior, softmax
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
emg=4

def make_dataset_dt2d(trainsC, emg=emg, syn=None, dts=dts, means=False, n=None):
    X,Y = make_dataset_2d(trainsC=trainsC,emg=emg,syn=syn,dt=dts[0],means=means,n=n)
    X = np.hstack((X, np.ones((len(X),1))*dts[0]))
    for dt in dts[1:]:
        X_,Y_ = make_dataset_2d(trainsC=trainsC,emg=emg,syn=syn,dt=dt,means=means,n=n)
        X_ = np.hstack((X_, np.ones((len(X_),1))*dt))
        X = np.vstack((X,X_))
        Y = np.vstack((Y,Y_))
    return X,Y

def train_models_dt2d(X,Y, kerneltype='add', symkern=False, num_restarts=1, prior1d=None, optimize=True, ARD=True, dtprior=False, constrain=False, sparse=None, m1d=None):
    k1 = GPy.kern.Matern52(input_dim=2, active_dims=[0,1], ARD=ARD)
    k2 = GPy.kern.Matern52(input_dim=2, active_dims=[2,3], ARD=ARD)
    kdt = GPy.kern.Matern52(input_dim=1, active_dims=[4], lengthscale=20)
    if m1d:
        k1.lengthscale = k2.lengthscale = m1d.Mat52.lengthscale
        k1.variance = k2.variance = m1d.Mat52.variance
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

    if sparse:
        m = GPy.models.SparseGPRegression(X,Y,k, mean_function=prior1d, num_inducing=sparse)
    else:
        m = GPy.models.GPRegression(X,Y,k, mean_function= prior1d)
    if m1d:
        m.Gaussian_noise.variance = m1d.Gaussian_noise.variance
    if optimize:
        m.optimize_restarts(num_restarts=num_restarts)

    return m

def get_chpairdt(wxyzdt, dts):
    w,x,y,z,dt = wxyzdt
    return [get_ch([w,x]), get_ch([y,z]), dt]

def get_chpairdtidx(wxyzdt, dts):
    w,x,y,z,dt = wxyzdt
    dt = dts[dt]
    return [get_ch([w,x]), get_ch([y,z]), dt]

def get_maxchpairdt(m, dts):
    X = np.array(list(itertools.product(range(2),range(5), range(2), range(5), dts)))
    means,_ = m.predict(X)
    maxidx = means.argmax()
    maxwxyz = np.unravel_index(maxidx, (2,5,2,5,len(dts)))
    maxchpair = get_chpairdtidx(maxwxyz, dts)
    return maxchpair

def plot_model_dt2d(m, k=-1, title="", plot_acq=False):
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

def train_model_seq_dt2d(trainsC, n_random_pts=10, n_total_pts=15, n_prior_queries=3, num_restarts=1, ARD=False, prior1d=None, m1d=None, fix=False, continue_opt=True, emg=emg, syn=None, dts=dts, dtprior=False, sa=True, symkern=False, kerneltype='mult', T=0.001, constrain=True, k=2):
    trains = trainsC.get_emgdct(emg)
    if dtprior:
        assert(continue_opt), "if dtprior is True, must set continue_opt to true"
        assert(prior1d is not None), "if dtprior is True, must give prior1d"
    X = []
    Y = []

    # PTS NEAR PRIOR'S 3 MAX CHS
    if m1d:
        # query pts around n_prior_queries max chs of prior1d
        nmaxchs = get_nmaxch(m1d, n=n_prior_queries)
        for xych1 in nmaxchs:
            ch1 = xy2ch[xych1[0]][xych1[1]]
            for xych2 in nmaxchs:
                ch2 = xy2ch[xych2[0]][xych2[1]]
                for dt in dts:
                    X.append(xych1+xych2+[dt])
                    if syn is None:
                        resp = random.choice(trainsC.get_resp(emg,dt,ch1,ch2).max(axis=1))
                    else:
                        resp = random.choice(trainsC.synergy(syn[0], syn[1], ch1, ch2, dt).max(axis=1))
                    Y.append(resp)
    else:
        # we need this so as to query the right total # of pts
        # (for loop below has - n_prior_queries**2)
        n_prior_queries = 0

    # We need this in case n_prior_queries**2*len(dts) > n_random_pts
    X = X[:n_random_pts]
    Y = Y[:n_random_pts]

    # RANDOM PTS
    for _ in range(n_random_pts - n_prior_queries**2*len(dts)):
        ch1 = random.choice(CHS)
        ch2 = random.choice(CHS)
        dt = random.choice(dts)
        X.append(ch2xy[ch1] + ch2xy[ch2] + [dt])
        if syn is None:
            resp = random.choice(trainsC.get_resp(emg,dt,ch1,ch2).max(axis=1))
        else:
            resp = random.choice(trainsC.synergy(syn[0], syn[1], ch1, ch2, dt).max(axis=1))
        Y.append(resp)
    #We save every model after each query
    models = []
    # Train initial model
    m = train_models_dt2d(np.array(X),np.array(Y)[:,None], prior1d=prior1d, m1d=m1d, ARD=ARD, kerneltype=kerneltype,symkern=symkern, constrain=constrain)
    models.append(m)

    # SEQUENTIAL QUERY PTS
    for _ in range(n_total_pts-n_random_pts):
        nextx = get_next_x(m, dts, sa=sa, T=T, k=k)
        X.append(nextx)
        ch1 = xy2ch[nextx[0]][nextx[1]]
        ch2 = xy2ch[nextx[2]][nextx[3]]
        dt = nextx[4]
        if syn is None:
            resp = random.choice(trainsC.get_resp(emg,dt,ch1,ch2).max(axis=1))
        else:
            resp = random.choice(trainsC.synergy(syn[0], syn[1], ch1, ch2, dt).max(axis=1))
        Y.append(resp)
        m = train_models_dt2d(np.array(X),np.array(Y)[:,None], prior1d=prior1d, m1d=m1d, ARD=ARD, kerneltype=kerneltype, symkern=symkern, constrain=constrain)

        models.append(m)
        
    dct = {
        'models': models,
        'nrnd': n_random_pts,
        'ntotal': n_total_pts
    }
    return dct

def get_next_x(m, dts, k=2, sa=False, T=0.001):
    X,acq = get_acq_map(m,dts,k=k)
    if sa:
        # SA is for simulated annealing (sample instead of taking max)
        sm = softmax(acq, T=T).flatten()
        nextidx = np.random.choice(range(len(acq)), p=sm)
    else:
        nextidx = acq.argmax()
    nextx = X[nextidx]
    return nextx

def get_acq_map(m, dts, k=2):
    # We use UCB, k is the "exploration" parameter
    X = np.array(list(itertools.product(range(2),range(5), range(2), range(5), dts)))
    mean,var = m.predict(X)
    std = np.sqrt(var)
    acq = mean + k*std
    return X,acq

def run_ch_stats_exps(trainsC, emg=emg, syn=None, dts=dts, uid='', jobid='', repeat=25, continue_opt=True, k=2, dtprior=False, ntotal=100, nrnd = [15,76,10], sa=True, multkern=True, symkern=False, ARD=False, T=0.001, constrain=True, n_prior_queries=3):
    if multkern: kerneltype='mult'
    else: kerneltype='add'
    if uid == '':
        uid = random.randrange(10000)
    assert(type(nrnd) is list and len(nrnd) == 3)
    trains = trainsC.get_emgdct(emg)
    nrnd = range(*nrnd)
    synstr = 'syn{}'.format(''.join([str(n) for n in syn]))
    dtsstr = 'dts{}'.format(''.join([str(n) for n in dts]))
    exppath = path.join('exps', '2d', 'exp{}'.format(uid), synstr, dtsstr, 'sa{}'.format(sa), 'multkern{}'.format(multkern), 'ARD{}'.format(ARD), 'constrain{}'.format(constrain), 'k{}'.format(k))
    if not path.isdir(exppath):
        os.makedirs(exppath)
    with open(os.path.join(exppath, 'jobid={}'.format(jobid)), 'w') as f:
        print("Writing to file: {}".format(os.path.join(exppath, 'jobid={}'.format(jobid))))
        f.write('sbatch jobid = {}'.format(jobid))
        
    n_models = 3 if dtprior else 2
    X = np.array(list(itertools.product(range(2),range(5), range(2), range(5), dts)))
    # Build 1d model for modelsprior
    if syn is None:
        X1d,Y1d = make_dataset_1d(trainsC, emg=emg)
        m1d1, = train_models_1d(X1d,Y1d, ARD=ARD)
        prior1d = build_prior(m1d1,input_dim=5)
    else:
        X1d,Y1d = make_dataset_1d(trainsC, emg=syn[0])
        m1d1, = train_models_1d(X1d,Y1d, ARD=ARD)
        X1d,Y1d = make_dataset_1d(trainsC, emg=syn[1])
        m1d2, = train_models_1d(X1d,Y1d, ARD=ARD)
        prior1d = build_prior(m1d1,m1d2,input_dim=5)

    # queriedchs contains 2 queried channels + dt (3) for all <repeat> runs of <ntotal>
    # queries with <nrnd> initial random pts for each of <n_models> models
    queriedchs = np.zeros((n_models, repeat, len(nrnd), ntotal, 3))
    maxchs = np.zeros((n_models, repeat, len(nrnd), ntotal, 3))
    vals = np.zeros((n_models, repeat, len(nrnd), ntotal, 100*len(dts)))
    for repeat in range(repeat):
        print("Repeat", repeat)
        for i,n1 in enumerate(nrnd):
            print(n1, "random init pts")
            modelsD = train_model_seq_dt2d(trainsC,n_random_pts=n1, n_total_pts=ntotal,
                                           num_restarts=1, continue_opt=continue_opt, ARD=ARD,
                                           dts=dts, emg=emg, syn=syn, sa=sa, k=k,
                                           kerneltype=kerneltype, symkern=symkern, T=T,
                                           constrain=constrain,n_prior_queries=n_prior_queries)
            modelspriorD = train_model_seq_dt2d(trainsC,n_random_pts=n1, n_total_pts=ntotal,
                                                num_restarts=1, continue_opt=continue_opt,
                                                prior1d=prior1d, m1d=m1d1, dts=dts, emg=emg,
                                                dtprior=False, sa=sa, kerneltype=kerneltype,
                                                syn=syn, symkern=symkern, ARD=ARD, T=T, k=k,
                                                constrain=constrain, n_prior_queries=n_prior_queries)
            models = modelsD['models']
            modelsprior = modelspriorD['models']
            queriedchs[0][repeat][i] = [get_chpairdt(xydt,dts) for xydt in models[-1].X]
            queriedchs[1][repeat][i] = [get_chpairdt(xydt,dts) for xydt in modelsprior[-1].X]
            for r,m in enumerate(models,n1-1):
                maxchs[0][repeat][i][r] = get_maxchpairdt(m,dts)
                vals[0][repeat][i][r] = m.predict(X)[0].reshape((-1))
            for r,m in enumerate(modelsprior,n1-1):
                maxchs[1][repeat][i][r] = get_maxchpairdt(m,dts)
                vals[1][repeat][i][r] = m.predict(X)[0].reshape((-1))
            if dtprior:
                modelsdtpriorD = train_model_seq_2d(trainsC,n_random_pts=n1, n_total_pts=ntotal,
                                                    num_restarts=1, continue_opt=continue_opt,
                                                    prior1d=m1d, dts=dts, emg=emg, dtprior=True,
                                                    sa=sa, kerneltype=kerneltype, k=k,
                                                    symkern=symkern, ARD=ARD, T=T, syn=syn,
                                                    n_prior_queries=n_prior_queries)
                modelsdtprior = modelsdtpriorD['models']
                queriedchs[2][repeat][i] = [get_chpairdt(xydt,dts) for xydt in modelsdtprior[-1].X]
                for r,m in enumerate(modelsdtprior,n1-1):
                    maxchs[2][repeat][i][r] = get_maxchpairdt(m,dts)
                    vals[2][repeat][i][r] = m.predict(X)[0].reshape((-1))
    dct = {
        'queriedchs': queriedchs,
        'maxchs': maxchs,
        'vals': vals,
        'nrnd': nrnd,
        'true_vals': np.concatenate([trainsC.build_f_grid(emg=emg,syn=syn,dt=dt).flatten() for dt in dts]),
        'ntotal': ntotal,
        'emg': emg,
        'syn': syn,
        'dts': dts,
        'uid': uid,
        'repeat': repeat,
        'true_chpairdt': trainsC.max_ch_dt2d(emg,dts),
        'multkern': multkern,
        'symkern': symkern,
        'k': k
    }
    filename = os.path.join(exppath, 'chrunsdt2d_dct.pkl')
    print("Saving stats dictionary to: {}".format(filename))
    with open(filename, 'wb') as f:
        pickle.dump(dct, f)
    return dct

if __name__ == "__main__":
    args = parser.parse_args()
    dts = args.dts
    emg = args.emg
    trainsC = Trains(emg=args.emg, clean_thresh=0.06)

    X1d,Y1d = make_dataset_1d(trainsC, emg=0)
    m1d0, = train_models_1d(X1d,Y1d, ARD=True)
    X1d,Y1d = make_dataset_1d(trainsC, emg=2)
    m1d2, = train_models_1d(X1d,Y1d, ARD=True)
    X1d,Y1d = make_dataset_1d(trainsC, emg=4)
    m1d4, = train_models_1d(X1d,Y1d, ARD=True)
    prior1d = build_prior(m1d0,m1d4,input_dim=5)

    X,Y = make_dataset_dt2d(trainsC,syn=(0,4),dts=[20,40,60])
    m = train_models_dt2d(X,Y,prior1d=prior1d, kerneltype='mult', m1d=m1d0)
    print(get_maxchpairdt(m, dts))

    mdct = train_model_seq_dt2d(trainsC, 50, 100, syn=(0,4), dts=(20,40,60), prior1d=prior1d, m1d=m1d0, sa=False, ARD=True, kerneltype='mult', constrain=True, n_prior_queries=0)
    
    plot_model_dt2d(mdct['models'][-1])
    plt.show()
