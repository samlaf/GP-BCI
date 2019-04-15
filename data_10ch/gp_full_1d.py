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

emg=2
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
    
def plot_model_1d(m, plot_f=False, title=None, plot_acq=False):
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
               plot_data=False, plot_raw=plot_f,
               title='Channels {}'.format(xy2ch[i]),
               lower=17, upper=83)
        # Plot data (m.plot plots all of the data in every slice, which is
        # wrong)
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
    X,_ = make_dataset_1d(trains, n=1)
    pred1 = m1.predict(X)[0]
    pred2 = m2.predict(X)[0]
    return LA.norm(pred1-pred2)
def linfdist(m1, m2):
    X,_ = make_dataset_1d(trains, n=1)
    pred1 = m1.predict(X)[0]
    pred2 = m2.predict(X)[0]
    return abs((pred1.max() - pred2.max())/pred2.max())

if __name__ == '__main__':
    # trainsC = Trains(emg=emg)
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

    # We train all models with n rnd start pts and m sequential pts
    # And compare them to the model trained with all datapts
    # Then compute statistics and plot them
    X,Y = make_dataset_1d(trains)
    mfull, = train_models_1d(X,Y, ARD=False)
    nrnd = range(5,30,5)
    nseq = range(0,30,5)
    N = 75
    l2s = np.zeros((len(nrnd),len(nseq),N))
    linfs = np.zeros((len(nrnd),len(nseq),N))
    for k in range(N):
        for i,n1 in enumerate(nrnd):
            for j,n2 in enumerate(nseq):
                models = train_model_seq(trains,n_random_pts=n1, n_total_pts=n1+n2, ARD=False)
                m = models[-1]
                l2 = l2dist(m, mfull)
                linf = linfdist(m, mfull)
                l2s[i][j][k] = l2
                linfs[i][j][k] = linf

    plt.imshow(l2s.mean(axis=2), extent=[0,30,30,5])
    plt.title("1d l2 dist to true gp mean")
    plt.ylabel("N random pts")
    plt.xlabel("N sequential")
    plt.colorbar()

    plt.figure()
    plt.imshow(linfs.mean(axis=2), extent=[0,30,30,5])
    plt.title("1d linf dist to true gp mean")
    plt.ylabel("N random pts")
    plt.xlabel("N sequential")
    plt.colorbar()
    plt.show()
