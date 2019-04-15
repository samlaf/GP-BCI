"""
We first learn the 1D GP.
The difference here with gp-2d-diff is that instead of learning the diff function (f-prior), we learn f directly but instead of initially querying n random points, we query the points that we think the 2d function will be minimum (by using our prior and assuming independence).

The problem with this approach is that we still have a lot of uncertainty in areas far from these "good" starting points, which we will need to "waste" time exploring (whereas the diff approach assumes that the difference will be a low order polynomial which we can completely learn by only querying in the "good" region... is the brain really this simple?)
"""

import numpy as np
import skopt
from skopt import gp_minimize, Optimizer
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from utils import my_plot_convergence, ei_with_prior
from skopt.learning import GaussianProcessRegressor as GPR
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skopt.space import Real, Integer
import itertools
from scipy.stats import multivariate_normal
from load_matlab import *
from collections import OrderedDict

# np.random.seed(1234)
# Note the 1e3 hack! The values in the data are too small, so
# we need this for numerical stability
SCALE=1e3

class GP:

    def __init__(self, n_dims = 2):
        self.trains = Trains()
        self.resf = None

    def plot_gp(self):
        if self.resf is None:
            self.minimize()

        _x = np.arange(0,2)
        _y = np.arange(0,5)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()

        predictions = -np.vectorize(self.predict)(x,y)
        real_values = np.vectorize(self.trains.xy2resp)(x,y)*SCALE

        plt.figure()
        ax = plt.subplot(1,1,1,projection='3d')
        surf = ax.bar3d(x,y,np.zeros_like(predictions).ravel(),1,1,predictions)

        # We also plot all of the real mean values
        ax.scatter(x,y,real_values, c='r')
        
        pts = self.resf.x_iters
        xpts = [x for x,y in pts]
        ypts = [y for x,y in pts]
        # Plot sampled points
        ax.scatter(xpts, ypts, 0,#-self.resf.func_vals,
                   marker='o', c='g', label="Observations")

    def plot_lml(self):
        plt.figure()
        theta0 = np.logspace(-2, 3, 49)
        theta1 = np.logspace(-2, 2, 50)
        Theta0, Theta1 = np.meshgrid(theta0, theta1)
        m = self.resf.models[-1]
        LML = [[m.log_marginal_likelihood(np.log([np.exp(m.kernel_.theta[0]),
                                                  np.exp(m.kernel_.theta[1]),
                                                  Theta0[i, j], Theta1[i, j]]))
        for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
        LML = np.array(LML).T

        vmin, vmax = (-LML).min(), (-LML).max()
        vmax = vmin + 50
        #level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
        level = np.around(np.linspace(vmin,vmax,250), decimals=2)
        plt.contour(Theta0, Theta1, -LML,
                    levels=level)
        #norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.colorbar()
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Length-scale")
        plt.ylabel("Noise-level")
        plt.title("Log-marginal-likelihood")
        plt.tight_layout()

    # We need this function since surrogate_m.predict only works on lists,
    # not meshgrid like datastructures
    def predict(self,x1,y1):
        # Note: We need to transform x for the gp prediction since skopt
        # normalizes the input space to [0,1]
        x_gp = self.resf.space.transform([[x1,y1]])
        return self.resf.models[-1].predict(x_gp)

    def debug(self):
        _x = np.arange(0,2)
        _y = np.arange(0,5)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        xy_pairs = list(zip(x,y))
        predictions = np.vectorize(self.predict)(x,y)
        real_values = - np.vectorize(self.trains.xy2resp)(x,y)*SCALE
        print("predict   real")
        for p,r in zip(predictions, real_values):
            print(p,r)
        return predictions, real_values

class NoisyGP(GP):
    
    def __init__(self, n_dims=2):
        super(NoisyGP, self).__init__()
        self.gpr = self.init_kernel(n_dims)

    def f(self, x):
        return -self.trains.sampleResp(x[0],x[1])*SCALE
        # ch = self.trains.xy2ch(x[0],x[1])
        # mean = self.trains.trains[ch][ch][0]['meanmax']
        # std = self.trains.trains[ch][ch][0]['stdmax']
        # return - random.normalvariate(mean,std)

    def minimize(self):
        spacef = [Integer(0, 1, name='x'),
                  Integer(0, 4, name='y')]
        resf = gp_minimize(self.f,
                           spacef,
                           base_estimator=self.gpr,
                           acq_func="EI",
                           n_calls=20,
                           n_random_starts=5,
                           noise=0,
                           acq_optimizer="sampling")
        self.resf = resf
        return resf

    def init_kernel(self, n_dims):
        cov_amplitude = skopt.learning.gaussian_process.kernels.ConstantKernel(1.0, (0.01, 1000.0))
        other_kernel  = skopt.learning.gaussian_process.kernels.Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=[(0.01, 100)] * n_dims, nu=2.5)
        gpr = skopt.learning.GaussianProcessRegressor(
            kernel               = cov_amplitude * other_kernel,
            normalize_y          = True,
            noise                = "gaussian",
            n_restarts_optimizer = 2)
        return gpr

class NoiseFreeGP(GP):
    
    def __init__(self, n_dims=2):
        super(NoiseFreeGP, self).__init__()
        self.gpr = self.init_kernel(n_dims)

    def f(self, x):
        return -self.trains.xy2resp(x[0],x[1])*SCALE

    def minimize(self):
        spacef = [Integer(0, 1, name='x'),
                  Integer(0, 4, name='y')]
        resf = gp_minimize(self.f,
                           spacef,
                           base_estimator=self.gpr,
                           acq_func="EI",
                           n_calls=10,
                           n_random_starts=5,
                           noise=0,
                           acq_optimizer="sampling")
        self.resf = resf
        return resf

    def init_kernel(self, n_dims):
        cov_amplitude = skopt.learning.gaussian_process.kernels.ConstantKernel(1.0, (0.01, 1000.0))
        other_kernel  = skopt.learning.gaussian_process.kernels.Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=[(0.01, 100)] * n_dims, nu=2.5)
        # We add a small amount of noise to help with numerical stability
        gpr = skopt.learning.GaussianProcessRegressor(
            kernel               = cov_amplitude * other_kernel,
            normalize_y          = False,
            noise                = 0,
            n_restarts_optimizer = 2)
        return gpr

gp1 = NoiseFreeGP()
gp1.plot_gp()
gp1.debug()
# gp = NoisyGP()
# gp.plot_gp()
# for xy,z in zip(gp.resf.x_iters, gp.resf.func_vals):
#     print(xy,z)
# print(gp.resf.func_vals)
# for m in gp.resf.models:
#     print(m.kernel_, m.noise_)
#     print(m.log_marginal_likelihood())
# gp.plot_lml()

#plt.show()

##### -------------------------------------------------- #####
#####                g function (2d)                     #####

class GP2:

    def __init__(self, use_initpts=False, use_prior=False):
        self.trains = Trains()
        self.resg = None
        self.use_initpts = use_initpts
        self.use_prior = use_prior

    def minimize(self):

        spaceg = [Integer(0, 1, name='x1'),
                  Integer(0, 4, name='y1'),
                  Integer(0, 1, name='x2'),
                  Integer(0, 4, name='y2')]

        if self.use_initpts or self.use_prior:
            # To use either, we need the single train gp
            gp1 = self.traingp1()

        def prior(x_pair):
            # Here we assume an independent additive prior
            # resp2d for (ch1,ch2) pair will be gp1(ch1) + gp1(ch2)
            y1 = gp1.predict(x_pair[0],x_pair[1]).item()
            y2 = gp1.predict(x_pair[2],x_pair[3]).item()
            return 1/2*(y1 + y2)

        if self.use_initpts:
            # We take the first use_initpts number of pts from gp1
            # Take all of their pairs, and use these as initial points
            vals = gp1.resf.func_vals
            xs = gp1.resf.x_iters
            sorted_xs = [x for _,x in sorted(zip(vals,xs))]
            unique_sorted_xs = list(OrderedDict.fromkeys([tuple(l) for l in sorted_xs]))
            xs = unique_sorted_xs[:self.use_initpts]
            pairs = list(itertools.product(xs,xs))
            x0 = [list(a)+list(b) for (a,b) in pairs]
            n_random_starts=0
        else:
            n_random_starts=10
            x0 = None

        if not self.use_prior:
            # If we don't have a prior, then we wil use the gp_minimize interface
            resg = gp_minimize(self.g,
                               spaceg,
                               base_estimator=self.gpr,
                               acq_func="EI",
                               n_calls=50,
                               n_random_starts=n_random_starts,
                               noise=0,
                               x0=x0,
                               acq_optimizer="sampling")

        else:
            # Here we use the prior from the single train gp1,
            # We need to use the ask-and-tell interface to use our own
            # acq function
            # we minimize 2resp = prior + res_gp (we learn the
            # residual only with the gp)

            opt = Optimizer(spaceg, base_estimator=self.gpr, acq_optimizer="sampling")

            g_vals = []
            sformat = "{:15}"*4
            print(sformat.format("x","real_value", "prior pred", "difference"))
            # We first query the first n (here 10) points randomly
            for _ in range(10):
                next_x = opt.ask()
                g_val = self.g(next_x)
                g_vals.append(g_val)
                diff_val = g_val - prior(next_x)
                print(next_x, g_val, prior(next_x), diff_val)
                opt.tell(next_x, diff_val)

            # We use xy for the GP
            aa = cc = range(2)
            bb = dd = range(5)
            xy = np.array([[a,b,c,d] for a in aa for b in bb
                           for c in cc for d in dd])
            xy_model = opt.space.transform(xy.tolist())
            priorxy = np.array([prior(xy_i) for xy_i in xy])

            for _ in range(40):
                # We need in case of noisy observations
                # We want the max value seen so far not to be a noisy
                # observation
                #So we take the mean of GP prediction (note that this
                #will be the same in noise-free case)
                y_max = self.resg.predict(xy_model).max()
                ei = ei_with_prior(xy_model, opt.models[-1], priorxy, y_opt=y_max)
                maxidx = ei.argmax()
                next_x = xy[maxidx].tolist()
                g_val = self.g(next_x)
                g_vals.append(g_val)
                diff_val = g_val - prior(next_x)
                print(next_x, g_val, diff_val)
                resg = opt.tell(next_x, diff_val)
                self.resg = resg

        return resg

class NoisyGP2(GP2):
    def __init__(self, n_dims=4, use_initpts=False, use_prior=False):
        super(NoisyGP2, self).__init__(use_initpts=use_initpts, use_prior=use_prior)
        #self.gpr = self.init_kernel(n_dims)

class NoiseFreeGP2(GP2):

    def __init__(self, n_dims=4, use_initpts=False, use_prior=False):
        super(NoiseFreeGP2, self).__init__(use_initpts=use_initpts, use_prior=use_prior)
        self.gpr = self.init_kernel(n_dims)

    def init_kernel(self, n_dims):
        cov_amplitude = skopt.learning.gaussian_process.kernels.ConstantKernel(1.0, (0.01, 1000.0))
        other_kernel  = skopt.learning.gaussian_process.kernels.Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=[(0.01, 100)] * n_dims, nu=2.5)
        # We add a small amount of noise to help with numerical stability
        gpr = skopt.learning.GaussianProcessRegressor(
            kernel               = cov_amplitude * other_kernel,
            normalize_y          = False,
            noise                = 0,
            n_restarts_optimizer = 2)
        return gpr

    def traingp1(self):
        gp1 = NoiseFreeGP()
        gp1.minimize()
        return gp1

    def g(self, x):
        x1,y1 = x[0], x[1]
        x2,y2 = x[2], x[3]
        ch1 = self.trains.xy2ch(x1,y1)
        ch2 = self.trains.xy2ch(x2,y2)
        resp = self.trains.trains[ch1][ch2][0]['meanmax']
        return -resp * SCALE

# gp2 = NoiseFreeGP2(use_prior=True)
# res_prior = gp2.minimize()
# gp2.use_prior = False
# res_noprior = gp2.minimize()
# my_plot_convergence(("prior",res_prior), ("noprior",res_noprior))
# plt.show()
