"""
Here we first learn the 1D GP, and build the 2D prior using its mean assuming independence.
We then learn the 2D GP on the difference (f-prior), while maximizing f itself when querying points (diff + prior is the function we want to minimize when building the acq function, not diff)
"""

import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from skopt.learning import GaussianProcessRegressor as GPR
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skopt.space import Real, Integer
import itertools
from utils import ei_with_prior
from skopt import Optimizer
from matplotlib import cm
from scipy.stats import multivariate_normal

noise_level = 0.1
n_dims = 1
np.random.seed(1234)

def f(x, noise_level=noise_level):
    return np.sin(x[0]) * (x[0]-2) + np.random.randn() * noise_level
def prior(x, a=1, noise_level=noise_level):
    return f([x[0]], noise_level=noise_level) + a*f([x[1]], noise_level=noise_level)
def prior_helper(x,y):
    return prior([x,y],noise_level)
@np.vectorize
def diff(x,y,b=100):
    #return -1/2*x*y

    #return b*(x+y)

    # if np.linalg.norm([x+2,y+5]) >= 2:
    #     return 0
    # else:
    #     return b*((x+2)**2 * (y+5)**2) - b*4

    mvn = multivariate_normal(mean=[-3,-5], cov=[[1,0],[0,1]])
    return -b*mvn.pdf([x,y])

    
def g(x, b=100, noise_level=noise_level):
    return prior(x) + diff(x[0],x[1],b) + np.random.randn() * noise_level
def g_helper(x,y,b=100,noise_level=noise_level):
    return g([x,y],b,noise_level)

spacef = [Integer(-8, 8, name='x')]

cov_amplitude = skopt.learning.gaussian_process.kernels.ConstantKernel(1.0, (0.01, 1000.0))
other_kernel  = skopt.learning.gaussian_process.kernels.Matern(
    length_scale=np.ones(n_dims),
    length_scale_bounds=[(0.1, 100)] * n_dims, nu=2.5)
#noise_kernel = skopt.learning.gaussian_process.kernels.WhiteKernel()
gpr = skopt.learning.GaussianProcessRegressor(
            kernel               = cov_amplitude * other_kernel,
            normalize_y          = True,
            noise                = noise_level,
            n_restarts_optimizer = 2)
resf = gp_minimize(f,
                   spacef,
                   base_estimator=gpr,
                   acq_func="EI",
                   n_calls=10,
                   n_random_starts=5,
                   noise=noise_level**2)

# TEST
for m in resf.models:
    print(m.kernel_)
print(resf.x_iters)

# Plot f(x) + contours
x = np.linspace(-8, 8, 400).reshape(-1, 1)
fx = [f(x_i, noise_level=0.0) for x_i in x]

surrogate_m = resf.models[-1]
# Note: We need to transform x for the gp prediction since skopt
# normalizes the input space to [0,1]
x_gp = resf.space.transform(x)
y_pred, sigma = surrogate_m.predict(x_gp, return_std=True)

plt.plot(x, fx, "r:", label="f(x) = (x-3)sin(x) + eps")
plt.plot(resf.x_iters, resf.func_vals, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                         (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc="b", ec="None")
plt.legend()
plt.grid(False)
plt.show()


##### -------------------------------------------------- #####
#####                g function (2d)                     #####

spaceg = [Integer(-8, 8, name='x'),
          Integer(-8, 8, name='y')]

n_dims = 2
cov_amplitude = skopt.learning.gaussian_process.kernels.ConstantKernel(1.0, (0.01, 1000.0))
other_kernel  = skopt.learning.gaussian_process.kernels.Matern(
    length_scale=np.ones(n_dims),
    length_scale_bounds=[(0.25, 100)] * n_dims, nu=2.5)
#noise_kernel = skopt.learning.gaussian_process.kernels.WhiteKernel()
gpr = skopt.learning.GaussianProcessRegressor(
            kernel               = cov_amplitude * other_kernel,
            normalize_y          = True,
            noise                = noise_level,
            n_restarts_optimizer = 2)
opt = Optimizer(spaceg, base_estimator=gpr, acq_optimizer="sampling")

g_vals = []
# We first query the first n (here 10) points randomly
for _ in range(10):
    next_x = opt.ask()
    g_val = g(next_x)
    g_vals.append(g_val)
    diff_val = g_val - prior(next_x, noise_level=0)
    print(next_x, diff_val)
    opt.tell(next_x, diff_val)
# We also give it the max from the 1D GP
x0 = [(-5, -5), (-5, -4), (-4, -5), (-4, -4)]
for next_x in x0:
    g_val = g(next_x)
    g_vals.append(g_val)
    diff_val = g_val - prior(next_x, noise_level=0)
    opt.tell(next_x, diff_val)

x = np.arange(-8,9)
y = np.arange(-8,9)
# We use xy for the GP
xy = np.array([[xi,yi] for xi in x for yi in y])
xy_model = opt.space.transform(xy.tolist())
priorxy = np.array([prior(xy_i) for xy_i in xy])

for _ in range(50):
    ei = ei_with_prior(xy_model, opt.models[-1], priorxy, y_opt=np.min(g_vals))
    maxidx = ei.argmax()
    next_x = xy[maxidx].tolist()
    g_val = g(next_x)
    g_vals.append(g_val)
    diff_val = g_val - prior(next_x)
    print(next_x, diff_val, g_val)
    opt.tell(next_x, diff_val)

# We use the meshgrid for plotting
rng = np.arange(-8,8,.1)
x_grid, y_grid = np.meshgrid(rng, rng)
z_grid_prior = g_helper(x_grid, y_grid, b=0, noise_level=0)
z_grid = g_helper(x_grid, y_grid, noise_level=0)
z_grid_diff = z_grid - z_grid_prior

diff_model = opt.models[-1]
x = np.arange(-8,8,.1)
y = np.arange(-8,8,.1)
# We use xy for the GP
xy = np.array([[xi,yi] for yi in y for xi in x])
xy_model = opt.space.transform(xy.tolist())
xy_pred, sigma = diff_model.predict(xy_model, return_std=True)
xy_pred = xy_pred.reshape(160,160)
sigma = sigma.reshape(160,160)
FC = sigma / sigma.max()

# Check that diff works
xy_diff = diff(x_grid, y_grid)

ax = plt.subplot(1,1,1,projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid,
                       #cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.plot_surface(x_grid, y_grid, xy_diff,
                       #cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.plot_surface(x_grid, y_grid, xy_pred,
                facecolors=cm.viridis(FC),
                linewidth=0, antialiased=False)

# Plot the observation points as scatter

# x_scat = np.array([p[0] for p in opt.Xi])
# y_scat = np.array([p[1] for p in opt.Xi])
# ax.plot(x_scat[:12], y_scat[:12], opt.yi[:12], 'r.', markersize=10, label='Prior Observations')
# ax.plot(x_scat[12:], y_scat[12:], opt.yi[12:], 'g.', markersize=10, label='Queried Observations')

plt.legend()
plt.show()




#PLOTS FOR PAPER
# plt.plot(range(10,100), priorconv, 'b.-', label='Using Prior')
# plt.plot(nopconv, 'r.-', label='No Prior')
# plt.title('Convergence Rates')
# plt.xlabel('Iterations')
# plt.ylabel('Min Value')
# plt.legend()

# plt.plot(sorted(resf.func_vals, reverse=True), 'b.-')
# plt.title('Convergence Rates 1D')
# plt.xlabel('Iterations')
# plt.ylabel('Min Value')

# We shift the grids to have the axes labeled from 0 to 16 instead of
# -8 to 8
# x_grid += 8
# y_grid += 8

# ax = plt.subplot(1,1,1,projection='3d')
# ax.plot_surface(x_grid, y_grid, z_grid,
#                 color=(.8,0,0),
#                 linewidth=0, antialiased=False)
# ax.set_zlim([-20,20])
# plt.savefig('real.pdf', format='pdf')

# ax = plt.subplot(1,1,1,projection='3d')
# ax.plot_surface(x_grid, y_grid, z_grid_prior,
#                 #cmap=cm.coolwarm,
#                 linewidth=0, antialiased=False)
# ax.set_zlim([-20,20])
# plt.savefig('prior.pdf', format='pdf')

# ax = plt.subplot(1,1,1,projection='3d')
# ax.plot_surface(x_grid, y_grid, xy_diff,
#                 color='g',
#                 linewidth=0, antialiased=False)
# ax.set_zlim([-20,20])
# plt.savefig('diff.pdf', format='pdf')
