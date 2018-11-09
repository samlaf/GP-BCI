import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from skopt.learning import GaussianProcessRegressor as GPR
import matplotlib.pyplot as plt
from skopt.space import Real, Integer
import itertools

noise_level = 0.1
n_dims = 1

def f(x, noise_level=noise_level):
    return np.sin(x[0]) * (x[0]-2) + np.random.randn() * noise_level
def g_helper(x,y,a=1,noise_level=noise_level):
    return g([x,y],a,noise_level)
def g(x, a=1, noise_level=noise_level):
    return f([x[0]]) + f([x[1]]) + a*(x[0]+x[1]) + np.random.randn() * noise_level

spacef = [Integer(-8, 8, name='x')]

cov_amplitude = skopt.learning.gaussian_process.kernels.ConstantKernel(1.0, (0.01, 1000.0))
other_kernel  = skopt.learning.gaussian_process.kernels.Matern(
    length_scale=np.ones(n_dims),
    length_scale_bounds=[(0.1, 100)] * n_dims, nu=2.5)
noise_kernel = skopt.learning.gaussian_process.kernels.WhiteKernel()
gpr = skopt.learning.GaussianProcessRegressor(
            kernel               = cov_amplitude * other_kernel,
            normalize_y          = True,
            noise                = noise_level,
            n_restarts_optimizer = 2)
# resf = gp_minimize(f,
#                    spacef,
#                    base_estimator=gpr,
#                    acq_func="EI",
#                    n_calls=10,
#                    n_random_starts=5,
#                    noise=noise_level**2)

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

# plt.plot(x, fx, "r:", label="f(x) = (x-3)sin(x) + eps")
# plt.plot(resf.x_iters, resf.func_vals, 'r.', markersize=10, label='Observations')
# plt.plot(x, y_pred, 'b-', label=u'Prediction')
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([y_pred - 1.9600 * sigma,
#                          (y_pred + 1.9600 * sigma)[::-1]]),
#          alpha=.5, fc="b", ec="None")
# plt.legend()
# plt.grid(False)
# plt.show()


##### -------------------------------------------------- #####
#####                g function (2d)                     #####

spaceg = [Integer(-8, 8, name='x'),
          Integer(-8, 8, name='y')]

# Prior for our gp approx to g
flattenedx = np.array(resf.x_iters).flatten()
x0 = list(itertools.product(flattenedx,flattenedx))
flattenedy = np.array(resf.func_vals)
ypairs = list(itertools.product(flattenedy,flattenedy))
y0 = [a+b for (a,b) in ypairs]

resg_prior = gp_minimize(g,
                   spaceg,
                   acq_func="EI",
                   n_calls=50,
                   n_random_starts=5,
                   noise=100*noise_level**2,
                   x0=x0,
                   y0=y0)
# resg = gp_minimize(g,
#                    spaceg,
#                    acq_func="EI",
#                    n_calls=50,
#                    n_random_starts=5,
#                    noise=noise_level**2)

x = np.arange(-8,8,.1)
y = np.arange(-8,8,.1)
x_grid, y_grid = np.meshgrid(x,y)
z_grid_prior = g_helper(x_grid, y_grid, a=0, noise_level=0)
z_grid = g_helper(x_grid, y_grid, noise_level=0)

ax = plt.subplot(1,1,1,projection='3d')
surf = ax.plot_surface(x_grid, y_grid, z_grid,
                       #cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Plot the observation points as scatter
x_scat = np.array([p[0] for p in resg_prior.x_iters])
y_scat = np.array([p[1] for p in resg_prior.x_iters])
z_scat = g_helper(x_scat, y_scat)
z_scat_prior = g_helper(x_scat,y_scat,a=0)
n_prior = len(x0)
ax.plot(x_scat[n_prior:], y_scat[n_prior:], z_scat[n_prior:], 'r.', markersize=10, label='Observations')
ax.plot(x_scat[:n_prior], y_scat[:n_prior], z_scat_prior[:n_prior], 'b.', markersize=10, label='Prior')

# ax = plt.subplot(1,2,2,projection='3d')
# surf = ax.plot_surface(x_grid, y_grid, z_grid,
#                        #cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
plt.legend()
plt.show()
