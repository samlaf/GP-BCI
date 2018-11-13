"""
Here we make tests to see how many queries it takes for a GP to learn low order polynomial functions in 2D
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skopt.acquisition import gaussian_ei
from skopt.space import Real, Integer
from skopt import gp_minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

noise_level = 1.0
XLIM = 8

@np.vectorize
def f_helper(x,y, noise_level=noise_level):
    return f([x,y], noise_level)
def f(x, noise_level=noise_level):
    #return x[0]+x[1]
    mvn = multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
    return -100*mvn.pdf(x)
    #return np.linalg.norm(x)
@np.vectorize
def gauss(x,y, noise_level=noise_level):
    mvn = multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
    return -100*mvn.pdf([x,y])

# Plot f(x) + contours
grid = np.arange(-XLIM,XLIM,.1)
xgrid, ygrid = np.meshgrid(grid,grid)
zgrid = f_helper(xgrid, ygrid, noise_level=0)
ax = plt.subplot(111, projection='3d')
ax.plot_surface(xgrid, ygrid, zgrid)


spacef = [Integer(-XLIM, XLIM, name='x'),
          Integer(-XLIM, XLIM, name='y')]

n_dims=2
cov_amplitude = skopt.learning.gaussian_process.kernels.ConstantKernel(1.0, (0.01, 1000.0))
other_kernel  = skopt.learning.gaussian_process.kernels.Matern(
    length_scale=np.ones(n_dims),
    length_scale_bounds=[(0.10, 100)] * n_dims, nu=2.5)
noise_kernel = skopt.learning.gaussian_process.kernels.WhiteKernel()
gpr = skopt.learning.GaussianProcessRegressor(
            kernel               = cov_amplitude * other_kernel,
            normalize_y          = True,
            noise                = noise_level,
            n_restarts_optimizer = 2)

res = gp_minimize(f,                  # the function to minimize
                  spacef,      # the bounds on each dimension of x
                  base_estimator=gpr,
                  acq_func="EI",      # the acquisition function
                  n_calls=25,         # the number of evaluations of f 
                  n_random_starts=10,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=123)   # the random seed

xy = np.array([[xi,yi]  for yi in grid for xi in grid])
xy_model = res.space.transform(xy.tolist())
xy_pred, sigma = res.models[-1].predict(xy_model, return_std=True)
xy_pred = xy_pred.reshape(160,160)
sigma = sigma.reshape(160,160)
FC = sigma / sigma.max()

ax.plot_surface(xgrid, ygrid, xy_pred,
                facecolors=cm.viridis(FC),
                linewidth=0, antialiased=False)

x_scat = np.array([p[0] for p in res.x_iters])
y_scat = np.array([p[1] for p in res.x_iters])
ax.plot(x_scat, y_scat, res.func_vals, 'r.', markersize=10, label='Observations')
plt.show()

# plt.rcParams["figure.figsize"] = (8, 8)

# x = np.linspace(-XLIM, XLIM, 400).reshape(-1, 1)
# x_gp = res.space.transform(x.tolist())
# fx = np.array([f(x_i, noise_level=0.0) for x_i in x])

# # Plot the 5 iterations following the 5 random points
# for n_iter in range(5):
#     gp = res.models[n_iter]
#     curr_x_iters = res.x_iters[:5+n_iter]
#     curr_func_vals = res.func_vals[:5+n_iter]

#     # Plot true function.
#     plt.subplot(5, 2, 2*n_iter+1)
#     plt.plot(x, fx, "r--", label="True (unknown)")
#     plt.fill(np.concatenate([x, x[::-1]]),
#              np.concatenate([fx - 1.9600 * noise_level, 
#                              fx[::-1] + 1.9600 * noise_level]),
#              alpha=.2, fc="r", ec="None")

#     # Plot GP(x) + contours
#     y_pred, sigma = gp.predict(x_gp, return_std=True)
#     plt.plot(x, y_pred, "g--", label=r"$\mu_{GP}(x)$")
#     plt.fill(np.concatenate([x, x[::-1]]),
#              np.concatenate([y_pred - 1.9600 * sigma, 
#                              (y_pred + 1.9600 * sigma)[::-1]]),
#              alpha=.2, fc="g", ec="None")

#     # Plot sampled points
#     plt.plot(curr_x_iters, curr_func_vals,
#              "r.", markersize=8, label="First 5 Random Observations")
#     plt.plot(curr_x_iters[5:], curr_func_vals[5:],
#              "b.", markersize=8, label="Queries")

#     # Adjust plot layout
#     plt.grid()

#     if n_iter == 0:
#         plt.legend(loc="best", prop={'size': 6}, numpoints=1)

#     if n_iter != 4:
#         plt.tick_params(axis='x', which='both', bottom='off', 
#                         top='off', labelbottom='off') 

#     # Plot EI(x)
#     plt.subplot(5, 2, 2*n_iter+2)
#     acq = gaussian_ei(x_gp, gp, y_opt=np.min(curr_func_vals))
#     plt.plot(x, acq, "b", label="EI(x)")
#     plt.fill_between(x.ravel(), -2.0, acq.ravel(), alpha=0.3, color='blue')

#     next_x = res.x_iters[5+n_iter]
#     next_acq = gaussian_ei(res.space.transform([next_x]), gp, y_opt=np.min(curr_func_vals))
#     plt.plot(next_x, next_acq, "bo", markersize=6, label="Next query point")

#     # Adjust plot layout
#     plt.ylim(0, 0.5)
#     plt.grid()

#     if n_iter == 0:
#         plt.legend(loc="best", prop={'size': 6}, numpoints=1)

#     if n_iter != 4:
#         plt.tick_params(axis='x', which='both', bottom='off', 
#                         top='off', labelbottom='off') 

# plt.show()
