import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

noise_level=0.1

def f1(x, noise_level=noise_level):
    return np.sin(x[0]) * (x[0]-3) + np.random.randn() * noise_level
def f_helper(x,y,noise_level=noise_level):
    return foo([x,y])
def f(x, noise_level=noise_level):
    return f1([x[0]]) + f1([x[1]]) +  1/5*x[0]*x[1] + np.random.randn() * noise_level
def foo(x, noise_level=noise_level):
    return f1([x[0]]) + f1([x[1]]) + np.random.randn() * noise_level


# def f(p):
#     """The function to predict."""
#     x,y = p
#     return x+y

x = np.arange(-8,8,.1)
y = np.arange(-8,8,.1)
x_grid, y_grid = np.meshgrid(x,y)
# x_grid = np.array([(x,y) for x in np.arange(-8,8,.1) for y in
# np.arange(-8,8,.1)])
z_grid = f_helper(x_grid, y_grid)
# y_grid = np.apply_along_axis(f,1,x_grid)
# yfoo_grid = np.apply_along_axis(foo,1,x_grid)
# X = np.array([(0,3),(2,7),(3,3)])
# y = np.apply_along_axis(f,1,X)

# # Instantiate a Gaussian Process model
# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# # Fit to data using Maximum Likelihood Estimation of the parameters
# gp.fit(X, y)

# # Make the prediction on the meshed x-axis (ask for MSE as well)
# y_pred, sigma = gp.predict(x_grid, return_std=True)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x_grid, y_grid, z_grid,
                       #cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax = fig.add_subplot(311, projection='3d')
ax.scatter(x_grid[:,0],x_grid[:,1],y_grid)
# ax1 = fig.add_subplot(312, projection='3d')
# ax1.scatter(x_grid[:,0],x_grid[:,1],yfoo_grid)
# ax2 = fig.add_subplot(313, projection='3d')
# ax2.scatter(x_grid[:,0],x_grid[:,1],y_pred)

plt.show()

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
# plt.figure()
# plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
# plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
# plt.plot(x, y_pred, 'b-', label=u'Prediction')
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([y_pred - 1.9600 * sigma,
#                         (y_pred + 1.9600 * sigma)[::-1]]),
#          alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('$x$')
# plt.ylabel('$f(x)$')
# plt.ylim(-10, 20)
# plt.legend(loc='upper left')

def GP(n=100):
    p0 = np.array((random.randint(0,3), random.randint(0,7)))
    
    
