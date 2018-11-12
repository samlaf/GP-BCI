import numpy as np
import matplotlib.pyplot as plt
plt.set_cmap("viridis")
from skopt.acquisition import gaussian_ei
from skopt import Optimizer
from utils import ei_with_prior

np.random.seed(1233)
noise_level = 0.1

# Our 1D toy problem, this is the function we are trying to
# minimize
def prior(x, noise_level=noise_level):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2))
def f(x, a=0.2, noise_level=noise_level):
    return prior(x) + a*(x[0]) + np.random.randn() * noise_level

# Plot f(x) + contours
x = np.linspace(-2, 2, 400).reshape(-1, 1)
fx = np.array([f(x_i, noise_level=0.0) for x_i in x])
priorx = np.array([prior(x_i) for x_i in x])
# plt.plot(x, fx, "r--", label="True (unknown)")
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx], 
#                          [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
#          alpha=.2, fc="r", ec="None")
# plt.legend()
# plt.grid()
# plt.show()

def plot_optimizer(opt, x, fx):
    model = opt.models[-1]
    x_model = opt.space.transform(x.tolist())

    # Plot true function.
    plt.plot(x, priorx, "b--", label="Prior")
    plt.plot(x, fx-priorx, "y--", label="Diff")
    plt.plot(x, fx, "r--", label="True (unknown)")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([fx - 1.9600 * noise_level, 
                             fx[::-1] + 1.9600 * noise_level]),
             alpha=.2, fc="r", ec="None")

    # Plot Model(x) + contours
    y_pred, sigma = model.predict(x_model, return_std=True)
    plt.plot(x, y_pred, "g--", label=r"$\mu(x)$")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma, 
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.2, fc="g", ec="None")

    # Plot sampled points
    plt.plot(opt.Xi, opt.yi,
             "r.", markersize=8, label="Observations")

    acq = gaussian_ei(x_model, model, y_opt=np.min(opt.yi))
    # shift down to make a better plot
    acq = 4*acq - 2
    plt.plot(x, acq, "b", label="EI(x)")
    plt.fill_between(x.ravel(), -2.0, acq.ravel(), alpha=0.3, color='blue')

    # Adjust plot layout
    plt.grid()
    plt.legend(loc='best')

opt = Optimizer([(-2.0, 2.0)], "GP", acq_optimizer="sampling")

f_vals = []
for _ in range(10):
    next_x = opt.ask()
    f_val = f(next_x)
    f_vals.append(f_val)
    diff_val = f_val - prior(next_x)
    opt.tell(next_x, diff_val)

x_model = opt.space.transform(x.tolist())
for _ in range(10):
    ei = ei_with_prior(x_model, opt.models[-1], priorx, y_opt=np.min(f_vals))
    maxidx = ei.argmax()
    next_x = x[maxidx].tolist()
    f_val = f(next_x)
    f_vals.append(f_val)
    diff_val = f_val - prior(next_x)
    opt.tell(next_x, diff_val)

plot_optimizer(opt, x, fx)
plt.show()
