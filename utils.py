import warnings
import numpy as np
from scipy.stats import norm

def ei_with_prior(X, model, priorx, y_opt=0.0, xi=0.01):
    """
    Similar to skopt's gaussian_ei function
(https://scikit-optimize.github.io/acquisition.m.html#skopt.acquisition.gaussian_ei)
except here we also use a prior function, and want to minimize model + prior instead of just model
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        mu, std = model.predict(X, return_std=True)

    mu = mu + priorx
    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore

    return values
