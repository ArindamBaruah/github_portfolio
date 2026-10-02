import numpy as np
from scipy.optimize import minimize


def fit_garch11(returns_pct):
    """Fit GARCH(1,1) via MLE on a 1D array of percent returns. Returns (omega, alpha, beta)."""
    r = np.asarray(returns_pct)

    def negloglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        n = len(r)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(r)
        for t in range(1, n):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r ** 2 / sigma2)
        return -ll

    res = minimize(negloglik, [0.05, 0.08, 0.90], method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    return res.x  # omega, alpha, beta


def garch_sigma2_series(returns_pct, omega, alpha, beta):
    """Recursive in-sample conditional variance sigma2_t, using only returns up to t-1 (no leakage)."""
    r = np.asarray(returns_pct)
    n = len(r)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(r)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def garch_one_step_forecast(last_return_pct, last_sigma2, omega, alpha, beta):
    return omega + alpha * last_return_pct ** 2 + beta * last_sigma2
