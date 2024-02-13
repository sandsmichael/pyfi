from typing import Tuple, Any

import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize

from nelson_siegel import NelsonSiegelCurve


def _assert_same_shape(t: np.ndarray, y: np.ndarray) -> None:
    assert t.shape == y.shape, "Mismatching shapes of time and values"


def betas_ns_ols(
    tau: float, t: np.ndarray, y: np.ndarray
) -> Tuple[NelsonSiegelCurve, Any]:
    """Calculate the best-fitting beta-values given tau
    for time-value pairs t and y and return a corresponding
    Nelson-Siegel curve instance.
    """
    _assert_same_shape(t, y)
    curve = NelsonSiegelCurve(0, 0, 0, tau)
    factors = curve.factor_matrix(t)
    print('factors:', factors)
    # print(factors.shape)
    lstsq_res = lstsq(factors, y, rcond=None)  #Return the least-squares solution to a linear matrix equation.
    beta = lstsq_res[0]
    # print(lstsq_res)
    print('beta:', beta)
    return NelsonSiegelCurve(beta[0], beta[1], beta[2], tau), lstsq_res


def errorfn_ns_ols(tau: float, t: np.ndarray, y: np.ndarray) -> float:
    """Sum of squares error function for a Nelson-Siegel model and
    time-value pairs t and y. All betas are obtained by ordinary
    least squares given tau.
    """
    print('tau:', tau)
    _assert_same_shape(t, y)
    curve, lstsq_res = betas_ns_ols(tau, t, y)
    return np.sum((curve(t) - y) ** 2)        # minimize the squared difference between the calibrated curve (curve at time t) and the observed y value


def calibrate_ns_ols(
    t: np.ndarray, y: np.ndarray, tau0: float = 2.0
) -> Tuple[NelsonSiegelCurve, Any]:
    """Calibrate a Nelson-Siegel curve to time-value pairs
    t and y, by optimizing tau and chosing all betas
    using ordinary least squares.
    """
    _assert_same_shape(t, y)
    opt_res = minimize(errorfn_ns_ols, x0=tau0, args=(t, y))
    curve, lstsq_res = betas_ns_ols(opt_res.x[0], t, y)
    return curve, opt_res

