from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sco
from curve_tools import discrete_df, continuous_df
from curve import Curve

np.seterr(divide='ignore', invalid='ignore')



class NelsonSiegel:
    # estimation of yield curve shapes based on real-world yields (par curve)
    
    def __init__(self, beta0: float, beta1: float, beta2: float, tau: float) -> None:
        self.beta0: float = self._is_positive_attr(beta0)    # level; long-term rate
        self.beta1: float = beta1                            # slope: steepeness, inversion, etc.
        self.beta2: float = beta2                            # shape: humped, maximum
        self.tau: float = self._is_positive_attr(tau)        # decay: exponential decay function; models more flexibile range of shapes; cannot be zero
        self.attr_list: list = ["beta0", "beta1", "beta2", "tau"]

    @staticmethod
    def _is_valid_curve(curve: Any) -> Curve:
        """Check if an attribute is an instance of Curve"""
        if not (isinstance(curve, Curve)):
            raise ValueError("Curve parameter must be an instance of Curve")
        return curve

    @staticmethod
    def _is_positive_attr(attr) -> float:
        if attr < 0:
            raise AttributeError("Beta0 and tau, must be positive")
        return attr

    def get_attr(self, attr) -> Union[list, float]:
        return self.__getattribute__(attr)

    def set_attr(self, attr, x) -> None:
        if attr in (["beta0", "tau"]):
            self.__setattr__(attr, self._is_positive_attr(x))
        self.__setattr__(attr, x)

    def print_model(self) -> None:
        print("Nelson Siegel Model")
        print(28 * "=")
        for attr in self.attr_list:
            print(attr + " =", self.get_attr(attr))
        print(28 * "_")

    @staticmethod
    def _calibration_func(x, curve) -> float:
        ns = NelsonSiegel(x[0], x[1], x[2], x[3])
        curve_estim = np.ones(len(curve.get_time))
        for i in range(len(curve.get_time)):
            curve_estim[i] = ns.d_rate(curve.get_time[i])
        sqr_error_sum = ((curve_estim - curve.get_rate) ** 2).sum()
        return sqr_error_sum

    def _print_fitting(self, res) -> None:
        self.print_model()
        print(28 * "=")
        print("Calibration Results")
        print(28 * "=")
        print(res.message)
        print("Mean Squared Error", res.fun)
        print("Number of Iterations", res.nit)
        print(28 * "_")

    def calibrate(self, curve) -> sco.OptimizeResult:
        self._is_valid_curve(curve)
        x0 = np.array([1, 0, 0, 1])
        boundaries = ((1e-6, np.inf), (-30, 30), (-30, 30), (1e-6, 30))
        calibration_result = sco.minimize(self._calibration_func, x0, method='L-BFGS-B', args=curve, bounds=boundaries)
        i = 0
        for attr in self.attr_list:
            self.set_attr(attr, calibration_result.x[i])
            i += 1
        self._print_fitting(calibration_result)
        return calibration_result

    def _time_decay(self, t) -> Union[np.ndarray, float]:
        return self.beta1 * (np.array(((1 - np.exp(-np.array(t, np.longdouble) / self.tau)) /
                                       (np.array(t, np.longdouble) / self.tau))))

    def _hump(self, t) -> Union[np.ndarray, float]:
        return self.beta2 * np.subtract(np.array(((1 - np.exp(-np.array(t, np.longdouble) / self.tau)) /
                                                  (np.array(t, np.longdouble) / self.tau))),
                                        np.array(np.exp(-np.array(t, np.longdouble) / self.tau)))

    def d_rate(self, t) -> Union[np.ndarray, float]:
        first_coefficient = self._time_decay(t)
        second_coefficient = self._hump(t)
        return self.beta0 + first_coefficient + second_coefficient

    def plot_calibrated(self, curve: Curve) -> None:
        fig = plt.figure(figsize=(20, 5))
        fig.suptitle("Nelson Siegel Curve")
        # fig.canvas.set_window_title('Nelson Siegel Curve')
        t = curve.get_time
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('t, years')
        ax1.set_ylabel('Yield')
        ax1.plot(t, self.d_rate(t), c="seagreen", label="Calibrated")
        ax1.plot(t, curve.get_rate, c="chocolate", label="Observed")
        plt.legend()
        plt.show()
        return fig

    def plot_model_params(self) -> None:
        t = np.linspace(0, 50, 1000)
        b0 = np.ones(1000) * self.beta0
        fig = plt.figure(figsize=(20, 8))        
        fig.suptitle("Nelson Siegel Parameters")
        # fig.canvas.set_window_title('Nelson Siegel Parameters')
        
        ax1 = fig.add_subplot(212)
        ax1.plot(t, self.d_rate(t))
        ax1.set_xlabel('t, years')
        ax1.set_ylabel('Yield')
        ax1.set_title('Nelson Siegel Model')
        ax2 = fig.add_subplot(231)
        ax2.plot(t, b0)
        ax2.set_ylabel('Yield')
        ax2.set_title('Long Term Rate')
        ax3 = fig.add_subplot(232)
        ax3.plot(t, self._time_decay(t))
        ax3.set_title('Time Decay Part')
        ax4 = fig.add_subplot(233)
        ax4.plot(t, self._hump(t))
        ax4.set_title('Hump Part')
        plt.show()
        return fig

    def plot_model(self) -> None:
        t = np.linspace(0, 50, 1000)
        b0 = np.ones(1000) * self.beta0

        fig = plt.figure(figsize=(20, 5))        
        fig.suptitle("Nelson Siegel BreakDown")
        # fig.canvas.set_window_title('Nelson Siegel Components')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('t, years')
        ax.set_ylabel('Yield')
        ax.plot(t, self.d_rate(t), label="Full Model")
        ax.plot(t, b0, label="Long Term Rate")
        ax.plot(t, self._hump(t), label="Hump")
        ax.plot(t, self._time_decay(t), label="Time Decay")
        plt.legend()
        plt.show()
        return fig

    def df_t(self, t) -> Union[np.ndarray, float]:
        return discrete_df(self.d_rate(t), t)

    def cdf_t(self, t) -> Union[np.ndarray, float]:
        return continuous_df(self.d_rate(t), t)

    def forward_rate(self, t_1, t_2) -> Union[np.ndarray, float]:
        return ((self.d_rate(t_2) * t_2) - (self.d_rate(t_1) * t_1)) / (t_2 - t_1)