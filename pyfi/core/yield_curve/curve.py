import numpy as np
import matplotlib.pyplot as plt

class Curve:

    def __init__(self, t: (np.ndarray, list), rt: (np.ndarray, list)) -> None:
        self._t = self._is_valid_attr(t)   # time index (maturities)
        self._rt = self._is_valid_attr(rt) # rates (yield curve)

    @property
    def get_time(self) -> (np.ndarray, list):
        """Time getter"""
        return self.__getattribute__("_t")

    @property
    def get_rate(self) -> (np.ndarray, list):
        """Rate getter"""
        return self.__getattribute__("_rt")

    @staticmethod
    def _is_valid_attr(attr) -> (np.ndarray, list):
        assert isinstance(attr, (np.ndarray, list)), "Class Constructor takes only numpy arrays or list as arguments"
        return attr

    def set_time(self, t: (np.ndarray, list)) -> None:
        self.__setattr__("_t", self._is_valid_attr(t))

    def set_rate(self, rt: (np.ndarray, list)) -> None:
        self.__setattr__("_rt", self._is_valid_attr(rt))

    def plot_curve(self) -> None:
        fig, ax = plt.subplots(1)
        fig.suptitle("Yield Curve Term Structure")
        # fig.canvas.set_window_title('Yield Curve Term Structure')
        # ax.set_xlabel('Time, t')
        # ax.set_ylabel('Simulated Asset Price')

        xi = list(range(len(self.get_time)))
        ax.plot(xi, self.get_rate, label="Yield Curve", lw=0.8, color="navy")
        ax.set_xticks(xi, self.get_time)

        ax.set_xlabel('Time')
        ax.set_ylabel('Yield')
        ax.fill_between(xi, self.get_rate,
                        facecolor="lightskyblue", alpha=0.3)
        plt.show()
        return fig