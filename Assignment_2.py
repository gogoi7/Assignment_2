from typing import Callable

import numpy as np


class ODESolver:
    """
    A base class used to represent an ODE solver.

    ...

    Attributes
    ----------
    f : Callable[[float, float], float]
        a function representing the ODE
    y_0 : float
        initial condition
    t_0 : float
        start time
    t_f : float
        stop time
    dt : float
        step size

    Methods
    -------
    solve():
        Solves the ODE and returns the numerical solution as a function of time.
    """

    def __init__(
        self,
        f: Callable[[float, float], float],
        y_0: float,
        t_0: float,
        t_f: float,
        dt: float,
    ):
        """
        Constructs all the necessary attributes for the ODE solver object.

        Parameters
        ----------
            f : Callable[[float, float], float]
                a function representing the ODE
            y_0 : float
                initial condition
            t_0 : float
                start time
            t_f : float
                stop time
            dt : float
                step size
        """

        if t_f <= t_0:
            raise ValueError("Stop time must be greater than start time.")

        self.f = f
        self.y_0 = y_0
        self.t_0 = t_0
        self.t_f = t_f
        self.dt = dt


class Euler(ODESolver):
    """
    A derived class used to represent an Euler solver for ODEs.
    Inherits from ODESolver.

    ...

    Methods
    -------
    solve():
        Solves the ODE using the Euler method and returns the numerical solution as a function of time.
    """

    def solve(self) -> np.ndarray:
        """
        Solves the ODE using the Euler method and returns the numerical solution as a function of time.

        Returns
        ----------
        np.ndarray
            a numpy array containing the numerical solution of the ODE
        """

        n_steps = int((self.t_f - self.t_0) / self.dt)
        t = np.linspace(self.t_0, self.t_f, n_steps)
        y = np.zeros(n_steps)
        y[0] = self.y_0

        for i in range(n_steps - 1):
            y[i + 1] = y[i] + self.dt * self.f(y[i], t[i])

        return y
