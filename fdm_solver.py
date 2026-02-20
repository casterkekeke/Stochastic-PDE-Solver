import numpy as np
from typing import Dict, Tuple, Union
from scipy.linalg import solve_banded


class BlackScholesPdeSolver:
    """
    Unified Finite Difference suite for European Option Pricing.
    Optimized via vectorized NumPy slicing and SciPy banded solvers.
    """

    def __init__(self, spot: float, strike: float, T: float, r: float, sigma: float, M: int = 200, N: int = 1000):
        self.spot, self.strike, self.t_exp, self.r, self.sigma = spot, strike, T, r, sigma
        self.m_space, self.n_time = M, N

        # Grid parameters
        self.s_max = 5.0 * self.spot
        self.ds = self.s_max / self.m_space
        self.dt = self.t_exp / self.n_time
        self.s_grid = np.linspace(0, self.s_max, self.m_space + 1)

    def _get_initial_condition(self, option_type: str) -> np.ndarray:
        if option_type.lower() == 'call':
            return np.maximum(self.s_grid - self.strike, 0)
        return np.maximum(self.strike - self.s_grid, 0)

    def _get_spatial_operator(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Discretizes the BS operator into tridiagonal components a, b, c."""
        s_int = self.s_grid[1:-1]
        a = (self.sigma ** 2 * s_int ** 2 / (2 * self.ds ** 2) - self.r * s_int / (2 * self.ds))
        b = (-self.sigma ** 2 * s_int ** 2 / (self.ds ** 2) - self.r)
        c = (self.sigma ** 2 * s_int ** 2 / (2 * self.ds ** 2) + self.r * s_int / (2 * self.ds))
        return a, b, c

    def solve_explicit(self, option_type: str = 'call', return_grid: bool = False) \
            -> Union[Dict[str, float], Tuple[np.ndarray, Dict[str, float]]]:
        v = self._get_initial_condition(option_type)
        dt_cfl = 0.45 * (self.ds ** 2) / (self.sigma ** 2 * self.s_max ** 2)
        l_dt = min(self.dt, dt_cfl)
        l_n = int(np.ceil(self.t_exp / l_dt))
        l_dt = self.t_exp / l_n

        a_ops, b_ops, c_ops = self._get_spatial_operator()
        a, b, c = l_dt * a_ops, 1 + l_dt * b_ops, l_dt * c_ops

        v_penultimate = None
        for n in range(1, l_n + 1):
            if n == l_n: v_penultimate = v.copy()
            v_old = v.copy()
            v[1:-1] = a * v_old[:-2] + b * v_old[1:-1] + c * v_old[2:]
            self._apply_boundary(v, n, l_dt, option_type)

        return (v.copy(), self._calculate_metrics(v, v_penultimate, self.dt)) if return_grid \
            else self._calculate_metrics(v, v_penultimate, self.dt)

    def solve_implicit(self, option_type: str = 'call', return_grid: bool = False) \
            -> Union[Dict[str, float], Tuple[np.ndarray, Dict[str, float]]]:
        v = self._get_initial_condition(option_type)
        a, b, c = self._get_spatial_operator()

        ab = np.zeros((3, self.m_space - 1))
        ab[0, 1:] = -self.dt * c[:-1]
        ab[1, :] = 1 - self.dt * b
        ab[2, :-1] = -self.dt * a[1:]

        if option_type.lower() == 'call':
            # Extract the exact c_{M-1} boundary coefficient
            c_bound = -self.dt * c[-1]
            ab[1, -1] += 2 * c_bound  # Modify main diagonal
            ab[2, -2] -= c_bound      # Modify lower diagonal
        elif option_type.lower() == 'put':
            c_bound = -self.dt * c[-1]
            ab[1, -1] += 2 * c_bound
            ab[2, -2] -= c_bound

        v_penultimate = None
        for n in range(1, self.n_time + 1):
            if n == self.n_time: v_penultimate = v.copy()

            rhs = v[1:-1].copy()

            # FIX: Re-inject the non-zero Dirichlet boundary for Puts at S=0
            if option_type.lower() == 'put':
                v_lower_future = self.strike * np.exp(-self.r * (n * self.dt))
                rhs[0] += self.dt * a[0] * v_lower_future

            v[1:-1] = solve_banded((1, 1), ab, rhs)
            self._apply_boundary(v, n, self.dt, option_type)

        return (v.copy(), self._calculate_metrics(v, v_penultimate, self.dt)) if return_grid \
            else self._calculate_metrics(v, v_penultimate, self.dt)

    def solve_crank_nicolson(self, option_type: str = 'call', return_grid: bool = False) \
            -> Union[Dict[str, float], Tuple[np.ndarray, Dict[str, float]]]:
        v = self._get_initial_condition(option_type)
        a, b, c = self._get_spatial_operator()
        dt_h = 0.5 * self.dt

        ab = np.zeros((3, self.m_space - 1))
        ab[0, 1:] = -dt_h * c[:-1]
        ab[1, :] = 1 - dt_h * b
        ab[2, :-1] = -dt_h * a[1:]

        if option_type.lower() == 'call':
            # Extract the exact c_{M-1} boundary coefficient
            c_bound = -dt_h * c[-1]
            ab[1, -1] += 2 * c_bound
            ab[2, -2] -= c_bound
        elif option_type.lower() == 'put':
            c_bound = -dt_h * c[-1]
            ab[1, -1] += 2 * c_bound
            ab[2, -2] -= c_bound

        v_penultimate = None
        for n in range(1, self.n_time + 1):
            if n == self.n_time: v_penultimate = v.copy()

            rhs = (dt_h * a * v[:-2] + (1 + dt_h * b) * v[1:-1] + dt_h * c * v[2:])

            # FIX: Re-inject the non-zero Dirichlet boundary for Puts at S=0
            if option_type.lower() == 'put':
                v_lower_future = self.strike * np.exp(-self.r * (n * self.dt))
                rhs[0] += dt_h * a[0] * v_lower_future

            v[1:-1] = solve_banded((1, 1), ab, rhs)
            self._apply_boundary(v, n, self.dt, option_type)

        return (v.copy(), self._calculate_metrics(v, v_penultimate, self.dt)) if return_grid \
            else self._calculate_metrics(v, v_penultimate, self.dt)

    def _apply_boundary(self, v, n, dt, option_type):
        tau = n * dt
        if option_type.lower() == 'call':
            v[0] = 0
            # Linearity Boundary: V[M] = 2*V[M-1] - V[M-2]
            # This ensures the slope remains constant at the edge
            v[-1] = 2 * v[-2] - v[-3]
        else:
            v[0] = self.strike * np.exp(-self.r * tau)
            v[-1] = 0
            v[-1] = 2 * v[-2] - v[-3]

    def _calculate_metrics(self, v_f, v_p, dt_u) -> Dict[str, float]:
        price = np.interp(self.spot, self.s_grid, v_f)
        idx = np.searchsorted(self.s_grid, self.spot)
        idx = max(1, min(idx, self.m_space - 1))
        delta = (v_f[idx + 1] - v_f[idx - 1]) / (2 * self.ds)
        gamma = (v_f[idx + 1] - 2 * v_f[idx] + v_f[idx - 1]) / (self.ds ** 2)
        theta = -(price - np.interp(self.spot, self.s_grid, v_p)) / dt_u
        return {"price": price, "delta": delta, "gamma": gamma, "theta": theta}