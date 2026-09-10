"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Stochastic version of the Rosenbrock problem: a robust optimization test problem.
"""
from typing import List

import numpy as np
import openturns as ot
from pymoo.core.variable import Real

from sb_arch_opt.stochastic_problem import StochasticArchOptProblem
from sb_arch_opt.uncertainty import *

__all__ = ['StochasticRosenbrock']


class StochasticRosenbrock(StochasticArchOptProblem):
    """
    The Rosenbrock function where the location of the valley is uncertain:

        f(x, u) = sum_i [ 100*(x_{i+1} - x_i^2)^2 + (u_i - x_i)^2 ]

    with one uncertain parameter u_i ~ N(mean, std) per term.

    With the default MEAN reduction the expected objective is minimized; since E[(u-x)^2] = (E[u]-x)^2 + Var[u],
    the optimum stays at x_i = mean and the optimal value is the sum of the parameter variances rather than 0.

    Note that only `_arch_evaluate_sample` is implemented, and that it is vectorized over all design points: the
    loop over uncertain-parameter samples is owned by `StochasticArchOptProblem`.

    A UQ method is configured independently and receives the parameter space when drawing samples / processing results:

    ```python
    method = PolynomialChaos(n_evaluations=40, degree=3, seed=42)
    problem = StochasticRosenbrock(n_var=2, uq_method=method)
    ```
    """

    def __init__(self, n_var=2, mean=1., std=.05, n=100, seed=42,
                 uq_method: UQMethod = None, obj_scalar: List[Scalarization] = None):
        if n_var < 2:
            raise ValueError('Need at least 2 design variables')
        self.mean = mean
        self.std = std
        self._n_param = n_var - 1

        # The problem and its UQ method must be defined over the same parameter space; when a method is supplied
        # it already carries the space it was built over, so take that one rather than building a second
        if uq_method is None:
            uq_method = MonteCarlo(n_evaluations=n, seed=seed)

        super().__init__(
            [Real(bounds=(-2.048, 2.048)) for _ in range(n_var)],
            param_space=self.get_parameter_space(n_var, mean, std),
            uq_method=uq_method,
            n_obj=1, obj_scalar=obj_scalar
        )

    @staticmethod
    def get_parameter_space(n_var=2, mean=1., std=.05) -> StochasticParameterSpace:
        """The uncertain valley location: one parameter per Rosenbrock term. Build this first if you want to
        construct the UQ method yourself, since a method is defined over a parameter space."""
        parameters = []
        for i in range(n_var-1):
            parameters.append(StochasticParameter(f'u{i}', ot.Normal(mean, std)))
        param_space = StochasticParameterSpace(parameters)
        return param_space

    def _is_conditionally_active(self) -> List[bool]:
        return [False]*self.n_var

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        pass

    def _get_n_valid_discrete(self) -> int:
        return 1

    def might_have_hidden_constraints(self):
        return False

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs):
        f = np.zeros((x.shape[0],))
        for i in range(self._n_param):
            f += 100 * (x[:, i+1] - x[:, i]**2) ** 2 + (parameters[i] - x[:, i]) ** 2
        f_out[:, 0] = f

    def _calc_pareto_front(self, *args, **kwargs):
        # E[(u-x)^2] is minimized at x = E[u], leaving the parameter variance
        return np.array([[self._n_param * self.std**2]])

    def _calc_pareto_set(self, *args, **kwargs):
        return np.full((1, self.n_var), self.mean)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_var={self.n_var}, std={self.std}, n={self.uq_method.n_evaluations})'