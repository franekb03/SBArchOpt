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

from sb_arch_opt.robust import StochasticArchOptProblem
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
    """

    def __init__(self, n_var=2, mean=1., std=.05, n=100, seed=42,
                 uq_method: UQMethod = None, obj_measure: List[RobustMeasure] = None):
        if n_var < 2:
            raise ValueError('Need at least 2 design variables')
        self.mean = mean
        self.std = std
        self._n_param = n_var - 1

        param_space = StochasticParameterSpace()
        for i in range(self._n_param):
            param_space.add_parameter(StochasticParameter(f'u{i}', ot.Normal(mean, std)))

        super().__init__(
            [Real(bounds=(-2.048, 2.048)) for _ in range(n_var)],
            param_space=param_space,
            uq_method=uq_method if uq_method is not None else MonteCarlo(n_evaluations=n, seed=seed),
            n_obj=1, obj_measure=obj_measure,
        )

    def _is_conditionally_active(self) -> List[bool]:
        return [False]*self.n_var

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        pass

    def _get_n_valid_discrete(self) -> int:
        return 1

    def might_have_hidden_constraints(self):
        return False

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, *args, sample, **kwargs):
        f = np.zeros((x.shape[0],))
        for i in range(self._n_param):
            f += 100*(x[:, i+1] - x[:, i]**2)**2 + (sample[i] - x[:, i])**2
        f_out[:, 0] = f

    def _calc_pareto_front(self, *args, **kwargs):
        # E[(u-x)^2] is minimized at x = E[u], leaving the parameter variance
        return np.array([[self._n_param * self.std**2]])

    def _calc_pareto_set(self, *args, **kwargs):
        return np.full((1, self.n_var), self.mean)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_var={self.n_var}, std={self.std}, n={self.uq_method.n_evaluations})'