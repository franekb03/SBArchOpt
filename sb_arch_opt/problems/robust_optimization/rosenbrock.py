"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Stochastic version of the Rosenbrock problem: a non-hierarchical robust optimization test problem.
"""
import numpy as np
from typing import *
from scipy.stats import norm
from pymoo.core.variable import Real
from sb_arch_opt.robust import RobustArchOptProblem, StochasticReduction
from sb_arch_opt.uncertainty import UncertainParameter

__all__ = ['StochasticRosenbrock']


class StochasticRosenbrock(RobustArchOptProblem):
    """
    The Rosenbrock function where the location of the valley is uncertain:

        f(x, u) = sum_i [ 100*(x_{i+1} - x_i^2)^2 + (u_i - x_i)^2 ]

    with one uncertain parameter u_i ~ N(mean, std) per term. All parameters are always active, so this problem
    exercises the uncertainty machinery without hierarchy; see `HierarchicalRobustProblem` for the hierarchical case.

    With the default MEAN reduction the expected objective is minimized; since E[(u-x)^2] = (E[u]-x)^2 + Var[u], the
    optimum stays at x_i = mean and the optimal value is the sum of the parameter variances rather than 0.
    """

    def __init__(self, n_var=2, mean=1., std=.05, n_mc=100, reduction=StochasticReduction.MEAN, k=2.):
        if n_var < 2:
            raise ValueError('Need at least 2 design variables')
        self.mean = mean
        self.std = std
        self._n_param = n_var - 1

        des_vars = [Real(bounds=(-2.048, 2.048)) for _ in range(n_var)]
        super().__init__(des_vars, n_obj=1, n_mc=n_mc, f_reduction=reduction, f_k=k)

    def _get_uncertain_parameters(self) -> List[UncertainParameter]:
        return [UncertainParameter(f'u{i}', norm(self.mean, self.std)) for i in range(self._n_param)]

    def _is_conditionally_active(self) -> List[bool]:
        return [False]*self.n_var

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        pass

    def _get_n_valid_discrete(self) -> int:
        return 1

    def _arch_evaluate_sample(self, x, is_active, u, is_param_active, f_out, g_out, h_out, *args, **kwargs):
        terms = np.zeros((x.shape[0], self._n_param))
        for i in range(self._n_param):
            terms[:, i] = 100*(x[:, i+1] - x[:, i]**2)**2 + (u[:, i] - x[:, i])**2
        f_out[:, 0] = np.sum(terms, axis=1)

    def _calc_pareto_front(self, *args, **kwargs):
        # E[(u-x)^2] is minimized at x = E[u], leaving the parameter variance
        return np.array([[self._n_param * self.std**2]])

    def _calc_pareto_set(self, *args, **kwargs):
        return np.full((1, self.n_var), self.mean)

    def might_have_hidden_constraints(self):
        return False

    def __repr__(self):
        return f'{self.__class__.__name__}(n_var={self.n_var}, std={self.std}, n_mc={self.n_mc})'
