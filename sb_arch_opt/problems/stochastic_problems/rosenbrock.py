"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Stochastic version of the Rosenbrock problem: a robust optimization test problem.
"""
import numpy as np
from typing import List
from pymoo.core.variable import Real

try:
    import openturns as ot
    from sb_arch_opt.stochastic_problem import StochasticArchOptProblem
    from sb_arch_opt.uncertainty import *
except ImportError:
    raise ImportError('Looks like SBArchOpt uncertainty package is not installed! Run: pip install sb-arch-opt[uncertainty]')

__all__ = ['StochasticRosenbrock']


class StochasticRosenbrock(StochasticArchOptProblem):
    """
    The Rosenbrock function where the location of the minimum is uncertain:

        f(x, u) = sum_i [ 100*(x_{i+1} - x_i^2)^2 + (u_i - x_i)^2 ]

    with one uncertain parameter u_i ~ N(mean, std) per term.

    With the default MEAN scalarization, the expected objective is minimized. Because
    E[(u-x)^2] = (E[u]-x)^2 + Var[u], the variance only adds a constant: for mean = 1,
    the optimum stays at x_i = 1 and the optimal value equals the sum of the parameter
    variances instead of 0.
    """

    def __init__(self, n_var=2, mean=1., std=.05, n=100, seed=42,
                 uq_method: UQMethod = None, obj_scalar: List[Scalarization] = None):
        if n_var < 2:
            raise ValueError('Need at least 2 design variables')
        self.mean = mean
        self.std = std
        self._n_param = n_var - 1

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