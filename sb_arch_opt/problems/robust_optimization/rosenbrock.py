import doctest

import numpy as np
from sb_arch_opt.problems.continuous import Rosenbrock
from pymoo.core.variable import Real
from sb_arch_opt.algo.pymoo_interface import plot
from pymoo.optimize import minimize
from sb_arch_opt.algo.arch_sbo.api import get_arch_sbo_gp
from sb_arch_opt.problems.problems_base import *
from pymoo.core.problem import Problem
import pymoo.gradient.toolbox as anp
from scipy.stats import multivariate_normal

__all__ = ['StochasticRosenbrock']

class StochasticRosenbrockProblem(Problem):
    def __init__(self, n_var=2, n_param=2, mean=1, std=0.05, seed=None):
        super().__init__(
            n_var=n_var, n_obj=1, n_ieq_constr=0, xl=-2.048, xu=2.048, vtype=float
        )
        assert n_param == n_var
        self.mean = mean
        self.std = std
        self.seed = seed

    def _evaluate(self, x, out, *args, **kwargs):
        terms = []
        param_realization = self.sample_param()
        for i in range(x.shape[1] - 1):
            val = 100 * (x[:, i + 1] - x[:, i] ** 2) ** 2 + (param_realization[i] - x[:, i]) ** 2
            terms.append(val)
        out["F"] = anp.sum(anp.column_stack(terms), axis=1)

    def sample_param(self):
        return multivariate_normal.rvs(self.mean, self.std, random_state=self.seed)

    def _calc_pareto_front(self):
        return 0.0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 1.0)

class StochasticRosenbrock(NoHierarchyWrappedProblem):
    def __init__(self):
        super().__init__(StochasticRosenbrockProblem(n_var=2))


if __name__ == "__main__":

    problem = StochasticRosenbrock()
    n_init = problem.n_var*10
    sbo = get_arch_sbo_gp(problem, n_parallel=4, init_size=n_init)
    n_infill = 100
    result_sbo = minimize(problem, sbo, termination=('n_eval', n_init + n_infill), seed=42, save_history=True)

    print(f"Iteration number: {len(result_sbo.history)}, minimum objective value: {result_sbo.opt.get("F")}")