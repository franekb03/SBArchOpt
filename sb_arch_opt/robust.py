"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Stochastic (robust) architecture optimization problems: the responses are evaluated for a number of samples of the
uncertain parameters, and the resulting statistics are reduced to the values the optimizer sees.
"""
from typing import Union, List, Optional

import numpy as np
from pymoo.core.variable import Variable

from sb_arch_opt.design_space import ArchDesignSpace
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.uncertainty import *

__all__ = ['StochasticArchOptProblem']


class StochasticArchOptProblem(ArchOptProblemBase):
    """
    Base class for a stochastic (robust) architecture optimization problem.

    Implement `_arch_evaluate_sample` and nothing else: it is called once per uncertain-parameter sample, with the
    full design vector matrix and one realization of the parameters, and fills the response matrices exactly like
    `_arch_evaluate` would. This class owns the loop over samples, the correction of design vectors, and the
    reduction of the sampled responses to the values the optimizer sees.

    The evaluation function stays vectorized over design points: it receives all n design vectors at once, so a
    cheap analytical problem can evaluate them in one numpy call, while an expensive problem is free to loop over
    (or parallelize across) the rows itself, exactly as a deterministic `_arch_evaluate` would.

    How each response is reduced is given per objective and per constraint as a `RobustMeasure`, which carries
    its own parameters: `Mean()`, `Margin(k=2.)`, `Quantile(q=.95)`. Any `RobustMeasure` subclass works, so a new
    measure needs no change here. Responses default to `Mean()`.

    The uncertainty propagation method is passed as an instance carrying its own settings, e.g.
    `MonteCarlo(n=100, seed=42)` or `PolynomialChaos(n=200, seed=42, degree=8)`.

        super().__init__(des_vars, param_space=param_space, uq_method=MonteCarlo(n=200, seed=42),
                         n_obj=1, n_ieq_constr=1,
                         obj_measure=[Margin(k=2.)], ieq_constr_measure=[Quantile(q=.95)])

    The statistics behind the reduced values stay available: a `StochasticResult` per design point is provided in
    the evaluation output under `out['stochastic']`, carrying the `StochasticOutput`s (and therefore mean, std,
    quantiles, and the fitted output distribution) of every response, plus whatever the UQ method produced beyond
    the samples (for polynomial chaos, the fitted `ot.FunctionalChaosResult` of each response).
    """

    #: Configuration arguments that were replaced by `uq_method` and the `*_measure` lists. Rejected explicitly
    #: because **kwargs is forwarded to pymoo, which silently ignores names it does not know.
    _RETIRED_KWARGS = ('obj_type', 'constr_type', 'ieq_constr_type', 'eq_constr_type',
                       'uq_method_type', 'uq_method_kwargs', 'n', 'seed')

    def __init__(self, des_vars: Union[List[Variable], ArchDesignSpace], param_space: StochasticParameterSpace,
                 uq_method: UQMethod, n_obj=1, n_ieq_constr=0, n_eq_constr=0,
                 obj_measure: List[RobustMeasure] = None,
                 ieq_constr_measure: List[RobustMeasure] = None,
                 eq_constr_measure: List[RobustMeasure] = None,
                 nan_policy: str = 'propagate', **kwargs):

        for retired in self._RETIRED_KWARGS:
            if retired in kwargs:
                raise TypeError(
                    f'{retired!r} is no longer accepted: pass a UQMethod instance as uq_method (e.g. '
                    f'MonteCarlo(n=100, seed=42)) and RobustMeasure lists as obj_measure / ieq_constr_measure / '
                    f'eq_constr_measure')

        if param_space is None or param_space.n_parameters == 0:
            raise ValueError('Define stochastic parameter space for the robust problem.')
        self.param_space = param_space

        self.obj_measure = self._check_measures(obj_measure, n_obj, 'obj_measure')
        self.ieq_constr_measure = self._check_measures(ieq_constr_measure, n_ieq_constr, 'ieq_constr_measure')
        self.eq_constr_measure = self._check_measures(eq_constr_measure, n_eq_constr, 'eq_constr_measure')
        self.measures = self.obj_measure + self.ieq_constr_measure + self.eq_constr_measure

        self.nan_policy = nan_policy

        if uq_method is None:
            raise ValueError('A UQ method must be specified, e.g. uq_method=MonteCarlo(n=100, seed=42)')
        if not isinstance(uq_method, UQMethod):
            raise ValueError(f'uq_method should be a UQMethod instance, got: {uq_method!r}')

        # Attach the problem structure the method could not know at construction time
        uq_method.bind(param_space, self.measures, n_obj, n_ieq_constr, n_eq_constr)
        self.uq_method = uq_method

        # Latest per-design-point statistics, also provided in the evaluation output
        self.stochastic_results: List[StochasticResult] = []

        super().__init__(des_vars, n_obj=n_obj, n_ieq_constr=n_ieq_constr, n_eq_constr=n_eq_constr, **kwargs)

    @staticmethod
    def _check_measures(measures: Optional[List[RobustMeasure]], n: int, name: str) -> List[RobustMeasure]:
        """Default unspecified responses to the expected value, and check the count"""
        if measures is None:
            return [Mean() for _ in range(n)]

        if len(measures) != n:
            raise ValueError(f'{name} should have {n} entries: {len(measures)}')
        for measure in measures:
            if not isinstance(measure, RobustMeasure):
                raise ValueError(f'{name} should contain RobustMeasure instances, got: {measure!r}')
        return list(measures)

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)

        # Provide the statistics the reduced values were derived from; pymoo carries extra output keys into the
        # population, so they remain available per individual after the optimization
        # Stochastic results may need larger memory space, becuase it carries all output sample values, dist etc.
        if len(self.stochastic_results) == len(out['X']):
            out['stochastic'] = list(self.stochastic_results)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        """
        Do not override this: implement `_arch_evaluate_sample` instead.
        """

        # Correction only runs before _arch_evaluate for explicit design spaces (see ArchOptProblemBase._evaluate),
        # so for an implicit design space the design vectors still need to be corrected and imputed here: otherwise
        # the evaluation function would see an all-True activeness matrix and un-imputed design vectors.
        if not self.design_space.is_explicit():
            self._correct_x_impute(x, is_active_out)

        u_samples = self.uq_method.get_samples()
        n_x, n_s = x.shape[0], u_samples.shape[0]

        f_s = np.zeros((n_x, n_s, self.n_obj))*np.nan
        g_s = np.zeros((n_x, n_s, self.n_ieq_constr))*np.nan
        h_s = np.zeros((n_x, n_s, self.n_eq_constr))*np.nan

        # Evaluate all design vectors for each realization of the uncertain parameters: the loop is over samples,
        # not over design points, so that the evaluation function stays vectorized over design points
        for sample_i in range(n_s):
            self._arch_evaluate_sample(
                x, is_active_out, f_s[:, sample_i, :], g_s[:, sample_i, :], h_s[:, sample_i, :],
                *args, sample=u_samples[sample_i, :], **kwargs)

        # Reduce the sampled responses of each design point to the values the optimizer sees
        nan_policy = self.nan_policy
        self.stochastic_results = []
        for x_i in range(n_x):
            result = self.uq_method.process_results(
                np.concatenate([f_s[x_i], g_s[x_i], h_s[x_i]], axis=1))
            self.stochastic_results.append(result)

            # Each output carries its own measure, so the same call reduces objectives, inequality constraints
            # and equality constraints alike - there is no per-kind parameter list to index into
            for f_i, output in enumerate(result.f):
                f_out[x_i, f_i] = output.reduce(nan_policy=nan_policy)
            for g_i, output in enumerate(result.g):
                g_out[x_i, g_i] = output.reduce(nan_policy=nan_policy)
            for h_i, output in enumerate(result.h):
                h_out[x_i, h_i] = output.reduce(nan_policy=nan_policy)

    def _arch_evaluate_sample(self, x: np.ndarray, is_active: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                              h_out: np.ndarray, *args, sample: np.ndarray, **kwargs):
        """
        Evaluate all design vectors for one realization of the uncertain parameters. Write results in the provided
        output matrices, exactly as for `_arch_evaluate`:
        - x (n x nx): corrected design vectors
        - is_active (n x nx): activeness of the design variables
        - sample (n_parameters): one realization of the uncertain parameters, in the order they were added to the
          StochasticParameterSpace; the same realization is used for every design vector in x
        - f_out (n x n_obj), g_out (n x n_ieq_constr), h_out (n x n_eq_constr): one sample of the responses

        Note that x and is_active are the same for every call: only sample varies.
        """
        raise NotImplementedError