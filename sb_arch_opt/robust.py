"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Stochastic (robust) architecture optimization problems: the responses are evaluated for a number of samples of the
uncertain parameters, and the resulting statistics are reduced to the values the optimizer sees.
"""
from typing import Union, List, Optional, Tuple

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

    Reduction is controlled per objective and per constraint through `StochasticMetricType`:
    - MEAN: the expected value
    - MARGIN: `mean + k*std`; since SBArchOpt objectives are always minimized and inequality constraints are
      satisfied when <= 0, this penalizes both a bad mean and a large spread for objectives and constraints alike
    - QUANTILE: the q'th quantile of the samples

    The statistics behind the reduced values stay available: a `StochasticResult` per design point is provided in
    the evaluation output under `out['stochastic']`, carrying the `StochasticOutput`s (and therefore mean, std,
    quantiles, and the fitted output distribution) of every response, plus whatever the UQ method produced beyond
    the samples (for polynomial chaos, the fitted `ot.FunctionalChaosResult` of each response).

    The uncertainty propagation method is selected with `uq_method_type` and configured through `n`, `seed` and
    `uq_method_kwargs`; register additional `UQMethod` subclasses in `UQ_METHODS` to add one.
    """

    #: Available uncertainty propagation methods, by type
    UQ_METHODS = {
        UQMethodType.MONTE_CARLO: MonteCarlo,
        UQMethodType.PCE: PolynomialChaos,
    }

    def __init__(self, des_vars: Union[List[Variable], ArchDesignSpace], param_space: StochasticParameterSpace,
                 uq_method_type: UQMethodType, n_obj=1, n_ieq_constr=0, n_eq_constr=0,
                 obj_type: List[Tuple[StochasticOutputType, Optional[float]]] = None, ieq_constr_type: List[Tuple[StochasticOutputType, Optional[float]]] = None,
                 eq_constr_type: List[Tuple[StochasticOutputType, Optional[float]]] = None,
                 n: int = 100, seed: int = None,
                 nan_policy: str = 'propagate', uq_method_kwargs: dict = None, **kwargs):

        if param_space is None or param_space.n_parameters == 0:
            raise ValueError('Define stochastic parameter space for the robust problem.')
        self.param_space = param_space

        if n_obj != 0 and obj_type is not None:
            if len(obj_type) != n_obj:
                raise ValueError(f'obj_type should have n_obj = {n_obj} entries: {len(obj_type)}')
        else:
            obj_type = [(StochasticOutputType.MEAN, None)]*n_obj

        if n_ieq_constr != 0 and ieq_constr_type is not None:
            if len(ieq_constr_type) != n_ieq_constr:
                raise ValueError(f'constr_type should have n_ieq_constr = {n_ieq_constr} entries: {len(ieq_constr_type)}')
        else:
            ieq_constr_type = [(StochasticOutputType.MEAN, None)]*n_ieq_constr

        if n_eq_constr != 0 and eq_constr_type is not None:
            if len(eq_constr_type) != n_eq_constr:
                raise ValueError(f'eq_constr_type should have n_eq_constr = {n_eq_constr} entries: {len(eq_constr_type)}')
        else:
            eq_constr_type = [(StochasticOutputType.MEAN, None)]*n_eq_constr

        self.obj_type = obj_type
        self.ieq_constr_type = ieq_constr_type
        self.eq_constr_type = eq_constr_type
        self.stochastic_output_type = obj_type + ieq_constr_type + eq_constr_type

        self.nan_policy = nan_policy

        if uq_method_type is None or uq_method_type == UQMethodType.NONE:
            raise ValueError('UQ method type must be specified')
        if n is None:
            raise ValueError('n must be specified: it is the number of expensive evaluations per design point')

        uq_method_class = self.UQ_METHODS.get(uq_method_type)
        if uq_method_class is None:
            raise ValueError(f'Unknown UQ method type: {uq_method_type}')

        # Method-specific settings, e.g. degree and n_metamodel_samples for polynomial chaos
        types = [t for t, _ in self.stochastic_output_type]
        self.uq_method = uq_method_class(param_space, types, n_obj, n_ieq_constr,
                                         n_eq_constr, n, seed, **(uq_method_kwargs or {}))

        # Latest per-design-point statistics, also provided in the evaluation output
        self.stochastic_results: List[StochasticResult] = []

        super().__init__(des_vars, n_obj=n_obj, n_ieq_constr=n_ieq_constr, n_eq_constr=n_eq_constr, **kwargs)

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

            for f_i, output in enumerate(result.f):
                _, param = self.obj_type[f_i]
                f_out[x_i, f_i] = output.reduce(param=param, nan_policy=nan_policy)
            for g_i, output in enumerate(result.g):
                _, param = self.ieq_constr_type[g_i]
                g_out[x_i, g_i] = output.reduce(param=param, nan_policy=nan_policy)
            for h_i, output in enumerate(result.h):
                _, param = self.ieq_constr_type[h_i]
                h_out[x_i, h_i] = output.reduce(param=param, nan_policy=nan_policy)

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