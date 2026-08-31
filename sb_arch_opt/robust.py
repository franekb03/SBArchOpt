"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from typing import *
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.uncertainty import StochasticReduction, reduce_samples

__all__ = ['RobustArchOptProblem', 'StochasticReduction']


class RobustArchOptProblem(ArchOptProblemBase):
    """
    Base class for an architecture optimization problem under uncertainty: the objectives and constraints are
    evaluated for a number of samples of the uncertain parameters, and the resulting statistics are reduced to the
    single value per objective/constraint that the optimizer sees.

    Implement `_arch_evaluate_sample` instead of `_arch_evaluate`: it is called once per uncertainty sample with one
    realization of the uncertain parameters.

    The reduction is controlled per objective and per constraint:
    - `StochasticReduction.MEAN`: the expected value
    - `StochasticReduction.MARGIN`: `mean + k*std`; since SBArchOpt objectives are always minimized and constraints
      are always satisfied when <= 0, this penalizes both a bad mean and a large spread
    - `StochasticReduction.QUANTILE`: the k'th quantile of the samples

    The underlying sample statistics stay available after evaluation: the sample mean and standard deviation are
    written to `out['F_mean']`, `out['F_std']` (and `G_mean`, `G_std`), so they are carried along by pymoo into the
    population.
    """

    def __init__(self, *args, n_mc=100, f_reduction: Union[StochasticReduction, List[StochasticReduction]] = None,
                 g_reduction: Union[StochasticReduction, List[StochasticReduction]] = None,
                 f_k: Union[float, List[float]] = 2., g_k: Union[float, List[float]] = 2.,
                 common_random_numbers=True, **kwargs):
        self.n_mc = n_mc
        self.common_random_numbers = common_random_numbers
        self._random_state = None
        super().__init__(*args, **kwargs)

        self.f_reduction = self._expand(f_reduction, self.n_obj, StochasticReduction.MEAN, 'f_reduction')
        self.g_reduction = self._expand(g_reduction, self.n_ieq_constr, StochasticReduction.MEAN, 'g_reduction')
        self.f_k = self._expand(f_k, self.n_obj, 2., 'f_k')
        self.g_k = self._expand(g_k, self.n_ieq_constr, 2., 'g_k')

        # Latest sample statistics, also written to the evaluation output
        self.f_mean = self.f_std = self.g_mean = self.g_std = None

    @staticmethod
    def _expand(value, n, default, name):
        if value is None:
            value = default
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != n:
                raise ValueError(f'{name} should have {n} entries: {len(value)}')
            return list(value)
        return [value]*n

    def set_random_state(self, random_state):
        """
        Fix the random state used for sampling the uncertain parameters. Together with common random numbers (the
        default), this makes the reduced objectives and constraints a deterministic function of x, which is what
        surrogate-based algorithms need: with freshly drawn samples per evaluation, the surrogate fits Monte Carlo
        noise instead of the underlying robust objective.
        """
        self._random_state = random_state

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):

        # Correction only runs before _arch_evaluate for explicit design spaces (see ArchOptProblemBase._evaluate),
        # so for an implicit design space is_active_out is still all-True here and has to be filled first: parameter
        # activeness is derived from the corrected design vectors and their activeness.
        if not self.design_space.is_explicit():
            self._correct_x_impute(x, is_active_out)

        is_param_active = self.get_param_is_active(x, is_active_out)
        u = self.sample_parameters(x, is_active_out, self.n_mc, random_state=self._random_state,
                                   common_random_numbers=self.common_random_numbers)

        n_x, n_mc = x.shape[0], self.n_mc
        f_samples = np.zeros((n_x, n_mc, self.n_obj))*np.nan
        g_samples = np.zeros((n_x, n_mc, self.n_ieq_constr))*np.nan
        h_samples = np.zeros((n_x, n_mc, self.n_eq_constr))*np.nan

        for i_sample in range(n_mc):
            self._arch_evaluate_sample(
                x, is_active_out, u[:, i_sample, :], is_param_active,
                f_samples[:, i_sample, :], g_samples[:, i_sample, :], h_samples[:, i_sample, :], *args, **kwargs)

        # Reduce the sampled values to what the optimizer sees
        for i_f in range(self.n_obj):
            f_out[:, i_f] = reduce_samples(
                f_samples[:, :, [i_f]], self.f_reduction[i_f], self.f_k[i_f])[:, 0]
        for i_g in range(self.n_ieq_constr):
            g_out[:, i_g] = reduce_samples(
                g_samples[:, :, [i_g]], self.g_reduction[i_g], self.g_k[i_g])[:, 0]
        if self.n_eq_constr > 0:
            h_out[:, :] = np.mean(h_samples, axis=1)

        # Keep the statistics the reduced values were derived from
        self.f_mean, self.f_std = np.mean(f_samples, axis=1), np.std(f_samples, axis=1)
        self.g_mean, self.g_std = np.mean(g_samples, axis=1), np.std(g_samples, axis=1)

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)

        # Provide the sample statistics alongside the reduced values; pymoo carries extra output keys into the
        # population, so mean and std remain available after the optimization
        out['F_mean'], out['F_std'] = self.f_mean, self.f_std
        if self.n_ieq_constr > 0:
            out['G_mean'], out['G_std'] = self.g_mean, self.g_std

    def _print_extra_stats(self):
        rates = self.design_space.get_param_activeness_rates()
        if rates is not None:
            import pandas as pd
            with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                                   'display.expand_frame_repr', False):
                print(rates)

    """##############################
    ### IMPLEMENT FUNCTIONS BELOW ###
    ##############################"""

    def _arch_evaluate_sample(self, x: np.ndarray, is_active: np.ndarray, u: np.ndarray,
                              is_param_active: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                              h_out: np.ndarray, *args, **kwargs):
        """
        Evaluate all design vectors for one realization of the uncertain parameters. Write results in the provided
        output matrices, exactly as for `_arch_evaluate`:
        - x (n x nx): corrected design vectors
        - is_active (n x nx): activeness of the design variables
        - u (n x n_param): one realization of the uncertain parameters; inactive parameters hold their nominal value
        - is_param_active (n x n_param): activeness of the uncertain parameters
        - f_out (n x n_obj), g_out (n x n_ieq_constr), h_out (n x n_eq_constr): one sample of the responses

        Note that x and is_active are the same for every sample: only u varies.
        """
        raise NotImplementedError
