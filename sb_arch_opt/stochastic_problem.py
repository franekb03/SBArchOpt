"""
MIT License

Copyright: (c) 2024, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
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
    """

    def __init__(self, des_vars: Union[List[Variable], ArchDesignSpace],
                 uq_method: UQMethod, n_obj=1, n_ieq_constr=0, n_eq_constr=0,
                 obj_measure: List[RobustMeasure] = None,
                 ieq_constr_measure: List[RobustMeasure] = None,
                 eq_constr_measure: List[RobustMeasure] = None,
                 nan_policy: str = 'propagate', **kwargs):

        self.obj_measure = self.check_measures(obj_measure, n_obj, 'obj_measure')
        self.ieq_constr_measure = self.check_measures(ieq_constr_measure, n_ieq_constr, 'ieq_constr_measure')
        self.eq_constr_measure = self.check_measures(eq_constr_measure, n_eq_constr, 'eq_constr_measure')
        self.measures = self.obj_measure + self.ieq_constr_measure + self.eq_constr_measure

        self.nan_policy = nan_policy

        if uq_method is None:
            raise ValueError('A UQ method must be specified, e.g. uq_method=MonteCarlo(n_evaluations=100, seed=42)')
        if not isinstance(uq_method, UQMethod):
            raise ValueError(f'uq_method should be a UQMethod instance, got: {uq_method!r}')

        if uq_method.param_space is None or uq_method.param_space.n_parameters == 0:
            raise ValueError('uq_method must contain valid parameter space')

        self.uq_method = uq_method
        # List for storing stochastic results object for each design point
        self.stochastic_results: List[StochasticResults] = []

        super().__init__(des_vars, n_obj=n_obj, n_ieq_constr=n_ieq_constr, n_eq_constr=n_eq_constr, **kwargs)

    @property
    def param_space(self):
        return self.uq_method.param_space

    @staticmethod
    def check_measures(measures: Optional[List[RobustMeasure]], n: int, name: str) -> List[RobustMeasure]:
        """Default unspecified responses to the expected value, and check the count"""
        if measures is None:
            # Mean is the default robust measure type
            return [Mean() for _ in range(n)]

        if len(measures) != n:
            raise ValueError(f'{name} should have {n} entries: {len(measures)}')
        for measure in measures:
            if not isinstance(measure, RobustMeasure):
                raise ValueError(f'{name} should contain RobustMeasure instances, got: {measure!r}')
        return list(measures)

    def _evaluate(self, x, out, *args, **kwargs):
        # The pymoo outputs are processed by the parent method
        super()._evaluate(x, out, *args, **kwargs)

        # Add stochastic result to pymoo out dictionary.
        if len(self.stochastic_results) == len(out['X']):
            out['stochastic'] = list(self.stochastic_results)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        """
        Evaluate architecture for the provided design vectors and samples.
        Implement _arch_evaluate_sample to evaluate architecture for single samples realizations.
        """

        if not self.design_space.is_explicit():
            self._correct_x_impute(x, is_active_out)

        samples = self.uq_method.get_samples()
        n_x, n_s = x.shape[0], samples.shape[0]

        f_s = np.zeros((n_x, n_s, self.n_obj))*np.nan
        g_s = np.zeros((n_x, n_s, self.n_ieq_constr))*np.nan
        h_s = np.zeros((n_x, n_s, self.n_eq_constr))*np.nan

        # Evaluate all design vectors for each realization of the uncertain parameters
        for i in range(n_s):
            self._arch_evaluate_sample(
                x, is_active_out, f_s[:, i, :], g_s[:, i, :], h_s[:, i, :], samples[i, :],*args, **kwargs)

        # Evaluate the stochastic result for all the evaluated design vectors and samples
        nan_policy = self.nan_policy
        self.stochastic_results = []
        for x_i in range(n_x):
            results = self.uq_method.process_results(
                np.concatenate([f_s[x_i], g_s[x_i], h_s[x_i]], axis=1))
            self.stochastic_results.append(results)

            # Reduce the sampled responses of each design point to the values the optimizer sees
            n_f, n_g = self.n_obj, self.n_ieq_constr
            for f_i, output in enumerate(results.outputs[:n_f]):
                f_out[x_i, f_i] = output.reduce(self.obj_measure[f_i], nan_policy=nan_policy)
            for g_i, output in enumerate(results.outputs[n_f:n_f+n_g]):
                g_out[x_i, g_i] = output.reduce(self.ieq_constr_measure[g_i], nan_policy=nan_policy)
            for h_i, output in enumerate(results.outputs[n_f+n_g:]):
                h_out[x_i, h_i] = output.reduce(self.eq_constr_measure[h_i], nan_policy=nan_policy)

    def _arch_evaluate_sample(self, x: np.ndarray, is_active: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                              h_out: np.ndarray, parameters: np.ndarray, *args, **kwargs):
        """
        Implement evaluation and write results in the provided output matrices:
        - x (design vectors): discrete variables have integer values, imputed design vectors can be output here (except
                              if using an explicit design space definition)
        - is_active (activeness): vector specifying for each design variable whether it was active or not
        - f (objectives): written as a minimization
        - g (inequality constraints): written as "<= 0"
        - h (equality constraints): written as "= 0"
        - sample: realization of single sample
        """

        raise NotImplementedError