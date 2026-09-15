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
                 param_space: StochasticParameterSpace,
                 uq_method: UQMethod, n_obj=1, n_ieq_constr=0, n_eq_constr=0,
                 obj_scalar: List[Scalarization] = None,
                 ieq_constr_scalar: List[Scalarization] = None,
                 eq_constr_scalar: List[Scalarization] = None,
                 **kwargs):

        self.obj_scalar = self.check_scalars(obj_scalar, n_obj)
        self.ieq_constr_scalar = self.check_scalars(ieq_constr_scalar, n_ieq_constr)
        self.eq_constr_scalar = self.check_scalars(eq_constr_scalar, n_eq_constr)
        self.scalars = self.obj_scalar + self.ieq_constr_scalar + self.eq_constr_scalar

        if uq_method is None:
            raise ValueError('A UQ method must be specified, e.g. uq_method=MonteCarlo(n_evaluations=100, seed=42)')
        if not isinstance(uq_method, UQMethod):
            raise ValueError(f'uq_method should be a UQMethod instance, got: {uq_method!r}')

        if param_space is None or param_space.n_parameters == 0:
            raise ValueError('param_space must contain valid parameter space')

        self.param_space = param_space
        self.uq_method = uq_method
        # List for storing stochastic results object for each design point
        self.stochastic_results: List[StochasticResults] = []

        super().__init__(des_vars, n_obj=n_obj, n_ieq_constr=n_ieq_constr, n_eq_constr=n_eq_constr, **kwargs)

    @staticmethod
    def check_scalars(scalars: Optional[List[Scalarization]], n: int) -> List[Scalarization]:
        """Default unspecified responses to the expected value, and check the count"""
        if scalars is None:
            # Mean is the default robust scalar type
            return [Mean() for _ in range(n)]

        if len(scalars) != n:
            raise ValueError(f'scalars should have {n} Scalarization entries, got {len(scalars)}')
        for scalar in scalars:
            if not isinstance(scalar, Scalarization):
                raise ValueError(f'scalars should contain Scalarization instances, got: {scalar!r}')
        return list(scalars)

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

        # Get samples and include deterministic parameter values for evaluation
        samples = self.uq_method.get_samples(self.param_space)
        # parameter_values = self.uq_method.param_space.include_deterministic_values(samples)

        n_x, n_s = x.shape[0], samples.shape[0]

        f_s = np.zeros((n_x, n_s, self.n_obj))*np.nan
        g_s = np.zeros((n_x, n_s, self.n_ieq_constr))*np.nan
        h_s = np.zeros((n_x, n_s, self.n_eq_constr))*np.nan

        # Evaluate all design vectors for each realization of the uncertain parameters
        for i in range(n_s):
            self._arch_evaluate_sample(
                x, is_active_out, f_s[:, i, :], g_s[:, i, :], h_s[:, i, :], samples[i, :],*args, **kwargs)

        # Evaluate the stochastic result for all the evaluated design vectors and samples
        self.stochastic_results = []
        for x_i in range(n_x):
            results = self.uq_method.process_results(np.concatenate([f_s[x_i], g_s[x_i], h_s[x_i]], axis=1), self.param_space)
            self.stochastic_results.append(results)

            # Reduce the sampled responses of each design point to the values the optimizer sees
            n_f, n_g = self.n_obj, self.n_ieq_constr
            for f_i, output in enumerate(results.outputs[:n_f]):
                f_out[x_i, f_i] = output.reduce(self.obj_scalar[f_i])
            for g_i, output in enumerate(results.outputs[n_f:n_f+n_g]):
                g_out[x_i, g_i] = output.reduce(self.ieq_constr_scalar[g_i])
            for h_i, output in enumerate(results.outputs[n_f+n_g:]):
                h_out[x_i, h_i] = output.reduce(self.eq_constr_scalar[h_i])

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