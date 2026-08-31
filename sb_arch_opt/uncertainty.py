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
import enum
import numpy as np
from typing import *
from pymoo.core.variable import Real

__all__ = ['UncertainParameter', 'DistributionInterface', 'StochasticReduction', 'ActiveIf',
           'reduce_samples']


class DistributionInterface:
    """
    Adapter around a probability distribution, exposing the two operations the uncertainty machinery needs:

    - `ppf(q)`: the inverse CDF (quantile function), used for sampling. Sampling goes through the inverse CDF
      rather than through the distribution's own random sampler so that a fixed block of quantiles can be reused
      across design points (common random numbers); see `ArchDesignSpace.sample_parameters`.
    - `mean()`: the nominal value, used to impute inactive parameters.

    Both `scipy.stats` frozen distributions (`.ppf`, `.mean`) and chaospy distributions (`.inv`, `cp.E`) are
    supported, so problems defined with either library work without conversion.
    """

    def __init__(self, dist):
        self._dist = dist

        if hasattr(dist, 'ppf'):  # scipy.stats frozen distribution
            self._ppf = dist.ppf
        elif hasattr(dist, 'inv'):  # chaospy distribution
            self._ppf = lambda q: np.asarray(dist.inv(q), dtype=float).reshape(np.shape(q))
        else:
            raise ValueError(f'Distribution should provide ppf (scipy) or inv (chaospy): {dist!r}')

    @property
    def dist(self):
        return self._dist

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Inverse CDF: map quantiles in [0, 1] to parameter values"""
        return np.asarray(self._ppf(q), dtype=float)

    def mean(self) -> float:
        dist = self._dist
        if hasattr(dist, 'mean'):  # scipy
            return float(dist.mean())

        import chaospy as cp  # chaospy
        return float(cp.E(dist))

    def bounds(self, tail=1e-4) -> Tuple[float, float]:
        """
        Bounds of the distribution's support. Unbounded distributions are truncated at the given tail probability,
        because SBArchOpt design variables (and therefore any surrogate over them) need finite bounds.
        """
        lower, upper = float(self.ppf(np.array(0.))), float(self.ppf(np.array(1.)))
        if not np.isfinite(lower):
            lower = float(self.ppf(np.array(tail)))
        if not np.isfinite(upper):
            upper = float(self.ppf(np.array(1.-tail)))
        return lower, upper

    def __repr__(self):
        return f'{self.__class__.__name__}({self._dist!r})'


# An activeness condition for an uncertain parameter, see UncertainParameter
ActiveIf = Union[
    None,                                        # always active
    int, str,                                    # active if design variable (index or name) is active
    Tuple[Union[int, str], Any],                 # idem, and the design variable takes (one of) the given value(s)
    Callable[[np.ndarray, np.ndarray], np.ndarray],  # (x, is_active) -> bool vector
]


class UncertainParameter(Real):
    """
    An uncertain parameter of an architecture: a quantity that influences the evaluation but that is *not* chosen
    by the optimizer. Uncertain parameters are sampled rather than optimized, and like design variables they can be
    hierarchical: a parameter may only exist for some architectures.

    Subclasses pymoo's `Real` so that bounds behave like those of a continuous design variable.

    Activeness is declared through `active_if`, which is resolved against the *corrected* design vectors and their
    activeness matrix (see `ArchDesignSpace.get_param_is_active`):

    - `None`: the parameter is always active
    - `i_dv` (index) or `'name'`: active whenever that design variable is active. Use this when the parameter sits
      on the same branch of the hierarchy as an existing design variable.
    - `(i_dv, value)` or `(i_dv, {v1, v2})`: active when that design variable is active *and* takes (one of) the
      given values. Use this when a branch carries a parameter but no design variable of its own; the activeness
      matrix alone cannot distinguish which option of a choice was selected.
    - a callable `(x, is_active) -> bool vector` for anything else.

    Note that the value test is always combined with the design variable's own activeness: inactive design
    variables are imputed to their lower bound, so testing the value alone would produce false positives.
    """

    def __init__(self, name: str, distribution=None, nominal: float = None, active_if: ActiveIf = None,
                 bounds: Tuple[float, float] = None):
        if distribution is None and nominal is None:
            raise ValueError(f'Provide a distribution or a nominal value for parameter {name!r}')

        self.name = name
        self.active_if = active_if
        self._nominal = nominal
        self._dist = DistributionInterface(distribution) if distribution is not None else None

        if bounds is None:
            bounds = self._dist.bounds() if self._dist is not None else (nominal, nominal)
        super().__init__(bounds=tuple(bounds))

    @property
    def distribution(self) -> Optional[DistributionInterface]:
        return self._dist

    @property
    def is_uncertain(self) -> bool:
        """Whether the parameter actually is uncertain: a parameter without distribution is a fixed value"""
        return self._dist is not None

    @property
    def nominal(self) -> float:
        """The value inactive occurrences of this parameter are imputed to"""
        if self._nominal is not None:
            return float(self._nominal)
        return self._dist.mean()

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Map quantiles in [0, 1] to parameter values; a parameter without distribution is constant"""
        if self._dist is None:
            return np.full(np.shape(q), self.nominal)
        return self._dist.ppf(q)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r}, {self._dist!r}, active_if={self.active_if!r})'


class StochasticReduction(enum.Enum):
    """
    How the sample statistics of a stochastic objective or constraint are reduced to the single value the optimizer
    sees. Mirrors adsg-core's `StochasticMetricType`.

    Note that SBArchOpt objectives are always minimized and constraints are always satisfied when <= 0, so unlike
    adsg-core no metric direction is needed: MARGIN is always `mean + k*std`, which penalizes both a bad mean and a
    large spread.
    """
    MEAN = 'mean'  # Expected value
    MARGIN = 'margin'  # mean + k*std: robust/reliability formulation
    QUANTILE = 'quantile'  # The k'th quantile of the samples


def reduce_samples(values: np.ndarray, reduction: StochasticReduction, k: float) -> np.ndarray:
    """
    Reduce sampled objective/constraint values (n_x x n_samples x n_f) to the values the optimizer sees (n_x x n_f).
    NaN samples (failed evaluations) propagate, so that hidden constraints keep working.
    """
    if reduction == StochasticReduction.MEAN:
        return np.mean(values, axis=1)

    if reduction == StochasticReduction.MARGIN:
        return np.mean(values, axis=1) + k*np.std(values, axis=1)

    if reduction == StochasticReduction.QUANTILE:
        if not 0. <= k <= 1.:
            raise ValueError(f'Quantile should be between 0 and 1: {k}')
        return np.quantile(values, k, axis=1)

    raise ValueError(f'Unknown reduction: {reduction}')
