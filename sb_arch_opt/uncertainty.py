import enum

import chaospy as cp
import numpy as np

class UncertainParameter:
    def __init__(self, name, nominal=None, distribution: cp.Distribution=None):
        if distribution is None and nominal is None:
            raise ValueError("Either distribution or nominal must be specified")
        self.name = name
        self.nominal = nominal
        self.distribution = distribution

    @property
    def is_uncertain(self):
        return self.distribution is not None

    def sample(self, n:int) -> np.ndarray:
        if not self.is_uncertain:
            if self.nominal is None:
                raise ValueError(f'Cannot sample parameter {self.name!r}: no distribution or nominal value set')
            return np.full(n, float(self.nominal))
        samples = np.asarray(self.distribution.sample(n))
        return samples.reshape((n,))


class StochasticMetricType(enum.Flag):
    NONE = 0
    MEAN = enum.auto()
    MARGIN = enum.auto()
    QUANTILE = enum.auto()