import enum
from typing import List

import openturns as ot
import numpy as np

class StochasticMetricType(enum.Flag):
    NONE = 0
    MEAN = enum.auto()
    MARGIN = enum.auto()
    QUANTILE = enum.auto()

class UQMethodType(enum.Flag):
    NONE = 0
    MONTE_CARLO = enum.auto()


class StochasticParameter:
    def __init__(self, name, distribution: ot.Distribution):
        self.name = name
        self.distribution = distribution

    def mean(self) -> float:
        return self.distribution.getMean()[0]

    def std(self) -> float:
        return self.distribution.getParameter()[0]

class StochasticOutput:
    def __init__(self, samples: ot.Sample, output_type: StochasticMetricType):
        self.samples = samples          # ot.Sample of shape (n, 1)
        self.output_type = output_type

    @classmethod
    def from_results(cls, results: ot.Sample, index: int, output_type: StochasticMetricType) -> "StochasticOutput":
        """Extract one objective's column from the full (n, n_outputs) results."""
        return cls(samples=results.getMarginal(index), output_type=output_type)

    def mean(self) -> float:
        return self.samples.computeMean()[0]

    def std(self) -> float:
        return self.samples.computeStandardDeviation()[0]

    def var(self) -> float:
        return self.samples.computeVariance()[0]

    def quantile(self, q: float) -> float:
        return self.samples.computeQuantile(q)[0]

    def prob_exceeds(self, threshold: float) -> float:
        arr = np.array(self.samples).flatten()
        return float(np.mean(arr > threshold))

    def margin(self, k: float = 1.645) -> float:
        """mean + k*sigma — the Gaussian-assumption margin formulation."""
        return self.mean() + k * self.std()

    def to_distribution(self) -> ot.Distribution:
        """Fit a continuous distribution if you need PDF/CDF rather than raw samples."""
        return ot.KernelSmoothing().build(self.samples)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.samples).flatten()

class StochasticParameterSpace:

    def __init__(self):
        self.sample = None
        self.parameters = []

    def add_parameter(self, parameter: StochasticParameter):
        self.parameters.append(parameter)

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    @property
    def joint_dist(self) -> ot.JointDistribution:
        return ot.JointDistribution(self.parameters, ot.IndependentCopula(self.n_parameters))

    def get_samples(self, n: int) -> np.ndarray:
        # Return array of shape (n, n_parameters)
        result = self.joint_dist.getSample(n)
        self.sample = result
        return np.array(result)



class UQMethod:
    def __init__(self, parameters_space: StochasticParameterSpace, output_type: List[StochasticMetricType], n_obj, n_ieq_constr, n_eq_constr):
        self.parameters_space = parameters_space
        self.output_type = output_type
        self.n_obj = n_obj
        self.n_ieq_constr = n_ieq_constr
        self.n_eq_constr = n_eq_constr
        self.f: List[StochasticOutput] = []
        self.g: List[StochasticOutput] = []
        self.h: List[StochasticOutput] = []

    def process_results(self, results):
        assert results != []

        self.f = [StochasticOutput.from_results(results, i, self.output_type[i]) for i in range(self.n_obj)]
        self.g = [StochasticOutput.from_results(results, i, self.output_type[i]) for i in range(self.n_obj, self.n_ieq_constr)]
        self.h = [StochasticOutput.from_results(results, i, self.output_type[i]) for i in range(self.n_ieq_constr, len(self.output_type))]

    def run(self, model):
        self._run(model)

    def _run(self, model):
        raise NotImplementedError


class MonteCarlo(UQMethod):
    def __init__(self, parameters_space: StochasticParameterSpace, output_type: List[StochasticMetricType], n_obj, n_ieq_constr, n_eq_constr, n: int, seed: int = None):
        self.n = n
        self.seed = seed
        super().__init__(parameters_space, output_type, n_obj, n_ieq_constr, n_eq_constr)

    def _run(self, model):
        """model: callable(sample: np.ndarray) -> float"""

        def wrapped(sample_point):
            f, g = model(np.array(sample_point))
            return [*f, *g]

        ot_model = ot.PythonFunction(self.parameters_space.n_parameters, len(self.output_type), wrapped)

        input_sample = self.parameters_space.joint_dist.getSample(self.n)
        output_sample = ot_model(input_sample)

        self.process_results(output_sample)

        return output_sample




# import openturns as ot
# import numpy as np
#
# # 1. Define each parameter's marginal distribution
# X1 = ot.Normal(0.85, 0.02)          # engine_efficiency
# X2 = ot.Uniform(1.2, 1.8)            # wing_load_factor
# X3 = ot.LogNormal(0.81, 0.05)        # fuel_density
# # ... add all your Z parameters here
#
# marginals = [X1, X2, X3]
#
# # 2. Combine into a joint distribution
# #    If independent, use an IndependentCopula; if correlated, swap in e.g. ot.NormalCopula(corr_matrix)
# copula = ot.IndependentCopula(len(marginals))
# joint_dist = ot.ComposedDistribution(marginals, copula)
# # (OT ≥1.20: ot.JointDistribution(marginals, copula) is the newer name for the same thing)
#
# # 3. Draw MC samples from the joint input distribution
# n = 10_000
# input_sample = joint_dist.getSample(n)
#
# # 4. Propagate through your model — wrap execute.py/mda.py as an ot.PythonFunction
# def model(X):
#     x1, x2, x3 = X
#     y = run_mda(x1, x2, x3)   # your existing OpenMDAO/mda.py call
#     return [y]
#
# ot_model = ot.PythonFunction(len(marginals), 1, model)
# output_sample = ot_model(input_sample)   # shape (n, 1)
#
# # 5. Get the output distribution and its quantile
# output_dist = ot.KernelSmoothing().build(output_sample)   # nonparametric fit
# q95 = output_dist.computeQuantile(0.95)[0]
#
# # Or skip the fit entirely and take the empirical quantile directly —
# # usually preferable for MC unless you specifically need a smooth PDF for something else:
# q95_empirical = output_sample.computeQuantile(0.95)[0]