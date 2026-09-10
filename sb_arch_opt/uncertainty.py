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
from typing import List, Optional, Dict

import openturns as ot
import numpy as np

__all__ = ['Scalarization', 'Mean', 'Margin', 'Quantile', 'StochasticParameter', 'StochasticParameterSpace',
           'StochasticOutput', 'StochasticResults', 'UQMethod', 'MonteCarlo', 'PolynomialChaos']



class Scalarization:
    """
    Parent class for all robust scalars etc. Margin, Mean, Quantile
    """

    def reduce(self, samples: ot.Sample) -> float:
        """Reduce an (n_samples x 1) sample of one response to a single value"""
        raise NotImplementedError


class Mean(Scalarization):
    """The expected value of the response"""

    def reduce(self, samples: ot.Sample) -> float:
        return float(samples.computeMean()[0])

class Margin(Scalarization):
    """Assumes the response is Gaussian"""

    def __init__(self, k: float = 1.645, direction: int =-1):
        self.k = k
        self.direction = direction

    def reduce(self, samples: ot.Sample) -> float:
        return float(samples.computeMean()[0] - self.k * np.sign(self.direction) * samples.computeStandardDeviation()[0])

class Quantile(Scalarization):
    """The q quantile of the response for arbitrary distribution"""

    def __init__(self, q: float = 0.95):
        if not 0. <= q <= 1.:
            raise ValueError(f'Quantile should be between 0 and 1: {q}')
        self.q = q

    def reduce(self, samples: ot.Sample) -> float:
        return float(samples.computeQuantile(self.q)[0])


class StochasticParameter:
    """A stochastic parameter used to evaluate architecture objectives or constraints."""

    def __init__(self, name, value: ot.DistributionImplementation):
        self.name = name
        self.value = value

    # def mean(self) -> float:
    #     return self.value.getMean()[0]
    #
    # def std(self) -> float:
    #     if isinstance(self.value, float):
    #         return 0.0
    #     return self.value.getStandardDeviation()[0]


class StochasticParameterSpace:
    """The joint distribution of all stochastic parameters of a problem."""

    def __init__(self, parameters: List[StochasticParameter]):
        self._sample = None
        self.parameters = parameters

    # def add_parameter(self, parameter: StochasticParameter):
    #     self.parameters.append(parameter)
    #     if not isinstance(parameter.value, float):
    #         self.stochastic_parameters.append(parameter)
    #     else:
    #         self.deterministic_parameters.append(parameter)

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    @property
    def parameter_names(self) -> List[str]:
        return [parameter.name for parameter in self.parameters]

    @property
    def joint_dist(self) -> ot.JointDistribution:
        """ Joint distribution of independent variables is chosen """
        return ot.JointDistribution([parameter.value for parameter in self.parameters],
                                    ot.IndependentCopula(self.n_parameters))


    def get_random_samples(self, n_samples: int) -> np.ndarray:
        """ Draw n samples of the stochastic parameters; returns an n x n_parameters matrix """
        result = self.joint_dist.getSample(n_samples)
        self._sample = result
        return np.array(result)

    def get_lhs_samples(self, n_samples: int) -> np.ndarray:
        """ Draw n samples of the stochastic parameters; returns an n x n_parameters matrix built with LHS"""
        lhs = ot.LHSExperiment(self.joint_dist, n_samples)
        result = lhs.generate()
        self._sample = result
        return np.array(result)

    # def include_deterministic_values(self, samples: np.ndarray) -> np.ndarray:
    #     """Add missing deterministic parameters to the sample."""
    #     extended_samples = np.zeros((samples.shape[0], self.n_parameters))
    #     n_d = 0
    #     for i, parameter in enumerate(self.parameters):
    #         if isinstance(parameter.value, float):
    #             extended_samples[:, i] = parameter.value
    #             n_d += 1
    #         else:
    #             extended_samples[:, i] = samples[:, i-n_d]
    #     return extended_samples



class StochasticOutput:
    """The sampled values of one response (objective or constraint) for one design point."""

    def __init__(self, output_samples: ot.Sample):
        self.output_samples = output_samples          # ot.Sample of shape (n, 1)

    @classmethod
    def from_results(cls, results: ot.Sample, index: int) -> 'StochasticOutput':
        """Extract one response's column from the full (n_samples, n_outputs) results of a single design point."""
        return cls(output_samples=results.getMarginal(index))

    def mean(self) -> float:
        return self.output_samples.computeMean()[0]

    def std(self) -> float:
        return self.output_samples.computeStandardDeviation()[0]

    def var(self) -> float:
        return self.output_samples.computeVariance()[0]

    def quantile(self, q: float) -> float:
        return self.output_samples.computeQuantile(q)[0]

    def prob_exceeds(self, threshold: float) -> float:
        arr = self.to_numpy()
        return float(np.mean(arr > threshold))

    def margin(self, k: float = 1.645, direction: int =-1) -> float:
        """mean + k*sigma - the Gaussian-assumption margin formulation."""
        return self.mean() - np.sign(direction) * k * self.std()

    def to_distribution(self) -> ot.Distribution:
        """Fit a continuous distribution if you need PDF/CDF rather than raw samples."""
        return ot.KernelSmoothing().build(self.output_samples)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.output_samples).flatten()

    def reduce(self, scalar: Scalarization) -> float:
        """
        Reduce the sampled values to the single value the optimizer sees, by applying the given scalar.
        """
        values = self.to_numpy()

        if values.size == 0 or not np.all(np.isfinite(values)):
            return np.nan
        samples = self.output_samples

        return scalar.reduce(samples)


class StochasticResults:
    """
    All sampled responses of a single design point.

    `method_result` optionally carries whatever the UQ method produced beyond the samples themselves; for polynomial
    chaos that is the list of `ot.FunctionalChaosResult`s, from which for example Sobol indices can be obtained.
    """

    def __init__(self, outputs: List[StochasticOutput], method_result=None):
        self.outputs = outputs
        self.method_result = method_result


class UQMethod:
    """
    Base class for an uncertainty propagation method.
    """

    def __init__(self, n_evaluations: int, seed: int = None):
        if n_evaluations is None:
            raise ValueError('n_evaluations must be specified: it is the number of expensive evaluations '
                             'per design point')
        self.n_evaluations = n_evaluations
        self.seed = seed

        self._samples: Optional[np.ndarray] = None

    def get_samples(self, param_space: StochasticParameterSpace) -> np.ndarray:
        if param_space is None:
            raise ValueError('No parameter space to sample')
        if self._samples is None:
            if self.seed is not None:
                ot.RandomGenerator.SetSeed(self.seed)
            self._samples = self._draw_samples(param_space)
        return self._samples

    def _draw_samples(self, param_space: StochasticParameterSpace) -> np.ndarray:
        """Draw the design of experiments in the parameter space; override to use a different design"""
        return param_space.get_random_samples(self.n_evaluations)

    def resample(self):
        """Draw a new design on the next evaluation"""
        self._samples = None

    def _check_results(self, results: np.ndarray):
        """The results of one design point must have one row per evaluated parameter sample"""
        if results.ndim != 2:
            raise ValueError(f'Expected a 2D (n_samples x n_outputs) results matrix, got {results.ndim}D')
        if results.shape[0] != self.n_evaluations:
            raise ValueError(f'Expected {self.n_evaluations} response rows, got {results.shape[0]}')

    def get_dictionary(self, param_space: StochasticParameterSpace, i_realization: int) -> Optional[Dict[str, float]]:
        if self._samples is None:
            return None
        samples = {}
        for j, parameter in enumerate(param_space.parameters):
            samples[parameter.name] = self._samples[i_realization, j]
        return samples

    def process_results(self, results: np.ndarray, **kwargs) -> StochasticResults:
        """
        Turn the responses of ONE design point (an n_samples x (n_obj+n_ieq_constr+n_eq_constr) matrix) into the
        stochastic result object that contains list of stochastic outputs.
        """
        raise NotImplementedError


class MonteCarlo(UQMethod):
    """
    Monte Carlo uncertainty propagation
    """

    def __init__(self, n_evaluations: int = 100, seed: int = 42):
        super().__init__(n_evaluations, seed)

    def _draw_samples(self, param_space: StochasticParameterSpace) -> np.ndarray:
        # Draw samples with LHS method
        samples = param_space.get_lhs_samples(self.n_evaluations)
        return samples

    def process_results(self, results: np.ndarray, **kwargs) -> StochasticResults:
        results = np.asarray(results, dtype=float)
        self._check_results(results)

        sample = ot.Sample(results)
        outputs = [StochasticOutput.from_results(sample, i) for i in range(results.shape[1])]
        return StochasticResults(outputs)


class PolynomialChaos(UQMethod):
    """
    Polynomial chaos expansion (PCE): a surrogate of each response as a function of the uncertain parameters is
    fitted from n_evaluations evaluations of the expensive model, and the statistics are then taken from that
    surrogate.

    The expensive model is evaluated `n_evaluations` times per design point, exactly like Monte Carlo, but the
    statistics come from `n_metamodel_samples` evaluations of the cheap fitted expansion instead of from the
    expensive ones. For a response that the expansion represents well this gives far more accurate quantiles and
    tail statistics for the same number of expensive evaluations.

    The fitted `ot.FunctionalChaosResult` of each response is kept on the `StochasticResult` (`method_result`), so
    Sobol sensitivity indices are available for free through `ot.FunctionalChaosSobolIndices`.
    """

    def __init__(self, n_evaluations: int, seed: int = None, degree: int = 3, n_metamodel_samples: int = 10000):
        self.degree = degree
        self.n_metamodel_samples = n_metamodel_samples
        self._metamodel_input: Optional[ot.Sample] = None
        super().__init__(n_evaluations, seed)

    def n_terms(self, param_space: StochasticParameterSpace) -> int:
        """Number of terms in the expansion, i.e. the minimum number of samples needed to fit it"""
        enumerate_function = ot.LinearEnumerateFunction(param_space.n_parameters)
        return int(enumerate_function.getStrataCumulatedCardinal(self.degree))

    def _validate(self, param_space: StochasticParameterSpace):
        # Needs the parameter space, since the number of terms depends on the number of parameters
        n_terms = self.n_terms(param_space)
        if self.n_evaluations < n_terms:
            raise ValueError(f'A degree-{self.degree} expansion in {param_space.n_parameters} '
                             f'parameters has {n_terms} terms, so it needs at least that many samples to fit: '
                             f'n_evaluations = {self.n_evaluations}')

    def _draw_samples(self, param_space: StochasticParameterSpace) -> np.ndarray:
        self._validate(param_space)
        # Draw samples with LHS method
        samples = param_space.get_lhs_samples(self.n_evaluations)
        return samples

    def _get_metamodel_input(self, param_space: StochasticParameterSpace) -> ot.Sample:
        if self._metamodel_input is None:
            self._metamodel_input = param_space.joint_dist.getSample(self.n_metamodel_samples)
        return self._metamodel_input

    def _build_algorithm(self, param_space: StochasticParameterSpace, input_sample: ot.Sample, output_sample: ot.Sample) -> ot.FunctionalChaosAlgorithm:
        distribution = param_space.joint_dist
        dimension = distribution.getDimension()

        polynomials = [ot.StandardDistributionPolynomialFactory(distribution.getMarginal(i))
                       for i in range(dimension)]
        enumerate_function = ot.LinearEnumerateFunction(dimension)
        basis = ot.OrthogonalProductPolynomialFactory(polynomials, enumerate_function)

        adaptive_strategy = ot.FixedStrategy(basis, self.n_terms(param_space))
        projection_strategy = ot.LeastSquaresStrategy()
        return ot.FunctionalChaosAlgorithm(input_sample, output_sample, distribution,
                                           adaptive_strategy, projection_strategy)

    def process_results(self, results: np.ndarray, **kwargs) -> StochasticResults:
        results = np.asarray(results, dtype=float)
        self._check_results(results)

        param_space = kwargs['param_space']

        input_sample = ot.Sample(self.get_samples(param_space))
        metamodel_input = self._get_metamodel_input(param_space)

        outputs, chaos_results = [], []
        for i_out in range(results.shape[1]):
            values = results[:, i_out]

            # A response with failed evaluations cannot be fitted: pass the raw samples through instead, so that
            # the problem's nan_policy decides what happens to this design point
            if not np.all(np.isfinite(values)):
                samples = ot.Sample(values.reshape((-1, 1)))
                chaos_results.append(None)

            else:
                algorithm = self._build_algorithm(param_space, input_sample, ot.Sample(values.reshape((-1, 1))))
                algorithm.run()
                chaos_result = algorithm.getResult()
                chaos_results.append(chaos_result)
                samples = chaos_result.getMetaModel()(metamodel_input)

            outputs.append(StochasticOutput(samples))
        return StochasticResults(outputs, method_result=chaos_results)