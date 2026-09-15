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
from typing import List, Optional

import openturns as ot
import numpy as np

__all__ = ['Scalarization', 'Mean', 'Margin', 'Quantile', 'StochasticParameter', 'StochasticParameterSpace',
           'StochasticOutput', 'StochasticResults', 'UQMethod', 'MonteCarlo', 'PolynomialChaos']



class Scalarization:
    """
    Parent class for all types of robust optimization problem. Do not instantiate this class directly, use its subclasses:

    - Mean
    - Margin
    - Quantile
    """

    def reduce(self, samples: ot.Sample) -> float:
        """Reduce an (n_samples x 1) sample of one response to a single value that the optimizer sees based on the optimization problem type."""
        raise NotImplementedError


class Mean(Scalarization):
    """Minimize the expectation of the objective or constraint function for example min(E[F(x)])"""

    def reduce(self, samples: ot.Sample) -> float:
        return float(samples.computeMean()[0])

class Margin(Scalarization):
    """
    Gaussian output distribution expected. Minimize for the objective or constraint function for a given
    confidence interval.
    --> min(E[F(x)] + k*sigma[F(x)]); The default k=1.645 is the one-sided 95% interval of a normal distribution.

    Requires:

    - :param k: coverage factor, i.e. the number of standard deviations added to the mean (default 1.645 covers 95% of a normal response on one side; 1, 2 and 3 give 84.1%, 97.7% and 99.9%).
    - :param direction: -1 for minimizing and +1 for maximizing.
    """

    def __init__(self, k: float = 1.645, direction: int =-1):
        self.k = k
        self.direction = direction

    def reduce(self, samples: ot.Sample) -> float:
        return float(samples.computeMean()[0] - self.k * np.sign(self.direction) * samples.computeStandardDeviation()[0])

class Quantile(Scalarization):
    """
    Makes no distributional assumption. Minimize the q quantile of the objective or constraint function --> min(F_q(x)), with q=0.95 by default.

    Requires:

    :param q: quantile level in [0, 1], i.e. the probability that the response falls below the returned value (default 0.95);
    unlike Margin this assumes nothing about the distribution shape, since the quantile is taken from the samples directly.

    """

    def __init__(self, q: float = 0.95):
        if not 0. <= q <= 1.:
            raise ValueError(f'Quantile should be between 0 and 1: {q}')
        self.q = q

    def reduce(self, samples: ot.Sample) -> float:
        return float(samples.computeQuantile(self.q)[0])


class StochasticParameter:
    """
     A stochastic parameter used to evaluate architecture objectives or constraints.

     The parameter carries two distinct things. `value` defines the parameter distribution defined by OpenTurns object.
     `sample` is one realization of it, written by `StochasticParameterSpace.param_realization` before each
     evaluation and overwritten on the next one.

     :param name: identifies the parameter
     :param value: OpenTurns distribution object
     :param ref: Optional reference object used for matching with the related class
     """

    def __init__(self, name, value: ot.DistributionImplementation, ref=None):
        self.name = name
        self.value = value
        self.ref = ref
        self._sample = None

    @property
    def sample(self) -> float:
        """
        The parameter realization of it, written by `StochasticParameterSpace.param_realization`
        """
        return self._sample

    @sample.setter
    def sample(self, value: float):
        self._sample = value




class StochasticParameterSpace:
    """
     The joint distribution of all stochastic parameters of a problem, which are assumed to be independent.
     The space does not store the sampled parameters as its attribute. The samples are processed by the chosen UQ method.

     :param parameters: the stochastic parameters
     """

    def __init__(self, parameters: List[StochasticParameter]):
        self._parameters = parameters

    @property
    def n_parameters(self) -> int:
        """Number of stochastic parameters"""
        return len(self._parameters)

    @property
    def parameter_names(self) -> List[str]:
        """List of stochastic parameter names"""
        return [parameter.name for parameter in self._parameters]

    @property
    def joint_dist(self) -> ot.JointDistribution:
        """Joint distribution of independent parameters"""
        return ot.JointDistribution([parameter.value for parameter in self._parameters],
                                    ot.IndependentCopula(self.n_parameters))

    def param_realization(self, samples: np.ndarray, i_realization: int) -> List[StochasticParameter]:
        """
        Return `i_realization` of `samples` (n_samples x n_parameters) as a list of stochastic parameters.
        """
        for j, param in enumerate(self._parameters):
            param.sample = samples[i_realization, j]
        return self._parameters

    def get_random_samples(self, n_samples: int) -> np.ndarray:
        """Draw n samples of the stochastic parameters; returns an n x n_parameters matrix"""
        result = self.joint_dist.getSample(n_samples)
        return np.array(result)

    def get_lhs_samples(self, n_samples: int) -> np.ndarray:
        """Draw n samples of the stochastic parameters; returns an n x n_parameters matrix built with LHS"""
        lhs = ot.LHSExperiment(self.joint_dist, n_samples)
        result = lhs.generate()
        return np.array(result)



class StochasticOutput:
    """
    The object representing the stochastic output of a single objective or constraint for a given design point.

    :param distribution: the output distribution for the specific objective or constraint.
    """

    def __init__(self, distribution: ot.Distribution, method_results=None):
        self.distribution = distribution
        self.method_results = method_results

    @classmethod
    def from_results(cls, results: ot.Sample, index: int, method_results=None) -> 'StochasticOutput':
        """Extract one response's column from the full (n_samples, n_outputs) results of a single design point."""
        dist = ot.KernelSmoothing().build(results.getMarginal(index))
        return cls(distribution=dist, method_results=method_results)

    @classmethod
    def from_output_samples(cls, output_samples: ot.Sample, method_results=None) -> 'StochasticOutput':
        """From the outputs of a single objective or constraint construct the object."""
        dist = ot.KernelSmoothing().build(output_samples)
        return cls(distribution=dist, method_results=method_results)

    @property
    def mean(self) -> float:
        return self.distribution.getMean()[0]

    @property
    def std(self) -> float:
        return self.distribution.getStandardDeviation()[0]

    def quantile(self, q: float) -> float:
        return self.distribution.computeQuantile(q)

    # def prob_exceeds(self, threshold: float) -> float:
    #     arr = self.to_numpy()
    #     return float(np.mean(arr > threshold))

    def margin(self, k: float = 1.645, direction: int =-1) -> float:
        """mean + k*sigma - the Gaussian-assumption margin formulation."""
        return self.mean - np.sign(direction) * k * self.std

    # @property
    # def to_distribution(self) -> ot.Distribution:
    #     """Fit a continuous distribution."""
    #     return ot.KernelSmoothing().build(self.output_samples)

    # def to_numpy(self) -> np.ndarray:
    #     return np.array(self.output_samples).flatten()

    # def reduce(self, scalar: Scalarization) -> float:
    #     """
    #     Reduce the sampled values to the single value the optimizer sees, by applying the given scalar.
    #     """
    #     values = self.to_numpy()
    #
    #     if values.size == 0 or not np.all(np.isfinite(values)):
    #         return np.nan
    #     samples = self.output_samples
    #
    #     return scalar.reduce(samples)

    def __str__(self):
        if self.std / self.mean < 1e-6:
            return f'{self.mean:.4g}'
        return f"(mean = {self.mean:.4g}, sigma = {self.std:.4g})"



# class StochasticResults:
#     """
#     Single object that stores all stochastic objectives or constraints for a given design point.
#
#     :param method_result: optionally carries whatever the UQ method produced beyond the samples themselves; for polynomial
#     chaos that is the list of `ot.FunctionalChaosResult`s, from which for example Sobol indices can be obtained.
#     """
#
#     def __init__(self, outputs: List[StochasticOutput], method_result=None):
#         self.outputs = outputs
#         self.method_result = method_result


class UQMethod:
    """
    Base class for an uncertainty propagation method.

    It is responsible for sampling the provided 'param_space' based on the provided number of evaluations and
    for performing uncertainty quantification study to return a list of StochasticResults for a given design vector.

    Subclass by overriding `_draw_samples` (sampling method) and `process_results` (statistics computation).

    :param n_evaluations: number of expensive evaluations spent per design point
    :param seed: optional seed the OpenTURNS generator before drawing, so the design is reproducible
    """


    def __init__(self, n_evaluations: int, seed: int = None):
        if n_evaluations is None:
            raise ValueError('n_evaluations must be specified: it is the number of expensive evaluations '
                             'per design point')
        self.n_evaluations = n_evaluations
        self.seed = seed

        self._samples: Optional[np.ndarray] = None

    def get_samples(self, param_space: StochasticParameterSpace) -> np.ndarray:
        """
        Get samples from the joint distribution provided by the param_space.
        """
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

    def process_results(self, results: np.ndarray, param_space: StochasticParameterSpace) -> list[StochasticOutput]:
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
        """Draw samples with LHS method"""
        samples = param_space.get_lhs_samples(self.n_evaluations)
        return samples

    def process_results(self, results: np.ndarray, param_space: StochasticParameterSpace = None) -> list[StochasticOutput]:
        results = np.asarray(results, dtype=float)

        sample = ot.Sample(results)
        outputs = [StochasticOutput.from_results(sample, i) for i in range(results.shape[1])]
        return outputs


class PolynomialChaos(UQMethod):
    """
     Polynomial chaos expansion (PCE): a surrogate model is fitted for each response as a function of the uncertain parameters.
     The trained model is then sampled with Monte Carlo to compute output statistics.

     The fitted `ot.FunctionalChaosResult` of each response is kept on the `StochasticResult` (`method_result`) with
     Sobol sensitivity indices available through `ot.FunctionalChaosSobolIndices`.

     :param degree: total degree of the expansion
     :param n_metamodel_samples: number of cheap evaluations of the trained model used to obtain the statistics
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
        """Check the expansion can be fitted from the budget"""
        n_terms = self.n_terms(param_space)
        if self.n_evaluations < n_terms:
            raise ValueError(f'A degree-{self.degree} expansion in {param_space.n_parameters} '
                             f'parameters has {n_terms} terms, so it needs at least that many samples to fit: '
                             f'n_evaluations = {self.n_evaluations}')

    def _draw_samples(self, param_space: StochasticParameterSpace) -> np.ndarray:
        """Draw samples with LHS method"""
        self._validate(param_space)
        samples = param_space.get_lhs_samples(self.n_evaluations)
        return samples

    def _get_metamodel_input(self, param_space: StochasticParameterSpace) -> ot.Sample:
        """Get samples that used for evaluating the surrogate model"""
        if self._metamodel_input is None:
            self._metamodel_input = param_space.joint_dist.getSample(self.n_metamodel_samples)
        return self._metamodel_input

    def _build_algorithm(self, param_space: StochasticParameterSpace, input_sample: ot.Sample, output_sample: ot.Sample) -> ot.FunctionalChaosAlgorithm:
        """Train surrogate model with the least-squares method"""
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

    def process_results(self, results: np.ndarray, param_space: StochasticParameterSpace) -> list[StochasticOutput]:
        results = np.asarray(results, dtype=float)

        input_sample = ot.Sample(self.get_samples(param_space))
        metamodel_input = self._get_metamodel_input(param_space)

        outputs = []
        for i_out in range(results.shape[1]):
            values = results[:, i_out]

            # A response with failed evaluations cannot be fitted: pass the raw samples through instead, so that
            # the problem's nan_policy decides what happens to this design point
            if not np.all(np.isfinite(values)):
                samples = ot.Sample(values.reshape((-1, 1)))
                chaos_result = None

            else:
                algorithm = self._build_algorithm(param_space, input_sample, ot.Sample(values.reshape((-1, 1))))
                algorithm.run()
                chaos_result = algorithm.getResult()
                samples = chaos_result.getMetaModel()(metamodel_input)

            outputs.append(StochasticOutput.from_output_samples(samples, chaos_result))
        return outputs