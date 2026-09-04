"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Uncertainty quantification building blocks for stochastic (robust) architecture optimization problems.
"""
from typing import List, Optional

import openturns as ot
import numpy as np

__all__ = ['RobustMeasure', 'Mean', 'Margin', 'Quantile', 'StochasticParameter', 'StochasticParameterSpace',
           'StochasticOutput', 'StochasticResult', 'UQMethod', 'MonteCarlo', 'PolynomialChaos']


class RobustMeasure:
    """
    Parent class for all robust measures etc. Margin, Mean, Quantile
    """

    def reduce(self, samples: ot.Sample) -> float:
        """Reduce an (n_samples x 1) sample of one response to a single value"""
        raise NotImplementedError


class Mean(RobustMeasure):
    """The expected value of the response"""

    def reduce(self, samples: ot.Sample) -> float:
        return float(samples.computeMean()[0])

class Margin(RobustMeasure):
    """Assumes the response is Gaussian"""

    def __init__(self, k: float = 1.645):
        self.k = k

    def reduce(self, samples: ot.Sample) -> float:
        return float(samples.computeMean()[0] + self.k*samples.computeStandardDeviation()[0])

class Quantile(RobustMeasure):
    """The q quantile of the response for arbitrary distribution"""

    def __init__(self, q: float = 0.95):
        if not 0. <= q <= 1.:
            raise ValueError(f'Quantile should be between 0 and 1: {q}')
        self.q = q

    def reduce(self, samples: ot.Sample) -> float:
        return float(samples.computeQuantile(self.q)[0])


class StochasticParameter:
    """An uncertain parameter: a quantity that influences the evaluation but is not chosen by the optimizer."""

    def __init__(self, name, distribution):
        self.name = name
        self.distribution = distribution

    def mean(self) -> float:
        return self.distribution.getMean()[0]

    def std(self) -> float:
        return self.distribution.getStandardDeviation()[0]


class StochasticParameterSpace:
    """The joint distribution of all uncertain parameters of a problem."""

    def __init__(self):
        self.sample = None
        self.parameters: List[StochasticParameter] = []

    def add_parameter(self, parameter: StochasticParameter):
        self.parameters.append(parameter)

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    @property
    def parameter_names(self) -> List[str]:
        return [parameter.name for parameter in self.parameters]

    @property
    def joint_dist(self) -> ot.JointDistribution:
        """ Joint distribution of independent variables is chosen """
        return ot.JointDistribution([parameter.distribution for parameter in self.parameters],
                                    ot.IndependentCopula(self.n_parameters))

    def get_samples(self, n: int) -> np.ndarray:
        """Draw n samples of all parameters; returns an n x n_parameters matrix"""
        result = self.joint_dist.getSample(n)
        self.sample = result
        return np.array(result)


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

    def margin(self, k: float = 1.645) -> float:
        """mean + k*sigma - the Gaussian-assumption margin formulation."""
        return self.mean() + k * self.std()

    def to_distribution(self) -> ot.Distribution:
        """Fit a continuous distribution if you need PDF/CDF rather than raw samples."""
        return ot.KernelSmoothing().build(self.output_samples)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.output_samples).flatten()

    def reduce(self, measure: RobustMeasure, nan_policy: str = 'propagate') -> float:
        """
        Reduce the sampled values to the single value the optimizer sees, by applying the given measure.

        The measure is passed in rather than stored: an output is just the sampled values, and which statistic
        of them the optimizer sees is the problem's business (see `StochasticArchOptProblem`).

        `nan_policy` controls what happens when some samples failed to evaluate (NaN, i.e. hidden constraints):
        - 'propagate': any failed sample makes the whole design point fail (the value becomes NaN). This is the
          conservative default and matches how SBArchOpt treats failed evaluations elsewhere.
        - 'omit': reduce over the samples that did evaluate; only an entirely failed design point becomes NaN.
        """
        values = self.to_numpy()

        if nan_policy == 'propagate':
            if values.size == 0 or not np.all(np.isfinite(values)):
                return np.nan
            samples = self.output_samples
        elif nan_policy == 'omit':
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return np.nan
            samples = ot.Sample(finite.reshape((-1, 1)))
        else:
            raise ValueError(f'Unknown nan_policy: {nan_policy!r} (expected "propagate" or "omit")')

        return measure.reduce(samples)


class StochasticResult:
    """
    All sampled responses of a single design point: the statistics behind the values the optimizer sees.

    `method_result` optionally carries whatever the UQ method produced beyond the samples themselves; for polynomial
    chaos that is the list of `ot.FunctionalChaosResult`s, from which for example Sobol indices can be obtained.
    """

    def __init__(self, outputs: List[StochasticOutput], method_result=None):
        self.outputs = outputs
        self.method_result = method_result


class UQMethod:
    """
    Base class for an uncertainty propagation method.

    A method only knows how many expensive evaluations per design point it is allowed to spend; everything that
    depends on the problem it is used for (the parameter space, the number of responses, the robust measures) is
    passed in as a method parameter, so the same instance carries no problem-specific state.
    """

    def __init__(self, n_evaluations: int, seed: int = None):
        if n_evaluations is None:
            raise ValueError('n_evaluations must be specified: it is the number of expensive evaluations '
                             'per design point')
        self.n_evaluations = n_evaluations
        self.seed = seed

        self._samples: Optional[np.ndarray] = None
        self._samples_space: Optional[StochasticParameterSpace] = None

    @property
    def n_samples(self) -> int:
        """Number of parameter samples this method evaluates per design point"""
        return self.n_evaluations

    def get_samples(self, parameters_space: StochasticParameterSpace) -> np.ndarray:
        """
        The parameter samples to evaluate, as an n_samples x n_parameters matrix.

        The design is drawn once and then reused, so that every design point is evaluated at the same realizations
        of the uncertain parameters (common random numbers): that is what makes design points comparable to each
        other, and a surrogate fitted through them smooth. Call `resample` to force a new design.

        The reuse is per parameter space: handing a *different* space redraws, since the cached design describes
        the parameters of the space it was drawn for and silently means something else for another one.
        """
        if parameters_space is None:
            raise ValueError('No parameter space to sample')
        if self._samples is None or self._samples_space is not parameters_space:
            if self.seed is not None:
                ot.RandomGenerator.SetSeed(self.seed)
            self._samples = self.draw_samples(parameters_space)
            self._samples_space = parameters_space
        return self._samples

    def draw_samples(self, parameters_space: StochasticParameterSpace) -> np.ndarray:
        """Draw the design of experiments in the parameter space; override to use a different design"""
        return parameters_space.get_samples(self.n_evaluations)

    def resample(self):
        """Draw a new design on the next evaluation"""
        self._samples = None
        self._samples_space = None

    def check_results(self, results: np.ndarray):
        """The results of one design point must have one row per evaluated parameter sample"""
        if results.ndim != 2:
            raise ValueError(f'Expected a 2D (n_samples x n_outputs) results matrix, got {results.ndim}D')
        if results.shape[0] != self.n_samples:
            raise ValueError(f'Expected {self.n_samples} response rows, got {results.shape[0]}')

    def process_results(self, results: np.ndarray,
                        parameters_space: StochasticParameterSpace) -> StochasticResult:
        """
        Turn the responses of ONE design point (an n_samples x (n_obj+n_ieq_constr+n_eq_constr) matrix) into the
        stochastic outputs of that design point, in the same column order.
        """
        raise NotImplementedError


class MonteCarlo(UQMethod):
    """
    Monte Carlo uncertainty propagation
    """

    def process_results(self, results: np.ndarray,
                        parameters_space: StochasticParameterSpace = None) -> StochasticResult:
        results = np.asarray(results, dtype=float)
        self.check_results(results)

        sample = ot.Sample(results)
        outputs = [StochasticOutput.from_results(sample, i) for i in range(results.shape[1])]
        return StochasticResult(outputs)


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

    Note that a response containing failed evaluations (NaN) cannot be fitted; those responses fall back to the raw
    samples, so the problem's `nan_policy` decides what happens to that design point as usual.
    """

    def __init__(self, n_evaluations: int, seed: int = None, degree: int = 3, n_metamodel_samples: int = 10000):
        self.degree = degree
        self.n_metamodel_samples = n_metamodel_samples
        self._metamodel_input: Optional[ot.Sample] = None
        self._metamodel_input_space: Optional[StochasticParameterSpace] = None
        super().__init__(n_evaluations, seed)

    def n_terms(self, parameters_space: StochasticParameterSpace) -> int:
        """Number of terms in the expansion, i.e. the minimum number of samples needed to fit it"""
        enumerate_function = ot.LinearEnumerateFunction(parameters_space.n_parameters)
        return int(enumerate_function.getStrataCumulatedCardinal(self.degree))

    def _validate(self, parameters_space: StochasticParameterSpace):
        # Needs the parameter space, since the number of terms depends on the number of parameters
        n_terms = self.n_terms(parameters_space)
        if self.n_evaluations < n_terms:
            raise ValueError(f'A degree-{self.degree} expansion in {parameters_space.n_parameters} '
                             f'parameters has {n_terms} terms, so it needs at least that many samples to fit: '
                             f'n_evaluations = {self.n_evaluations}')

    def draw_samples(self, parameters_space: StochasticParameterSpace) -> np.ndarray:
        self._validate(parameters_space)

        # A Latin hypercube covers the parameter space more evenly than plain Monte Carlo, which matters when the
        # design is used to fit an expansion rather than to average over
        experiment = ot.LHSExperiment(parameters_space.joint_dist, self.n_evaluations, False, True)
        samples = experiment.generate()
        parameters_space.sample = samples
        return np.array(samples)

    def _get_metamodel_input(self, parameters_space: StochasticParameterSpace) -> ot.Sample:
        # Like the expensive design, drawn once and reused, and redrawn for a different parameter space
        if self._metamodel_input is None or self._metamodel_input_space is not parameters_space:
            self._metamodel_input = parameters_space.joint_dist.getSample(self.n_metamodel_samples)
            self._metamodel_input_space = parameters_space
        return self._metamodel_input

    def _build_algorithm(self, input_sample: ot.Sample, output_sample: ot.Sample,
                         parameters_space: StochasticParameterSpace) -> ot.FunctionalChaosAlgorithm:
        distribution = parameters_space.joint_dist
        dimension = distribution.getDimension()

        polynomials = [ot.StandardDistributionPolynomialFactory(distribution.getMarginal(i))
                       for i in range(dimension)]
        enumerate_function = ot.LinearEnumerateFunction(dimension)
        basis = ot.OrthogonalProductPolynomialFactory(polynomials, enumerate_function)

        adaptive_strategy = ot.FixedStrategy(basis, self.n_terms(parameters_space))
        projection_strategy = ot.LeastSquaresStrategy()
        return ot.FunctionalChaosAlgorithm(input_sample, output_sample, distribution,
                                           adaptive_strategy, projection_strategy)

    def process_results(self, results: np.ndarray,
                        parameters_space: StochasticParameterSpace) -> StochasticResult:
        results = np.asarray(results, dtype=float)
        self.check_results(results)

        input_sample = ot.Sample(self.get_samples(parameters_space))
        metamodel_input = self._get_metamodel_input(parameters_space)

        outputs, chaos_results = [], []
        for i_out in range(results.shape[1]):
            values = results[:, i_out]

            # A response with failed evaluations cannot be fitted: pass the raw samples through instead, so that
            # the problem's nan_policy decides what happens to this design point
            if not np.all(np.isfinite(values)):
                samples = ot.Sample(values.reshape((-1, 1)))
                chaos_results.append(None)

            else:
                algorithm = self._build_algorithm(input_sample, ot.Sample(values.reshape((-1, 1))),
                                                  parameters_space)
                algorithm.run()
                chaos_result = algorithm.getResult()
                chaos_results.append(chaos_result)
                samples = chaos_result.getMetaModel()(metamodel_input)

            outputs.append(StochasticOutput(samples))

        return StochasticResult(outputs, method_result=chaos_results)