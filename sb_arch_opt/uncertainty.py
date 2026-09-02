"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Uncertainty quantification building blocks for stochastic (robust) architecture optimization problems.
"""
import enum
from typing import List, Optional

import openturns as ot
import numpy as np

__all__ = ['StochasticOutputType', 'UQMethodType', 'StochasticParameter', 'StochasticParameterSpace',
           'StochasticOutput', 'StochasticResult', 'UQMethod', 'MonteCarlo', 'PolynomialChaos']


class StochasticOutputType(enum.Flag):
    NONE = 0
    MEAN = enum.auto()
    MARGIN = enum.auto()
    QUANTILE = enum.auto()


class UQMethodType(enum.Flag):
    NONE = 0
    MONTE_CARLO = enum.auto()
    PCE = enum.auto()


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
        # Note: the marginals are the underlying ot.Distribution objects, not the StochasticParameter wrappers
        return ot.JointDistribution([parameter.distribution for parameter in self.parameters],
                                    ot.IndependentCopula(self.n_parameters))

    def get_samples(self, n: int) -> np.ndarray:
        """Draw n samples of all parameters; returns an n x n_parameters matrix"""
        result = self.joint_dist.getSample(n)
        self.sample = result
        return np.array(result)


class StochasticOutput:
    """The sampled values of one response (objective or constraint) for one design point."""

    def __init__(self, output_samples: ot.Sample, output_type: StochasticOutputType):
        self.output_samples = output_samples          # ot.Sample of shape (n, 1)
        self.output_type = output_type

    @classmethod
    def from_results(cls, results: ot.Sample, index: int, output_type: StochasticOutputType) -> 'StochasticOutput':
        """Extract one response's column from the full (n_samples, n_outputs) results of a single design point."""
        return cls(output_samples=results.getMarginal(index), output_type=output_type)

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

    def reduce(self, k: float = 1.645, q: float = .95, nan_policy: str = 'propagate') -> float:
        """
        Reduce the sampled values to the single value the optimizer sees.

        Note that SBArchOpt objectives are always minimized and inequality constraints are satisfied when <= 0, so
        MARGIN is always `mean + k*std`: it penalizes both a bad mean and a large spread, for objectives and
        constraints alike.

        `nan_policy` controls what happens when some samples failed to evaluate (NaN, i.e. hidden constraints):
        - 'propagate': any failed sample makes the whole design point fail (the value becomes NaN). This is the
          conservative default and matches how SBArchOpt treats failed evaluations elsewhere.
        - 'omit': reduce over the samples that did evaluate; only an entirely failed design point becomes NaN.
        """
        values = self.to_numpy()

        # Note: the statistics are computed by OpenTURNS rather than numpy, so that a reduced value is exactly
        # reproducible from this object's mean()/std()/quantile()/margin() afterwards
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

        output_type = self.output_type
        if output_type in (None, StochasticOutputType.NONE, StochasticOutputType.MEAN):
            return float(samples.computeMean()[0])

        if output_type == StochasticOutputType.MARGIN:
            return float(samples.computeMean()[0]) + k * float(samples.computeStandardDeviation()[0])

        if output_type == StochasticOutputType.QUANTILE:
            if not 0. <= q <= 1.:
                raise ValueError(f'Quantile should be between 0 and 1: {q}')
            return self.quantile(q)

        raise ValueError(f'Unknown stochastic metric type: {output_type}')


class StochasticResult:
    """
    All sampled responses of a single design point: the statistics behind the values the optimizer sees.

    Provided per design point in the evaluation output (`out['stochastic']`), so mean, std, quantiles and the fitted
    output distribution stay available after the optimization has run.

    `method_result` optionally carries whatever the UQ method produced beyond the samples themselves; for polynomial
    chaos that is the list of `ot.FunctionalChaosResult`s, from which for example Sobol indices can be obtained.
    """

    def __init__(self, f: List[StochasticOutput], g: List[StochasticOutput], h: List[StochasticOutput],
                 method_result=None):
        self.f = f
        self.g = g
        self.h = h
        self.method_result = method_result


class UQMethod:
    """
    Base class for an uncertainty propagation method.

    A UQ method has two jobs:
    - `get_samples`: provide the parameter values at which the expensive model must be evaluated (the design)
    - `process_results`: turn the responses collected at that design, for ONE design point, into `StochasticOutput`s

    The loop over samples is driven by the problem, so the problem's evaluation function stays vectorized over
    design points. Note that this sample-in / responses-out interface is what OpenTURNS' own UQ algorithms consume:
    `FunctionalChaosAlgorithm` (PCE), `KrigingAlgorithm` and the sample-based `SobolIndicesAlgorithm` constructors
    all take an (inputSample, outputSample) pair rather than a callable.

    The design of n samples is drawn once and reused for every design point and every evaluation (common random
    numbers), so that the reduced objective is a deterministic function of x. This matters for surrogate-based
    optimization: with freshly drawn samples per evaluation, the surrogate fits noise instead of the robust
    objective. Call `resample()` to explicitly draw a new design.

    Subclass and implement `process_results` to add a method, overriding `_draw_samples` if it needs a design other
    than a plain draw from the joint distribution; use `_split_outputs` to map the per-response outputs onto
    objectives, inequality constraints and equality constraints.
    """

    def __init__(self, parameters_space: StochasticParameterSpace, output_type: List[StochasticOutputType],
                 n_obj: int, n_ieq_constr: int, n_eq_constr: int, n: int, seed: int = None):
        self.parameters_space = parameters_space
        self.output_type = output_type
        self.n_obj = n_obj
        self.n_ieq_constr = n_ieq_constr
        self.n_eq_constr = n_eq_constr
        self.n = n
        self.seed = seed
        self._samples: Optional[np.ndarray] = None

        n_expected = n_obj + n_ieq_constr + n_eq_constr
        if len(output_type) != n_expected:
            raise ValueError(f'Expected {n_expected} output types (n_obj + n_ieq_constr + n_eq_constr), '
                             f'got {len(output_type)}')

    @property
    def n_samples(self) -> int:
        """Number of parameter samples this method evaluates per design point"""
        return self.n

    def get_samples(self) -> np.ndarray:
        """The parameter samples to evaluate, as an n_samples x n_parameters matrix"""
        if self._samples is None:
            if self.seed is not None:
                ot.RandomGenerator.SetSeed(self.seed)
            self._samples = self._draw_samples()
        return self._samples

    def resample(self):
        """Draw a new design on the next evaluation"""
        self._samples = None

    def _draw_samples(self) -> np.ndarray:
        """Draw the design; override for a design other than a plain draw from the joint distribution"""
        return self.parameters_space.get_samples(self.n)

    def process_results(self, results: np.ndarray) -> StochasticResult:
        """
        Turn the responses of ONE design point (an n_samples x (n_obj+n_ieq_constr+n_eq_constr) matrix) into the
        stochastic outputs of that design point.
        """
        raise NotImplementedError

    def _check_results(self, results: np.ndarray):
        if results.shape[1] != len(self.output_type):
            raise ValueError(f'Expected {len(self.output_type)} response columns, got {results.shape[1]}')
        if results.shape[0] != self.n_samples:
            raise ValueError(f'Expected {self.n_samples} response rows, got {results.shape[0]}')

    def _split_outputs(self, outputs: List[StochasticOutput], method_result=None) -> StochasticResult:
        i_g0 = self.n_obj
        i_h0 = self.n_obj + self.n_ieq_constr
        return StochasticResult(outputs[:i_g0], outputs[i_g0:i_h0], outputs[i_h0:], method_result=method_result)


class MonteCarlo(UQMethod):
    """
    Monte Carlo uncertainty propagation: the responses are evaluated for n samples of the uncertain parameters, and
    the statistics are taken directly over those samples.
    """

    def process_results(self, results: np.ndarray) -> StochasticResult:
        self._check_results(results)
        sample = ot.Sample(np.asarray(results, dtype=float))
        outputs = [StochasticOutput.from_results(sample, i, self.output_type[i])
                   for i in range(len(self.output_type))]
        return self._split_outputs(outputs)


class PolynomialChaos(UQMethod):
    """
    Polynomial chaos expansion (PCE): a surrogate of each response as a function of the uncertain parameters is
    fitted from n evaluations of the expensive model, and the statistics are then taken from that surrogate.

    The expensive model is evaluated `n` times per design point, exactly like Monte Carlo, but the statistics come
    from `n_metamodel_samples` evaluations of the cheap fitted expansion instead of from the n expensive ones. For a
    response that the expansion represents well this gives far more accurate quantiles and tail statistics for the
    same number of expensive evaluations.

    The fitted `ot.FunctionalChaosResult` of each response is kept on the `StochasticResult` (`method_result`), so
    Sobol sensitivity indices are available for free through `ot.FunctionalChaosSobolIndices`.

    Note that a response containing failed evaluations (NaN) cannot be fitted; those responses fall back to the raw
    samples, so the problem's `nan_policy` decides what happens to that design point as usual.
    """

    def __init__(self, parameters_space: StochasticParameterSpace, output_type: List[StochasticOutputType],
                 n_obj: int, n_ieq_constr: int, n_eq_constr: int, n: int, seed: int = None,
                 degree: int = 3, n_metamodel_samples: int = 10000):
        self.degree = degree
        self.n_metamodel_samples = n_metamodel_samples
        self._metamodel_input: Optional[ot.Sample] = None
        super().__init__(parameters_space, output_type, n_obj, n_ieq_constr, n_eq_constr, n, seed)

        n_terms = self.n_terms
        if n < n_terms:
            raise ValueError(f'A degree-{degree} expansion in {parameters_space.n_parameters} parameters has '
                             f'{n_terms} terms, so it needs at least that many samples to fit: n = {n}')

    @property
    def n_terms(self) -> int:
        """Number of terms in the expansion, i.e. the minimum number of samples needed to fit it"""
        enumerate_function = ot.LinearEnumerateFunction(self.parameters_space.n_parameters)
        return int(enumerate_function.getStrataCumulatedCardinal(self.degree))

    def _draw_samples(self) -> np.ndarray:
        # A Latin hypercube covers the parameter space more evenly than plain Monte Carlo, which matters when the
        # design is used to fit an expansion rather than to average over
        experiment = ot.LHSExperiment(self.parameters_space.joint_dist, self.n, False, True)
        samples = experiment.generate()
        self.parameters_space.sample = samples
        return np.array(samples)

    def _get_metamodel_input(self) -> ot.Sample:
        if self._metamodel_input is None:
            self._metamodel_input = self.parameters_space.joint_dist.getSample(self.n_metamodel_samples)
        return self._metamodel_input

    def _build_algorithm(self, input_sample: ot.Sample, output_sample: ot.Sample) -> ot.FunctionalChaosAlgorithm:
        distribution = self.parameters_space.joint_dist
        dimension = distribution.getDimension()

        polynomials = [ot.StandardDistributionPolynomialFactory(distribution.getMarginal(i))
                       for i in range(dimension)]
        enumerate_function = ot.LinearEnumerateFunction(dimension)
        basis = ot.OrthogonalProductPolynomialFactory(polynomials, enumerate_function)

        adaptive_strategy = ot.FixedStrategy(basis, self.n_terms)
        projection_strategy = ot.LeastSquaresStrategy()
        return ot.FunctionalChaosAlgorithm(input_sample, output_sample, distribution,
                                           adaptive_strategy, projection_strategy)

    def process_results(self, results: np.ndarray) -> StochasticResult:
        self._check_results(results)
        results = np.asarray(results, dtype=float)

        input_sample = ot.Sample(self.get_samples())
        metamodel_input = self._get_metamodel_input()

        outputs, chaos_results = [], []
        for i_out in range(results.shape[1]):
            values = results[:, i_out]

            # A response with failed evaluations cannot be fitted: pass the raw samples through instead, so that
            # the problem's nan_policy decides what happens to this design point
            if not np.all(np.isfinite(values)):
                samples = ot.Sample(values.reshape((-1, 1)))
                chaos_results.append(None)

            else:
                algorithm = self._build_algorithm(input_sample, ot.Sample(values.reshape((-1, 1))))
                algorithm.run()
                chaos_result = algorithm.getResult()
                chaos_results.append(chaos_result)
                samples = chaos_result.getMetaModel()(metamodel_input)

            outputs.append(StochasticOutput(samples, self.output_type[i_out]))

        return self._split_outputs(outputs, method_result=chaos_results)