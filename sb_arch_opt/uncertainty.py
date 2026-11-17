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
from __future__ import annotations

import copy

import numpy as np
from typing import List, Optional, Union, Any

try:
    import openturns as ot
    HAS_UNCERTAINTY = True
except ImportError:
    HAS_UNCERTAINTY = False


def check_dependency():
    if not HAS_UNCERTAINTY:
        raise ImportError(
            'Looks like SBArchOpt uncertainty package is not installed! Run: pip install sb-arch-opt[uncertainty]')


__all__ = ['Scalarization', 'Mean', 'Margin', 'Quantile', 'StochasticParameter', 'StochasticParameterSpace',
           'StochasticOutput', 'UQMethod', 'MonteCarlo', 'PolynomialChaos', 'EvaluationOutput', 'AdaptivePolynomialChaos', 'HAS_UNCERTAINTY', 'check_dependency']

EvaluationOutput = Union['StochasticOutput', float]
"""Output either distribution or numeric value."""

class StochasticParameter:
    """
     A stochastic parameter used to evaluate architecture objectives or constraints.

     The parameter carries two distinct things. `dist` defines the parameter distribution defined by OpenTurns object.
     `sample` is one realization of it.

     :param name: identifies the parameter
     :param dist: OpenTurns distribution object
     :param ref: Optional reference object used for matching with the related class
     """

    def __init__(self, name, dist: ot.DistributionImplementation, ref=None):
        check_dependency()
        self.name = name
        self.dist = dist
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
        check_dependency()
        self._parameters = parameters

    @classmethod
    def from_joint_dist(cls, joint_dist: ot.JointDistribution):
        parameters = []
        for i in range(joint_dist.getDimension()):
            dist = joint_dist.getMarginal(i)
            parameters.append(StochasticParameter(dist.getName(), dist.getImplementation()))
        return cls(parameters)

    @property
    def n_parameters(self) -> int:
        """Number of stochastic parameters"""
        return len(self._parameters)

    @property
    def parameters(self) -> List[StochasticParameter]:
        """The stochastic parameters, in the order of the sample columns"""
        return list(self._parameters)

    @property
    def parameter_names(self) -> List[str]:
        """List of stochastic parameter names"""
        return [parameter.name for parameter in self._parameters]

    @property
    def joint_dist(self) -> ot.JointDistribution:
        """Joint distribution of independent parameters"""
        return ot.JointDistribution([parameter.dist for parameter in self._parameters],
                                    ot.IndependentCopula(self.n_parameters))

    def param_realization(self, samples: np.ndarray, i_realization: int) -> List[StochasticParameter]:
        """
        Return `i_realization` of `samples` (n_samples x n_parameters) as a list of stochastic parameters.
        """
        realization = []
        for j, param in enumerate(self._parameters):
            realized = copy.copy(param)
            realized.sample = float(samples[i_realization, j])
            realization.append(realized)
        return realization

    def get_random_samples(self, n_samples: int) -> np.ndarray:
        """Draw n samples of the stochastic parameters; returns an n x n_parameters matrix"""
        result = self.joint_dist.getSample(n_samples)
        return np.array(result)

    def get_lhs_samples(self, n_samples: int) -> np.ndarray:
        """Draw n samples of the stochastic parameters; returns an n x n_parameters matrix built with LHS"""
        lhs = ot.LHSExperiment(self.joint_dist, n_samples)
        result = lhs.generate()
        return np.array(result)

    def get_samples_from_qmc(self, sequence: ot.LowDiscrepancySequenceImplementation, n_samples) -> np.ndarray:
        exp = ot.LowDiscrepancyExperiment(sequence, self.joint_dist, n_samples, True)
        exp.setRandomize(True)
        result = exp.generate()
        return np.array(result)


class StochasticOutput:
    """
    The object representing the stochastic output of a single objective or constraint for a given design point.

    :param dist: the output distribution for the specific objective or constraint.
    """

    def __init__(self, dist: ot.Distribution, samples: np.ndarray, method_results=None):
        check_dependency()
        self.dist = dist
        self.samples = samples
        self.method_results = method_results

    @staticmethod
    def build_distribution(output_samples: ot.Sample) -> Union[ot.Distribution, float]:
        values = np.asarray(output_samples, dtype=float).ravel()

        if values.size == 0 or not np.all(np.isfinite(values)):
            return np.nan

        mean, std = float(np.mean(values)), float(np.std(values))
        if std <= 1e-12*max(abs(mean), 1.):
            return mean
        else:
            return ot.KernelSmoothing().build(output_samples)

    @classmethod
    def from_output_samples(cls, output_samples: ot.Sample, method_results=None) -> EvaluationOutput:
        """
        Construct the object from evaluation output samples of a single objective or constraint.
        If the variance of the provided samples is statistically insignificant,
        a float value is returned to represent deterministic output. If any
        sample is nan, np.nan is returned.
        """
        values = np.asarray(output_samples, dtype=float).ravel()
        dist = cls.build_distribution(output_samples)
        if isinstance(dist, float):
            return dist
        return cls(dist=dist, samples=values, method_results=method_results)

    @property
    def mean(self) -> float:
        return self.dist.getMean()[0]

    @property
    def std(self) -> float:
        return self.dist.getStandardDeviation()[0]

    def quantile(self, q: float) -> float:
        return self.dist.computeQuantile(q)[0]

    def margin(self, k: float = 1.645, direction: int =-1) -> float:
        return self.mean - np.sign(direction) * k * self.std

    def __repr__(self):
        return f"(mean = {self.mean:.4g}, sigma = {self.std:.4g})"

    def __str__(self):
        return repr(self)


class Scalarization:
    """
    Parent class for all types of stochastic optimization problem. Do not instantiate this class directly, use its subclasses:

    - Mean
    - Margin
    - Quantile
    """

    def __init__(self):
        check_dependency()

    def scalarize(self, output: StochasticOutput) -> float:
        """Reduce an (n_samples x 1) sample of one response to a single value that the optimizer sees based on the optimization problem type."""
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return repr(self)


class Mean(Scalarization):
    """Minimize the expectation of the objective or constraint function for example min(E[F(x)])"""

    def scalarize(self, output: StochasticOutput) -> float:
        return float(output.mean)


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
        super().__init__()
        self.k = k
        self.direction = direction

    def scalarize(self, output: StochasticOutput) -> float:
        return float(output.margin(self.k, self.direction))

    def __repr__(self):
        return f"Margin with k = {self.k}"


class Quantile(Scalarization):
    """
    Makes no distributional assumption. Minimize the q quantile of the objective or constraint function --> min(F_q(x)), with q=0.95 by default.

    Requires:

    :param q: quantile level in [0, 1], i.e. the probability that the response falls below the returned value (default 0.95);
    unlike Margin this assumes nothing about the distribution shape, since the quantile is taken from the samples directly.

    """

    def __init__(self, q: float = 0.95, direction: int =-1):
        super().__init__()
        if not 0. <= q <= 1.:
            raise ValueError(f'Quantile should be between 0 and 1: {q}')
        self.q = q
        self.direction = direction

    def scalarize(self, output: StochasticOutput) -> float:
        q = self.q if self.direction == -1 else 1 - self.q
        return float(output.quantile(q))

    def __repr__(self):
        return f"Quantile with q = {self.q}"


class UQMethod:
    """
    Base class for an uncertainty propagation methods.

    It is responsible for sampling the provided 'param_space' based on the provided number of evaluations and
    for performing uncertainty quantification study to return a list of StochasticOutputs or floats for a given design vector.

    Subclass by overriding `_draw_samples` (sampling method) and `process_results` (statistics computation).

    :param n_evaluations: number of expensive evaluations spent per design point
    :param seed: optional seed the OpenTURNS generator before drawing, so the design is reproducible
    """

    def __init__(self, n_evaluations: Optional[int] = None, seed: int = 42):
        check_dependency()
        if n_evaluations is not None and n_evaluations <= 0:
            raise ValueError('n_evaluations must be specified: it is the number of expensive evaluations '
                             'per design point')
        self.n_evaluations = n_evaluations
        self.seed = seed

    @property
    def adaptive_uq_method(self) -> bool:
        return False

    def get_samples(self, param_space: StochasticParameterSpace, n_evaluations: Optional[int] = None) -> np.ndarray:
        """
        Get samples from the joint distribution provided by the param_space. Every draw reseeds the generator to
        `self.seed`, so the same realizations are reproduced across calls (common random numbers); set `seed=None`
        to draw different ones.
        """
        if n_evaluations is None:
            n_evaluations = self.n_evaluations
        if n_evaluations is None:
            raise ValueError('n_evaluations must be specified')
        if n_evaluations <= 0:
            raise ValueError('n_evaluations must be positive')

        # Reseed before every draw so the same realizations are reproduced across calls (common random numbers);
        # leave the generator alone when seed is None so repeated calls draw new samples.
        if self.seed is not None:
            ot.RandomGenerator.SetSeed(self.seed)
        return self._draw_samples(param_space, n_evaluations)

    @staticmethod
    def _draw_samples(param_space: StochasticParameterSpace, n_evaluations: int) -> np.ndarray:
        """Draw the design of experiments in the parameter space; override to use a different design"""
        return param_space.get_lhs_samples(n_evaluations)

    def process_results(self, results, *args, **kwargs) -> list[EvaluationOutput]:
        """
        Turn the responses of one design point (an n_samples x (n_obj+n_ieq_constr+n_eq_constr) matrix) into the
        list of EvaluationOutput objects.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self):
        return repr(self)


class MonteCarlo(UQMethod):
    """
    Monte Carlo uncertainty propagation
    """

    @staticmethod
    def _draw_samples(param_space: StochasticParameterSpace, n_evaluations: int) -> np.ndarray:
        """Draw samples with LHS method"""
        return param_space.get_lhs_samples(n_evaluations)

    def process_results(self, results, *args, **kwargs) -> list[EvaluationOutput]:
        results = np.asarray(results, dtype=float)

        sample = ot.Sample(results)
        outputs = [StochasticOutput.from_output_samples(sample[:, i]) for i in range(results.shape[1])]
        return outputs

    def __repr__(self) -> str:
        return 'Monte Carlo'


class PolynomialChaos(UQMethod):
    """
     Polynomial chaos expansion (PCE): a surrogate model is fitted for each response as a function of the uncertain parameters.
     The trained model is then sampled with Monte Carlo to compute output statistics.

     The fitted `ot.FunctionalChaosResult` of each response is kept on the `StochasticOutput` (`method_result`) with
     Sobol sensitivity indices available through `ot.FunctionalChaosSobolIndices`.

     :param degree: total degree of the expansion
     :param n_metamodel_samples: number of cheap evaluations of the trained model used to obtain the statistics
     """

    def __init__(self, degree: int = 3, n_metamodel_samples: int = 10000, **kwargs):
        self.degree = degree
        self.n_metamodel_samples = n_metamodel_samples
        super().__init__(**kwargs)

    def n_terms(self, param_space: StochasticParameterSpace) -> int:
        """Number of terms in the expansion, i.e. the minimum number of samples needed to fit it"""
        enumerate_function = ot.LinearEnumerateFunction(param_space.n_parameters)
        return enumerate_function.getBasisSizeFromTotalDegree(self.degree)

    def validate(self, param_space: StochasticParameterSpace):
        """Check the expansion can be fitted from the budget"""
        n_terms = self.n_terms(param_space)
        if self.n_evaluations < n_terms:
            raise ValueError(f'A degree-{self.degree} expansion in {param_space.n_parameters} '
                             f'parameters has {n_terms} terms, so it needs at least that many samples to fit: '
                             f'n_evaluations = {self.n_evaluations}')

    @staticmethod
    def _draw_samples(param_space: StochasticParameterSpace, n_evaluations: int) -> np.ndarray:
        """Draw samples with LHS method"""
        return param_space.get_random_samples(n_evaluations)

    @staticmethod
    def make_polynomials(distribution: ot.JointDistribution) -> list:
        return [ot.StandardDistributionPolynomialFactory(distribution.getMarginal(i))
                for i in range(distribution.getDimension())]

    def _build_algorithm(self, param_space: StochasticParameterSpace, input_sample: ot.Sample, output_sample: ot.Sample) -> ot.FunctionalChaosAlgorithm:
        """Train surrogate model with the least-squares method"""
        self.validate(param_space)
        distribution = param_space.joint_dist
        dimension = distribution.getDimension()

        polynomials = self.make_polynomials(distribution)
        enumerate_function = ot.LinearEnumerateFunction(dimension)
        basis = ot.OrthogonalProductPolynomialFactory(polynomials, enumerate_function)

        adaptive_strategy = ot.FixedStrategy(basis, self.n_terms(param_space))
        projection_strategy = ot.LeastSquaresStrategy()
        return ot.FunctionalChaosAlgorithm(input_sample, output_sample, distribution,
                                           adaptive_strategy, projection_strategy)

    def process_results(self, results, param_space: Optional[StochasticParameterSpace]=None, *args, **kwargs) -> list[EvaluationOutput]:
        if param_space is None:
            raise RuntimeError('Parameter space not specified')

        results = np.asarray(results, dtype=float)
        input_sample = ot.Sample(self.get_samples(param_space))
        metamodel_input = ot.Sample(self.get_samples(param_space, self.n_evaluations))

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

    def __repr__(self) -> str:
        return 'Polynomial Chaos Expansion'


class AdaptiveUQMethod(UQMethod):
    def __init__(self, n_add: int=5, **kwargs):
        if n_add <= 0:
            raise ValueError('n_add must be positive')
        self.n_add = n_add
        super().__init__(**kwargs)

    @property
    def adaptive_uq_method(self) -> bool:
        return True

    @property
    def eval_kwargs(self) -> dict:
        return {}

    @staticmethod
    def _draw_samples(param_space: StochasticParameterSpace, n_evaluations: int) -> np.ndarray:
        sequence = ot.SobolSequence(param_space.n_parameters)
        return param_space.get_samples_from_qmc(sequence, n_evaluations)


class AdaptivePolynomialChaos(AdaptiveUQMethod):
    """Adaptive sparse PCE"""

    def __init__(self, init_degree: int = 1, threshold: float = 0.95,
                 n_metamodel_samples: int = 10000, hyperbolic_trunc: bool = False, max_degree: int=30,
                 max_size: Optional[float] = 1e5, q_early_stop: bool = True, **kwargs):
        self.init_degree = init_degree
        self.threshold = threshold
        self.n_metamodel_samples = n_metamodel_samples
        self.hyperbolic_trunc = hyperbolic_trunc
        self.max_size = max_size
        self.q_early_stop = q_early_stop
        self.max_degree = max_degree

        self._q_norms = [0.4, 0.5, 0.6, 0.75, 1.0]

        super().__init__(**kwargs)

    @property
    def q_norms(self):
        if self.hyperbolic_trunc:
            return self._q_norms
        return [1.0]

    @q_norms.setter
    def q_norms(self, q_norms: list[float]):
        self._q_norms = q_norms

    @property
    def eval_kwargs(self) -> dict:
        return dict(degrees=None, r2_array=None, chaos_results=None, qs=None)

    def _build_algorithm(self, param_space: StochasticParameterSpace, x_train: ot.Sample,
                         y_train: ot.Sample, degree: int, q: float,
                         polynomials: Optional[list] = None,
                         *args, **kwargs) -> ot.FunctionalChaosAlgorithm:
        """Sparse PCE"""
        distribution = param_space.joint_dist
        dimension = distribution.getDimension()
        if polynomials is None:
            polynomials = PolynomialChaos.make_polynomials(distribution)
        if self.hyperbolic_trunc:
            enumerate_function = ot.HyperbolicAnisotropicEnumerateFunction(dimension, q)
        else:
            enumerate_function = ot.LinearEnumerateFunction(dimension)
        n_terms = enumerate_function.getBasisSizeFromTotalDegree(degree)
        basis = ot.OrthogonalProductPolynomialFactory(polynomials, enumerate_function)
        adaptive_strategy = ot.FixedStrategy(basis, n_terms)

        selection = ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut())
        projection_strategy = ot.LeastSquaresStrategy(selection)
        return ot.FunctionalChaosAlgorithm(x_train, y_train, distribution,
                                           adaptive_strategy, projection_strategy)

    def _fit(self, param_space, x_train, y_obj, degree, q, polynomials):
        """Fit one candidate PCE"""
        try:
            algorithm = self._build_algorithm(param_space, x_train, y_obj, degree, q,
                                              polynomials=polynomials)
            algorithm.run()
            result = algorithm.getResult()
            history = result.getErrorHistory()
            q2 = 1.0 - min(history) if len(history) > 0 else -np.inf
            return (q2 if np.isfinite(q2) else -np.inf), result
        except Exception:
            return -np.inf, None

    def train_and_validate(self, samples: Any, outputs: Any,
                           param_space: StochasticParameterSpace, *,
                           degrees=None, r2_array=None, chaos_results=None, qs=None):
        x_train = ot.Sample(np.asarray(samples))
        y_train = ot.Sample(np.asarray(outputs).reshape(len(outputs), -1))
        n_obj = y_train.getDimension()
        n_samples, dim = x_train.getSize(), x_train.getDimension()

        # Initialize fresh evaluation lists or reuse from previous run
        if degrees is None or len(degrees) != n_obj:
            degrees = np.full(n_obj, self.init_degree, dtype=int)
        degrees = np.asarray(degrees, dtype=int).copy()

        if r2_array is None or len(r2_array) != n_obj:
            r2_array = np.full(n_obj, -np.inf)
        r2_array = np.asarray(r2_array, dtype=float).copy()

        if chaos_results is None or len(chaos_results) != n_obj:
            chaos_results = [None] * n_obj
        chaos_results = list(chaos_results)

        if qs is None or len(qs) != n_obj:
            qs = np.full(n_obj, np.nan)
        qs = np.asarray(qs, dtype=float).copy()

        # Helpers
        polynomials = PolynomialChaos.make_polynomials(param_space.joint_dist)
        q_values = np.sort(np.asarray(self.q_norms, dtype=float))
        max_size = np.inf if self.max_size is None else self.max_size
        sizes = {}

        def size_of(p, q):
            if (p, q) not in sizes:
                enum = ot.HyperbolicAnisotropicEnumerateFunction(dim, q)
                sizes[(p, q)] = enum.getBasisSizeFromTotalDegree(p)
            return sizes[(p, q)]

        # Degree and q sweep per metric
        for j in range(n_obj):
            # converged in earlier round
            if r2_array[j] >= self.threshold:
                continue

            y_obj = y_train.getMarginal(j)

            # nan policy
            if not np.all(np.isfinite(np.asarray(y_obj))):
                r2_array[j] = 1.0
                continue

            # Reuse results from previous run
            start = int(np.clip(degrees[j], self.init_degree, self.max_degree))
            best_q2, p_hist = -np.inf, []

            # Degree loop
            for degree in range(start, self.max_degree + 1):
                best_q2_p, q_hist, prev_size = -np.inf, [], None

                # q loop
                for q in q_values:
                    size = size_of(degree, q)
                    # Candidate basis too large
                    if size > max_size:
                        break

                    q2, result = self._fit(param_space, x_train, y_obj, degree, q, polynomials)
                    q_hist.append(q2)

                    best_q2_p = max(best_q2_p, q2)
                    if q2 > best_q2:
                        best_q2 = q2
                        chaos_results[j], degrees[j], qs[j] = result, degree, q
                    if best_q2 >= self.threshold:
                        break
                    if self.q_early_stop and self._dropped_twice(q_hist):
                        break

                if not q_hist or best_q2 >= self.threshold:
                    break
                p_hist.append(best_q2_p)
                if self._dropped_twice(p_hist):
                    break

            r2_array[j] = best_q2

        eval_kwargs = dict(degrees=degrees, r2_array=r2_array,
                           chaos_results=chaos_results, qs=qs)

        converged = bool(np.all(r2_array >= self.threshold))

        if converged:
            print(f'Degrees: {degrees}, q: {qs}, evaluations: {n_samples}')
            return True, x_train, eval_kwargs
        else:
            n_tot = x_train.getSize() + self.n_add
            new_samples = self.get_samples(param_space, n_tot)
            return False, new_samples, eval_kwargs

    @staticmethod
    def _dropped_twice(hist) -> bool:
        return len(hist) >= 3 and hist[-1] <= hist[-2] <= hist[-3]

    def process_results(self, results, param_space: Optional[StochasticParameterSpace]=None,
                        chaos_results: Optional[List[Optional[ot.FunctionalChaosResult]]]=None, *args, **kwargs) -> list[EvaluationOutput]:

        if param_space is None or chaos_results is None:
            raise ValueError('param_space and results are required')

        metamodel_input = self.get_samples(param_space, self.n_metamodel_samples)

        outputs = []
        for result in chaos_results:
            # Execute nan policy
            if result is None:
                samples = ot.Sample(np.full((self.n_metamodel_samples, 1), np.nan))
            else:
                samples = result.getMetaModel()(metamodel_input)

            outputs.append(StochasticOutput.from_output_samples(samples, result))
        return outputs

    def __repr__(self) -> str:
        return 'Adaptive Polynomial Chaos Expansion'