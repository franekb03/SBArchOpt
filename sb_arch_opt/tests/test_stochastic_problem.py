import pytest
import numpy as np
import openturns as ot
from pymoo.core.variable import Real, Choice

from sb_arch_opt.stochastic_problem import StochasticArchOptProblem
from sb_arch_opt.uncertainty import *
from sb_arch_opt.problems.robust_optimization.rosenbrock import StochasticRosenbrock


class VectorizedProblem(StochasticArchOptProblem):
    """Only implements _arch_evaluate_sample, vectorized over all design points"""

    def __init__(self, n=100, seed=42, uq_method=None, **kwargs):
        # A supplied method already carries the space it was built over; the problem must use that same one
        if uq_method is None:
            param_space = self.get_parameter_space()
            uq_method = MonteCarlo(param_space, n_evaluations=n, seed=seed)

        super().__init__([Real(bounds=(-2., 2.)), Real(bounds=(-2., 2.))],
                         uq_method=uq_method, n_obj=1, **kwargs)

    @staticmethod
    def get_parameter_space():
        param_space = StochasticParameterSpace()
        param_space.add_parameter(InputParameter('u0', ot.Normal(1., .05)))
        return param_space

    def _is_conditionally_active(self):
        return [False, False]

    def _correct_x(self, x, is_active):
        pass

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs):
        f_out[:, 0] = (parameters[0] - x[:, 0]) ** 2 + x[:, 1] ** 2


class HierarchicalProblem(StochasticArchOptProblem):
    """Hierarchical (implicit) design space, a constraint, and the per-row evaluation pattern"""

    def __init__(self, n=50, seed=1, **kwargs):
        param_space = StochasticParameterSpace()
        param_space.add_parameter(InputParameter('payload', ot.Normal(2., .3)))
        param_space.add_parameter(InputParameter('bsfc', ot.Normal(.42, .075)))
        super().__init__([Choice(options=['electric', 'hybrid']), Real(bounds=(.2, 1.)), Real(bounds=(.1, .4))],
                         uq_method=MonteCarlo(param_space, n_evaluations=n, seed=seed),
                         n_obj=1, n_ieq_constr=1, **kwargs)

    def _is_conditionally_active(self):
        return [False, False, True]

    def _correct_x(self, x, is_active):
        is_active[:, 2] = x[:, 0] == 1  # fuel fraction only active for the hybrid branch

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs):
        payload, bsfc = parameters
        for i in range(x.shape[0]):  # per-row pattern, as used by several problems in the repo
            mass = .5 + x[i, 1] + payload
            energy = x[i, 2]*130./(bsfc+.1) if x[i, 0] == 1 else x[i, 1]*40.
            f_out[i, 0] = mass**1.5 / energy
            g_out[i, 0] = mass - 3.5


"""### Parameters and parameter space ###"""


def test_parameter_mean_and_std():
    param = InputParameter('u', ot.Normal(5., .3))
    assert param.mean() == pytest.approx(5.)
    assert param.std() == pytest.approx(.3)  # not the mean: getParameter()[0] would give 5.


@pytest.mark.parametrize('value', [1.225, 5, np.float64(2.5)])
def test_deterministic_parameter_stays_a_plain_value(value):
    """A deterministic parameter is kept as a number rather than wrapped in a distribution: it has no orthonormal
    polynomial family to contribute to a chaos basis, so it cannot be a marginal of the joint distribution. Note
    an int is accepted too - it is the most natural way to write a constant"""
    param = InputParameter('rho', value)

    assert isinstance(param.value, float)
    assert param.mean() == pytest.approx(float(value))
    assert param.std() == pytest.approx(0.)


def test_space_separates_stochastic_from_deterministic_parameters():
    space = StochasticParameterSpace()
    u = InputParameter('u', ot.Normal(0., 1.))
    rho = InputParameter('rho', 1.225)
    space.add_parameter(u)
    space.add_parameter(rho)

    assert space.parameters == [u, rho]
    assert space.stochastic_parameters == [u]
    assert space.deterministic_parameters == [rho]
    assert space.n_parameters == 2
    assert space.n_stochastic_parameters == 1


def test_deterministic_parameters_are_reinserted_into_a_realization():
    """Only the stochastic parameters are drawn; the constants are put back so that a realization covers every
    parameter of the space and can be indexed by position"""
    space = StochasticParameterSpace()
    space.add_parameter(InputParameter('u', ot.Normal(0., 1.)))
    space.add_parameter(InputParameter('rho', 1.225))

    samples = space.get_random_samples(20)
    assert samples.shape == (20, 1)  # the Dirac column is not drawn at all
    assert samples[:, 0].std() > 0.

    extended = space.include_deterministic_values(samples)
    assert extended.shape == (20, 2)
    assert np.all(extended[:, 1] == 1.225)
    assert np.allclose(extended[:, 0], samples[:, 0])


def test_deterministic_values_are_reinserted_in_the_right_columns():
    """The constants can sit anywhere in the space, so the stochastic columns have to be shifted past them"""
    space = StochasticParameterSpace()
    space.add_parameter(InputParameter('d0', 1.))
    space.add_parameter(InputParameter('s0', ot.Normal(10., 1.)))
    space.add_parameter(InputParameter('d1', 2.))
    space.add_parameter(InputParameter('s1', ot.Normal(20., 1.)))

    samples = space.get_random_samples(8)
    assert samples.shape == (8, 2)

    extended = space.include_deterministic_values(samples)
    assert extended.shape == (8, 4)
    assert np.all(extended[:, 0] == 1.)
    assert np.all(extended[:, 2] == 2.)
    assert np.allclose(extended[:, 1], samples[:, 0])
    assert np.allclose(extended[:, 3], samples[:, 1])


def test_parameter_space_joint_dist():
    space = StochasticParameterSpace()
    space.add_parameter(InputParameter('a', ot.Normal(0., 1.)))
    space.add_parameter(InputParameter('b', ot.Uniform(0., 1.)))
    space.add_parameter(InputParameter('c', 2.0))

    assert space.n_parameters == 3
    assert space.n_stochastic_parameters == 2
    assert space.parameter_names == ['a', 'b', 'c']
    assert space.joint_dist.getDimension() == 2

    samples = space.get_random_samples(20)
    samples_with_deterministic = space.include_deterministic_values(samples)
    assert samples.shape == (20, 2)
    assert samples_with_deterministic.shape == (20, 3)


def test_parameter_space_lhs_samples_are_stratified():
    """An LHS puts exactly one point in each of the n equiprobable strata of every marginal; a random design
    does not, which is what distinguishes the two draws the space offers"""
    space = _space(2)

    lhs, random = space.get_lhs_samples(40), space.get_random_samples(40)
    assert lhs.shape == random.shape == (40, 2)

    assert all(_n_occupied_strata(space, lhs, i) == 40 for i in range(2))
    assert any(_n_occupied_strata(space, random, i) < 40 for i in range(2))


def _n_occupied_strata(space, samples, i_param):
    """How many of the n equiprobable strata of marginal `i_param` contain at least one sample"""
    n = samples.shape[0]
    cdf = np.array([space.parameters[i_param].value.computeCDF(v) for v in samples[:, i_param]])
    return len(np.unique(np.floor(cdf*n).astype(int)))


"""### Reduction ###"""


def _output(values):
    return StochasticOutput(ot.Sample(np.array(values, dtype=float).reshape((-1, 1))))


def _space(n=1):
    space = StochasticParameterSpace()
    for i in range(n):
        space.add_parameter(InputParameter(f'u{i}', ot.Normal(0., 1.)))
    return space


def test_reduce_mean():
    assert _output([1., 2., 3.]).reduce(Mean()) == pytest.approx(2.)


def test_reduce_margin_matches_margin_method():
    out = _output([1., 2., 3., 4.])
    assert out.reduce(Margin(k=2.)) == pytest.approx(out.margin(2.))


def test_reduce_quantile_matches_quantile_method():
    out = _output(np.linspace(0., 10., 101))
    assert out.reduce(Quantile(q=.9)) == pytest.approx(out.quantile(.9))


def test_measure_parameters_validated_at_construction():
    """A bad configuration should fail when the problem is built, not on the first evaluation"""
    with pytest.raises(ValueError):
        Quantile(q=2.)


def test_margin_direction_follows_the_optimization_direction():
    """The margin is the conservative value, so which tail that is depends on the direction: for a minimized
    response it is mean + k*std, for a maximized one mean - k*std"""
    out = _output([1., 2., 3., 4., 5.])
    mean, std = out.mean(), out.std()

    assert out.reduce(Margin(k=2.)) == pytest.approx(mean + 2.*std)  # minimization is the default
    assert out.reduce(Margin(k=2., direction=-1)) == pytest.approx(mean + 2.*std)
    assert out.reduce(Margin(k=2., direction=1)) == pytest.approx(mean - 2.*std)


def test_margin_penalizes_spread_in_both_directions():
    """Same mean, less scatter -> a better value whichever way the response is optimized"""
    wide, narrow = _output([1., 3., 5.]), _output([2.5, 3., 3.5])
    assert wide.mean() == pytest.approx(narrow.mean())

    assert narrow.reduce(Margin(k=2.)) < wide.reduce(Margin(k=2.))  # minimized: lower is better
    assert narrow.reduce(Margin(k=2., direction=1)) > wide.reduce(Margin(k=2., direction=1))  # maximized


def test_measures_carry_their_own_parameters():
    """Two measures of the same kind but different parameters reduce the same samples differently"""
    out = _output(np.linspace(0., 10., 101))
    assert out.reduce(Quantile(q=.5)) != out.reduce(Quantile(q=.95))
    assert out.reduce(Margin(k=1.)) != out.reduce(Margin(k=3.))


def test_custom_measure_needs_no_library_change():
    """The point of the RobustMeasure hierarchy: a new measure is one class, wired in nowhere"""
    class WorstCase(RobustMeasure):
        def reduce(self, samples):
            return float(samples.getMax()[0])

    assert _output([1., 5., 3.]).reduce(WorstCase()) == pytest.approx(5.)


def test_reduce_rejects_unknown_nan_policy():
    with pytest.raises(ValueError):
        _output([1., 2.]).reduce(Mean(), nan_policy='whatever')


"""### UQ methods ###"""


def test_uq_method_is_constructed_over_a_parameter_space():
    """A method is defined by what it propagates: the space is bound at construction, not passed per call"""
    space = _space(2)
    method = MonteCarlo(space, n_evaluations=10)

    assert method.param_space is space
    assert method.n_samples == 10
    assert method.get_samples().shape == (10, 2)


def test_uq_method_needs_a_parameter_space():
    with pytest.raises(ValueError, match='parameter space'):
        MonteCarlo(None, n_evaluations=10)

    with pytest.raises(ValueError, match='parameter space'):
        MonteCarlo(StochasticParameterSpace(), n_evaluations=10)


def test_uq_method_needs_a_number_of_evaluations():
    with pytest.raises(ValueError, match='n_evaluations'):
        MonteCarlo(_space(), n_evaluations=None)


@pytest.mark.parametrize('method_class', [MonteCarlo, PolynomialChaos])
def test_uq_methods_draw_a_latin_hypercube(method_class):
    """Both methods use an LHS rather than a plain random draw: it covers the parameter space more evenly for the
    same number of expensive evaluations, which matters both for averaging over and for fitting an expansion"""
    space = _space(2)
    samples = method_class(space, n_evaluations=40, seed=42).get_samples()

    assert samples.shape == (40, 2)
    assert all(_n_occupied_strata(space, samples, i) == 40 for i in range(2))


def test_process_results_gives_one_output_per_column():
    method = MonteCarlo(_space(), n_evaluations=3)

    result = method.process_results(np.array([[1., 10., 20.], [2., 11., 21.], [3., 12., 22.]]))
    assert len(result.outputs) == 3
    assert result.outputs[0].mean() == pytest.approx(2.)
    assert result.outputs[1].mean() == pytest.approx(11.)
    assert result.outputs[2].mean() == pytest.approx(21.)


def test_process_results_checks_the_number_of_sample_rows():
    method = MonteCarlo(_space(), n_evaluations=3)
    with pytest.raises(ValueError):
        method.process_results(np.array([[1.], [2.]]))


"""### The evaluation loop ###"""


def test_user_only_implements_arch_evaluate_sample():
    """The whole point: a problem implements _arch_evaluate_sample and gets real objective values back."""
    problem = VectorizedProblem(n=200)
    out = problem.evaluate(np.array([[1., 0.], [0., 0.]]), return_as_dictionary=True)

    assert out['F'].shape == (2, 1)
    assert np.all(np.isfinite(out['F']))
    # E[(u-x)^2] + x1^2 = (E[u]-x0)^2 + var + x1^2
    assert out['F'][0, 0] == pytest.approx(.05**2, abs=2e-3)
    assert out['F'][1, 0] == pytest.approx(1. + .05**2, abs=2e-2)


def test_evaluate_sample_receives_one_realization_per_sample():
    """The hook is called once per row of the method's design, and gets that row - not an index into it"""
    problem = VectorizedProblem(n=25)
    seen = []

    original = problem._arch_evaluate_sample

    def _recording(x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs):
        seen.append(np.asarray(parameters).copy())
        return original(x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs)

    problem._arch_evaluate_sample = _recording
    problem.evaluate(np.array([[1., 0.], [0., 0.]]), return_as_dictionary=True)

    assert len(seen) == 25  # once per sample, not once per design point
    assert all(realization.shape == (1,) for realization in seen)  # one row, not the whole design
    assert len({tuple(realization) for realization in seen}) == 25  # and a different row each time
    assert np.allclose(np.array(seen), problem.uq_method.get_samples())


class MixedParameterProblem(StochasticArchOptProblem):
    """One stochastic and one deterministic parameter, to pin the shape of a realization"""

    def __init__(self, n=10, **kwargs):
        space = StochasticParameterSpace()
        space.add_parameter(InputParameter('u', ot.Normal(1., .05)))
        space.add_parameter(InputParameter('rho', 1.225))
        self.seen = []
        super().__init__([Real(bounds=(-2., 2.))], uq_method=MonteCarlo(space, n_evaluations=n, seed=42),
                         n_obj=1, **kwargs)

    def _is_conditionally_active(self):
        return [False]

    def _correct_x(self, x, is_active):
        pass

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs):
        self.seen.append(np.asarray(parameters).copy())
        f_out[:, 0] = parameters[0]*parameters[1] + x[:, 0]


def test_evaluate_sample_realization_covers_every_parameter():
    """Regression: the whole (n_samples x n_parameters) design was handed over on every call instead of one row,
    so the loop index went unused and every sample evaluated identically"""
    problem = MixedParameterProblem(n=10)
    out = problem.evaluate(np.array([[0.]]), return_as_dictionary=True)

    assert len(problem.seen) == 10
    for realization in problem.seen:
        assert realization.shape == (2,)  # one value per parameter of the space, deterministic included
        assert realization[1] == 1.225

    # The stochastic parameter really varies, so the response does too
    assert len({realization[0] for realization in problem.seen}) == 10
    assert out['stochastic'][0].outputs[0].std() > 0.
    assert out['F'][0, 0] == pytest.approx(1.*1.225, abs=.05)


def test_multiple_design_points_at_once():
    """The design point dimension must be handled: an ot.PythonFunction-style driver cannot do this."""
    problem = VectorizedProblem(n=20)
    x = np.random.RandomState(0).uniform(-2., 2., size=(7, 2))
    out = problem.evaluate(x, return_as_dictionary=True)
    assert out['F'].shape == (7, 1)
    assert np.all(np.isfinite(out['F']))


def test_evaluation_is_deterministic():
    """Common random numbers: F must be a deterministic function of x, or a surrogate fits Monte Carlo noise."""
    problem = VectorizedProblem(n=50, seed=7)
    x = np.array([[.5, .5]])
    assert np.allclose(problem.evaluate(x, return_as_dictionary=True)['F'],
                       problem.evaluate(x, return_as_dictionary=True)['F'])


def test_seed_is_applied():
    x = np.array([[.5, .5]])
    f_a = VectorizedProblem(n=50, seed=3).evaluate(x, return_as_dictionary=True)['F']
    f_b = VectorizedProblem(n=50, seed=3).evaluate(x, return_as_dictionary=True)['F']
    f_c = VectorizedProblem(n=50, seed=4).evaluate(x, return_as_dictionary=True)['F']

    assert np.allclose(f_a, f_b)  # same seed -> same result
    assert not np.allclose(f_a, f_c)  # different seed -> different result


def test_resample_draws_new_samples():
    problem = VectorizedProblem(n=50, seed=3)
    x = np.array([[.5, .5]])
    f_a = problem.evaluate(x, return_as_dictionary=True)['F'].copy()
    problem.uq_method.resample()
    problem.uq_method.seed = None
    assert not np.allclose(f_a, problem.evaluate(x, return_as_dictionary=True)['F'])


def test_correction_runs_for_implicit_design_space():
    """Correction only happens up front for explicit design spaces, so the loop has to do it: otherwise the
    evaluation function sees an all-True activeness matrix and un-imputed design vectors."""
    problem = HierarchicalProblem()
    assert not problem.design_space.is_explicit()

    out = problem.evaluate(np.array([[0, .6, .25], [1, .6, .25]]), return_as_dictionary=True)
    assert np.all(out['is_active'] == np.array([
        [True, True, False],   # electric: fuel fraction inactive
        [True, True, True],    # hybrid: active
    ]))


def test_hierarchical_problem_with_constraint():
    problem = HierarchicalProblem(obj_measure=[Margin(k=2.)],
                                  ieq_constr_measure=[Quantile(q=.95)])
    out = problem.evaluate(np.array([[0, .6, .25], [1, .6, .25]]), return_as_dictionary=True)

    assert out['F'].shape == (2, 1)
    assert out['G'].shape == (2, 1)
    assert np.all(np.isfinite(out['F']))
    assert np.all(np.isfinite(out['G']))


class AllResponseKindsProblem(StochasticArchOptProblem):
    """One objective, one inequality constraint and one equality constraint, each with a different measure"""

    def __init__(self, **kwargs):
        param_space = StochasticParameterSpace()
        param_space.add_parameter(InputParameter('u', ot.Normal(1., .2)))
        kwargs.setdefault('obj_measure', [Mean()])
        kwargs.setdefault('ieq_constr_measure', [Mean()])
        kwargs.setdefault('eq_constr_measure', [Margin(k=3.)])
        super().__init__([Real(bounds=(0., 1.))],
                         uq_method=MonteCarlo(param_space, n_evaluations=200, seed=5),
                         n_obj=1, n_ieq_constr=1, n_eq_constr=1, **kwargs)

    def _is_conditionally_active(self):
        return [False]

    def _correct_x(self, x, is_active):
        pass

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs):
        f_out[:, 0] = parameters[0] + x[:, 0]
        g_out[:, 0] = parameters[0] - x[:, 0]
        h_out[:, 0] = parameters[0] * x[:, 0]


def test_equality_constraints_use_their_own_measure():
    """Regression: the equality-constraint loop used to index the inequality-constraint parameter list, so H was
    reduced with the wrong measure (or raised IndexError when the two lists differed in length)."""
    problem = AllResponseKindsProblem()
    out = problem.evaluate(np.array([[.5]]), return_as_dictionary=True)

    result = out['stochastic'][0]
    f_output, g_output, h_output = result.outputs  # one objective, one inequality and one equality constraint
    # H is declared as Margin(k=3.) while G is Mean(): the two must not agree
    assert out['H'][0, 0] == pytest.approx(h_output.margin(3.))
    assert out['G'][0, 0] == pytest.approx(g_output.mean())
    assert out['H'][0, 0] != pytest.approx(h_output.mean())
    assert out['F'][0, 0] == pytest.approx(f_output.mean())


def test_measure_counts_checked_per_response_kind():
    with pytest.raises(ValueError):
        AllResponseKindsProblem(eq_constr_measure=[Mean(), Mean()])


"""### Statistics carried in the output ###"""


def test_statistics_available_per_design_point():
    problem = VectorizedProblem(n=100)
    out = problem.evaluate(np.array([[1., 0.], [0., 0.]]), return_as_dictionary=True)

    assert len(out['stochastic']) == 2
    for result in out['stochastic']:
        assert isinstance(result, StochasticResults)
        assert len(result.outputs) == 1
        assert len(result.outputs[0].to_numpy()) == 100


def test_reported_statistics_reproduce_the_reduced_value():
    """The statistics handed back must be exactly what the optimizer saw, not a differently-estimated version."""
    problem = HierarchicalProblem(obj_measure=[Margin(k=2.)],
                                  ieq_constr_measure=[Quantile(q=.95)])
    out = problem.evaluate(np.array([[1, .6, .25]]), return_as_dictionary=True)

    result = out['stochastic'][0]
    assert result.outputs[0].margin(2.) == pytest.approx(out['F'][0, 0])
    assert result.outputs[1].quantile(.95) == pytest.approx(out['G'][0, 0])


"""### Hidden constraints ###"""


class FailingProblem(VectorizedProblem):
    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs):
        f_out[:, 0] = x[:, 0] + parameters[0]
        if parameters[0] > 1.02:  # some samples fail to evaluate
            f_out[:, 0] = np.nan


def test_nan_policy_propagate_fails_the_design_point():
    problem = FailingProblem(n=50, seed=3, nan_policy='propagate')
    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)

    assert not np.isfinite(out['F'][0, 0])
    assert problem.get_failed_points(out)[0]


def test_nan_policy_omit_reduces_over_surviving_samples():
    problem = FailingProblem(n=50, seed=3, nan_policy='omit')
    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)

    values = out['stochastic'][0].outputs[0].to_numpy()
    assert np.any(~np.isfinite(values))  # some samples did fail
    assert np.isfinite(out['F'][0, 0])  # but the design point still has a value
    assert out['F'][0, 0] == pytest.approx(np.mean(values[np.isfinite(values)]))


"""### Configuration checks ###"""


def test_obj_measure_count_checked():
    space = _space()
    with pytest.raises(ValueError):
        StochasticArchOptProblem([Real(bounds=(0., 1.))],
                                 uq_method=MonteCarlo(space, n_evaluations=10), n_obj=1,
                                 obj_measure=[Mean(), Mean()])


def test_ieq_constr_measure_checked_against_n_ieq_constr():
    """Regression: this used to be validated against n_obj"""
    with pytest.raises(ValueError):
        HierarchicalProblem(ieq_constr_measure=[Mean(), Mean()])

    # a matching number of constraint measures is accepted
    HierarchicalProblem(ieq_constr_measure=[Mean()])


def test_measures_must_be_robust_measure_instances():
    with pytest.raises(ValueError):
        HierarchicalProblem(ieq_constr_measure=['quantile'])


def test_uq_method_required():
    with pytest.raises(ValueError):
        StochasticArchOptProblem([Real(bounds=(0., 1.))],
                                 uq_method=None, n_obj=1)

    with pytest.raises(ValueError):
        StochasticArchOptProblem([Real(bounds=(0., 1.))],
                                 uq_method='monte carlo', n_obj=1)


"""### Polynomial chaos ###"""


class QuadraticProblem(StochasticArchOptProblem):
    """f = (u - x0)^2 + x1^2: at x = (mean(u), 0) the expected value is exactly Var[u].
    Quadratic in u, so a polynomial chaos expansion represents it essentially exactly."""

    def __init__(self, uq_method_class=MonteCarlo, n=50, fail=False, method_kwargs=None, **kwargs):
        param_space = StochasticParameterSpace()
        param_space.add_parameter(InputParameter('u', ot.Normal(1., .05)))
        self.fail = fail
        super().__init__([Real(bounds=(-2., 2.)), Real(bounds=(-2., 2.))],
                         uq_method=uq_method_class(param_space, n_evaluations=n, seed=42,
                                                   **(method_kwargs or {})),
                         n_obj=1, obj_measure=[Mean()], **kwargs)

    def _is_conditionally_active(self):
        return [False, False]

    def _correct_x(self, x, is_active):
        pass

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, parameters, *args, **kwargs):
        f_out[:, 0] = (parameters[0] - x[:, 0]) ** 2 + x[:, 1] ** 2
        if self.fail and parameters[0] > 1.02:
            f_out[:, 0] = np.nan


def test_pce_is_selected_through_the_method_registry():
    problem = QuadraticProblem(uq_method_class=PolynomialChaos, n=50)
    assert isinstance(problem.uq_method, PolynomialChaos)
    assert isinstance(QuadraticProblem(n=50).uq_method, MonteCarlo)


def test_pce_kwargs_are_passed_through():
    problem = QuadraticProblem(uq_method_class=PolynomialChaos, n=50,
                               method_kwargs={'degree': 4, 'n_metamodel_samples': 500})
    assert problem.uq_method.degree == 4
    assert problem.uq_method.n_metamodel_samples == 500


def test_pce_n_terms():
    """The number of terms follows from the degree and the parameter space the method was built over"""
    assert PolynomialChaos(_space(3), n_evaluations=200, degree=8).n_terms == 165  # degree 8 in 3 dimensions
    assert PolynomialChaos(_space(1), n_evaluations=200, degree=8).n_terms == 9  # degree 8 in 1 dimension


def test_pce_needs_enough_samples_to_fit_the_expansion():
    """A degree-8 expansion in 3 parameters has 165 terms, so 10 samples cannot fit it. Both the degree and the
    space are known at construction, so this fails there rather than at the first evaluation"""
    with pytest.raises(ValueError, match='165 terms'):
        PolynomialChaos(_space(3), n_evaluations=10, degree=8)





def test_pce_is_more_accurate_than_monte_carlo():
    """The point of PCE: far better statistics for the same number of expensive evaluations. Both methods draw
    the same kind of LHS design, so what is compared here is fitting an expansion against averaging over it."""
    x = np.array([[1., 0.]])
    exact = .05**2  # E[(u-x)^2] at x = E[u] is Var[u]

    f_mc = QuadraticProblem(MonteCarlo, n=20).evaluate(x, return_as_dictionary=True)['F'][0, 0]
    f_pce = QuadraticProblem(PolynomialChaos, n=20).evaluate(x, return_as_dictionary=True)['F'][0, 0]

    assert abs(f_pce - exact) < abs(f_mc - exact)
    assert f_pce == pytest.approx(exact, abs=1e-4)


def test_pce_is_deterministic():
    x = np.array([[.5, .5]])
    problem = QuadraticProblem(PolynomialChaos, n=50)
    assert np.allclose(problem.evaluate(x, return_as_dictionary=True)['F'],
                       problem.evaluate(x, return_as_dictionary=True)['F'])


def test_pce_provides_chaos_result_for_sensitivity_analysis():
    """The fitted expansion is kept, so Sobol indices are available for free"""
    problem = QuadraticProblem(PolynomialChaos, n=50)
    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)

    chaos_results = out['stochastic'][0].method_result
    assert len(chaos_results) == 1
    assert isinstance(chaos_results[0], ot.FunctionalChaosResult)

    sobol = ot.FunctionalChaosSobolIndices(chaos_results[0])
    assert sobol.getSobolIndex(0) == pytest.approx(1., abs=1e-6)  # single parameter explains everything


def test_pce_falls_back_to_raw_samples_when_evaluations_fail():
    """A response with failed evaluations cannot be fitted; nan_policy then decides as usual"""
    x = np.array([[.5, .5]])

    problem = QuadraticProblem(PolynomialChaos, n=50, fail=True, nan_policy='propagate')
    out = problem.evaluate(x, return_as_dictionary=True)
    assert out['stochastic'][0].method_result[0] is None  # not fitted
    assert not np.isfinite(out['F'][0, 0])
    assert problem.get_failed_points(out)[0]

    problem = QuadraticProblem(PolynomialChaos, n=50, fail=True, nan_policy='omit')
    out = problem.evaluate(x, return_as_dictionary=True)
    assert np.isfinite(out['F'][0, 0])


def _mixed_space(*values):
    space = StochasticParameterSpace()
    for i, value in enumerate(values):
        space.add_parameter(InputParameter(f'p{i}', value))
    return space


def test_pce_counts_terms_over_the_stochastic_parameters_only():
    """Regression: n_terms counted every parameter while the basis holds only the stochastic ones, so a
    deterministic parameter silently raised the degree - here degree 2 asked a 1-D basis for 6 terms, which is a
    degree-5 expansion - and inflated the sample requirement with it"""
    assert PolynomialChaos(_mixed_space(ot.Normal(1., .05)), n_evaluations=40, degree=2).n_terms == 3
    assert PolynomialChaos(_mixed_space(ot.Normal(1., .05), 1.225), n_evaluations=40, degree=2).n_terms == 3


def test_pce_needs_enough_samples_for_the_stochastic_parameters_only():
    """A deterministic parameter adds no terms, so it must not inflate the sample requirement either"""
    space = _mixed_space(ot.Normal(0., 1.), 1., 2., 3.)

    PolynomialChaos(space, n_evaluations=9, degree=8)  # degree 8 in 1 stochastic dimension: 9 terms

    with pytest.raises(ValueError, match='9 terms'):
        PolynomialChaos(space, n_evaluations=8, degree=8)


def test_pce_fits_a_response_with_a_deterministic_parameter():
    """Regression: a Dirac marginal broke the chaos basis outright (NotYetImplementedException from the enumerate
    function). Deterministic parameters are now kept out of the joint distribution entirely"""
    space = _mixed_space(ot.Normal(1., .05), 1.225)
    method = PolynomialChaos(space, n_evaluations=40, seed=42, degree=2)

    samples = method.get_samples()
    assert samples.shape == (40, 1)  # only the stochastic parameter is drawn

    values = space.include_deterministic_values(samples)
    result = method.process_results((values[:, 0]**2 + values[:, 1]).reshape((-1, 1)))

    chaos_result = result.method_result[0]
    assert chaos_result is not None  # actually fitted, not passed through
    assert chaos_result.getMetaModel().getInputDimension() == 1  # the constant is not an input
    assert chaos_result.getCoefficients().getSize() == 3  # degree 2 in one dimension
    assert result.outputs[0].mean() == pytest.approx(1. + .05**2 + 1.225, abs=1e-3)


def test_pce_handles_a_deterministic_parameter_in_any_column():
    """The constants can sit anywhere in the space, so which columns are stochastic must not matter"""
    space = _mixed_space(2.5, ot.Normal(1., .05))
    method = PolynomialChaos(space, n_evaluations=40, seed=42, degree=2)

    values = space.include_deterministic_values(method.get_samples())
    result = method.process_results((values[:, 1]**2 + values[:, 0]).reshape((-1, 1)))

    assert result.method_result[0] is not None
    assert result.outputs[0].mean() == pytest.approx(1. + .05**2 + 2.5, abs=1e-3)


def test_pce_sobol_indices_are_over_the_stochastic_parameters():
    """The expansion is built over the stochastic parameters only, so Sobol index i refers to the i-th of those
    rather than to parameter i of the space"""
    space = _mixed_space(ot.Normal(1., .3), 5.0, ot.Normal(2., .1))
    method = PolynomialChaos(space, n_evaluations=60, seed=42, degree=2)

    values = space.include_deterministic_values(method.get_samples())
    result = method.process_results((values[:, 0]**2).reshape((-1, 1)))  # only p0 matters

    chaos_result = result.method_result[0]
    assert chaos_result.getMetaModel().getInputDimension() == 2  # the constant is not an input

    sobol = ot.FunctionalChaosSobolIndices(chaos_result)
    assert sobol.getSobolIndex(0) == pytest.approx(1., abs=1e-6)  # p0
    assert sobol.getSobolIndex(1) == pytest.approx(0., abs=1e-6)  # p2, not the constant


def test_problem_rejects_an_entirely_deterministic_space():
    """Nothing varies, so there is no uncertainty to propagate and the problem is a deterministic one"""
    space = _mixed_space(1.5, 2.5)

    with pytest.raises(ValueError, match='stochastic parameters'):
        StochasticArchOptProblem([Real(bounds=(0., 1.))],
                                 uq_method=MonteCarlo(space, n_evaluations=10), n_obj=1)


def test_pce_uses_all_samples_it_asked_for():
    problem = QuadraticProblem(PolynomialChaos, n=40)
    assert problem.uq_method.get_samples().shape == (40, 1)

    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)
    # statistics come from the cheap metamodel, not from the 40 expensive evaluations
    assert len(out['stochastic'][0].outputs[0].to_numpy()) == problem.uq_method.n_metamodel_samples


"""### Test problem ###"""


def test_stochastic_rosenbrock():
    problem = StochasticRosenbrock(n_var=3, std=.05, n=500)
    assert problem.n_var == 3
    assert problem.param_space.n_parameters == 2

    # At the optimum the expected objective equals the sum of the parameter variances
    out = problem.evaluate(problem.pareto_set(), return_as_dictionary=True)
    assert out['F'][0, 0] == pytest.approx(problem.pareto_front()[0, 0], abs=1e-3)

    # Moving away from the optimum makes it worse
    f_off = problem.evaluate(np.array([[.5, .5, .5]]), return_as_dictionary=True)['F']
    assert f_off[0, 0] > out['F'][0, 0]

def test_samples_are_drawn_once_and_reused():
    """Common random numbers: every design point sees the same realizations"""
    method = MonteCarlo(_space(), n_evaluations=20, seed=42)
    assert np.all(method.get_samples() == method.get_samples())


def test_problem_exposes_the_bound_parameter_space():
    """The problem has no parameter space of its own: it reads the one its method propagates, so the two cannot
    disagree"""
    problem = VectorizedProblem(n=10)

    assert problem.uq_method.param_space is problem.param_space
    assert problem.param_space.parameter_names == ['u0']

    with pytest.raises(AttributeError):
        problem.param_space = _space()


def test_problem_rejects_a_method_without_parameters():
    """UQMethod already refuses an empty space at construction, so this only bites when one is swapped in
    afterwards - but that is exactly when it would otherwise fail deep inside the first evaluation"""
    method = MonteCarlo(_space(), n_evaluations=10)
    method.param_space = StochasticParameterSpace()

    with pytest.raises(ValueError, match='parameter space'):
        StochasticArchOptProblem([Real(bounds=(0., 1.))], uq_method=method, n_obj=1)