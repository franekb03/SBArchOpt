import pytest
import numpy as np
import openturns as ot
from pymoo.core.variable import Real, Choice

from sb_arch_opt.robust import StochasticArchOptProblem
from sb_arch_opt.uncertainty import *
from sb_arch_opt.problems.robust_optimization.rosenbrock import StochasticRosenbrock


class VectorizedProblem(StochasticArchOptProblem):
    """Only implements _arch_evaluate_sample, vectorized over all design points"""

    def __init__(self, n=100, seed=42, uq_method=None, **kwargs):
        param_space = StochasticParameterSpace()
        param_space.add_parameter(StochasticParameter('u0', ot.Normal(1., .05)))
        super().__init__([Real(bounds=(-2., 2.)), Real(bounds=(-2., 2.))], param_space=param_space,
                         uq_method=uq_method if uq_method is not None else MonteCarlo(n_evaluations=n, seed=seed),
                         n_obj=1, **kwargs)

    def _is_conditionally_active(self):
        return [False, False]

    def _correct_x(self, x, is_active):
        pass

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, *args, sample, **kwargs):
        f_out[:, 0] = (sample[0] - x[:, 0])**2 + x[:, 1]**2


class HierarchicalProblem(StochasticArchOptProblem):
    """Hierarchical (implicit) design space, a constraint, and the per-row evaluation pattern"""

    def __init__(self, n=50, seed=1, **kwargs):
        param_space = StochasticParameterSpace()
        param_space.add_parameter(StochasticParameter('payload', ot.Normal(2., .3)))
        param_space.add_parameter(StochasticParameter('bsfc', ot.Normal(.42, .075)))
        super().__init__([Choice(options=['electric', 'hybrid']), Real(bounds=(.2, 1.)), Real(bounds=(.1, .4))],
                         param_space=param_space, uq_method=MonteCarlo(n_evaluations=n, seed=seed),
                         n_obj=1, n_ieq_constr=1, **kwargs)

    def _is_conditionally_active(self):
        return [False, False, True]

    def _correct_x(self, x, is_active):
        is_active[:, 2] = x[:, 0] == 1  # fuel fraction only active for the hybrid branch

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, *args, sample, **kwargs):
        payload, bsfc = sample
        for i in range(x.shape[0]):  # per-row pattern, as used by several problems in the repo
            mass = .5 + x[i, 1] + payload
            energy = x[i, 2]*130./(bsfc+.1) if x[i, 0] == 1 else x[i, 1]*40.
            f_out[i, 0] = mass**1.5 / energy
            g_out[i, 0] = mass - 3.5


"""### Parameters and parameter space ###"""


def test_parameter_mean_and_std():
    param = StochasticParameter('u', ot.Normal(5., .3))
    assert param.mean() == pytest.approx(5.)
    assert param.std() == pytest.approx(.3)  # not the mean: getParameter()[0] would give 5.


def test_parameter_space_joint_dist():
    space = StochasticParameterSpace()
    space.add_parameter(StochasticParameter('a', ot.Normal(0., 1.)))
    space.add_parameter(StochasticParameter('b', ot.Uniform(0., 1.)))

    assert space.n_parameters == 2
    assert space.parameter_names == ['a', 'b']
    assert space.joint_dist.getDimension() == 2

    samples = space.get_samples(20)
    assert samples.shape == (20, 2)


"""### Reduction ###"""


def _output(values, measure):
    return StochasticOutput(ot.Sample(np.array(values, dtype=float).reshape((-1, 1))), measure)


def _space(n=1):
    space = StochasticParameterSpace()
    for i in range(n):
        space.add_parameter(StochasticParameter(f'u{i}', ot.Normal(0., 1.)))
    return space


def test_reduce_mean():
    assert _output([1., 2., 3.], Mean()).reduce() == pytest.approx(2.)


def test_reduce_margin_matches_margin_method():
    out = _output([1., 2., 3., 4.], Margin(k=2.))
    assert out.reduce() == pytest.approx(out.margin(2.))


def test_reduce_quantile_matches_quantile_method():
    out = _output(np.linspace(0., 10., 101), Quantile(q=.9))
    assert out.reduce() == pytest.approx(out.quantile(.9))


def test_measure_parameters_validated_at_construction():
    """A bad configuration should fail when the problem is built, not on the first evaluation"""
    with pytest.raises(ValueError):
        Quantile(q=2.)


def test_measures_carry_their_own_parameters():
    """Two outputs of the same kind but different parameters reduce differently, with nothing passed at reduce"""
    values = np.linspace(0., 10., 101)
    assert _output(values, Quantile(q=.5)).reduce() != _output(values, Quantile(q=.95)).reduce()
    assert _output(values, Margin(k=1.)).reduce() != _output(values, Margin(k=3.)).reduce()


def test_custom_measure_needs_no_library_change():
    """The point of the RobustMeasure hierarchy: a new measure is one class, wired in nowhere"""
    class WorstCase(RobustMeasure):
        def reduce(self, samples):
            return float(samples.getMax()[0])

    assert _output([1., 5., 3.], WorstCase()).reduce() == pytest.approx(5.)


def test_reduce_rejects_unknown_nan_policy():
    with pytest.raises(ValueError):
        _output([1., 2.], Mean()).reduce(nan_policy='whatever')


def test_uq_method_checks_measure_count_on_bind():
    method = MonteCarlo(n_evaluations=10)
    with pytest.raises(ValueError):
        method.add_config(_space(), [Mean()], n_obj=1, n_ieq_constr=1, n_eq_constr=0)


def test_uq_method_must_be_bound_before_use():
    with pytest.raises(RuntimeError):
        MonteCarlo(n_evaluations=10).get_samples()


def test_process_results_splits_columns():
    method = MonteCarlo(n_evaluations=3)
    method.add_config(_space(), [Mean()]*3, n_obj=1, n_ieq_constr=2, n_eq_constr=0)

    result = method.process_results(np.array([[1., 10., 20.], [2., 11., 21.], [3., 12., 22.]]))
    assert len(result.f) == 1
    assert len(result.g) == 2
    assert len(result.h) == 0
    assert result.f[0].mean() == pytest.approx(2.)
    assert result.g[0].mean() == pytest.approx(11.)
    assert result.g[1].mean() == pytest.approx(21.)


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
        param_space.add_parameter(StochasticParameter('u', ot.Normal(1., .2)))
        kwargs.setdefault('obj_measure', [Mean()])
        kwargs.setdefault('ieq_constr_measure', [Mean()])
        kwargs.setdefault('eq_constr_measure', [Margin(k=3.)])
        super().__init__([Real(bounds=(0., 1.))], param_space=param_space,
                         uq_method=MonteCarlo(n_evaluations=200, seed=5),
                         n_obj=1, n_ieq_constr=1, n_eq_constr=1, **kwargs)

    def _is_conditionally_active(self):
        return [False]

    def _correct_x(self, x, is_active):
        pass

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, *args, sample, **kwargs):
        f_out[:, 0] = sample[0] + x[:, 0]
        g_out[:, 0] = sample[0] - x[:, 0]
        h_out[:, 0] = sample[0] * x[:, 0]


def test_equality_constraints_use_their_own_measure():
    """Regression: the equality-constraint loop used to index the inequality-constraint parameter list, so H was
    reduced with the wrong measure (or raised IndexError when the two lists differed in length)."""
    problem = AllResponseKindsProblem()
    out = problem.evaluate(np.array([[.5]]), return_as_dictionary=True)

    result = out['stochastic'][0]
    # H is declared as Margin(k=3.) while G is Mean(): the two must not agree
    assert out['H'][0, 0] == pytest.approx(result.h[0].margin(3.))
    assert out['G'][0, 0] == pytest.approx(result.g[0].mean())
    assert out['H'][0, 0] != pytest.approx(result.h[0].mean())


def test_measure_counts_checked_per_response_kind():
    with pytest.raises(ValueError):
        AllResponseKindsProblem(eq_constr_measure=[Mean(), Mean()])


"""### Statistics carried in the output ###"""


def test_statistics_available_per_design_point():
    problem = VectorizedProblem(n=100)
    out = problem.evaluate(np.array([[1., 0.], [0., 0.]]), return_as_dictionary=True)

    assert len(out['stochastic']) == 2
    for result in out['stochastic']:
        assert isinstance(result, StochasticResult)
        assert len(result.f) == 1
        assert len(result.f[0].to_numpy()) == 100


def test_reported_statistics_reproduce_the_reduced_value():
    """The statistics handed back must be exactly what the optimizer saw, not a differently-estimated version."""
    problem = HierarchicalProblem(obj_measure=[Margin(k=2.)],
                                  ieq_constr_measure=[Quantile(q=.95)])
    out = problem.evaluate(np.array([[1, .6, .25]]), return_as_dictionary=True)

    result = out['stochastic'][0]
    assert result.f[0].margin(2.) == pytest.approx(out['F'][0, 0])
    assert result.g[0].quantile(.95) == pytest.approx(out['G'][0, 0])


"""### Hidden constraints ###"""


class FailingProblem(VectorizedProblem):
    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, *args, sample, **kwargs):
        f_out[:, 0] = x[:, 0] + sample[0]
        if sample[0] > 1.02:  # some samples fail to evaluate
            f_out[:, 0] = np.nan


def test_nan_policy_propagate_fails_the_design_point():
    problem = FailingProblem(n=50, seed=3, nan_policy='propagate')
    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)

    assert not np.isfinite(out['F'][0, 0])
    assert problem.get_failed_points(out)[0]


def test_nan_policy_omit_reduces_over_surviving_samples():
    problem = FailingProblem(n=50, seed=3, nan_policy='omit')
    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)

    values = out['stochastic'][0].f[0].to_numpy()
    assert np.any(~np.isfinite(values))  # some samples did fail
    assert np.isfinite(out['F'][0, 0])  # but the design point still has a value
    assert out['F'][0, 0] == pytest.approx(np.mean(values[np.isfinite(values)]))


"""### Configuration checks ###"""


def test_requires_parameter_space():
    with pytest.raises(ValueError):
        StochasticArchOptProblem([Real(bounds=(0., 1.))], param_space=StochasticParameterSpace(),
                                 uq_method=MonteCarlo(n_evaluations=10), n_obj=1)


def test_obj_measure_count_checked():
    with pytest.raises(ValueError):
        StochasticArchOptProblem([Real(bounds=(0., 1.))], param_space=_space(),
                                 uq_method=MonteCarlo(n_evaluations=10), n_obj=1,
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
        StochasticArchOptProblem([Real(bounds=(0., 1.))], param_space=_space(),
                                 uq_method=None, n_obj=1)

    with pytest.raises(ValueError):
        StochasticArchOptProblem([Real(bounds=(0., 1.))], param_space=_space(),
                                 uq_method='monte carlo', n_obj=1)


"""### Polynomial chaos ###"""


class QuadraticProblem(StochasticArchOptProblem):
    """f = (u - x0)^2 + x1^2: at x = (mean(u), 0) the expected value is exactly Var[u].
    Quadratic in u, so a polynomial chaos expansion represents it essentially exactly."""

    def __init__(self, uq_method_class=MonteCarlo, n=50, fail=False, method_kwargs=None, **kwargs):
        param_space = StochasticParameterSpace()
        param_space.add_parameter(StochasticParameter('u', ot.Normal(1., .05)))
        self.fail = fail
        super().__init__([Real(bounds=(-2., 2.)), Real(bounds=(-2., 2.))], param_space=param_space,
                         uq_method=uq_method_class(n=n, seed=42, **(method_kwargs or {})),
                         n_obj=1, obj_measure=[Mean()], **kwargs)

    def _is_conditionally_active(self):
        return [False, False]

    def _correct_x(self, x, is_active):
        pass

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, *args, sample, **kwargs):
        f_out[:, 0] = (sample[0] - x[:, 0])**2 + x[:, 1]**2
        if self.fail and sample[0] > 1.02:
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
    space = StochasticParameterSpace()
    for name in ['a', 'b', 'c']:
        space.add_parameter(StochasticParameter(name, ot.Normal(0., 1.)))
    method = PolynomialChaos(n_evaluations=200, degree=8)
    method.add_config(_space(3), [Mean()], n_obj=1, n_ieq_constr=0, n_eq_constr=0)
    assert method.n_terms == 165  # degree 8 in 3 dimensions



def test_pce_is_more_accurate_than_monte_carlo():
    """The point of PCE: far better statistics for the same number of expensive evaluations."""
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


def test_pce_uses_all_samples_it_asked_for():
    problem = QuadraticProblem(PolynomialChaos, n=40)
    assert problem.uq_method.get_samples().shape == (40, 1)

    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)
    # statistics come from the cheap metamodel, not from the 40 expensive evaluations
    assert len(out['stochastic'][0].f[0].to_numpy()) == problem.uq_method.n_metamodel_samples


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