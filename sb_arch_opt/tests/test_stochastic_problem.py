import pytest
import numpy as np
import openturns as ot
from sb_arch_opt.uncertainty import *
from sb_arch_opt.stochastic_problem import StochasticArchOptProblem
from sb_arch_opt.problems.robust_optimization.rosenbrock import StochasticRosenbrock
from sb_arch_opt.tests.conftest import VectorizedProblem, HierarchicalProblem, make_space
from pymoo.core.variable import Real


def _output(values):
    return StochasticOutput(ot.Sample(np.array(values, dtype=float).reshape((-1, 1))))


def _n_occupied_strata(space, samples, i_param):
    # An LHS puts exactly one point in each of the n equiprobable strata of every marginal
    n = samples.shape[0]
    cdf = np.array([space.param_realization(0)[i_param].value.computeCDF(v) for v in samples[:, i_param]])
    return len(np.unique(np.floor(cdf*n).astype(int)))


def test_parameter_space():
    space = StochasticParameterSpace([
        StochasticParameter('a', ot.Normal(0., 1.)),
        StochasticParameter('b', ot.Uniform(0., 1.)),
    ])

    assert space.n_parameters == 2
    assert space.parameter_names == ['a', 'b']
    assert space.joint_dist.getDimension() == 2

    for samples in [space.get_random_samples(20), space.get_lhs_samples(20)]:
        assert samples.shape == (20, 2)
        assert np.all(np.isfinite(samples))

    # The two draws differ in how evenly they cover each marginal
    assert all(_n_occupied_strata(space, space.get_lhs_samples(40), i) == 40 for i in range(2))
    assert any(_n_occupied_strata(space, space.get_random_samples(40), i) < 40 for i in range(2))


def test_scalars():
    out = _output([1., 2., 3., 4., 5.])
    mean, std = out.mean(), out.std()

    assert out.reduce(Mean()) == pytest.approx(mean)
    assert out.reduce(Quantile(q=.9)) == pytest.approx(out.quantile(.9))
    assert out.reduce(Margin(k=2.)) == pytest.approx(out.margin(2.))

    # The margin is the conservative value, so which tail that is depends on the direction
    assert out.reduce(Margin(k=2.)) == pytest.approx(mean + 2.*std)
    assert out.reduce(Margin(k=2., direction=-1)) == pytest.approx(mean + 2.*std)
    assert out.reduce(Margin(k=2., direction=1)) == pytest.approx(mean - 2.*std)

    # Each instance carries its own parameters
    assert out.reduce(Quantile(q=.5)) != out.reduce(Quantile(q=.95))
    assert out.reduce(Margin(k=1.)) != out.reduce(Margin(k=3.))

    with pytest.raises(ValueError):
        Quantile(q=2.)


def test_scalars_penalize_spread():
    wide, narrow = _output([1., 3., 5.]), _output([2.5, 3., 3.5])
    assert wide.mean() == pytest.approx(narrow.mean())

    assert narrow.reduce(Margin(k=2.)) < wide.reduce(Margin(k=2.))  # minimized: lower is better
    assert narrow.reduce(Margin(k=2., direction=1)) > wide.reduce(Margin(k=2., direction=1))  # maximized


def test_custom_scalar():
    class WorstCase(Scalarization):
        def reduce(self, samples):
            return float(samples.getMax()[0])

    assert _output([1., 5., 3.]).reduce(WorstCase()) == pytest.approx(5.)


@pytest.mark.parametrize('method_class', [MonteCarlo, PolynomialChaos])
def test_uq_method_samples(method_class):
    space = make_space(ot.Normal(0., 1.), ot.Normal(0., 1.))
    method = method_class(n_evaluations=40, seed=42)

    samples = method.get_samples(space)
    assert samples.shape == (40, 2)

    # Drawn once and reused, so every design point sees the same realizations (common random numbers)
    assert np.all(method.get_samples(space) == samples)

    # Both methods use an LHS rather than a plain random draw
    assert all(_n_occupied_strata(space, samples, i) == 40 for i in range(2))

    method.resample()
    assert method.get_samples(space).shape == (40, 2)

    with pytest.raises(ValueError):
        method.get_samples(None)

    samples = method.get_samples(space)
    for i in range(5):
        values = method.get_dictionary(space, i)
        assert list(values) == ['a', 'b']
        assert values['a'] == samples[i, 0]
        assert values['b'] == samples[i, 1]


def test_uq_method_requires_n_evaluations():
    with pytest.raises(ValueError):
        MonteCarlo(n_evaluations=None)

def test_user_only_implements_arch_evaluate_sample(stochastic_problem):
    out = stochastic_problem.evaluate(np.array([[1., 0.], [0., 0.]]), return_as_dictionary=True)

    assert out['F'].shape == (2, 1)
    assert np.all(np.isfinite(out['F']))
    # E[(u-x0)^2] + x1^2 = (E[u]-x0)^2 + var + x1^2
    assert out['F'][0, 0] == pytest.approx(.05**2, abs=2e-3)
    assert out['F'][1, 0] == pytest.approx(1. + .05**2, abs=2e-2)


def test_evaluate_sample_receives_one_realization_per_sample():
    problem = VectorizedProblem(n=25)
    problem.evaluate(np.array([[1., 0.], [0., 0.]]), return_as_dictionary=True)

    seen = problem.seen_parameters
    assert len(seen) == 25  # once per sample, not once per design point
    assert all(realization.shape == (1,) for realization in seen)  # one row, not the whole design
    assert len({tuple(realization) for realization in seen}) == 25  # a different row each time
    assert np.allclose(np.array(seen), problem.uq_method.get_samples(problem.param_space))


def test_multiple_design_points_at_once(stochastic_problem):
    x = np.random.RandomState(0).uniform(-2., 2., size=(7, 2))
    out = stochastic_problem.evaluate(x, return_as_dictionary=True)

    assert out['F'].shape == (7, 1)
    assert np.all(np.isfinite(out['F']))
    assert len(np.unique(out['F'])) == 7


def test_evaluation_is_deterministic_and_seeded():
    x = np.array([[.5, .5]])
    f_a = VectorizedProblem(n=50, seed=3).evaluate(x, return_as_dictionary=True)['F']
    f_b = VectorizedProblem(n=50, seed=3).evaluate(x, return_as_dictionary=True)['F']
    f_c = VectorizedProblem(n=50, seed=4).evaluate(x, return_as_dictionary=True)['F']

    assert np.allclose(f_a, f_b)
    assert not np.allclose(f_a, f_c)


def test_resample_draws_new_samples():
    problem = VectorizedProblem(n=50, seed=3)
    x = np.array([[.5, .5]])
    f_a = problem.evaluate(x, return_as_dictionary=True)['F'].copy()

    problem.uq_method.resample()
    problem.uq_method.seed = None
    assert not np.allclose(f_a, problem.evaluate(x, return_as_dictionary=True)['F'])


def test_correction_runs_for_implicit_design_space(hierarchical_problem):
    # Correction only happens up front for explicit design spaces, so the loop has to do it: otherwise the
    # evaluation function sees an all-True activeness matrix and un-imputed design vectors
    x = np.array([[0, .5, .3], [1, .5, .3]])
    out = hierarchical_problem.evaluate(x, return_as_dictionary=True)

    assert np.all(out['is_active'] == [[True, True, False], [True, True, True]])
    assert out['X'][0, 2] == pytest.approx(.25)  # imputed to the middle of its bounds
    assert np.all(np.isfinite(out['F']))


def test_hierarchical_problem_with_constraint(hierarchical_problem):
    x = np.array([[0, .5, .3], [1, .9, .35]])
    out = hierarchical_problem.evaluate(x, return_as_dictionary=True)

    assert out['F'].shape == (2, 1)
    assert out['G'].shape == (2, 1)
    assert np.all(np.isfinite(out['F']))
    assert np.all(np.isfinite(out['G']))


def test_response_kinds_use_their_own_scalar(all_response_kinds_problem):
    problem = all_response_kinds_problem
    out = problem.evaluate(np.array([[.5]]), return_as_dictionary=True)
    result = out['stochastic'][0]

    assert len(result.outputs) == 3
    assert out['F'][0, 0] == pytest.approx(result.outputs[0].reduce(Mean()))
    assert out['G'][0, 0] == pytest.approx(result.outputs[1].reduce(Mean()))
    assert out['H'][0, 0] == pytest.approx(result.outputs[2].reduce(Margin(k=3.)))

    # The equality constraint uses a margin, so it is above its own mean
    assert out['H'][0, 0] > result.outputs[2].mean()


def test_scalar_counts_checked_per_response_kind():
    with pytest.raises(ValueError):
        HierarchicalProblem(ieq_constr_scalar=[Mean(), Mean()])
    with pytest.raises(ValueError):
        HierarchicalProblem(obj_scalar=[Mean(), Mean()])
    with pytest.raises(ValueError):
        HierarchicalProblem(ieq_constr_scalar=['quantile'])

    HierarchicalProblem(ieq_constr_scalar=[Mean()])  # a matching count is accepted


def test_problem_configuration_checks():
    space = make_space(ot.Normal(0., 1.))

    with pytest.raises(ValueError):
        StochasticArchOptProblem([Real(bounds=(0., 1.))], param_space=space, uq_method=None, n_obj=1)
    with pytest.raises(ValueError):
        StochasticArchOptProblem([Real(bounds=(0., 1.))], param_space=space, uq_method='monte carlo', n_obj=1)
    with pytest.raises(ValueError):
        StochasticArchOptProblem([Real(bounds=(0., 1.))], param_space=None, uq_method=MonteCarlo(10), n_obj=1)
    with pytest.raises(ValueError):
        StochasticArchOptProblem([Real(bounds=(0., 1.))], param_space=StochasticParameterSpace([]),
                                 uq_method=MonteCarlo(10), n_obj=1)


def test_statistics_available_per_design_point(stochastic_problem):
    out = stochastic_problem.evaluate(np.array([[1., 0.], [0., 0.]]), return_as_dictionary=True)

    assert len(out['stochastic']) == 2
    for result in out['stochastic']:
        assert isinstance(result, StochasticResults)
        assert len(result.outputs) == 1
        assert len(result.outputs[0].to_numpy()) == 100

    # The realizations really reach the model, so the response scatters
    values = out['stochastic'][0].outputs[0].to_numpy()
    assert len(set(values.tolist())) == 100
    assert values.std() > 0.


def test_reported_statistics_reproduce_the_reduced_value():
    problem = VectorizedProblem(n=200, obj_scalar=[Margin(k=2.)])
    out = problem.evaluate(np.array([[1., 0.], [.5, .5]]), return_as_dictionary=True)

    for i, result in enumerate(out['stochastic']):
        output = result.outputs[0]
        assert out['F'][i, 0] == pytest.approx(output.reduce(Margin(k=2.)))
        assert out['F'][i, 0] == pytest.approx(output.mean() + 2.*output.std())


def test_stochastic_output_statistics():
    out = _output(np.linspace(0., 10., 101))

    assert out.mean() == pytest.approx(5.)
    assert out.var() == pytest.approx(out.std()**2)
    assert out.quantile(.5) == pytest.approx(5., abs=.1)
    assert out.prob_exceeds(5.) == pytest.approx(.5, abs=.02)
    assert len(out.to_numpy()) == 101
    assert out.to_distribution().computeCDF(5.) == pytest.approx(.5, abs=.05)


def test_a_failed_sample_fails_the_design_point():
    # Any non-finite sample discards the design point: with a single failed evaluation the model is not
    # trustworthy at that point, and SBArchOpt treats NaN as a hidden-constraint violation
    problem = VectorizedProblem(n=50, fail=True)
    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)

    values = out['stochastic'][0].outputs[0].to_numpy()
    assert np.any(~np.isfinite(values))
    assert not np.isfinite(out['F'][0, 0])
    assert problem.get_failed_points(out)[0]


def test_pce_is_more_accurate_than_monte_carlo():
    # Both methods draw the same kind of LHS design, so this compares fitting an expansion against averaging
    # over it. f = (u-x0)^2 + x1^2 is quadratic in u, so the expansion is essentially exact.
    x = np.array([[1., 0.]])
    exact = .05**2  # E[(u-x)^2] at x = E[u] is Var[u]

    f_mc = VectorizedProblem(n=20, obj_scalar=[Mean()]).evaluate(x, return_as_dictionary=True)['F'][0, 0]
    f_pce = VectorizedProblem(uq_method=PolynomialChaos(20, seed=42), obj_scalar=[Mean()]) \
        .evaluate(x, return_as_dictionary=True)['F'][0, 0]

    assert abs(f_pce - exact) < abs(f_mc - exact)
    assert f_pce == pytest.approx(exact, abs=1e-4)


def test_pce_through_the_problem():
    # Regression: the problem called process_results without the parameter space, which PCE requires
    problem = VectorizedProblem(uq_method=PolynomialChaos(50, seed=42), obj_scalar=[Mean()])
    x = np.array([[.5, .5]])
    out = problem.evaluate(x, return_as_dictionary=True)

    assert np.all(np.isfinite(out['F']))
    assert np.allclose(out['F'], problem.evaluate(x, return_as_dictionary=True)['F'])

    # Statistics come from the cheap metamodel, not from the 50 expensive evaluations
    assert len(out['stochastic'][0].outputs[0].to_numpy()) == problem.uq_method.n_metamodel_samples


def test_pce_terms_and_sample_requirement():
    space = make_space(*[ot.Normal(0., 1.)]*3)

    assert PolynomialChaos(200, degree=8).n_terms(space) == 165  # degree 8 in 3 dimensions
    assert PolynomialChaos(200, degree=8).n_terms(make_space(ot.Normal(0., 1.))) == 9  # ... in 1 dimension

    with pytest.raises(ValueError):
        PolynomialChaos(10, degree=8).get_samples(space)  # 165 terms cannot be fitted from 10 samples


def test_pce_provides_chaos_result_for_sensitivity_analysis():
    problem = VectorizedProblem(uq_method=PolynomialChaos(50, seed=42), obj_scalar=[Mean()])
    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)

    chaos_results = out['stochastic'][0].method_result
    assert len(chaos_results) == 1
    assert isinstance(chaos_results[0], ot.FunctionalChaosResult)

    sobol = ot.FunctionalChaosSobolIndices(chaos_results[0])
    assert sobol.getSobolIndex(0) == pytest.approx(1., abs=1e-6)  # a single parameter explains everything


def test_pce_falls_back_to_raw_samples_when_evaluations_fail():
    # A response with failed evaluations cannot be fitted; the design point then fails as usual
    problem = VectorizedProblem(uq_method=PolynomialChaos(50, seed=42), obj_scalar=[Mean()], fail=True)
    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)

    assert out['stochastic'][0].method_result[0] is None
    assert not np.isfinite(out['F'][0, 0])


def test_stochastic_rosenbrock():
    problem = StochasticRosenbrock(n_var=3, std=.05, n=500)
    assert problem.n_var == 3
    assert problem.param_space.n_parameters == 2
    assert repr(problem)

    # At the optimum the expected objective equals the sum of the parameter variances
    out = problem.evaluate(problem.pareto_set(), return_as_dictionary=True)
    assert out['F'][0, 0] == pytest.approx(problem.pareto_front()[0, 0], abs=1e-3)

    # Moving away from the optimum makes it worse
    f_off = problem.evaluate(np.array([[.5, .5, .5]]), return_as_dictionary=True)['F']
    assert f_off[0, 0] > out['F'][0, 0]


def test_stochastic_rosenbrock_with_pce():
    problem = StochasticRosenbrock(n_var=3, uq_method=PolynomialChaos(40, seed=42, degree=2))
    out = problem.evaluate(problem.pareto_set(), return_as_dictionary=True)

    assert out['F'][0, 0] == pytest.approx(problem.pareto_front()[0, 0], abs=1e-3)