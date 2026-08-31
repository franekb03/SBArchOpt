import pytest
import numpy as np
from typing import *
from scipy.stats import norm, uniform
from pymoo.core.variable import Real, Integer, Choice
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.design_space import ArchDesignSpace, ImplicitArchDesignSpace
from sb_arch_opt.uncertainty import UncertainParameter, StochasticReduction, reduce_samples
from sb_arch_opt.robust import RobustArchOptProblem
from sb_arch_opt.problems.robust_optimization import HierarchicalRobustProblem, StochasticRosenbrock


class BranchingDesignSpace(ArchDesignSpace):
    """x0 selects a branch; x1 is only active in branch 0, x2 only in branch 1"""

    def __init__(self, params: List[UncertainParameter] = None, is_param_cond: List[bool] = None,
                 param_is_active_func=None):
        self._params = params or []
        self._is_param_cond = is_param_cond
        self._param_is_active_func = param_is_active_func
        super().__init__()

    def is_explicit(self) -> bool:
        return False

    def _get_variables(self):
        return [Choice(options=['a', 'b']), Real(bounds=(0., 1.)), Integer(bounds=(0, 3))]

    def _get_dv_names(self):
        return ['branch', 'cont_a', 'int_b']

    def _get_uncertain_parameters(self):
        return self._params

    def _is_param_conditionally_active(self):
        return self._is_param_cond

    def _get_param_is_active(self, x, is_active):
        if self._param_is_active_func is not None:
            return self._param_is_active_func(x, is_active)

    def _is_conditionally_active(self):
        return [False, True, True]

    def _correct_x(self, x, is_active):
        is_a = x[:, 0] == 0
        is_active[:, 1] = is_a
        is_active[:, 2] = ~is_a

    def _quick_sample_discrete_x(self, n):
        raise RuntimeError

    def _get_n_valid_discrete(self):
        return 5  # branch a (1 discrete combo) + branch b (4 int options)

    def _get_n_active_cont_mean(self):
        pass

    def _get_n_correct_discrete(self):
        pass

    def _get_n_active_cont_mean_correct(self):
        pass

    def _gen_all_discrete_x(self):
        x = np.array([[0, .5, 0]] + [[1, .5, i] for i in range(4)], dtype=float)
        is_active = np.ones(x.shape, dtype=bool)
        is_active[0, 2] = False
        is_active[1:, 1] = False
        return x, is_active


def _x_both_branches():
    return np.array([[0, .5, 0], [1, .5, 2]], dtype=float)


"""### UncertainParameter ###"""


def test_parameter_from_distribution():
    param = UncertainParameter('payload', norm(2., .3))
    assert param.name == 'payload'
    assert param.is_uncertain
    assert param.nominal == pytest.approx(2.)

    # Unbounded distributions are truncated so that bounds are finite
    xl, xu = param.bounds
    assert np.isfinite(xl) and np.isfinite(xu)
    assert xl < 2. < xu

    assert param.ppf(np.array([.5])) == pytest.approx([2.])


def test_parameter_explicit_nominal_and_bounds():
    param = UncertainParameter('rho', norm(1.225, .05), nominal=1.3, bounds=(1., 1.5))
    assert param.nominal == pytest.approx(1.3)
    assert param.bounds == (1., 1.5)


def test_parameter_without_distribution_is_constant():
    param = UncertainParameter('fixed', nominal=1.5)
    assert not param.is_uncertain
    assert param.nominal == pytest.approx(1.5)
    assert np.all(param.ppf(np.array([0., .5, 1.])) == 1.5)


def test_parameter_needs_distribution_or_nominal():
    with pytest.raises(ValueError):
        UncertainParameter('nope')


def test_parameter_rejects_unknown_distribution():
    with pytest.raises(ValueError):
        UncertainParameter('nope', distribution=object())


"""### Design space integration ###"""


def test_no_parameters_by_default():
    """A design space that declares no uncertain parameters must stay completely inert: this is what keeps
    existing (deterministic) problems, including adsg-core's DSGDesignSpace, working unchanged."""
    ds = BranchingDesignSpace()
    assert ds.n_param == 0
    assert ds.param_names == []

    x, is_active = ds.correct_x(_x_both_branches())
    assert ds.get_param_is_active(x, is_active).shape == (2, 0)
    assert ds.sample_parameters(x, is_active, 10).shape == (2, 10, 0)
    assert ds.param_imputation_ratio == 1.
    assert ds.get_param_activeness_rates() is None
    ds.check_param_conditionality()


def test_always_active_parameters():
    ds = BranchingDesignSpace([UncertainParameter('a', norm(0., 1.)), UncertainParameter('b', norm(5., 1.))])
    assert ds.n_param == 2
    assert ds.param_names == ['a', 'b']

    x, is_active = ds.correct_x(_x_both_branches())
    assert np.all(ds.get_param_is_active(x, is_active))
    assert not np.any(ds.is_param_conditionally_active)
    assert ds.param_imputation_ratio == pytest.approx(1.)


def test_duplicate_parameter_names_rejected():
    ds = BranchingDesignSpace([UncertainParameter('a', norm(0., 1.)), UncertainParameter('a', norm(1., 1.))])
    with pytest.raises(ValueError):
        getattr(ds, 'uncertain_params')


"""### active_if resolution ###"""


def test_active_if_mirrors_design_variable_by_name():
    ds = BranchingDesignSpace([
        UncertainParameter('always', norm(0., 1.)),
        UncertainParameter('on_a', norm(0., 1.), active_if='cont_a'),
        UncertainParameter('on_b', norm(0., 1.), active_if='int_b'),
    ])
    x, is_active = ds.correct_x(_x_both_branches())
    is_param_active = ds.get_param_is_active(x, is_active)

    assert np.all(is_param_active == np.array([[True, True, False], [True, False, True]]))


def test_active_if_by_index_matches_by_name():
    by_name = BranchingDesignSpace([UncertainParameter('p', norm(0., 1.), active_if='int_b')])
    by_index = BranchingDesignSpace([UncertainParameter('p', norm(0., 1.), active_if=2)])

    x, is_active = by_name.correct_x(_x_both_branches())
    assert np.all(by_name.get_param_is_active(x, is_active) == by_index.get_param_is_active(x, is_active))


def test_active_if_on_selected_value():
    """A branch may carry a parameter but no design variable; then activeness has to key off the selected value"""
    ds = BranchingDesignSpace([
        UncertainParameter('on_branch_a', norm(0., 1.), active_if=('branch', 0)),
        UncertainParameter('on_branch_b', norm(0., 1.), active_if=('branch', 1)),
    ])
    x, is_active = ds.correct_x(_x_both_branches())
    assert np.all(ds.get_param_is_active(x, is_active) == np.array([[True, False], [False, True]]))


def test_active_if_on_set_of_values():
    ds = BranchingDesignSpace([UncertainParameter('p', norm(0., 1.), active_if=('int_b', {1, 2}))])
    x = np.array([[1, .5, 0], [1, .5, 1], [1, .5, 2], [1, .5, 3]], dtype=float)
    x, is_active = ds.correct_x(x)
    assert np.all(ds.get_param_is_active(x, is_active).ravel() == [False, True, True, False])


def test_active_if_value_test_respects_activeness():
    """Inactive design variables are imputed to their lower bound, so a value test that ignored activeness would
    report the parameter active for every architecture where the design variable does not exist at all."""
    ds = BranchingDesignSpace([UncertainParameter('p', norm(0., 1.), active_if=('int_b', 0))])

    x, is_active = ds.correct_x(_x_both_branches())
    # Branch a: int_b is inactive and imputed to its lower bound (0), which is exactly the tested value
    assert x[0, 2] == 0
    assert not is_active[0, 2]

    is_param_active = ds.get_param_is_active(x, is_active)
    assert not is_param_active[0, 0]  # inactive design variable -> parameter inactive
    assert not is_param_active[1, 0]  # branch b, but int_b == 2 != 0


def test_active_if_callable():
    ds = BranchingDesignSpace([
        UncertainParameter('p', norm(0., 1.), active_if=lambda x, is_act: x[:, 1] > .3),
    ])
    x = np.array([[0, .2, 0], [0, .8, 0]], dtype=float)
    x, is_active = ds.correct_x(x)
    assert np.all(ds.get_param_is_active(x, is_active).ravel() == [False, True])


def test_active_if_unknown_reference():
    ds = BranchingDesignSpace([UncertainParameter('p', norm(0., 1.), active_if='does_not_exist')])
    x, is_active = ds.correct_x(_x_both_branches())
    with pytest.raises(ValueError):
        ds.get_param_is_active(x, is_active)

    ds = BranchingDesignSpace([UncertainParameter('p', norm(0., 1.), active_if=99)])
    with pytest.raises(ValueError):
        ds.get_param_is_active(x, is_active)


def test_structural_param_is_active_overrides_links():
    """A design space can compute parameter activeness structurally (as the adsg-core adapter would), which takes
    precedence over the declarative conditions."""
    def _structural(x, is_active):
        return np.tile(np.array([[True, False]]), (x.shape[0], 1))

    ds = BranchingDesignSpace(
        [UncertainParameter('a', norm(0., 1.), active_if='cont_a'), UncertainParameter('b', norm(0., 1.))],
        param_is_active_func=_structural)

    x, is_active = ds.correct_x(_x_both_branches())
    assert np.all(ds.get_param_is_active(x, is_active) == np.array([[True, False], [True, False]]))


def test_structural_param_is_active_shape_checked():
    ds = BranchingDesignSpace(
        [UncertainParameter('a', norm(0., 1.))],
        param_is_active_func=lambda x, is_active: np.ones((x.shape[0], 3), dtype=bool))

    x, is_active = ds.correct_x(_x_both_branches())
    with pytest.raises(RuntimeError):
        ds.get_param_is_active(x, is_active)


"""### Conditionality bookkeeping ###"""


def test_param_conditionality_deduced_from_all_discrete_x():
    ds = BranchingDesignSpace([
        UncertainParameter('always', norm(0., 1.)),
        UncertainParameter('on_a', norm(0., 1.), active_if='cont_a'),
    ])
    assert np.all(ds.is_param_conditionally_active == [False, True])


def test_param_conditionality_declared_length_checked():
    ds = BranchingDesignSpace([UncertainParameter('a', norm(0., 1.))], is_param_cond=[False, True])
    with pytest.raises(ValueError):
        getattr(ds, 'is_param_conditionally_active')


def test_check_param_conditionality_catches_inconsistency():
    """Declaring a parameter unconditional while it does become inactive is a bug: catch it at declaration time
    rather than as a quietly wrong variance later on."""
    ds = BranchingDesignSpace(
        [UncertainParameter('on_a', norm(0., 1.), active_if='cont_a')], is_param_cond=[False])
    with pytest.raises(RuntimeError):
        ds.check_param_conditionality()

    ds_ok = BranchingDesignSpace(
        [UncertainParameter('on_a', norm(0., 1.), active_if='cont_a')], is_param_cond=[True])
    ds_ok.check_param_conditionality()


def test_param_activeness_rates_and_imputation_ratio():
    ds = BranchingDesignSpace([
        UncertainParameter('always', norm(0., 1.)),
        UncertainParameter('on_a', norm(0., 1.), active_if='cont_a'),
    ])
    rates = ds.get_param_activeness_rates()
    assert list(rates.index) == ['always', 'on_a']
    assert rates['active_rate']['always'] == pytest.approx(1.)
    assert rates['active_rate']['on_a'] == pytest.approx(1/5)  # active for 1 of the 5 valid discrete vectors

    assert ds.param_imputation_ratio == pytest.approx(2 / (1 + 1/5))


"""### Imputation and sampling ###"""


def test_impute_u_sets_inactive_to_nominal():
    ds = BranchingDesignSpace([
        UncertainParameter('a', norm(2., .3)),
        UncertainParameter('b', norm(5., 1.), active_if='cont_a'),
    ])
    u = np.zeros((2, 2))
    is_param_active = np.array([[True, True], [True, False]])
    ds.impute_u(u, is_param_active)

    assert np.all(u[0, :] == 0.)  # active: untouched
    assert u[1, 0] == 0.
    assert u[1, 1] == pytest.approx(5.)  # inactive: nominal


def test_sample_parameters_shape_and_imputation():
    ds = BranchingDesignSpace([
        UncertainParameter('always', norm(2., .3)),
        UncertainParameter('on_a', norm(5., 1.), active_if='cont_a'),
        UncertainParameter('on_b', uniform(0., 1.), active_if='int_b'),
    ])
    x, is_active = ds.correct_x(_x_both_branches())
    u = ds.sample_parameters(x, is_active, 25, random_state=42)
    assert u.shape == (2, 25, 3)

    # Always-active parameter varies for both design points
    assert len(np.unique(u[0, :, 0])) == 25
    assert len(np.unique(u[1, :, 0])) == 25

    # Inactive parameters hold their nominal value and therefore contribute no variation
    assert np.all(u[0, :, 2] == pytest.approx(ds.u_nominal[2]))
    assert np.all(u[1, :, 1] == pytest.approx(ds.u_nominal[1]))


def test_common_random_numbers():
    ds = BranchingDesignSpace([UncertainParameter('a', norm(2., .3))])
    x, is_active = ds.correct_x(_x_both_branches())

    u_crn = ds.sample_parameters(x, is_active, 20, random_state=42)
    assert np.allclose(u_crn[0, :, 0], u_crn[1, :, 0])

    u_indep = ds.sample_parameters(x, is_active, 20, random_state=42, common_random_numbers=False)
    assert not np.allclose(u_indep[0, :, 0], u_indep[1, :, 0])


def test_sampling_is_reproducible():
    ds = BranchingDesignSpace([UncertainParameter('a', norm(2., .3))])
    x, is_active = ds.correct_x(_x_both_branches())
    assert np.allclose(ds.sample_parameters(x, is_active, 20, random_state=7),
                       ds.sample_parameters(x, is_active, 20, random_state=7))


"""### Implicit design space and problem plumbing ###"""


def test_implicit_design_space_parameters():
    ds = ImplicitArchDesignSpace(
        [Choice(options=['a', 'b']), Real(bounds=(0., 1.))],
        correct_x_func=lambda x, is_active: is_active.__setitem__((slice(None), 1), x[:, 0] == 0),
        is_conditional_func=lambda: [False, True],
        uncertain_params=[UncertainParameter('p', norm(0., 1.), active_if='cont')],
        dv_names=['branch', 'cont'],
    )
    x, is_active = ds.correct_x(np.array([[0, .5], [1, .5]], dtype=float))
    assert np.all(ds.get_param_is_active(x, is_active).ravel() == [True, False])


def test_problem_exposes_parameters():
    problem = HierarchicalRobustProblem(n_mc=5)
    assert problem.n_param == 5
    assert problem.param_names == ['payload', 'drag', 'eta_bat', 'bsfc', 'gen_loss']
    assert np.all(problem.is_param_conditionally_active == [False, False, True, True, True])

    # Uncertain parameters must not leak into the design vectors seen by the optimizer
    assert problem.n_var == 4
    assert len(problem.des_vars) == 4
    assert problem.xl.shape == (4,)


def test_deterministic_problem_unaffected():
    """Regression guard for the adsg-core path: a problem that declares no uncertain parameters is untouched."""
    class Deterministic(ArchOptProblemBase):
        def __init__(self):
            super().__init__([Real(bounds=(0., 1.)), Real(bounds=(0., 1.))], n_obj=1)

        def _is_conditionally_active(self):
            return [False, False]

        def _correct_x(self, x, is_active):
            pass

        def _arch_evaluate(self, x, is_active_out, f_out, g_out, h_out, *args, **kwargs):
            f_out[:, 0] = np.sum(x, axis=1)

    problem = Deterministic()
    assert problem.n_param == 0
    out = problem.evaluate(np.array([[.25, .25]]), return_as_dictionary=True)
    assert out['F'][0, 0] == pytest.approx(.5)
    assert 'F_std' not in out


"""### Reductions and the robust problem ###"""


def test_reduce_samples():
    values = np.array([[[1.], [2.], [3.]]])  # 1 x 3 samples x 1
    assert reduce_samples(values, StochasticReduction.MEAN, 0.)[0, 0] == pytest.approx(2.)
    assert reduce_samples(values, StochasticReduction.MARGIN, 1.)[0, 0] == pytest.approx(2.+np.std([1., 2., 3.]))
    assert reduce_samples(values, StochasticReduction.QUANTILE, .5)[0, 0] == pytest.approx(2.)

    with pytest.raises(ValueError):
        reduce_samples(values, StochasticReduction.QUANTILE, 2.)


def test_robust_problem_margin_reduction():
    problem = HierarchicalRobustProblem(n_mc=200)
    problem.set_random_state(42)
    x = np.array([[0, .6, .4, .25], [1, .6, .4, .25]])

    out = problem.evaluate(x, return_as_dictionary=True)
    assert out['F'].shape == (2, 1)
    assert np.allclose(out['F'], out['F_mean'] + 2*out['F_std'])
    assert np.all(out['F_std'] > 0)
    assert np.allclose(out['G'], out['G_mean'] + 2*out['G_std'])


def test_robust_problem_is_deterministic_with_fixed_random_state():
    """Surrogate-based algorithms need f(x) to be a deterministic function of x; with freshly drawn samples per
    evaluation the surrogate would fit Monte Carlo noise."""
    problem = HierarchicalRobustProblem(n_mc=50)
    problem.set_random_state(1)
    x = np.array([[0, .6, .4, .25], [1, .6, .4, .25]])

    f1 = problem.evaluate(x, return_as_dictionary=True)['F']
    f2 = problem.evaluate(x, return_as_dictionary=True)['F']
    assert np.allclose(f1, f2)


def test_robust_problem_fills_activeness_for_implicit_design_space():
    """Correction only runs up front for explicit design spaces, so the robust problem has to correct before
    deriving parameter activeness; otherwise every parameter would look active."""
    problem = HierarchicalRobustProblem(n_mc=5)
    out = problem.evaluate(np.array([[0, .6, .4, .25], [1, .6, .4, .25]]), return_as_dictionary=True)

    assert np.all(out['is_active'] == np.array([
        [True, True, True, False],
        [True, True, False, True],
    ]))


def test_inactive_parameters_do_not_affect_evaluation():
    """An inactive parameter does not exist for that architecture, so changing its distribution must not change
    the result of evaluating that architecture."""
    x_electric = np.array([[0, .6, .4, .25]])

    problem = HierarchicalRobustProblem(n_mc=100)
    problem.set_random_state(3)
    f_ref = problem.evaluate(x_electric, return_as_dictionary=True)['F']

    # bsfc and gen_loss are inactive for the electric branch
    problem2 = HierarchicalRobustProblem(n_mc=100)
    problem2.set_random_state(3)
    params = problem2.design_space.uncertain_params
    params[3]._dist = UncertainParameter('bsfc', norm(10., 5.)).distribution
    params[4]._dist = UncertainParameter('gen_loss', norm(-3., 2.)).distribution
    f_alt = problem2.evaluate(x_electric, return_as_dictionary=True)['F']

    assert np.allclose(f_ref, f_alt)


def test_reduction_length_checked():
    with pytest.raises(ValueError):
        StochasticRosenbrock(n_var=2, reduction=[StochasticReduction.MEAN, StochasticReduction.MEAN])


def test_stochastic_rosenbrock():
    problem = StochasticRosenbrock(n_var=3, std=.05, n_mc=500)
    problem.set_random_state(42)
    assert problem.n_var == 3
    assert problem.n_param == 2
    assert not np.any(problem.is_param_conditionally_active)

    # At the optimum the expected objective equals the sum of the parameter variances
    x_opt = problem.pareto_set()
    out = problem.evaluate(x_opt, return_as_dictionary=True)
    assert out['F'][0, 0] == pytest.approx(problem.pareto_front()[0, 0], abs=2e-3)

    # Moving away from the optimum makes it worse
    f_off = problem.evaluate(np.array([[.5, .5, .5]]), return_as_dictionary=True)['F']
    assert f_off[0, 0] > out['F'][0, 0]


def test_print_stats_with_parameters(capsys):
    HierarchicalRobustProblem(n_mc=5).print_stats()
    captured = capsys.readouterr().out
    assert 'n_param' in captured
    assert 'HIER_param' in captured
