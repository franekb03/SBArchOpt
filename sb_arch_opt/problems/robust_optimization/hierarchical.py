"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Test problems for architecture optimization under uncertainty, featuring uncertain parameters that are only active
for some architectures.
"""
import numpy as np
from typing import *
from scipy.stats import norm, uniform
from pymoo.core.variable import Real, Choice
from sb_arch_opt.robust import RobustArchOptProblem, StochasticReduction
from sb_arch_opt.uncertainty import UncertainParameter

__all__ = ['HierarchicalRobustProblem']


class HierarchicalRobustProblem(RobustArchOptProblem):
    """
    A small hierarchical problem under uncertainty, structured like a powertrain selection: a categorical choice
    selects one of two branches, and each branch has its own continuous design variable *and* its own uncertain
    parameter. Two more parameters are always active.

    Design variables:
    - x0: powertrain (categorical: 0 = electric, 1 = hybrid)
    - x1: size (continuous, always active)
    - x2: battery fraction (continuous, active only for electric)
    - x3: fuel fraction (continuous, active only for hybrid)

    Uncertain parameters:
    - payload, drag: always active
    - eta_bat: active only for electric (same branch as x2)
    - bsfc: active only for hybrid (same branch as x3)

    The objective is minimized and formulated as a margin (`mean + k*std`), so an architecture is only good if it is
    both good on average and insensitive to the uncertain parameters. The hybrid branch reaches a better mean but a
    larger relative spread (its extra uncertain parameters, bsfc and gen_loss, feed straight into the objective),
    which is the kind of trade the robust formulation is meant to expose.
    """

    def __init__(self, n_mc=50, k=2.):
        des_vars = [
            Choice(options=['electric', 'hybrid']),
            Real(bounds=(0.2, 1.0)),
            Real(bounds=(0.2, 0.6)),
            Real(bounds=(0.1, 0.4)),
        ]

        super().__init__(
            des_vars, n_obj=1, n_ieq_constr=1, n_mc=n_mc,
            f_reduction=StochasticReduction.MARGIN, f_k=k,
            g_reduction=StochasticReduction.MARGIN, g_k=k,
        )

    def _get_dv_names(self) -> Optional[List[str]]:
        return ['powertrain', 'size', 'bat_frac', 'fuel_frac']

    def _get_uncertain_parameters(self) -> List[UncertainParameter]:
        return [
            # Always active
            UncertainParameter('payload', norm(2., .3)),
            UncertainParameter('drag', norm(1., .08)),

            # Active on the same branch as an existing design variable: mirror that design variable's activeness
            UncertainParameter('eta_bat', norm(.92, .03), active_if='bat_frac'),
            UncertainParameter('bsfc', norm(.42, .075), active_if='fuel_frac'),

            # Active for the hybrid branch, referenced by the value of the choice rather than by a design variable
            UncertainParameter('gen_loss', uniform(.02, .06), active_if=('powertrain', 1)),
        ]

    def _is_param_conditionally_active(self) -> Optional[List[bool]]:
        return [False, False, True, True, True]

    def _is_conditionally_active(self) -> List[bool]:
        return [False, False, True, True]

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        is_electric = x[:, 0] == 0
        is_active[:, 2] = is_electric  # bat_frac only for electric
        is_active[:, 3] = ~is_electric  # fuel_frac only for hybrid

    def _get_n_valid_discrete(self) -> int:
        return 2

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        x = np.array([
            [0, .6, .4, .25],
            [1, .6, .4, .25],
        ])
        is_active = np.ones(x.shape, dtype=bool)
        is_active[0, 3] = False
        is_active[1, 2] = False
        return x, is_active

    def _arch_evaluate_sample(self, x, is_active, u, is_param_active, f_out, g_out, h_out, *args, **kwargs):
        payload, drag, eta_bat, bsfc, gen_loss = (u[:, i] for i in range(5))
        is_electric = x[:, 0] == 0

        mass = .5 + x[:, 1] + payload
        power = drag * mass**1.5

        # Electric: energy from the battery, moderately sensitive to the uncertain parameters
        energy_electric = x[:, 2] * 40. * eta_bat

        # Hybrid: more energy available, but the fuel burn depends strongly on the uncertain BSFC
        energy_hybrid = x[:, 3] * 130. * (1. - gen_loss) / (bsfc + .1)

        energy = np.where(is_electric, energy_electric, energy_hybrid)

        # Minimized: the inverse of endurance
        f_out[:, 0] = power / energy
        # Mass should stay under a limit
        g_out[:, 0] = mass - 3.5

    def might_have_hidden_constraints(self):
        return False

    def __repr__(self):
        return f'{self.__class__.__name__}(n_mc={self.n_mc})'
