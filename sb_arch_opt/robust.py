from typing import Union, List

from pymoo.core.variable import Variable

from sb_arch_opt.design_space import ArchDesignSpace
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.uncertainty import *


class RobustArchOptProblem(ArchOptProblemBase):
    def __init__(self, des_vars: Union[List[Variable], ArchDesignSpace], n_obj=1, n_ieq_constr=0, n_eq_constr=0,
                 obj_type: List[StochasticMetricType] = None, constr_type: List[StochasticMetricType] = None,
                 **kwargs):

        if len(obj_type) != n_obj:
            raise ValueError("obj type must be equal to n_obj")
        if len(constr_type) != n_obj:
            raise ValueError("constr type must be equal to n_obj")


        super().__init__(des_vars, n_obj, n_ieq_constr, n_eq_constr, **kwargs)

    def _get_uncertain_parameters(self) -> List[UncertainParameter]:
        raise NotImplementedError
