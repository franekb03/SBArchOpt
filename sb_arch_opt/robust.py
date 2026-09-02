from typing import Union, List, Dict, Tuple

from pymoo.core.variable import Variable

from adsg_core import MetricNode
from sb_arch_opt.design_space import ArchDesignSpace
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.uncertainty import *

__all__ = ["StochasticArchOptProblem"]

class StochasticArchOptProblem(ArchOptProblemBase):
    """
    Class for stochastic optimization problem.
    """
    def __init__(self, des_vars: Union[List[Variable], ArchDesignSpace], param_space: StochasticParameterSpace, uq_method_type: UQMethodType, n_obj=1, n_ieq_constr=0, n_eq_constr=0,
                 obj_type: List[StochasticMetricType] = None, constr_type: List[StochasticMetricType] = None,
                 **kwargs):
        if param_space is None or param_space.n_parameters == 0:
            raise ValueError("Define stochastic parameter space for the robust problem.")

        self.param_space = param_space

        if len(obj_type) != n_obj:
            raise ValueError("obj type must be equal to n_obj")
        if len(constr_type) != n_obj:
            raise ValueError("constr type must be equal to n_obj")

        self.obj_type = obj_type
        self.constr_type = constr_type

        self.stochastic_output_type = []
        for obj in obj_type:
            self.stochastic_output_type.append(obj)
        for constr in constr_type:
            self.stochastic_output_type.append(constr)


        if uq_method_type is None:
            raise ValueError("UQ method type must be specified")

        if uq_method_type is UQMethodType.MONTE_CARLO:
            if kwargs["n"] is None:
                raise ValueError("n must be specified for Monte Carlo evaluation")
            if kwargs["seed"] is None:
                raise ValueError("seed must be specified for Monte Carlo evaluation")
            self.uq_method = MonteCarlo(param_space, self.stochastic_output_type, n_obj, n_ieq_constr, n_eq_constr, kwargs["n"], kwargs["seed"])
        else:
            raise ValueError("Unknown UQ method type")

        super().__init__(des_vars, n_obj, n_ieq_constr, n_eq_constr, **kwargs)

    def propagate_uncertainty(self, uq_method: str, func, dsg: DSGType, metric_node: MetricNode, **kwargs) -> Dict[MetricNode, float]:
        """
        This function propagates uncertainty using a UQ method for a single design vector.
        This function should be called inside _evaluate() per each objective
        Input:
            - uq_method: UQMethod instance
            - func: Objective/Constraint function with the following format func(dsg: DSGType, param_sample: Dict[UncertainParameterNode, float]) -> float
            - dsg: DSG instance
            - metric_node: MetricNode instance
            **kwargs: N samples for Monte Carlo evaluation or any other relevant parameters for the chosen UQ method.
        Output:
            - Returns a mapping from metric node to float
        """
        stochastic_metric_node = {}
        mean, std = UQMethod.run(dsg, uq_method, func, **kwargs)
        dsg.set_metric_statistics(metric_node, mean=mean, std=std, method=uq_method, n_samples=kwargs.get('n'))
        stochastic_metric_node[metric_node] = self.process_stochastic_qoi(metric_node=metric_node, mean=mean, std=std)
        return stochastic_metric_node

    def make_ot_model(self, x_out, is_active_out, f_out, g_out, h_out, *args, **kwargs):
        def wrapped(sample_list):
            sample = np.array(sample_list)
            result = self._arch_evaluate_sample(
                x_out, is_active_out, f_out, g_out, h_out, *args, sample=sample, **kwargs
            )
            return [result]

        return ot.PythonFunction(self.param_space.n_parameters, len(self.stochastic_output), wrapped)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates a set of design vectors (provided as matrix). Outputs:
        X: imputed design vectors
        is_active: activeness matrix, specifying for each design variable whether it is active or not
        F: objective values
        G: inequality constraint values
        H: equality constraint values
        """
        # Prepare output matrices for evaluation
        x_out: np.ndarray = x.copy()
        self.design_space.round_x_discrete(x_out)
        is_active_out = np.ones(x.shape, dtype=bool)

        f_out = np.zeros((x.shape[0], self.n_obj))*np.nan
        g_out = np.zeros((x.shape[0], self.n_ieq_constr))*np.nan
        h_out = np.zeros((x.shape[0], self.n_eq_constr))*np.nan

        # If the design space definition is explicit, it means that that is all we need to correct and impute, and we
        # prevent subsequent changing of the inputs
        if self.design_space.is_explicit():
            self._correct_x_impute(x_out, is_active_out)
            x_out.setflags(write=False)
            is_active_out.setflags(write=False)

        # Call evaluation function
        self._arch_evaluate(x_out, is_active_out, f_out, g_out, h_out, *args, **kwargs)

        # Provide outputs to pymoo
        out['X'] = x_out
        out['is_active'] = is_active_out
        out['F'] = f_out
        if self.n_ieq_constr > 0:
            out['G'] = g_out
        if self.n_eq_constr > 0:
            out['H'] = h_out

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        def model(sample):
            return self._arch_evaluate_sample(
                x, is_active_out, f_out, g_out, h_out, *args, sample, **kwargs
            )

        self.uq_method.run(model)



    def _arch_evaluate_sample(self, x_out, is_active_out, f_out, g_out, h_out, *args, sample, **kwargs) -> Tuple[List[float], List[float]]:
        raise NotImplementedError

