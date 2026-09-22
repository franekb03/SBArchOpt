# Stochastic Problem Definition

Stochastic (robust) optimization uses [OpenTURNS](https://openturns.github.io/) for the uncertainty propagation,
which is an optional dependency:

```
pip install sb-arch-opt[uncertainty]
```



::: sb_arch_opt.stochastic_problem.StochasticArchOptProblem
    handler: python
    options:
        heading_level: 2
        members:
            - design_space
            - param_space
            - uq_method
            - obj_scalar
            - ieq_constr_scalar
            - eq_constr_scalar
            - scalars
            - stochastic_results
            - _arch_evaluate
            - _arch_evaluate_sample

::: sb_arch_opt.uncertainty.Scalarization
    handler: python

::: sb_arch_opt.uncertainty.Mean
    handler: python

::: sb_arch_opt.uncertainty.Margin
    handler: python
    options:
        heading_level: 2
        members:
            - k
            - direction

::: sb_arch_opt.uncertainty.Quantile
    handler: python
    options:
        heading_level: 2
        members:
            - q

::: sb_arch_opt.uncertainty.StochasticParameter
    handler: python
    options:
        heading_level: 2
        members:
            - name
            - value
            - _sample
            - sample

::: sb_arch_opt.uncertainty.StochasticParameterSpace
    handler: python
    options:
        heading_level: 2
        members:
            - _parameters
            - n_parameters
            - parameter_names
            - joint_dist
            - param_realization
            - get_random_samples
            - get_lhs_samples

::: sb_arch_opt.uncertainty.StochasticOutput
    handler: python
    options:
        heading_level: 2
        members:
            - output_samples
            - from_results
            - mean
            - std
            - var
            - quantile
            - prob_exceeds
            - margin
            - to_distribution
            - to_numpy
            - reduce
            - __str__

::: sb_arch_opt.uncertainty.StochasticResults
    handler: python
    options:
        heading_level: 2
        members:
            - outputs
            - method_result