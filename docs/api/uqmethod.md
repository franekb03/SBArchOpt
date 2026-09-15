# Uncertainty Quantification Methods

::: sb_arch_opt.uncertainty.UQMethod
    handler: python
    options:
        heading_level: 2
        members:
            - n_evaluations
            - seed
            - _samples
            - get_samples
            - _draw_samples
            - resample
            - process_results

::: sb_arch_opt.uncertainty.MonteCarlo
    handler: python
    options:
        heading_level: 2
        members:
            - _draw_samples
            - process_results

::: sb_arch_opt.uncertainty.PolynomialChaos
    handler: python
    options:
        heading_level: 2
        members:
            - degree
            - n_metamodel_samples
            - _metamodel_input
            - n_terms
            - _validate
            - _draw_samples
            - _get_metamodel_input
            - _build_algorithm
            - process_results
            