import tests.spacetime.test1 as t1
import tests.spacetime.test2 as t2
import tests.spacetime.test3 as t3
import tests.spacetime.test4 as t4
import tests.spacetime.test5 as t5
import tests.spacetime.test6 as t6
import tests.spacetime.test7 as t7
import tests.spacetime.test8 as t8
import tests.spacetime.test9 as t9
import tests.spacetime.test10 as t10
import tests.spacetime.test11 as t11
import tests.spacetime.test12 as t12
import tests.spacetime.test13 as t13


if __name__ == "__main__":
    t1.test_single_source_isotropy_relativistic()
    t2.test_two_source_interference_relativistic()
    t3.test_light_cone_speed()
    t4.test_energy_conservation()
    t5.test_kappa_scaling()
    t6.test_cardinal_vs_diagonal()
    t7.test_global_time_dilation_frequency_scaling()
    t8.test_curvature_induced_time_dilation_local()
    t9.test_backreaction_stability_and_effect()
    t10.test_metric_weighted_unitarity_fixed_lapse()
    t11.test_weighted_norm_residual_separates_external_vs_backreaction()
    t12.test_gravity_field_causal_propagation()
    t13.test_coupled_gravity_delayed_frequency_shift()