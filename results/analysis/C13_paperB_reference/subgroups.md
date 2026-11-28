# Subgroups: Dice on fractured femurs (missed = 0)

Smallest third: each part's own lower tertile of released mask area among fractured femurs. Paired: the fractured femur of a two-femur x-ray whose other femur is healthy. Wilcoxon (Pratt); Holm across the pairs, per subgroup and scope.

## oof (area cut 72691 px)

| Subgroup | n | S10_m0_convnext_unet | R2_random_ref | R3_nearest_ref | R4_refrac |
|---|---|---|---|---|---|
| smallest third of masks | 227 | 0.712 (0.680–0.740) | 0.724 (0.696–0.749) | 0.712 (0.681–0.740) | 0.713 (0.681–0.740) |
| larger two thirds | 453 | 0.809 (0.796–0.820) | 0.809 (0.797–0.820) | 0.807 (0.794–0.819) | 0.809 (0.796–0.821) |
| fractured, healthy partner | 334 | 0.784 (0.766–0.802) | 0.790 (0.773–0.807) | 0.784 (0.764–0.801) | 0.780 (0.760–0.799) |
| fractured, single | 346 | 0.769 (0.750–0.786) | 0.772 (0.755–0.788) | 0.768 (0.750–0.785) | 0.774 (0.757–0.790) |

| Subgroup | A vs B | Difference (95% CI) | p | p (Holm) |
|---|---|---|---|---|
| smallest third of masks | R2_random_ref vs S10_m0_convnext_unet | +0.012 (-0.003 to +0.029) | 0.876 | 1.000 |
| smallest third of masks | R3_nearest_ref vs R2_random_ref | -0.012 (-0.024 to -0.002) | 0.377 | 1.000 |
| smallest third of masks | R4_refrac vs R3_nearest_ref | +0.001 (-0.009 to +0.011) | 0.876 | 1.000 |
| smallest third of masks | R4_refrac vs S10_m0_convnext_unet | +0.001 (-0.014 to +0.015) | 0.461 | 1.000 |
| larger two thirds | R2_random_ref vs S10_m0_convnext_unet | +0.000 (-0.007 to +0.008) | 0.447 | 1.000 |
| larger two thirds | R3_nearest_ref vs R2_random_ref | -0.002 (-0.008 to +0.003) | 0.733 | 1.000 |
| larger two thirds | R4_refrac vs R3_nearest_ref | +0.001 (-0.004 to +0.008) | 0.650 | 1.000 |
| larger two thirds | R4_refrac vs S10_m0_convnext_unet | +0.000 (-0.008 to +0.009) | 0.379 | 1.000 |
| fractured, healthy partner | R2_random_ref vs S10_m0_convnext_unet | +0.006 (-0.004 to +0.017) | 0.414 | 0.980 |
| fractured, healthy partner | R3_nearest_ref vs R2_random_ref | -0.007 (-0.014 to -0.000) | 0.141 | 0.564 |
| fractured, healthy partner | R4_refrac vs R3_nearest_ref | -0.003 (-0.011 to +0.004) | 0.840 | 0.980 |
| fractured, healthy partner | R4_refrac vs S10_m0_convnext_unet | -0.004 (-0.016 to +0.008) | 0.327 | 0.980 |
| fractured, single | R2_random_ref vs S10_m0_convnext_unet | +0.003 (-0.007 to +0.013) | 0.133 | 0.534 |
| fractured, single | R3_nearest_ref vs R2_random_ref | -0.004 (-0.012 to +0.003) | 0.649 | 1.000 |
| fractured, single | R4_refrac vs R3_nearest_ref | +0.006 (-0.001 to +0.013) | 0.589 | 1.000 |
| fractured, single | R4_refrac vs S10_m0_convnext_unet | +0.005 (-0.004 to +0.014) | 0.551 | 1.000 |

## test_ensemble (area cut 49444 px)

| Subgroup | n | S10_m0_convnext_unet | R2_random_ref | R3_nearest_ref | R4_refrac |
|---|---|---|---|---|---|
| smallest third of masks | 49 | 0.661 (0.620–0.697) | 0.653 (0.613–0.688) | 0.659 (0.619–0.694) | 0.656 (0.616–0.691) |
| larger two thirds | 96 | 0.786 (0.754–0.814) | 0.785 (0.753–0.812) | 0.784 (0.752–0.810) | 0.781 (0.747–0.810) |
| fractured, healthy partner | 64 | 0.731 (0.689–0.767) | 0.727 (0.685–0.763) | 0.731 (0.691–0.766) | 0.730 (0.689–0.765) |
| fractured, single | 81 | 0.754 (0.719–0.783) | 0.751 (0.717–0.780) | 0.750 (0.714–0.779) | 0.746 (0.708–0.778) |

| Subgroup | A vs B | Difference (95% CI) | p | p (Holm) |
|---|---|---|---|---|
| smallest third of masks | R2_random_ref vs S10_m0_convnext_unet | -0.007 (-0.013 to -0.002) | 0.006 | 0.026 |
| smallest third of masks | R3_nearest_ref vs R2_random_ref | +0.006 (+0.001 to +0.011) | 0.016 | 0.047 |
| smallest third of masks | R4_refrac vs R3_nearest_ref | -0.003 (-0.010 to +0.003) | 0.136 | 0.136 |
| smallest third of masks | R4_refrac vs S10_m0_convnext_unet | -0.005 (-0.013 to +0.004) | 0.044 | 0.089 |
| larger two thirds | R2_random_ref vs S10_m0_convnext_unet | -0.001 (-0.007 to +0.005) | 0.531 | 1.000 |
| larger two thirds | R3_nearest_ref vs R2_random_ref | -0.001 (-0.007 to +0.004) | 0.285 | 1.000 |
| larger two thirds | R4_refrac vs R3_nearest_ref | -0.002 (-0.016 to +0.006) | 0.512 | 1.000 |
| larger two thirds | R4_refrac vs S10_m0_convnext_unet | -0.005 (-0.019 to +0.005) | 0.996 | 1.000 |
| fractured, healthy partner | R2_random_ref vs S10_m0_convnext_unet | -0.004 (-0.011 to +0.003) | 0.205 | 0.560 |
| fractured, healthy partner | R3_nearest_ref vs R2_random_ref | +0.004 (-0.001 to +0.010) | 0.078 | 0.312 |
| fractured, healthy partner | R4_refrac vs R3_nearest_ref | -0.002 (-0.006 to +0.003) | 0.187 | 0.560 |
| fractured, healthy partner | R4_refrac vs S10_m0_convnext_unet | -0.001 (-0.008 to +0.005) | 0.196 | 0.560 |
| fractured, single | R2_random_ref vs S10_m0_convnext_unet | -0.002 (-0.008 to +0.003) | 0.679 | 1.000 |
| fractured, single | R3_nearest_ref vs R2_random_ref | -0.002 (-0.007 to +0.004) | 0.540 | 1.000 |
| fractured, single | R4_refrac vs R3_nearest_ref | -0.003 (-0.019 to +0.007) | 0.501 | 1.000 |
| fractured, single | R4_refrac vs S10_m0_convnext_unet | -0.007 (-0.025 to +0.004) | 0.575 | 1.000 |

