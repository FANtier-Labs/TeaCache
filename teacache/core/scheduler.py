"""Official FLUX-dev relative-L1 calibration (ali-vilab/TeaCache, Apache-2.0)."""
COEFFICIENTS = (4.98651651e2, -2.83781631e2, 5.58554382e1, -3.82021401, 2.64230861e-1)


def rescale(distance: float) -> float:
    result = 0.0
    for coefficient in COEFFICIENTS:
        result = result * distance + coefficient
    return result
