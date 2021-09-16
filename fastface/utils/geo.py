import math

import numpy as np


def get_rotation_matrix(degree: float) -> np.ndarray:
    rad = math.radians(degree)
    sinq = math.sin(rad)
    cosq = math.cos(rad)
    return np.array([[cosq, sinq], [-1 * sinq, cosq]], dtype=np.float32)
