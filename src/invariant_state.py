from math import atan, atan2, cos, sin

import numpy as np


def coordinate_in_path_ref(p, dir, x):

    theta = atan2(dir[1], dir[0])

    x = np.array(x) - np.array(p)
    sin_th = sin(theta)
    cos_th = cos(theta)

    R = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
    R = R.T

    return R @ x


def coordinate_in_global_ref(p, dir, x):
    theta = atan2(dir[1], dir[0])
    sin_th = sin(theta)
    cos_th = cos(theta)

    R = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
    x = R @ x + p

    return x


def coordinate_in_path_ref_3D(p, x, t, n, b):
    B = np.column_stack((t, n, b))
    x = np.array(x) - np.array(p)
    return B.T @ x


def coordinate_in_global_ref_3D(p, x, t, n, b):
    B = np.column_stack((t, n, b))
    x = B @ x + p
    return x
