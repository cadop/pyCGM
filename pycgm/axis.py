from math import pi, sin, cos, radians
import numpy as np
import math


def get_spine_angle(axis_p, axis_d):
    r"""Spine angle calculation.

    This function takes in two axes and returns three angles and uses the
    inverse Euler rotation matrix in YXZ order.
    Returns the angles in degrees.

    .. math::
        \[ alpha = \arcsin{(axisD_{y} \cdot axisP_{z})} \]
        \[ gamma = \arcsin{(-(axisD_{y} \cdot axisP_{x}) / \cos{\alpha})} \]
        \[ beta = \arcsin{(-(axisD_{x} \cdot axisP_{z}) / \cos{\alpha})} \]

    Parameters
    ----------
    axis_p : list
        Shows the unit vector of axis_p, the position of the proximal axis.
    axis_d : list
        Shows the unit vector of axis_d, the position of the distal axis.

    Returns
    -------
    angle : list
        Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list.

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import get_spine_angle
    >>> axis_p = [[ 0.04,   0.99,  0.06, 749.24],
    ...        [ 0.99, -0.04, -0.05, 321.12],
    ...        [-0.05,  0.07, -0.99, 145.12],
    ...        [0, 0, 0, 1]]
    >>> axis_d = [[-0.18, -0.98,-0.02, 541.68],
    ...        [ 0.71, -0.11,  -0.69, 112.48],
    ...        [ 0.67, -0.14,   0.72, 155.77],
    ...        [0, 0, 0, 1]]
    >>> np.around(get_spine_angle(axis_p,axis_d), 2)
    array([ 2.97,  9.13, 39.78])
    """
    # this angle calculation is for spine angle.

    alpha = np.arcsin(
            (axis_d[1][0] * axis_p[2][0])
            + (axis_d[1][1] * axis_p[2][1])
            + (axis_d[1][2] * axis_p[2][2])
    )

    gamma = np.arcsin(
        (
            (-1 * axis_d[1][0] * axis_p[0][0])
            + (-1 * axis_d[1][1] * axis_p[0][1])
            + (-1 * axis_d[1][2] * axis_p[0][2])
        ) / np.cos(alpha)
    )

    beta = np.arcsin(
        (
            (-1 * axis_d[0][0] * axis_p[2][0])
            + (-1 * axis_d[0][1] * axis_p[2][1])
            + (-1 * axis_d[0][2] * axis_p[2][2])
        ) / np.cos(alpha)
    )

    angle = [180.0 * beta / pi, 180.0 * gamma / pi, 180.0 * alpha / pi]

    return angle
