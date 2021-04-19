# -*- coding: utf-8 -*-

from math import pi
import numpy as np

def get_angle(axis_p, axis_d):
    """Normal angle calculation.
    This function takes in two axes and returns three angles and uses the
    inverse Euler rotation matrix in YXZ order.
    Returns the angles in degrees.
    As we use arc sin we have to care about if the angle is in area between -pi/2 to pi/2
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
    >>> from .axis import get_angle
    >>> axis_p = [[ 0.04,   0.99,  0.06],
    ...         [ 0.99, -0.04, -0.05],
    ...         [-0.05,  0.07, -0.99]]
    >>> axis_d = [[-0.18, -0.98, -0.02],
    ...         [ 0.71, -0.11,  -0.69],
    ...         [ 0.67, -0.14,   0.72 ]]
    >>> np.around(get_angle(axis_p, axis_d), 2)
    array([-174.82,  -39.26,  100.54])
    """
    # this is the angle calculation which is in order Y-X-Z

    # alpha is abduction angle.

    ang = (-1*axis_d[2][0]*axis_p[1][0])+(-1*axis_d[2][1]*axis_p[1][1])+(-1*axis_d[2][2]*axis_p[1][2])
    alpha = np.nan
    if -1 <= ang <= 1:
        alpha = np.arcsin(ang)

    # ensure the abduction angle is between -pi/2 and pi/2
    # beta is flextion angle
    # gamma is rotation angle

    if (-pi/2) < alpha < (pi/2):
        beta = np.arctan2(
                (axis_d[2][0]*axis_p[0][0])+(axis_d[2][1]*axis_p[0][1])+(axis_d[2][2]*axis_p[0][2]),
                (axis_d[2][0]*axis_p[2][0])+(axis_d[2][1]*axis_p[2][1])+(axis_d[2][2]*axis_p[2][2])
            )

        gamma = np.arctan2(
                (axis_d[1][0]*axis_p[1][0])+(axis_d[1][1]*axis_p[1][1])+(axis_d[1][2]*axis_p[1][2]),
                (axis_d[0][0]*axis_p[1][0])+(axis_d[0][1]*axis_p[1][1])+(axis_d[0][2]*axis_p[1][2])
            )
    else:
        beta = np.arctan2(
                -1*((axis_d[2][0]*axis_p[0][0])+(axis_d[2][1]*axis_p[0][1])+(axis_d[2][2]*axis_p[0][2])),
                (axis_d[2][0]*axis_p[2][0])+(axis_d[2][1]*axis_p[2][1])+(axis_d[2][2]*axis_p[2][2])
            )

        gamma = np.arctan2(
                -1*((axis_d[1][0]*axis_p[1][0])+(axis_d[1][1]*axis_p[1][1])+(axis_d[1][2]*axis_p[1][2])),
                (axis_d[0][0]*axis_p[1][0])+(axis_d[0][1]*axis_p[1][1])+(axis_d[0][2]*axis_p[1][2])
            )

    angle = [180.0*beta/pi, 180.0*alpha/pi, 180.0*gamma/pi]

    return angle
