from math import pi, sin, cos, radians
import numpy as np
import math

def rotmat(x=0, y=0, z=0):
    r"""Rotation Matrix.

    This function creates and returns a rotation matrix.

    Parameters
    ----------
    x, y, z : float, optional
        Angle, which will be converted to radians, in
        each respective axis to describe the rotations.
        The default is 0 for each unspecified angle.

    Returns
    -------
    r_xyz : list
        The product of the matrix multiplication.

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import rotmat
    >>> x = 0.5
    >>> y = 0.3
    >>> z = 0.8
    >>> np.around(rotmat(x, y, z), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  , -0.01,  0.01],
    [ 0.01,  1.  , -0.01],
    [-0.01,  0.01,  1.  ]])
    >>> x = 0.5
    >>> np.around(rotmat(x), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  ,  0.  ,  0.  ],
    [ 0.  ,  1.  , -0.01],
    [ 0.  ,  0.01,  1.  ]])
    >>> x = 1
    >>> y = 1
    >>> np.around(rotmat(x,y), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  ,  0.  ,  0.02],
    [ 0.  ,  1.  , -0.02],
    [-0.02,  0.02,  1.  ]])
    """
    x, y, z = math.radians(x), math.radians(y), math.radians(z)
    r_x = [ [1,0,0],[0,math.cos(x),math.sin(x)*-1],[0,math.sin(x),math.cos(x)] ]
    r_y = [ [math.cos(y),0,math.sin(y)],[0,1,0],[math.sin(y)*-1,0,math.cos(y)] ]
    r_z = [ [math.cos(z),math.sin(z)*-1,0],[math.sin(z),math.cos(z),0],[0,0,1] ]
    r_xy = np.matmul(r_x,r_y)
    r_xyz = np.matmul(r_xy,r_z)

    return r_xyz

def get_angle(axis_p, axis_d):
    r"""Normal angle calculation.

    This function takes in two axes and returns three angles and uses the
    inverse Euler rotation matrix in YXZ order.

    Returns the angles in degrees.

    As we use arcsin we have to care about if the angle is in area between -pi/2 to pi/2
    
    :math:`\alpha = \arcsin{(-axis\_d_{z} \cdot axis\_p_{y})}`

    If alpha is between -pi/2 and pi/2

    :math:`\beta = \arctan2{((axis\_d_{z} \cdot axis\_p_{x}), axis\_d_{z} \cdot axis\_p_{z})}`
    
    :math:`\gamma = \arctan2{((axis\_d_{y} \cdot axis\_p_{y}), axis\_d_{x} \cdot axis\_p_{y})}`

    Otherwise

    :math:`\beta = \arctan2{(-(axis\_d_{z} \cdot axis\_p_{x}), axis\_d_{z} \cdot axis\_p_{z})}`

    :math:`\gamma = \arctan2{(-(axis\_d_{y} \cdot axis\_p_{y}), axis\_d_{x} \cdot axis\_p_{y})}`

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
    >>> axis_p = [[ 0.04,   0.99,  0.06, 429.67],
    ...         [ 0.99, -0.04, -0.05, 275.15],
    ...         [-0.05,  0.07, -0.99, 1452.95],
    ...         [0, 0, 0, 1]]
    >>> axis_d = [[-0.18, -0.98, -0.02, 64.09],
    ...         [ 0.71, -0.11,  -0.69, 275.83],
    ...         [ 0.67, -0.14,   0.72, 1463.78],
    ...         [0, 0, 0, 1]]
    >>> np.around(get_angle(axis_p, axis_d), 2)
    array([-174.82,  -39.26,  100.54])
    """
    # this is the angle calculation which order is Y-X-Z, alpha is the abdcution angle.

    ang = (
        (-1 * axis_d[2][0] * axis_p[1][0])
        + (-1 * axis_d[2][1] * axis_p[1][1])
        + (-1 * axis_d[2][2] * axis_p[1][2])
    )

    alpha = np.nan
    if -1 <= ang <= 1:
        alpha = np.arcsin(ang)

    # check the abduction angle is in the area between -pi/2 and pi/2
    # beta is flextion angle, gamma is rotation angle

    if -1.57079633 < alpha < 1.57079633:
        beta = np.arctan2(
            (axis_d[2][0] * axis_p[0][0])
            + (axis_d[2][1] * axis_p[0][1])
            + (axis_d[2][2] * axis_p[0][2]),

            (axis_d[2][0] * axis_p[2][0])
            + (axis_d[2][1] * axis_p[2][1])
            + (axis_d[2][2] * axis_p[2][2])
        )

        gamma = np.arctan2(
            (axis_d[1][0] * axis_p[1][0])
            + (axis_d[1][1] * axis_p[1][1])
            + (axis_d[1][2] * axis_p[1][2]),

            (axis_d[0][0] * axis_p[1][0])
            + (axis_d[0][1] * axis_p[1][1])
            + (axis_d[0][2] * axis_p[1][2])
        )
    else:
        beta = np.arctan2(
            -1 * (
                (axis_d[2][0] * axis_p[0][0])
                + (axis_d[2][1] * axis_p[0][1])
                + (axis_d[2][2] * axis_p[0][2])
            ),
                (axis_d[2][0] * axis_p[2][0])
                + (axis_d[2][1] * axis_p[2][1])
                + (axis_d[2][2] * axis_p[2][2])
        )
        gamma = np.arctan2(
            -1 * (
                (axis_d[1][0] * axis_p[1][0])
                + (axis_d[1][1] * axis_p[1][1])
                + (axis_d[1][2] * axis_p[1][2])
            ),
                (axis_d[0][0] * axis_p[1][0])
                + (axis_d[0][1] * axis_p[1][1])
                + (axis_d[0][2] * axis_p[1][2])
        )

    angle = [180.0*beta/pi, 180.0*alpha / pi, 180.0*gamma/pi]

    return angle