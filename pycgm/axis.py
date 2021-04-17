# -*- coding: utf-8 -*-
import numpy as np


def pelvis_axis(rasi, lasi, rpsi, lpsi, sacr=None):
    r"""Make the Pelvis Axis.

    Takes in RASI, LASI, RPSI, LPSI, and optional SACR markers.
    Calculates the pelvis axis.

    Markers used: RASI, LASI, RPSI, LPSI
    Other landmarks used: sacrum

    Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure
    [1]_ and then normalized.
    Pelvis Y_axis: LASI-RASI x,y,z positions, then normalized.
    Pelvis Z_axis: Cross product of x_axis and y_axis.

    :math:`$o = m_{rasi} + m_{lasi} / 2$`

    :math:`$y = \frac{m_{lasi} - m_{rasi}}{||m_{lasi} - m_{rasi}||}$`

    :math:`x = \frac{(m_{origin} - m_{sacr}) - ((m_{origin} - m_{sacr}) \dot y) * y}{||(m_{origin} - m_{sacr}) - ((m_{origin} - m_{sacr}) \cdot y) \times y||}`

    :math:`z = x \times y`

    Parameters
    ----------
    rasi: array
        1x3 RASI marker
    lasi: array
        1x3 LASI marker
    rpsi: array
        1x3 RPSI marker
    lpsi: array
        1x3 LPSI marker
    sacr: array, optional
        1x3 SACR marker. If not present, RPSI and LPSI are used instead.

    Returns
    -------
    pelvis : array
        4x4 affine matrix with pelvis x, y, z axes and pelvis origin.

    .. math::

        \begin{bmatrix}
            \hat{x}_x & \hat{x}_y & \hat{x}_z & o_x \\
            \hat{y}_x & \hat{y}_y & \hat{y}_z & o_y \\
            \hat{z}_x & \hat{z}_y & \hat{z}_z & o_z \\
            0 & 0 & 0 & 1 \\
        \end{bmatrix}

    References
    ----------
    .. [1] M. P. Kadaba, H. K. Ramakrishnan, and M. E. Wootten, “Measurement of
            lower extremity kinematics during level walking,” J. Orthop. Res.,
            vol. 8, no. 3, pp. 383–392, May 1990, doi: 10.1002/jor.1100080310.

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import pelvis_axis
    >>> rasi = np.array([ 395.36,  428.09, 1036.82])
    >>> lasi = np.array([ 183.18,  422.78, 1033.07])
    >>> rpsi = np.array([ 341.41,  246.72, 1055.99])
    >>> lpsi = np.array([ 255.79,  241.42, 1057.30])
    >>> [arr.round(2) for arr in pelvis_axis(rasi, lasi, rpsi, lpsi, None)] # doctest: +NORMALIZE_WHITESPACE
    [array([-2.0000e-02,  9.9000e-01, -1.2000e-01,  2.8927e+02]),
    array([-1.0000e+00, -3.0000e-02, -2.0000e-02,  4.2543e+02]),
    array([-2.00000e-02,  1.20000e-01,  9.90000e-01,  1.03494e+03]),
    array([0., 0., 0., 1.])]
    """
    # Get the Pelvis Joint Centre

    if sacr is None:
        sacr = (rpsi + lpsi) / 2.0

    # REQUIRED LANDMARKS:
    # sacrum

    # Origin is Midpoint between RASI and LASI
    o = (rasi+lasi)/2.0

    b1 = o - sacr
    b2 = lasi - rasi

    # y is normalized b2
    y = b2 / np.linalg.norm(b2)

    b3 = b1 - (np.dot(b1, y) * y)
    x = b3/np.linalg.norm(b3)

    # Z-axis is cross product of x and y vectors.
    z = np.cross(x, y)

    pelvis = np.zeros((4, 4))
    pelvis[3, 3] = 1.0
    pelvis[0, :3] = x
    pelvis[1, :3] = y
    pelvis[2, :3] = z
    pelvis[:3, 3] = o

    return pelvis
