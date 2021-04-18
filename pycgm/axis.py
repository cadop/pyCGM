# -*- coding: utf-8 -*-
import math
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


def hip_joint_center(pelvis, subject):
    u"""Get the right and left hip joint center.

    Takes in a 4x4 affine matrix of pelvis axis and subject measurements
    dictionary. Calculates and returns the left and right hip joint centers.

    Subject Measurement values used: MeanLegLength, R_AsisToTrocanterMeasure,
    InterAsisDistance, L_AsisToTrocanterMeasure

    Hip Joint Center: Computed using Hip Joint Center Calculation [1]_.

    Parameters
    ----------
    pelvis : array
        A 4x4 affine matrix with pelvis x, y, z axes and pelvis origin.
    subject : dict
        A dictionary containing subject measurements.

    Returns
    -------
    hip_jc : array
        A 2x3 array that contains two 1x3 arrays
        containing the x, y, z components of the left and right hip joint
        centers.

    References
    ----------
    .. [1] Davis, R. B., III, Õunpuu, S., Tyburski, D. and Gage, J. R. (1991).
            A gait analysis data collection and reduction technique.
            Human Movement Science 10 575–87.

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import hip_joint_center
    >>> vsk = {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.51,
    ...        'L_AsisToTrocanterMeasure': 72.51, 'InterAsisDistance': 215.90}
    >>> pelvis_axis = np.array([
    ...     [0.14, 0.98, -0.11, 251.60],
    ...     [-0.99, 0.13, -0.02, 391.74],
    ...     [0, 0.1, 0.99, 1032.89],
    ...     [0, 0, 0, 1]
    ... ])
    >>> np.around(hip_joint_center(pelvis_axis,vsk), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[181.71, 340.33, 936.18],
    [307.36, 323.83, 938.72]])
    """

    # Requires
    # pelvis axis

    pel_origin = pelvis[:3, 3]

    # Model's eigen value
    #
    # LegLength
    # MeanLegLength
    # mm (marker radius)
    # interAsisMeasure

    # Set the variables needed to calculate the joint angle
    # Half of marker size
    mm = 7.0

    mean_leg_length = subject['MeanLegLength']
    right_asis_to_trochanter = subject['R_AsisToTrocanterMeasure']
    left_asis_to_trochanter = subject['L_AsisToTrocanterMeasure']
    interAsisMeasure = subject['InterAsisDistance']
    C = (mean_leg_length * 0.115) - 15.3
    theta = 0.500000178813934
    beta = 0.314000427722931
    aa = interAsisMeasure/2.0
    S = -1

    # Hip Joint Center Calculation (ref. Davis_1991)

    # Left: Calculate the distance to translate along the pelvis axis
    L_Xh = (-left_asis_to_trochanter - mm) * \
        math.cos(beta) + C * math.cos(theta) * math.sin(beta)
    L_Yh = S*(C*math.sin(theta) - aa)
    L_Zh = (-left_asis_to_trochanter - mm) * \
        math.sin(beta) - C * math.cos(theta) * math.cos(beta)

    # Right:  Calculate the distance to translate along the pelvis axis
    R_Xh = (-right_asis_to_trochanter - mm) * \
        math.cos(beta) + C * math.cos(theta) * math.sin(beta)
    R_Yh = (C*math.sin(theta) - aa)
    R_Zh = (-right_asis_to_trochanter - mm) * \
        math.sin(beta) - C * math.cos(theta) * math.cos(beta)

    # get the unit pelvis axis
    pelvis_xaxis = pelvis[0, :3]
    pelvis_yaxis = pelvis[1, :3]
    pelvis_zaxis = pelvis[2, :3]
    pelvis_axis = pelvis[:3, :3]

    # multiply the distance to the unit pelvis axis
    left_hip_jc_x = pelvis_xaxis * L_Xh
    left_hip_jc_y = pelvis_yaxis * L_Yh
    left_hip_jc_z = pelvis_zaxis * L_Zh
    # left_hip_jc = left_hip_jc_x + left_hip_jc_y + left_hip_jc_z

    left_hip_jc = np.asarray([
        left_hip_jc_x[0]+left_hip_jc_y[0]+left_hip_jc_z[0],
        left_hip_jc_x[1]+left_hip_jc_y[1]+left_hip_jc_z[1],
        left_hip_jc_x[2]+left_hip_jc_y[2]+left_hip_jc_z[2]
    ])

    left_hip_jc = pelvis_axis.T @ np.array([L_Xh, L_Yh, L_Zh])

    R_hipJCx = pelvis_xaxis * R_Xh
    R_hipJCy = pelvis_yaxis * R_Yh
    R_hipJCz = pelvis_zaxis * R_Zh
    right_hip_jc = R_hipJCx + R_hipJCy + R_hipJCz

    right_hip_jc = pelvis_axis.T @ np.array([R_Xh, R_Yh, R_Zh])

    left_hip_jc = left_hip_jc+pel_origin
    right_hip_jc = right_hip_jc+pel_origin

    hip_jc = np.array([left_hip_jc, right_hip_jc])

    return hip_jc


def hip_axis(r_hip_jc, l_hip_jc, pelvis_axis):
    r"""Make the hip axis.

    Takes in the x, y, z positions of left and right hip joint center and
    pelvis axis to calculate the hip axis.

    hip origin: Midpoint of left and right hip joint centers.

    Hip axis: sets the pelvis orientation to the hip center axis (i.e.
    midpoint of left and right hip joint centers)

    Parameters
    ----------
    l_hip_jc, r_hip_jc : array
        left and right hip joint center with x, y, z position in an array.
    pelvis_axis : array
        4x4 affine matrix with pelvis x, y, z axes and pelvis origin.

    Returns
    -------
    axis : array
        4x4 affine matrix with hip x, y, z axes and hip origin.

    .. math::

        \begin{bmatrix}
            \hat{x}_x & \hat{x}_y & \hat{x}_z & o_x \\
            \hat{y}_x & \hat{y}_y & \hat{y}_z & o_y \\
            \hat{z}_x & \hat{z}_y & \hat{z}_z & o_z \\
            0 & 0 & 0 & 1 \\
        \end{bmatrix}


    Examples
    --------
    >>> import numpy as np
    >>> from .axis import hip_axis
    >>> r_hip_jc = [182.57, 339.43, 935.52]
    >>> l_hip_jc = [308.38, 322.80, 937.98]
    >>> pelvis_axis = np.array([
    ...     [0.14, 0.98, -0.11, 251.60],
    ...     [-0.99, 0.13, -0.02, 391.74],
    ...     [0, 0.1, 0.99, 1032.89],
    ...     [0, 0, 0, 1]
    ... ])
    >>> [np.around(arr, 2) for arr in hip_axis(l_hip_jc,r_hip_jc, pelvis_axis)] #doctest: +NORMALIZE_WHITESPACE
    [array([ 1.4000e-01,  9.8000e-01, -1.1000e-01,  2.4548e+02]),
    array([-9.9000e-01,  1.3000e-01, -2.0000e-02,  3.3112e+02]),
    array([0.0000e+00, 1.0000e-01, 9.9000e-01, 9.3675e+02]),
    array([0., 0., 0., 1.])]
    """

    # Get shared hip axis, it is inbetween the two hip joint centers
    hipaxis_center = (np.asarray(r_hip_jc) + np.asarray(l_hip_jc)) / 2.0

    # convert pelvis_axis to x,y,z axis to use more easy
    pelvis_x_axis = pelvis_axis[0, :3]
    pelvis_y_axis = pelvis_axis[1, :3]
    pelvis_z_axis = pelvis_axis[2, :3]

    axis = np.zeros((4, 4))
    axis[3, 3] = 1.0
    axis[0, :3] = pelvis_x_axis
    axis[1, :3] = pelvis_y_axis
    axis[2, :3] = pelvis_z_axis
    axis[:3, 3] = hipaxis_center

    return axis
