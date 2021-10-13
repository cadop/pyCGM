# -*- coding: utf-8 -*-
"""
This file provides helper functions for static calculations.

Created on Tue Jul 28 16:55:25 2015

@author: cadop
"""
import numpy as np

def get_iad(rasi, lasi):
    r"""Get the Inter ASIS Distance (IAD)

    Calculates the Inter ASIS Distance from a given frame.

    Given the markers RASI and LASI, the Inter ASIS Distance is
    defined as:

    .. math::

        InterASISDist = \sqrt{(RASI_x-LASI_x)^2 + (RASI_y-LASI_y)^2 +
                        (RASI_z-LASI_z)^2}

    where :math:`RASI_x` is the x-coordinate of the RASI marker in frame.

    Parameters
    ----------
    rasi : array
        Array of marker data.
    lasi : array
        Array of marker data.

    Returns
    -------
    Inter Asis Distance : float
        The euclidian distance between the two markers.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import get_iad
    >>> rasi = np.array([395.37, 428.1, 1036.83])
    >>> lasi = np.array([183.19, 422.79, 1033.07])
    >>> np.around(get_iad(rasi, lasi), 2)
    212.28
    """

    return np.sqrt(np.sum([(rasi[i] - lasi[i])**2 for i in range(3)]))

def calculate_head_angle(head):
    r"""Static Head Calculation function

    This function first calculates the x, y, z axes of the head by
    subtracting the head origin from the given head axes. Then the offset
    angles are calculated. This is a y-rotation from the rotational matrix
    created by the matrix multiplication of the distal
    axis and the inverse of the proximal axis.

    The head axis is calculated as follows:

    ..math::

        head\_axis = \begin{bmatrix}
            head_{x1}-origin_x & head_{x2}-origin_y & head_{x3}-origin_z \\
            head_{y1}-origin_x & head_{y2}-origin_y & head_{y3}-origin_z \\
            head_{z1}-origin_x & head_{z2}-origin_y & head_{z3}-origin_z
            \end{bmatrix}\\

    The offset angle is defined as:

    .. math::

        \[ result = \arctan{\frac{M[0][2]}{M[2][2]}} \]

    where M is the rotation matrix produced from multiplying axisD and
    :math:`axisP^{-1}`

    Parameters
    ----------
    head : array
        An array containing the head axis and head origin.

    Returns
    -------
    offset : float
        The head offset angle for static calibration.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import calculate_head_angle
    >>> head = np.array([[[100.33, 83.39, 1484.08],
    ...        [98.97, 83.58, 1483.77],
    ...        [99.35, 82.64, 1484.76]],
    ...        [99.58, 82.79, 1483.8]], dtype=list)
    >>> np.around(calculate_head_angle(head), 2)
    0.28
    """

    # Calculate head_axis as above in the function description
    # [[head_axis_x1 - origin_x, ...], ...]
    head_axis = np.array([[head[0][y][x] - head[1][x] for x in range(3)]
                          for y in range(3)])

    # Inversion of [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    # calculate_head_angle permanently assumes an incorrect axis.
    inverted_global_axis = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

    # Calculate rotational matrix.
    rotation_matrix = np.matmul(head_axis, inverted_global_axis)

    # Return arctangent of angle y.
    with np.errstate(invalid='ignore', divide='ignore'):
        sine_y = rotation_matrix[0][2]
        cosine_y = np.nan_to_num(rotation_matrix[2][2])
        return np.arctan(sine_y/cosine_y)

def foot_joint_center(rtoe, ltoe, static_info, ankle_joint_center):
    (r"""Calculate the foot joint center and axis function.

    Takes in a dictionary of xyz positions and marker names, the ankle axis and
    knee axis.
    Calculates the foot joint axis by rotating the incorrect foot joint axes
    about the offset angle.
    Returns the foot axis origin and axis.

    In the case of the foot joint center, we've already made 2 kinds of axes
    for the static offset angle and then, we call this static offset angle as
    an input of this function for the dynamic trial.

    Special Cases:

    (anatomically uncorrected foot axis)
    If flat foot, make the reference markers instead of HEE marker whose height
    is the same as TOE marker's height. Else use the HEE marker for making Z
    axis.

    Markers used: RTOE,LTOE
    Other landmarks used: ANKLE_FLEXION_AXIS
    Subject Measurement values used: RightStaticRotOff, RightStaticPlantFlex,
    LeftStaticRotOff, LeftStaticPlantFlex

    The incorrect foot joint axes for both feet are calculated using the
    following calculations:
        z-axis = ankle joint center - TOE marker
        y-flex = ankle joint center flexion - ankle joint center
        x-axis = y-flex cross z-axis
        y-axis = z-axis cross x-axis
    Calculate the foot joint axis by rotating incorrect foot joint axes
    about offset angle.

    .. math::

        z_{axis} = ankle\_joint\_center - toe\_marker\\
        flexion\_axis = ankle\_joint\_center\_flexion - ankle\_joint\_center\\
        x_{axis} = flexion\_axis \times z_{axis}\\
        y_{axis} = z_{axis} \times x_{axis}\\
        \\
        \text{Rotated about the } y_{axis} \text{:}\\
        rotation = \begin{bmatrix}
        cos(\beta) * x_{axis}[0] + sin(\beta) * z_{axis}[0] & cos(\beta)
        * x_{axis}[1] + sin(\beta) * z_{axis}[1] & cos(\beta) * x_{axis}[2]
        + sin(\beta) * z_{axis}[2\\
        y_{axis}[0] & y_{axis}[1] & y_{axis}[2]\\
        -sin(\beta) * x_{axis}[0] + cos(\beta) * z_{axis}[0] & -sin(\beta)
        * x_{axis}[1] + cos(\beta) * z_{axis}[1] & -sin(\beta) * x_{axis}[2]
        + cos(\beta) * z_{axis}[2]
        \end{bmatrix}\\
        \\
        \text{Rotated about the } x_{axis} \text{:}\\
        rotation = \begin{bmatrix}
        cos(\alpha) * rotation[1][0] - sin(\alpha) * rotation[2][0]
        & cos(\alpha) * rotation[1][1] - sin(\alpha) * rotation[2][1]
        & cos(\alpha) * rotation[1][2] - sin(\alpha) * rotation[2][2]\\
        y_{axis}[0] & y_{axis}[1] & y_{axis}[2]\\
        sin(\alpha) * rotation[1][0] + cos(\alpha) * rotation[2][0]
        & sin(\alpha) * rotation[1][1] + cos(\alpha) * rotation[2][1]
        & sin(\alpha) * rotation[1][2] + cos(\alpha) * rotation[2][2]
        \end{bmatrix}

    Parameters
    ----------
    rtoe : array
        Array of marker data.
    ltoe : array
        Array of marker data.
    static_info : array
        An array containing offset angles.
    ankle_joint_center : array
        An array containing the x,y,z axes marker positions of the
        ankle joint center.

    Returns
    -------
    rtoe, ltoe, foot_axis : array
        Returns a list that contain the toe (right and left) markers in
        1x3 arrays of xyz values and a 2x3x3 array composed of the foot axis
        center x, y, and z axis components. The xyz axis components are 2x3
        arrays consisting of the axis center in the first dimension and the
        direction of the axis in the second dimension.

    Modifies
    --------
    Axis changes the following in the static info.

    You can set the static_info with the button and this will calculate the
    offset angles.
    The first setting, the foot axis shows the uncorrected foot anatomical
    reference axis(Z_axis point to the AJC from TOE).

    If you press the static_info button so if static_info is not None,
    then the static offset angles are applied to the reference axis.
    The reference axis is Z axis point to HEE from TOE

    Examples
    --------
    >>> import numpy as np
    >>> from .static import foot_joint_center
    >>> rtoe = np.array([442.82, 381.62, 42.66])
    >>> ltoe = np.array([39.44, 382.45, 41.79])
    >>> static_info = [[0.03, 0.15, 0],
    ...                [0.01, 0.02, 0]]
    >>> ankle_joint_center = [np.array([393.76, 247.68, 87.74]),
    ...                       np.array([98.75, 219.47, 80.63]),
    ...                       [[np.array([394.48, 248.37, 87.72]),
    ...                         np.array([393.07, 248.39, 87.62]),
    ...                         np.array([393.69, 247.78, 88.73])],
    ...                        [np.array([98.47, 220.43, 80.53]),
    ...                         np.array([97.79, 219.21, 80.76]),
    ...                         np.array([98.85, 219.60, 81.62])]]]
    >>> delta = 0
    >>> [np.around(arr,2) for arr in """
    """foot_joint_center(rtoe, ltoe,static_info,ankle_joint_center)] """
    """#doctest: +NORMALIZE_WHITESPACE
    [array([442.82, 381.62,  42.66]), array([ 39.44, 382.45,  41.79]), """
    """array([[[442.89, 381.76,  43.65],
            [441.89, 382.  ,  42.67],
            [442.45, 380.7 ,  42.82]],
           [[ 39.51, 382.68,  42.76],
            [ 38.5 , 382.15,  41.93],
            [ 39.76, 381.53,  41.99]]])]
    """)
    # REQUIRED MARKERS:
    # RTOE
    # LTOE
    toes = [rtoe, ltoe]

    # REQUIRE JOINT CENTER & AXIS
    # KNEE JOINT CENTER
    # ANKLE JOINT CENTER
    # ANKLE FLEXION AXIS

    # Dealing with the Incorrect Axis
    foot_axis = []
    for side in range(2):  # calculate the left and right foot axes
        # Z_axis taken from toe marker to Ankle Joint Center; then normalized.
        z_axis = np.array(ankle_joint_center[side]) - np.array(toes[side])
        with np.errstate(invalid='ignore'):
            z_axis = np.divide(z_axis, np.nan_to_num(np.linalg.norm(z_axis)))

        # Calculated flexion axis from Ankle Joint Center; then normalized.
        flexion_axis = (np.array(ankle_joint_center[2][side][1])
                        - np.array(ankle_joint_center[side]))
        with np.errstate(invalid='ignore'):
            flexion_axis = np.divide(flexion_axis,
                                     np.nan_to_num(np.linalg.norm(flexion_axis)))

        # X_axis taken from cross product of Z_axis and the flexion axis.
        x_axis = np.cross(flexion_axis, z_axis)
        with np.errstate(invalid='ignore'):
            x_axis = np.divide(x_axis, np.nan_to_num(np.linalg.norm(x_axis)))

        # Y_axis take from cross product of Z_axis and X_axis (perpendicular).
        # Then normalized.
        y_axis = np.cross(z_axis, x_axis)
        with np.errstate(invalid='ignore'):
            y_axis = np.divide(y_axis, np.nan_to_num(np.linalg.norm(y_axis)))

        # Apply static offset angle to the incorrect foot axes
        # static offset angle are taken from static_info variable in radians.
        # N.B. This replaces a procedure that converted from radians to degrees
        # and back to radians, the result was then rounded to 5 decimal places.
        alpha = static_info[side][0]
        beta = static_info[side][1]

        # Rotate incorrect foot axis around y axis first.
        rotated_axis = [
            [(np.cos(beta)*x_axis[x]+np.sin(beta)*z_axis[x]) for x in range(3)],
            y_axis,
            [(-np.sin(beta)*x_axis[x]+np.cos(beta)*z_axis[x]) for x in range(3)]]

        # Rotate incorrect foot axis around x axis next.
        rotated_axis = [rotated_axis[0],
                        [(np.cos(alpha) * rotated_axis[1][x]
                         - np.sin(alpha) * rotated_axis[2][x]) for x in range(3)],
                        [(np.sin(alpha) * rotated_axis[1][x]
                         + np.cos(alpha) * rotated_axis[2][x]) for x in range(3)]]

        # Attach each axis to the origin
        foot_axis.append([np.array(axis)
                          + np.array(toes[side]) for axis in rotated_axis])

    return [rtoe, ltoe, foot_axis]
