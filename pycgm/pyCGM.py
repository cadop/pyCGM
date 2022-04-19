# -*- coding: utf-8 -*-
#pyCGM

# Copyright (c) 2015 Mathew Schwartz <umcadop@gmail.com>
# Core Developers: Seungeun Yeon, Mathew Schwartz
# Contributors Filipe Alves Caixeta, Robert Van-wesep
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#pyCGM

"""
This file is used in joint angle and joint center calculations.
"""

import sys
import os
from math import sin, cos, pi, sqrt
import math
import numpy as np
from .pycgmIO import *

# Lowerbody Coordinate System

def calc_axis_pelvis(rasi, lasi, rpsi, lpsi, sacr=None):
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
    >>> np.set_printoptions(suppress=True)
    >>> from .pyCGM import calc_axis_pelvis
    >>> rasi = np.array([ 395.36,  428.09, 1036.82])
    >>> lasi = np.array([ 183.18,  422.78, 1033.07])
    >>> rpsi = np.array([ 341.41,  246.72, 1055.99])
    >>> lpsi = np.array([ 255.79,  241.42, 1057.30])
    >>> [arr.round(2) for arr in calc_axis_pelvis(rasi, lasi, rpsi, lpsi, None)] # doctest: +NORMALIZE_WHITESPACE
    [array([ -0.02,   0.99,  -0.12, 289.27]), 
    array([ -1.  ,  -0.03,  -0.02, 425.43]), 
    array([  -0.02,    0.12,    0.99, 1034.94]), 
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


def calc_joint_center_hip(pelvis, mean_leg_length, right_asis_to_trochanter, left_asis_to_trochanter, inter_asis_distance):
    u"""Calculate the right and left hip joint center.

    Takes in a 4x4 affine matrix of pelvis axis and subject measurements
    dictionary. Calculates and returns the right and left hip joint centers.

    Subject Measurement values used:
        MeanLegLength

        R_AsisToTrocanterMeasure

        InterAsisDistance

        L_AsisToTrocanterMeasure

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
        containing the x, y, z components of the right and left hip joint
        centers.

    References
    ----------
    .. [1] Davis, R. B., III, Õunpuu, S., Tyburski, D. and Gage, J. R. (1991).
            A gait analysis data collection and reduction technique.
            Human Movement Science 10 575–87.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)
    >>> from .pyCGM import calc_joint_center_hip
    >>> mean_leg_length = 940.0 
    >>> right_asis_to_trochanter = 72.51
    >>> left_asis_to_trochanter = 72.51
    >>> inter_asis_distance = 215.90
    >>> pelvis_axis = np.array([
    ...     [0.14, 0.98, -0.11, 251.60],
    ...     [-0.99, 0.13, -0.02, 391.74],
    ...     [0, 0.1, 0.99, 1032.89],
    ...     [0, 0, 0, 1]
    ... ])
    >>> np.around(calc_joint_center_hip(pelvis_axis, mean_leg_length, right_asis_to_trochanter, left_asis_to_trochanter, inter_asis_distance), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[307.36, 323.83, 938.72],
           [181.71, 340.33, 936.18]])
    """

    # Requires
    # pelvis axis

    pelvis = np.asarray(pelvis)
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

    C = (mean_leg_length * 0.115) - 15.3
    theta = 0.500000178813934
    beta = 0.314000427722931
    aa = inter_asis_distance/2.0
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

    left_hip_jc = np.matmul(pelvis_axis.T, np.array([L_Xh, L_Yh, L_Zh]))

    R_hipJCx = pelvis_xaxis * R_Xh
    R_hipJCy = pelvis_yaxis * R_Yh
    R_hipJCz = pelvis_zaxis * R_Zh
    right_hip_jc = R_hipJCx + R_hipJCy + R_hipJCz

    right_hip_jc = np.matmul(pelvis_axis.T, np.array([R_Xh, R_Yh, R_Zh]))

    left_hip_jc = left_hip_jc+pel_origin
    right_hip_jc = right_hip_jc+pel_origin

    hip_jc = np.array([right_hip_jc, left_hip_jc])

    return hip_jc


def calc_axis_hip(r_hip_jc, l_hip_jc, pelvis_axis):
    r"""Make the hip axis.

    Takes in the x, y, z positions of right and left hip joint center and
    pelvis axis to calculate the hip axis.

    Hip origin: Midpoint of right and left hip joint centers.

    Hip axis: Sets the pelvis orientation to the hip center axis (i.e.
    midpoint of right and left hip joint centers)

    Parameters
    ----------
    r_hip_jc, l_hip_jc : array
        right and left hip joint center with x, y, z position in an array.
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
    >>> np.set_printoptions(suppress=True)
    >>> from .pyCGM import calc_axis_hip
    >>> r_hip_jc = [182.57, 339.43, 935.52]
    >>> l_hip_jc = [308.38, 322.80, 937.98]
    >>> pelvis_axis = np.array([
    ...     [0.14, 0.98, -0.11, 251.60],
    ...     [-0.99, 0.13, -0.02, 391.74],
    ...     [0, 0.1, 0.99, 1032.89],
    ...     [0, 0, 0, 1]
    ... ])
    >>> [np.around(arr, 2) for arr in calc_axis_hip(l_hip_jc,r_hip_jc, pelvis_axis)] #doctest: +NORMALIZE_WHITESPACE
    [array([  0.14,   0.98,  -0.11, 245.48]),
    array([ -0.99,   0.13,  -0.02, 331.12]),
    array([  0.  ,   0.1 ,   0.99, 936.75]), 
    array([0., 0., 0., 1.])]
    """

    # Get shared hip axis, it is inbetween the two hip joint centers
    pelvis_axis = np.asarray(pelvis_axis)
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


def calc_axis_knee(rthi, lthi, rkne, lkne, r_hip_jc, l_hip_jc, rkne_width, lkne_width):
    """Calculate the knee joint center and axis.

    Takes in markers that correspond to (x, y, z) positions of the current
    frame, the hip joint center, and knee widths.

    Markers used: RTHI, LTHI, RKNE, LKNE, r_hip_jc, l_hip_jc

    Subject Measurement values used: RightKneeWidth, LeftKneeWidth

    Knee joint center: Computed using Knee Axis Calculation [1]_.

    Parameters
    ----------
    rthi : array
        1x3 RTHI marker
    lthi : array
        1x3 LTHI marker
    rkne : array
        1x3 RKNE marker
    lkne : array
        1x3 LKNE marker
    r_hip_jc : array
        4x4 affine matrix containing the right hip joint center.
    l_hip_jc : array
        4x4 affine matrix containing the left hip joint center.
    rkne_width : float
        The width of the right knee
    lkne_width : float
        The width of the left knee

    Returns
    -------
    [r_axis, l_axis] : array
        An array of two 4x4 affine matrices representing the right and left
        knee axes and joint centers.

    References
    ----------
    .. [1] Baker, R. (2013). Measuring walking : a handbook of clinical gait
            analysis. Mac Keith Press.

    Notes
    -----
    Delta is changed suitably to knee.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)
    >>> rthi = np.array([426.50, 262.65, 673.66])
    >>> lthi = np.array([51.93, 320.01, 723.03])
    >>> rkne = np.array([416.98, 266.22, 524.04])
    >>> lkne = np.array([84.62, 286.69, 529.39])
    >>> l_hip_jc = [182.57, 339.43, 935.52]
    >>> r_hip_jc = [309.38, 322.80, 937.98]
    >>> rkne_width = 105.0
    >>> lkne_width = 105.0
    >>> [arr.round(2) for arr in calc_axis_knee(rthi, lthi, rkne, lkne, l_hip_jc, r_hip_jc, rkne_width, lkne_width)] #doctest: +NORMALIZE_WHITESPACE
    [array([[  0.3 ,   0.95,   0.  , 365.09],
        [ -0.87,   0.28,  -0.4 , 282.84],
        [ -0.38,   0.12,   0.92, 500.13],
        [  0.  ,   0.  ,   0.  ,   1.  ]]),
     array([[  0.11,   0.98,  -0.15, 139.57],
        [ -0.92,   0.16,   0.35, 277.13],
        [  0.37,   0.1 ,   0.93, 508.67],
        [  0.  ,   0.  ,   0.  ,   1.  ]])]
    """
    # Get Global Values
    mm = 7.0
    R_delta = (rkne_width/2.0) + mm
    L_delta = (lkne_width/2.0) + mm

    # Determine the position of kneeJointCenter using calc_joint_center function
    R = calc_joint_center(rthi, r_hip_jc, rkne, R_delta)
    L = calc_joint_center(lthi, l_hip_jc, lkne, L_delta)

    # Z axis is Thigh bone calculated by the hipJC and  kneeJC
    # the axis is then normalized
    axis_z = r_hip_jc-R

    # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
    # and calculated by each point's vector cross vector.
    # the axis is then normalized.
    # axis_x = cross(axis_z,thi_kne_R)
    axis_x = np.cross(axis_z, rkne-r_hip_jc)

    # Y axis is determined by cross product of axis_z and axis_x.
    # the axis is then normalized.
    axis_y = np.cross(axis_z, axis_x)

    Raxis = np.asarray([axis_x, axis_y, axis_z])

    # Z axis is Thigh bone calculated by the hipJC and  kneeJC
    # the axis is then normalized
    axis_z = l_hip_jc-L

    # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
    # and calculated by each point's vector cross vector.
    # the axis is then normalized.
    # axis_x = cross(thi_kne_L,axis_z)
    # using hipjc instead of thigh marker
    axis_x = np.cross(lkne-l_hip_jc, axis_z)

    # Y axis is determined by cross product of axis_z and axis_x.
    # the axis is then normalized.
    axis_y = np.cross(axis_z, axis_x)

    Laxis = np.asarray([axis_x, axis_y, axis_z])

    # Clear the name of axis and then nomalize it.
    R_knee_x_axis = Raxis[0]
    R_knee_x_axis = R_knee_x_axis/np.linalg.norm(R_knee_x_axis)
    R_knee_y_axis = Raxis[1]
    R_knee_y_axis = R_knee_y_axis/np.linalg.norm(R_knee_y_axis)
    R_knee_z_axis = Raxis[2]
    R_knee_z_axis = R_knee_z_axis/np.linalg.norm(R_knee_z_axis)
    L_knee_x_axis = Laxis[0]
    L_knee_x_axis = L_knee_x_axis/np.linalg.norm(L_knee_x_axis)
    L_knee_y_axis = Laxis[1]
    L_knee_y_axis = L_knee_y_axis/np.linalg.norm(L_knee_y_axis)
    L_knee_z_axis = Laxis[2]
    L_knee_z_axis = L_knee_z_axis/np.linalg.norm(L_knee_z_axis)

    r_axis = np.zeros((4, 4))
    r_axis[3, 3] = 1.0
    r_axis[0, :3] = R_knee_x_axis
    r_axis[1, :3] = R_knee_y_axis
    r_axis[2, :3] = R_knee_z_axis
    r_axis[:3, 3] = R

    l_axis = np.zeros((4, 4))
    l_axis[3, 3] = 1.0
    l_axis[0, :3] = L_knee_x_axis
    l_axis[1, :3] = L_knee_y_axis
    l_axis[2, :3] = L_knee_z_axis
    l_axis[:3, 3] = L

    return np.asarray([r_axis, l_axis])


def calc_axis_ankle(rtib, ltib, rank, lank, r_knee_JC, l_knee_JC, rank_width, lank_width, rtib_torsion, ltib_torsion):
    """Calculate the ankle joint center and axis.

    Takes in markers that correspond to (x, y, z) positions of the current
    frame, the knee joint centers, ankle widths, and tibial torsions.

    Markers used: RTIB, LTIB, RANK, LANK, r_knee_JC, l_knee_JC

    Subject Measurement values used:
        RightKneeWidth

        LeftKneeWidth

        RightTibialTorsion

        LeftTibialTorsion

    Ankle Axis: Computed using Ankle Axis Calculation [1]_.

    Parameters
    ----------
    rtib : array
        1x3 RTIB marker
    ltib : array
        1x3 LTIB marker
    rank : array
        1x3 RANK marker
    lank : array
        1x3 LANK marker
    r_knee_JC : array
        The (x,y,z) position of the right knee joint center.
    l_knee_JC : array
        The (x,y,z) position of the left knee joint center.
    rank_width : float
        The width of the right ankle
    lank_width : float
        The width of the left ankle
    rtib_torsion : float
        Right tibial torsion angle
    ltib_torsion : float
        Left tibial torsion angle

    Returns
    -------
    [r_axis, l_axis] : array
        An array of two 4x4 affine matrices representing the right and left
        ankle axes and joint centers.

    References
    ----------
    .. [1] Baker, R. (2013). Measuring walking : a handbook of clinical gait
            analysis. Mac Keith Press.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)
    >>> rank_width = 70.0
    >>> lank_width = 70.0
    >>> rtib_torsion = 0.0
    >>> ltib_torsion = 0.0
    >>> rtib = np.array([433.97, 211.93, 273.30])
    >>> ltib = np.array([50.04, 235.90, 364.32])
    >>> rank = np.array([422.77, 217.74, 92.86])
    >>> lank = np.array([58.57, 208.54, 86.16])
    >>> knee_JC = np.array([[365.09, 282.84, 500.13],
    ...                     [139.57, 277.13, 508.67]])
    >>> [np.around(arr, 2) for arr in calc_axis_ankle(rtib, ltib, rank, lank, knee_JC[0], knee_JC[1], rank_width, lank_width, rtib_torsion, ltib_torsion)] #doctest: +NORMALIZE_WHITESPACE
                [array([[  0.69,   0.73,  -0.02, 392.33],
                        [ -0.72,   0.68,  -0.11, 246.32],
                        [ -0.07,   0.09,   0.99,  88.31],
                        [  0.  ,   0.  ,   0.  ,   1.  ]]),
                 array([[ -0.28,   0.96,  -0.1 ,  98.76],
                        [ -0.96,  -0.26,   0.13, 219.53],
                        [  0.09,   0.13,   0.99,  80.85],
                        [  0.  ,   0.  ,   0.  ,   1.  ]])]
    """
    # Get Global Values
    mm = 7.0
    R_delta = (rank_width/2.0)+mm
    L_delta = (lank_width/2.0)+mm

    # REQUIRED MARKERS:
    # RTIB
    # LTIB
    # RANK
    # LANK
    # r_knee_JC
    # l_knee_JC

    # This is Torsioned Tibia and this describe the ankle angles
    # Tibial frontal plane being defined by ANK,TIB and KJC

    # Determine the position of ankleJointCenter using calc_joint_center function
    R = calc_joint_center(rtib, r_knee_JC, rank, R_delta)
    L = calc_joint_center(ltib, l_knee_JC, lank, L_delta)

    # Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)
    # Right axis calculation

    # Z axis is shank bone calculated by the ankleJC and  kneeJC
    axis_z = r_knee_JC-R

    # X axis is perpendicular to the points plane which is determined by ANK,TIB and KJC markers.
    # and calculated by each point's vector cross vector.
    # tib_ank_R vector is making a tibia plane to be assumed as rigid segment.
    tib_ank_R = rtib-rank
    axis_x = np.cross(axis_z, tib_ank_R)

    # Y axis is determined by cross product of axis_z and axis_x.
    axis_y = np.cross(axis_z, axis_x)

    Raxis = [axis_x, axis_y, axis_z]

    # Left axis calculation

    # Z axis is shank bone calculated by the ankleJC and  kneeJC
    axis_z = l_knee_JC-L

    # X axis is perpendicular to the points plane which is determined by ANK,TIB and KJC markers.
    # and calculated by each point's vector cross vector.
    # tib_ank_L vector is making a tibia plane to be assumed as rigid segment.
    tib_ank_L = ltib-lank
    axis_x = np.cross(tib_ank_L, axis_z)

    # Y axis is determined by cross product of axis_z and axis_x.
    axis_y = np.cross(axis_z, axis_x)

    Laxis = [axis_x, axis_y, axis_z]

    # Clear the name of axis and then normalize it.
    R_ankle_x_axis = Raxis[0]
    R_ankle_x_axis_div = np.linalg.norm(R_ankle_x_axis)
    R_ankle_x_axis = [R_ankle_x_axis[0]/R_ankle_x_axis_div, R_ankle_x_axis[1] /
                      R_ankle_x_axis_div, R_ankle_x_axis[2]/R_ankle_x_axis_div]

    R_ankle_y_axis = Raxis[1]
    R_ankle_y_axis_div = np.linalg.norm(R_ankle_y_axis)
    R_ankle_y_axis = [R_ankle_y_axis[0]/R_ankle_y_axis_div, R_ankle_y_axis[1] /
                      R_ankle_y_axis_div, R_ankle_y_axis[2]/R_ankle_y_axis_div]

    R_ankle_z_axis = Raxis[2]
    R_ankle_z_axis_div = np.linalg.norm(R_ankle_z_axis)
    R_ankle_z_axis = [R_ankle_z_axis[0]/R_ankle_z_axis_div, R_ankle_z_axis[1] /
                      R_ankle_z_axis_div, R_ankle_z_axis[2]/R_ankle_z_axis_div]

    L_ankle_x_axis = Laxis[0]
    L_ankle_x_axis_div = np.linalg.norm(L_ankle_x_axis)
    L_ankle_x_axis = [L_ankle_x_axis[0]/L_ankle_x_axis_div, L_ankle_x_axis[1] /
                      L_ankle_x_axis_div, L_ankle_x_axis[2]/L_ankle_x_axis_div]

    L_ankle_y_axis = Laxis[1]
    L_ankle_y_axis_div = np.linalg.norm(L_ankle_y_axis)
    L_ankle_y_axis = [L_ankle_y_axis[0]/L_ankle_y_axis_div, L_ankle_y_axis[1] /
                      L_ankle_y_axis_div, L_ankle_y_axis[2]/L_ankle_y_axis_div]

    L_ankle_z_axis = Laxis[2]
    L_ankle_z_axis_div = np.linalg.norm(L_ankle_z_axis)
    L_ankle_z_axis = [L_ankle_z_axis[0]/L_ankle_z_axis_div, L_ankle_z_axis[1] /
                      L_ankle_z_axis_div, L_ankle_z_axis[2]/L_ankle_z_axis_div]

    # Put both axis in array
    Raxis = [R_ankle_x_axis, R_ankle_y_axis, R_ankle_z_axis]
    Laxis = [L_ankle_x_axis, L_ankle_y_axis, L_ankle_z_axis]

    # Rotate the axes about the tibia torsion.
    rtib_torsion = np.radians(rtib_torsion)
    ltib_torsion = np.radians(ltib_torsion)

    Raxis = [[math.cos(rtib_torsion)*Raxis[0][0]-math.sin(rtib_torsion)*Raxis[1][0],
              math.cos(rtib_torsion)*Raxis[0][1] -
              math.sin(rtib_torsion)*Raxis[1][1],
              math.cos(rtib_torsion)*Raxis[0][2]-math.sin(rtib_torsion)*Raxis[1][2]],
             [math.sin(rtib_torsion)*Raxis[0][0]+math.cos(rtib_torsion)*Raxis[1][0],
             math.sin(rtib_torsion)*Raxis[0][1] +
              math.cos(rtib_torsion)*Raxis[1][1],
             math.sin(rtib_torsion)*Raxis[0][2]+math.cos(rtib_torsion)*Raxis[1][2]],
             [Raxis[2][0], Raxis[2][1], Raxis[2][2]]]

    Laxis = [[math.cos(ltib_torsion)*Laxis[0][0]-math.sin(ltib_torsion)*Laxis[1][0],
              math.cos(ltib_torsion)*Laxis[0][1] -
              math.sin(ltib_torsion)*Laxis[1][1],
              math.cos(ltib_torsion)*Laxis[0][2]-math.sin(ltib_torsion)*Laxis[1][2]],
             [math.sin(ltib_torsion)*Laxis[0][0]+math.cos(ltib_torsion)*Laxis[1][0],
             math.sin(ltib_torsion)*Laxis[0][1] +
              math.cos(ltib_torsion)*Laxis[1][1],
             math.sin(ltib_torsion)*Laxis[0][2]+math.cos(ltib_torsion)*Laxis[1][2]],
             [Laxis[2][0], Laxis[2][1], Laxis[2][2]]]

    r_axis = np.zeros((4, 4))
    r_axis[3, 3] = 1.0
    r_axis[0, :3] = Raxis[0]
    r_axis[1, :3] = Raxis[1]
    r_axis[2, :3] = Raxis[2]
    r_axis[:3, 3] = R

    l_axis = np.zeros((4, 4))
    l_axis[3, 3] = 1.0
    l_axis[0, :3] = Laxis[0]
    l_axis[1, :3] = Laxis[1]
    l_axis[2, :3] = Laxis[2]
    l_axis[:3, 3] = L

    # Both of axis in array.
    axis = np.array([r_axis, l_axis])

    return axis


def calc_axis_foot(rtoe, ltoe, r_ankle_axis, l_ankle_axis, r_static_rot_off, l_static_rot_off, r_static_plant_flex, l_static_plant_flex):
    """Calculate the foot joint center and axis.

    Takes in markers that correspond to (x, y, z) positions of the current
    frame, the right and left ankle axes, right and left static rotation
    offset angles, and the right and left static plantar flexion angles.

    Calculates the foot joint axis by rotating incorrect foot joint axes about
    offset angle.

    Markers used: RTOE, LTOE

    Other landmarks used: ankle axis

    Subject Measurement values used: 
        RightStaticRotOff

        RightStaticPlantFlex

        LeftStaticRotOff

        LeftStaticPlantFlex

    Parameters
    ----------
    rtoe : array
        1x3 RTOE marker
    ltoe : array
        1x3 LTOE marker
    r_ankle_axis : array
        4x4 affine matrix with right ankle x, y, z axes and origin.
    l_ankle_axis : array
        4x4 affine matrix with left ankle x, y, z axes and origin.
    r_static_rot_off : float
        Right static offset angle.
    l_static_rot_off : float
        Left static offset angle.
    r_static_plant_flex : float
        Right static plantar flexion angle.
    l_static_plant_flex : float
        Left static plantar flexion angle.

    Returns
    -------
    [r_axis, l_axis] : array
        A list of two 4x4 affine matrices representing the right and left
        foot axes and origins.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)
    >>> from .pyCGM import calc_axis_foot
    >>> r_static_rot_off = 0.01
    >>> l_static_rot_off = 0.00
    >>> r_static_plant_flex = 0.27
    >>> l_static_plant_flex = 0.20
    >>> rtoe = np.array([442.81, 381.62, 42.66])
    >>> ltoe = np.array([39.43, 382.44, 41.78])
    >>> r_ankle_axis = np.array([[  0.69,   0.73,  -0.02, 392.33],
    ...                          [ -0.72,   0.68,  -0.11, 246.32],
    ...                          [ -0.07,   0.09,   0.99,  88.31],
    ...                          [  0.  ,   0.  ,   0.  ,   1.  ]])
    >>> l_ankle_axis = np.array([[ -0.28,   0.96,  -0.1 ,  98.76],
    ...                         [  -0.96,  -0.26,   0.13, 219.53],
    ...                         [   0.09,   0.13,   0.99,  80.85],
    ...                         [   0.  ,   0.  ,   0.  ,   1.  ]])
    >>> [np.around(arr, 2) for arr in calc_axis_foot(rtoe, ltoe, r_ankle_axis, l_ankle_axis, r_static_rot_off, l_static_rot_off, r_static_plant_flex, l_static_plant_flex)] #doctest: +NORMALIZE_WHITESPACE
    [array([[  0.02,   0.03,   1.  , 442.81],
            [ -0.94,   0.34,   0.01, 381.62],
            [ -0.34,  -0.94,   0.04,  42.66],
            [  0.  ,   0.  ,   0.  ,   1.  ]]), 
     array([[  0.13,   0.07,   0.99,  39.43],
            [ -0.94,  -0.31,   0.14, 382.44],
            [  0.31,  -0.95,   0.02,  41.78],
            [  0.  ,   0.  ,   0.  ,   1.  ]])]

    """
    # Required joint centers and axes:
    # Knee joint center
    # Ankle joint center
    # Ankle flexion axis

    rtoe, ltoe, r_ankle_axis, l_ankle_axis = map(np.asarray, [rtoe, ltoe, r_ankle_axis, l_ankle_axis])

    ankle_JC_R      = r_ankle_axis[:3, 3]
    ankle_JC_L      = l_ankle_axis[:3, 3]
    ankle_flexion_R = r_ankle_axis[1, :3] + ankle_JC_R
    ankle_flexion_L = l_ankle_axis[1, :3] + ankle_JC_L

    # Toe axis's origin is marker position of TOE
    right_origin = rtoe
    left_origin  = ltoe

    # Right

    # Z-axis is from TOE marker to AJC, then normalized
    R_axis_z     = ankle_JC_R - rtoe
    R_axis_z_div = np.linalg.norm(R_axis_z)
    R_axis_z     = np.divide(R_axis_z, R_axis_z_div)

    # Bring the flexion axis of ankle axes from AnkleJointCenter function, then normalize it
    y_flex_R     = ankle_flexion_R - ankle_JC_R
    y_flex_R_div = np.linalg.norm(y_flex_R)
    y_flex_R     = y_flex_R/y_flex_R_div

    # X-axis is calculated as a cross product of z-axis and ankle flexion axis
    R_axis_x     = np.cross(y_flex_R, R_axis_z)
    R_axis_x_div = np.linalg.norm(R_axis_x)
    R_axis_x     = np.divide(R_axis_x, R_axis_x_div)

    # Y-axis is then perpendicularly calculated from z-axis and x-axis, then normalized
    R_axis_y     = np.cross(R_axis_z, R_axis_x)
    R_axis_y_div = np.linalg.norm(R_axis_y)
    R_axis_y     = np.divide(R_axis_y, R_axis_y_div)

    R_foot_axis  = [R_axis_x, R_axis_y, R_axis_z]

    # Left

    # Z-axis is from TOE marker to AJC, then normalized
    L_axis_z     = ankle_JC_L - ltoe
    L_axis_z_div = np.linalg.norm(L_axis_z)
    L_axis_z     = np.divide(L_axis_z, L_axis_z_div)

    # Bring the flexion axis of ankle axes from AnkleJointCenter function, then normalize it
    y_flex_L     = ankle_flexion_L - ankle_JC_L
    y_flex_L_div = np.linalg.norm(y_flex_L)
    y_flex_L     = np.divide(y_flex_L, y_flex_L_div)

    # X-axis is calculated as a cross product of z-axis and ankle flexion axis
    L_axis_x     = np.cross(y_flex_L, L_axis_z)
    L_axis_x_div = np.linalg.norm(L_axis_x)
    L_axis_x     = np.divide(L_axis_x, L_axis_x_div)

    # Y-axis is then perpendicularly calculated from z-axis and x-axis, then normalized
    L_axis_y     = np.cross(L_axis_z, L_axis_x)
    L_axis_y_div = np.linalg.norm(L_axis_y)
    L_axis_y     = np.divide(L_axis_y, L_axis_y_div)

    L_foot_axis  = [L_axis_x, L_axis_y, L_axis_z]

    foot_axis    = [R_foot_axis, L_foot_axis]

    # Apply static offset angle to the incorrect foot axes

    # Static offset angles are taken from static_info variable in radians
    R_alpha = r_static_rot_off
    R_beta  = r_static_plant_flex
    L_alpha = l_static_rot_off
    L_beta  = l_static_plant_flex

    R_alpha = np.around(math.degrees(R_alpha), decimals=5)
    R_beta  = np.around(math.degrees(R_beta), decimals=5)
    L_alpha = np.around(math.degrees(L_alpha), decimals=5)
    L_beta  = np.around(math.degrees(L_beta), decimals=5)

    R_alpha = -math.radians(R_alpha)
    R_beta  = math.radians(R_beta)
    L_alpha = math.radians(L_alpha)
    L_beta  = math.radians(L_beta)

    R_axis = R_foot_axis
    L_axis = L_foot_axis

    # First, rotate incorrect foot axis around y-axis

    # Right
    R_rotmat = [
                    [(math.cos(R_beta) * R_axis[0][0] + math.sin(R_beta) * R_axis[2][0]),
                     (math.cos(R_beta) * R_axis[0][1] + math.sin(R_beta) * R_axis[2][1]),
                     (math.cos(R_beta) * R_axis[0][2] + math.sin(R_beta) * R_axis[2][2])],
                     [R_axis[1][0], R_axis[1][1], R_axis[1][2]],
                    [(-1 * math.sin(R_beta) * R_axis[0][0] + math.cos(R_beta) * R_axis[2][0]),
                     (-1 * math.sin(R_beta) * R_axis[0][1] + math.cos(R_beta) * R_axis[2][1]),
                     (-1 * math.sin(R_beta) * R_axis[0][2] + math.cos(R_beta) * R_axis[2][2])]
               ]
    # Left
    L_rotmat = [
                    [(math.cos(L_beta) * L_axis[0][0] + math.sin(L_beta) * L_axis[2][0]),
                     (math.cos(L_beta) * L_axis[0][1] + math.sin(L_beta) * L_axis[2][1]),
                     (math.cos(L_beta) * L_axis[0][2] + math.sin(L_beta) * L_axis[2][2])],
                     [L_axis[1][0], L_axis[1][1], L_axis[1][2]],
                    [(-1 * math.sin(L_beta) * L_axis[0][0] + math.cos(L_beta) * L_axis[2][0]),
                     (-1 * math.sin(L_beta) * L_axis[0][1] + math.cos(L_beta) * L_axis[2][1]),
                     (-1 * math.sin(L_beta) * L_axis[0][2] + math.cos(L_beta) * L_axis[2][2])]
               ]

    # Next, rotate incorrect foot axis around x-axis

    # Right
    right_axis = np.asarray([[R_rotmat[0][0], R_rotmat[0][1], R_rotmat[0][2]],
                             [
                                 (math.cos(R_alpha) * R_rotmat[1][0] - math.sin(R_alpha) * R_rotmat[2][0]),
                                 (math.cos(R_alpha) * R_rotmat[1][1] - math.sin(R_alpha) * R_rotmat[2][1]),
                                 (math.cos(R_alpha) * R_rotmat[1][2] - math.sin(R_alpha) * R_rotmat[2][2])
                             ],
                             [
                                 (math.sin(R_alpha) * R_rotmat[1][0] + math.cos(R_alpha) * R_rotmat[2][0]),
                                 (math.sin(R_alpha) * R_rotmat[1][1] + math.cos(R_alpha) * R_rotmat[2][1]),
                                 (math.sin(R_alpha) * R_rotmat[1][2] + math.cos(R_alpha) * R_rotmat[2][2])
                            ]])

    # Left
    left_axis = np.asarray([[L_rotmat[0][0], L_rotmat[0][1], L_rotmat[0][2]],
                            [
                                (math.cos(L_alpha) * L_rotmat[1][0] - math.sin(L_alpha) * L_rotmat[2][0]),
                                (math.cos(L_alpha) * L_rotmat[1][1] - math.sin(L_alpha) * L_rotmat[2][1]),
                                (math.cos(L_alpha) * L_rotmat[1][2] - math.sin(L_alpha) * L_rotmat[2][2])
                            ],
                            [
                                (math.sin(L_alpha) * L_rotmat[1][0] + math.cos(L_alpha) * L_rotmat[2][0]),
                                (math.sin(L_alpha) * L_rotmat[1][1] + math.cos(L_alpha) * L_rotmat[2][1]),
                                (math.sin(L_alpha) * L_rotmat[1][2] + math.cos(L_alpha) * L_rotmat[2][2])
                            ]])

    # Attach each axis to the origin
    r_foot_axis = np.zeros((4, 4))
    r_foot_axis[3, 3] = 1.0
    r_foot_axis[:3, :3] = right_axis
    r_foot_axis[:3, 3] = right_origin

    l_foot_axis = np.zeros((4, 4))
    l_foot_axis[3, 3] = 1.0
    l_foot_axis[:3, :3] = left_axis
    l_foot_axis[:3, 3] = left_origin

    foot_axis = np.array([r_foot_axis, l_foot_axis])

    return foot_axis

# Upperbody Coordinate System

def calc_axis_head(lfhd, rfhd, lbhd, rbhd, head_offset):
    """Calculate the head joint center and axis.

    Takes in markers that correspond to (x, y, z) positions of the current
    frame, and the head offset. 

    Calculates the head joint center and axis.

    Markers used: LFHD, RFHD, LBHD, RBHD

    Subject Measurement values used: HeadOffset

    Parameters
    ----------
    lfhd : array
        1x3 LFHD marker
    rfhd : array
        1x3 RFHD marker
    lbhd : array
        1x3 LBHD marker
    rbhd : array
        1x3 RBHD marker
    head_offset : float
        Static head offset angle.

    Returns
    -------
    head_axis : array
        4x4 affine matrix with head (x, y, z) axes and origin.


    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)
    >>> from .pyCGM import calc_axis_head
    >>> head_offset = 0.25
    >>> rfhd = np.array([325.82, 402.55, 1722.49])
    >>> lfhd = np.array([184.55, 409.68, 1721.34])
    >>> rbhd = np.array([304.39, 242.91, 1694.97])
    >>> lbhd = np.array([197.86, 251.28, 1696.90])
    >>> [np.around(arr, 2) for arr in calc_axis_head(lfhd, rfhd, lbhd, rbhd, head_offset)] #doctest: +NORMALIZE_WHITESPACE
    [array([ 0.03, 1.  ,  -0.09, 255.18]), 
    array([ -1.  , 0.03,  -0.  , 406.12]), 
    array([ -0.  , 0.09,   1.  , 1721.92]), 
      array([0.,   0.,     0.,      1.])]
    """

    head_offset = -1*head_offset

    # get the midpoints of the head to define the sides
    front = (lfhd + rfhd)/2.0
    back = (lbhd + rbhd)/2.0
    left = (lfhd + lbhd)/2.0
    right = (rfhd + rbhd)/2.0

    # Get the vectors from the sides with primary x axis facing front
    # First get the x direction
    x_axis = np.subtract(front, back)
    x_axis_norm = np.nan_to_num(np.linalg.norm(x_axis))
    if x_axis_norm:
        x_axis = np.divide(x_axis, x_axis_norm)

    # get the direction of the y axis
    y_axis = np.subtract(left, right)
    y_axis_norm = np.nan_to_num(np.linalg.norm(y_axis))
    if y_axis_norm:
        y_axis = np.divide(y_axis, y_axis_norm)

    # get z axis by cross-product of x axis and y axis.
    z_axis = np.cross(x_axis, y_axis)
    z_axis_norm = np.nan_to_num(np.linalg.norm(z_axis))
    if z_axis_norm:
        z_axis = np.divide(z_axis, z_axis_norm)

    # make sure all x,y,z axis is orthogonal each other by cross-product
    y_axis = np.cross(z_axis, x_axis)
    y_axis_norm = np.nan_to_num(np.linalg.norm(y_axis))
    if y_axis_norm:
        y_axis = np.divide(y_axis, y_axis_norm)

    x_axis = np.cross(y_axis, z_axis)
    x_axis_norm = np.nan_to_num(np.linalg.norm(x_axis))
    if x_axis_norm:
        x_axis = np.divide(x_axis, x_axis_norm)

# rotate the head axis around y axis about head offset angle.
    x_axis_rot = [x_axis[0]*math.cos(head_offset)+z_axis[0]*math.sin(head_offset),
            x_axis[1]*math.cos(head_offset)+z_axis[1]*math.sin(head_offset),
            x_axis[2]*math.cos(head_offset)+z_axis[2]*math.sin(head_offset)]
    y_axis_rot = [y_axis[0],y_axis[1],y_axis[2]]
    z_axis_rot = [x_axis[0]*-1*math.sin(head_offset)+z_axis[0]*math.cos(head_offset),
            x_axis[1]*-1*math.sin(head_offset)+z_axis[1]*math.cos(head_offset),
            x_axis[2]*-1*math.sin(head_offset)+z_axis[2]*math.cos(head_offset)]

    # Create the return matrix
    head_axis = np.zeros((4, 4))
    head_axis[3, 3] = 1.0
    head_axis[0, :3] = x_axis_rot
    head_axis[1, :3] = y_axis_rot
    head_axis[2, :3] = z_axis_rot
    head_axis[:3, 3] = front

    return head_axis


def calc_axis_thorax(clav, c7, strn, t10):
    r"""Make the Thorax Axis.


    Takes in CLAV, C7, STRN, T10 markers.
    Calculates the thorax axis.

    :math:`upper = (\textbf{m}_{clav} + \textbf{m}_{c7}) / 2.0`

    :math:`lower = (\textbf{m}_{strn} + \textbf{m}_{t10}) / 2.0`

    :math:`\emph{front} = (\textbf{m}_{clav} + \textbf{m}_{strn}) / 2.0`

    :math:`back = (\textbf{m}_{t10} + \textbf{m}_{c7}) / 2.0`

    :math:`\hat{z} = \frac{lower - upper}{||lower - upper||}`

    :math:`\hat{x} = \frac{\emph{front} - back}{||\emph{front} - back||}`

    :math:`\hat{y} = \frac{ \hat{z} \times \hat{x} }{||\hat{z} \times \hat{x}||}`

    :math:`\hat{z} = \frac{\hat{x} \times \hat{y} }{||\hat{x} \times \hat{y} ||}`

    Parameters
    ----------
    clav: array
        1x3 CLAV marker
    c7: array
        1x3 C7 marker
    strn: array
        1x3 STRN marker
    t10: array
        1x3 T10 marker

    Returns
    -------
    thorax : array
        4x4 affine matrix with thorax x, y, z axes and thorax origin.

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
    >>> np.set_printoptions(suppress=True)
    >>> from .pyCGM import calc_axis_thorax
    >>> c7 = np.array([256.78, 371.28, 1459.70])
    >>> t10 = np.array([228.64, 192.32, 1279.64])
    >>> clav = np.array([256.78, 371.28, 1459.70])
    >>> strn = np.array([251.67, 414.10, 1292.08])
    >>> [np.around(arr, 2) for arr in calc_axis_thorax(clav, c7, strn, t10)] #doctest: +NORMALIZE_WHITESPACE
    [array([ 0.07,  0.93, -0.37,  256.27]), 
    array([  0.99, -0.1 , -0.06,  364.8 ]), 
    array([ -0.09, -0.36, -0.93, 1462.29]), 
      array([0.,    0.,    0.,      1.])]
    """

    clav, c7, strn, t10 = map(np.asarray, [clav, c7, strn, t10])

    # Set or get a marker size as mm
    marker_size = (14.0) / 2.0

    # Get the midpoints of the upper and lower sections, as well as the front and back sections
    upper = (clav + c7)/2.0
    lower = (strn + t10)/2.0
    front = (clav + strn)/2.0
    back = (t10 + c7)/2.0

    # Get the direction of the primary axis Z (facing down)
    z_direc = lower - upper
    z = z_direc/np.linalg.norm(z_direc)

    # The secondary axis X is from back to front
    x_direc = front - back
    x = x_direc/np.linalg.norm(x_direc)

    # make sure all the axes are orthogonal to each other by cross-product
    y_direc = np.cross(z, x)
    y = y_direc/np.linalg.norm(y_direc)
    x_direc = np.cross(y, z)
    x = x_direc/np.linalg.norm(x_direc)
    z_direc = np.cross(x, y)
    z = z_direc/np.linalg.norm(z_direc)

    # move the axes about offset along the x axis.
    offset = x * marker_size

    # Add the CLAV back to the vector to get it in the right position before translating it
    o = clav - offset

    thorax = np.zeros((4, 4))
    thorax[3, 3] = 1.0
    thorax[0, :3] = x
    thorax[1, :3] = y
    thorax[2, :3] = z
    thorax[:3, 3] = o

    return thorax


def calc_marker_wand(rsho, lsho, thorax_axis):
    """Calculate the wand marker position.

    Takes in markers that correspond to (x, y, z) positions of the current
    frame, and the thorax axis.

    Calculates the wand marker position.

    Markers used: RSHO, LSHO

    Other landmarks used: thorax axis

    Parameters
    ----------
    rsho : array
        1x3 RSHO marker
    lsho : array
        1x3 LSHO marker
    thorax_axis : array
        4x4 affine matrix with thorax (x, y, z) axes and origin.

    Returns
    -------
    wand : array
        A list of two 1x3 arrays representing the right and left wand markers.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)
    >>> from .pyCGM import calc_marker_wand
    >>> rsho = np.array([428.88, 270.55, 1500.73])
    >>> lsho = np.array([68.24, 269.01, 1510.10])
    >>> thorax_axis = np.array([[ 0.09,  1.  ,  0.01,  256.14],
    ...                         [ 1.  , -0.09, -0.07,  364.3 ],
    ...                         [-0.06, -9.98, -1.  , 1459.65],
    ...                         [ 0.  ,  0.  ,  0.  ,    1.  ]])
    >>> [np.around(arr, 2) for arr in calc_marker_wand(rsho, lsho, thorax_axis)] #doctest: +NORMALIZE_WHITESPACE
    [array([ 255.91,  364.31, 1460.62]),
     array([ 256.42,  364.27, 1460.61])]
    """

    thorax_axis = np.asarray(thorax_axis)
    thorax_origin = thorax_axis[:3, 3]

    axis_x_vec = thorax_axis[0, :3]
    axis_x_vec = axis_x_vec / np.linalg.norm(axis_x_vec)

    # Calculate for getting a wand marker

    RSHO_vec = rsho - thorax_origin
    LSHO_vec = lsho - thorax_origin
    RSHO_vec = RSHO_vec/np.linalg.norm(RSHO_vec)
    LSHO_vec = LSHO_vec/np.linalg.norm(LSHO_vec)

    r_wand = np.cross(RSHO_vec, axis_x_vec)
    r_wand = r_wand/np.linalg.norm(r_wand)
    r_wand = thorax_origin + r_wand

    l_wand = np.cross(axis_x_vec, LSHO_vec)
    l_wand = l_wand/np.linalg.norm(l_wand)
    l_wand = thorax_origin + l_wand

    wand = np.array([r_wand, l_wand])

    return wand


def calc_joint_center_shoulder(rsho, lsho, thorax_axis, r_wand, l_wand, r_sho_off, l_sho_off):
    """Calculate the shoulder joint center.

    Takes in markers that correspond to (x, y, z) positions of the current
    frame, the thorax axis, the right and left wand markers, and the right and
    left shoulder offset angles.

    Parameters
    ----------
    rsho : array
        1x3 RSHO marker
    lsho : array
        1x3 LSHO marker
    thorax_axis : array
        4x4 affine matrix with thorax (x, y, z) axes and origin.
    r_wand : array
        1x3 right wand marker
    l_wand : array
        1x3 left wand marker
    r_sho_off : float
        Right shoulder static offset angle
    l_sho_off : float
        Left shoulder static offset angle

    Returns
    -------
    shoulder_JC : array
        4x4 affine matrix containing the right and left shoulders joint centers.


    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)
    >>> from .pyCGM import calc_joint_center_shoulder
    >>> rsho = np.array([428.88, 270.55, 1500.73])
    >>> lsho = np.array([68.24, 269.01, 1510.10])
    >>> thorax_axis = np.array([[ 0.07,  0.93, -0.37,  256.27], 
    ...                        [  0.99, -0.1 , -0.06,  364.8 ], 
    ...                        [ -0.09, -0.36, -0.93, 1462.29], 
    ...                        [  0.,    0.,    0.,      1.]])
    >>> r_wand = [255.92, 364.32, 1460.62]
    >>> l_wand = [256.42, 364.27, 1460.61]
    >>> r_sho_off = 40.0
    >>> l_sho_off = 40.0
    >>> [np.around(arr, 2) for arr in calc_joint_center_shoulder(rsho, lsho, thorax_axis, r_wand, l_wand, r_sho_off, l_sho_off)] #doctest: +NORMALIZE_WHITESPACE
    [array([[   1.  ,    0.  ,    0.  ,  419.62],
            [   0.  ,    1.  ,    0.  ,  293.35],
            [   0.  ,    0.  ,    1.  , 1540.77],
            [   0.  ,    0.  ,    0.  ,    1.  ]]),
     array([[   1.  ,    0.  ,    0.  ,   79.26],
            [   0.  ,    1.  ,    0.  ,  290.54],
            [   0.  ,    0.  ,    1.  , 1550.4 ],
            [   0.  ,    0.  ,    0.  ,    1.  ]])]
    """

    thorax_axis = np.asarray(thorax_axis)
    thorax_origin = thorax_axis[:3, 3]

    # Get Subject Measurement Values
    mm = 7.0
    R_delta = (r_sho_off + mm)
    L_delta = (l_sho_off + mm)

    # REQUIRED MARKERS:
    # RSHO
    # LSHO

    R_Sho_JC = calc_joint_center(r_wand, thorax_origin, rsho, R_delta)
    L_Sho_JC = calc_joint_center(l_wand, thorax_origin, lsho, L_delta)

    r_sho_jc = np.identity(4)
    r_sho_jc[:3, 3] = R_Sho_JC

    l_sho_jc = np.identity(4)
    l_sho_jc[:3, 3] = L_Sho_JC

    shoulder_JC = np.array([r_sho_jc, l_sho_jc])

    return shoulder_JC


def calc_axis_shoulder(thorax_axis, r_sho_jc, l_sho_jc, r_wand, l_wand):
    """Make the Shoulder axis.

    Takes in the thorax axis, right and left shoulder joint center,
    and right and left wand markers.

    Calculates the right and left shoulder joint axes.

    Parameters
    ----------
    thorax_axis : array
        4x4 affine matrix with thorax (x, y, z) axes and origin.
    r_sho_jc : array
       The (x, y, z) position of the right shoulder joint center. 
    l_sho_jc : array
       The (x, y, z) position of the left shoulder joint center. 
    r_wand : array
        1x3 right wand marker
    l_wand : array
        1x3 left wand marker

    Returns
    -------
    shoulder : array
        A list of two 4x4 affine matrices respresenting the right and left
        shoulder axes and origins.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)
    >>> from .pyCGM import calc_axis_shoulder
    >>> thorax = np.array([[  0.07,  0.93, -0.37,  256.27], 
    ...                    [  0.99, -0.1 , -0.06,  364.8 ], 
    ...                    [ -0.09, -0.36, -0.93, 1462.29], 
    ...                    [  0.,    0.,    0.,      1.]])
    >>> r_sho_jc = np.array([[   1.  ,    0.  ,    0.  ,  419.62],
    ...                      [   0.  ,    1.  ,    0.  ,  293.35],
    ...                      [   0.  ,    0.  ,    1.  , 1540.77],
    ...                      [   0.  ,    0.  ,    0.  ,    1.  ]])
    >>> l_sho_jc = np.array([[   1.  ,    0.  ,    0.  ,   79.26],
    ...                      [   0.  ,    1.  ,    0.  ,  290.54],
    ...                      [   0.  ,    0.  ,    1.  , 1550.4 ],
    ...                      [   0.  ,    0.  ,    0.  ,    1.  ]])
    >>> wand = [[255.92, 364.32, 1460.62],
    ...        [ 256.42, 364.27, 1460.61]]
    >>> [np.around(arr, 2) for arr in calc_axis_shoulder(thorax, r_sho_jc, l_sho_jc, wand[0], wand[1])] #doctest: +NORMALIZE_WHITESPACE
    [array([[  -0.51,   -0.79,    0.33,  419.62],
            [  -0.2 ,    0.49,    0.85,  293.35],
            [  -0.84,    0.37,   -0.4 , 1540.77],
            [   0.  ,    0.  ,    0.  ,    1.  ]]),
     array([[   0.49,   -0.82,    0.3 ,   79.26],
            [  -0.23,   -0.46,   -0.86,  290.54],
            [   0.84,    0.35,   -0.42, 1550.4 ],
            [   0.  ,    0.  ,    0.  ,    1.  ]])]
    """

    thorax_axis = np.asarray(thorax_axis)
    r_sho_jc = np.asarray(r_sho_jc)
    l_sho_jc = np.asarray(l_sho_jc)

    thorax_origin = thorax_axis[:3, 3]

    R_shoulderJC = r_sho_jc[:3, 3]
    L_shoulderJC = l_sho_jc[:3, 3]

    R_wand = r_wand
    L_wand = l_wand

    R_wand_direc = R_wand - thorax_origin
    L_wand_direc = L_wand - thorax_origin
    R_wand_direc = R_wand_direc/np.linalg.norm(R_wand_direc)
    L_wand_direc = L_wand_direc/np.linalg.norm(L_wand_direc)

    # Right

    # Get the direction of the primary axis Z,X,Y
    z_direc = thorax_origin - R_shoulderJC
    z_direc = z_direc/np.linalg.norm(z_direc)
    y_direc = R_wand_direc * -1
    x_direc = np.cross(y_direc, z_direc)
    x_direc = x_direc/np.linalg.norm(x_direc)
    y_direc = np.cross(z_direc, x_direc)
    y_direc = y_direc/np.linalg.norm(y_direc)

    # backwards to account for marker size
    x_axis = x_direc
    y_axis = y_direc
    z_axis = z_direc

    r_sho = np.zeros((4, 4))
    r_sho[3, 3] = 1.0
    r_sho[0, :3] = x_axis
    r_sho[1, :3] = y_axis
    r_sho[2, :3] = z_axis
    r_sho[:3, 3] = R_shoulderJC

    # Left

    # Get the direction of the primary axis Z,X,Y
    z_direc = thorax_origin - L_shoulderJC
    z_direc = z_direc/np.linalg.norm(z_direc)
    y_direc = L_wand_direc
    x_direc = np.cross(y_direc, z_direc)
    x_direc = x_direc/np.linalg.norm(x_direc)
    y_direc = np.cross(z_direc, x_direc)
    y_direc = y_direc/np.linalg.norm(y_direc)

    # backwards to account for marker size
    x_axis = x_direc
    y_axis = y_direc
    z_axis = z_direc

    l_sho = np.zeros((4, 4))
    l_sho[3, 3] = 1.0
    l_sho[0, :3] = x_axis
    l_sho[1, :3] = y_axis
    l_sho[2, :3] = z_axis
    l_sho[:3, 3] = L_shoulderJC

    shoulder = np.array([r_sho, l_sho])

    return shoulder


def calc_axis_elbow(relb, lelb, rwra, rwrb, lwra, lwrb, r_shoulder_jc, l_shoulder_jc, r_elbow_width, l_elbow_width, r_wrist_width, l_wrist_width, mm):
    """Calculate the elbow joint center and axis.

    Takes in markers that correspond to (x, y, z) positions of the current
    frame, the shoulder joint center, elbow widths, wrist widths, and the
    marker size in millimeters..

    Markers used: relb, lelb, rwra, rwrb, lwra, lwrb.

    Subject Measurement values used: r_elbow_width, l_elbow_width, r_wrist_width,
    l_wrist_width.

    Parameters
    ----------
    relb : array
        1x3 RELB marker
    lelb : array
        1x3 LELB marker
    rwra : array
        1x3 RWRA marker
    rwrb : array
        1x3 RWRB marker
    lwra : array
        1x3 LWRA marker
    lwrb : array
        1x3 LWRB marker
    r_shoulder_jc : ndarray
        A 4x4 identity matrix that holds the right shoulder joint center
    l_shoulder_jc : ndarray
        A 4x4 identity matrix that holds the left shoulder joint center
    r_elbow_width : float
        The width of the right elbow
    l_elbow_width : float
        The width of the left elbow
    r_wrist_width : float
        The width of the right wrist
    l_wrist_width : float
        The width of the left wrist
    mm : float
        The thickness of the marker in millimeters

    Returns
    -------
    [r_axis, l_axis, r_wri_origin, l_wri_origin] : array
        An array consisting of a 4x4 affine matrix representing the
        right elbow axis, a 4x4 affine matrix representing the left 
        elbow axis, a 4x4 affine matrix representing the right wrist
        origin, and a 4x4 affine matrix representing the left wrist origin.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import calc_axis_elbow
    >>> np.set_printoptions(suppress=True)
    >>> relb = np.array([ 658.90, 326.07, 1285.28])
    >>> lelb = np.array([-156.32, 335.25, 1287.39])
    >>> rwra = np.array([ 776.51, 495.68, 1108.38])
    >>> rwrb = np.array([ 830.90, 436.75, 1119.11])
    >>> lwra = np.array([-249.28, 525.32, 1117.09])
    >>> lwrb = np.array([-311.77, 477.22, 1125.16])
    >>> shoulder_jc = [np.array([[1., 0., 0.,  429.66],
    ...                          [0., 1., 0.,  275.06],
    ...                          [0., 0., 1., 1453.95],
    ...                          [0., 0., 0.,    1.  ]]),
    ...                np.array([[1., 0., 0.,   64.51],
    ...                          [0., 1., 0.,  274.93],
    ...                          [0., 0., 1., 1463.63],
    ...                          [0., 0., 0.,    1.  ]])]
    >>> [np.around(arr, 2) for arr in calc_axis_elbow(relb, lelb, rwra, rwrb, lwra, lwrb, shoulder_jc[0], shoulder_jc[1], 74.0, 74.0, 55.0, 55.0, 7.0)] #doctest: +NORMALIZE_WHITESPACE
    [array([[   0.14,   -0.99,   -0.  ,  633.66],
            [   0.69,    0.1 ,    0.72,  304.95],
            [  -0.71,   -0.1 ,    0.69, 1256.07],
            [   0.  ,    0.  ,    0.  ,    1.  ]]), 
    array([[   -0.15,   -0.99,   -0.06, -129.16],
            [   0.72,   -0.07,   -0.69,  316.86],
            [   0.68,   -0.15,    0.72, 1258.06],
            [   0.  ,    0.  ,    0.  ,    1.  ]]), 
    array([[    1.  ,    0.  ,    0.  ,  793.32],
            [   0.  ,    1.  ,    0.  ,  451.29],
            [   0.  ,    0.  ,    1.  , 1084.43],
            [   0.  ,    0.  ,    0.  ,    1.  ]]),  
    array([[    1.  ,    0.  ,    0.  , -272.46],
            [   0.  ,    1.  ,    0.  ,  485.79],
            [   0.  ,    0.  ,    1.  , 1091.37],
            [   0.  ,    0.  ,    0.  ,    1.  ]])]
    """
    relb, lelb, rwra, rwrb, lwra, lwrb, r_shoulder_jc, l_shoulder_jc = map(np.asarray, [relb, lelb, rwra, rwrb, lwra, lwrb, r_shoulder_jc, l_shoulder_jc])

    r_elbow_width *= -1
    r_delta = (r_elbow_width/2.0)-mm
    l_delta = (l_elbow_width/2.0)+mm

    rwri = (rwra + rwrb) / 2.0
    lwri = (lwra + lwrb) / 2.0

    rsjc = r_shoulder_jc[:3, 3]
    lsjc = l_shoulder_jc[:3, 3]

    # make the construction vector for finding the elbow joint center
    r_con_1     = np.subtract(rsjc, relb)
    r_con_1_div = np.linalg.norm(r_con_1)
    r_con_1     = np.divide(r_con_1, r_con_1_div)

    r_con_2     = np.subtract(rwri, relb)
    r_con_2_div = np.linalg.norm(r_con_2)
    r_con_2     = np.divide(r_con_2, r_con_2_div)

    r_cons_vec     = np.cross(r_con_1, r_con_2)
    r_cons_vec_div = np.linalg.norm(r_cons_vec)
    r_cons_vec     = np.divide(r_cons_vec, r_cons_vec_div)

    r_cons_vec = r_cons_vec * 500 + relb

    l_con_1     = np.subtract(lsjc, lelb)
    l_con_1_div = np.linalg.norm(l_con_1)
    l_con_1     = np.divide(l_con_1, l_con_1_div)

    l_con_2     = np.subtract(lwri, lelb)
    l_con_2_div = np.linalg.norm(l_con_2)
    l_con_2     = np.divide(l_con_2, l_con_2_div)

    l_cons_vec     = np.cross(l_con_1, l_con_2)
    l_cons_vec_div = np.linalg.norm(l_cons_vec)

    l_cons_vec = np.divide(l_cons_vec, l_cons_vec_div)

    l_cons_vec = l_cons_vec * 500 + lelb

    rejc = calc_joint_center(r_cons_vec, rsjc, relb, r_delta)
    lejc = calc_joint_center(l_cons_vec, lsjc, lelb, l_delta)

    # this is radius axis for humerus
    # right
    x_axis     = np.subtract(rwra, rwrb)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis     = np.divide(x_axis, x_axis_div)

    z_axis     = np.subtract(rejc, rwri)
    z_axis_div = np.linalg.norm(z_axis)
    z_axis     = np.divide(z_axis, z_axis_div)

    y_axis     = np.cross(z_axis, x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis     = np.divide(y_axis, y_axis_div)

    x_axis     = np.cross(y_axis, z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis     = np.divide(x_axis, x_axis_div)

    r_radius = [x_axis, y_axis, z_axis]

    # left
    x_axis     = np.subtract(lwra, lwrb)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis     = np.divide(x_axis, x_axis_div)

    z_axis     = np.subtract(lejc, lwri)
    z_axis_div = np.linalg.norm(z_axis)
    z_axis     = np.divide(z_axis, z_axis_div)

    y_axis     = np.cross(z_axis, x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis     = np.divide(y_axis, y_axis_div)

    x_axis     = np.cross(y_axis, z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis     = np.divide(x_axis, x_axis_div)

    l_radius = [x_axis, y_axis, z_axis]

    # calculate wrist joint center for humerus
    r_wrist_width = (r_wrist_width/2.0 + mm)
    l_wrist_width = (l_wrist_width/2.0 + mm)

    rwjc = [
                rwri[0] + r_wrist_width * r_radius[1][0],
                rwri[1] + r_wrist_width * r_radius[1][1],
                rwri[2] + r_wrist_width * r_radius[1][2]
           ]

    lwjc = [
                lwri[0] - l_wrist_width * l_radius[1][0],
                lwri[1] - l_wrist_width * l_radius[1][1],
                lwri[2] - l_wrist_width * l_radius[1][2]
           ]

    # recombine the humerus axis
    # right
    z_axis     = np.subtract(rsjc, rejc)
    z_axis_div = np.linalg.norm(z_axis)
    z_axis     = np.divide(z_axis, z_axis_div)

    x_axis     = np.subtract(rwjc, rejc)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis     = np.divide(x_axis, x_axis_div)

    y_axis     = np.cross(x_axis, z_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis     = np.divide(y_axis, y_axis_div)

    x_axis     = np.cross(y_axis, z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis     = np.divide(x_axis, x_axis_div)

    r_axis = np.zeros((4, 4))
    r_axis[3, 3] = 1.0
    r_axis[0, :3] = x_axis
    r_axis[1, :3] = y_axis
    r_axis[2, :3] = z_axis
    r_axis[:3, 3] = rejc

    # left
    z_axis     = np.subtract(lsjc, lejc)
    z_axis_div = np.linalg.norm(z_axis)
    z_axis     = np.divide(z_axis, z_axis_div)

    x_axis     = np.subtract(lwjc, lejc)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis     = np.divide(x_axis, x_axis_div)

    y_axis     = np.cross(x_axis, z_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis     = np.divide(y_axis, y_axis_div)

    x_axis     = np.cross(y_axis, z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis     = np.divide(x_axis, x_axis_div)

    l_axis = np.zeros((4, 4))
    l_axis[3, 3] = 1.0
    l_axis[0, :3] = x_axis
    l_axis[1, :3] = y_axis
    l_axis[2, :3] = z_axis
    l_axis[:3, 3] = lejc

    r_wri_origin = np.identity(4)
    r_wri_origin[:3, 3] = rwjc

    l_wri_origin = np.identity(4)
    l_wri_origin[:3, 3] = lwjc

    return np.asarray([r_axis, l_axis, r_wri_origin, l_wri_origin])


def calc_axis_wrist(r_elbow, l_elbow, r_wrist_jc, l_wrist_jc):
    r"""Calculate the wrist joint center and axis.

    Takes in the right and left elbow axes, 
    and the right and left wrist joint centers.

    Parameters
    ----------
    r_elbow : array
        4x4 affine matrix representing the right elbow axis and origin
    l_elbow : array
        4x4 affine matrix representing the left elbow axis and origin
    r_wrist_jc : array
        4x4 affine matrix representing the right wrist joint center
    l_wrist_jc : array
        4x4 affine matrix representing the left wrist joint center

    Returns
    --------
    [r_axis, l_axis] : array
        A list of two 4x4 affine matrices representing the right hand axis as
        well as the left hand axis.

    Notes
    -----
    .. math::
        \begin{matrix}
            o_{L} = \textbf{m}_{LWJC} & o_{R} = \textbf{m}_{RWJC} \\
            \hat{y}_{L} = Elbow\_Flex_{L} & \hat{y}_{R} =  Elbow\_Flex_{R} \\
            \hat{z}_{L} = \textbf{m}_{LEJC} - \textbf{m}_{LWJC} & \hat{z}_{R} = \textbf{m}_{REJC} - \textbf{m}_{RWJC} \\
            \hat{x}_{L} = \hat{y}_{L} \times \hat{z}_{L} & \hat{x}_{R} = \hat{y}_{R} \times \hat{z}_{R} \\
            \hat{z}_{L} = \hat{x}_{L} \times \hat{y}_{L} & \hat{z}_{R} = \hat{x}_{R} \times \hat{y}_{R} \\
        \end{matrix}

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import calc_axis_wrist
    >>> np.set_printoptions(suppress=True)
    >>> r_elbow = np.array([[ 0.15, -0.99,  0.  ,  633.66],
    ...                     [ 0.69,  0.1,   0.72,  304.95],
    ...                     [-0.71, -0.1,   0.7 , 1256.07],
    ...                     [  0.  , 0. ,   0.  ,    1.  ]])
    >>> l_elbow = np.array([[-0.16, -0.98, -0.06, -129.16],
    ...                     [ 0.71, -0.07, -0.69,  316.86],
    ...                     [ 0.67, -0.14,  0.72, 1258.06],
    ...                     [ 0.  ,  0.  ,  0.  ,    1.  ]])
    >>> r_wrist_jc = np.array([[793.77, 450.44, 1084.12,  793.32],
    ...                        [794.01, 451.38, 1085.15,  451.29],
    ...                        [792.75, 450.76, 1085.05, 1084.43],
    ...                        [  0.,     0.,      0.,      1.]])
    >>> l_wrist_jc = np.array([[-272.92, 485.01, 1090.96, -272.45],
    ...                        [-271.74, 485.72, 1090.67,  485.8],
    ...                        [-271.94, 485.19, 1091.96, 1091.36],
    ...                        [   0.,     0.,      0.,      1.]])
    >>> [np.around(arr, 2) for arr in calc_axis_wrist(r_elbow, l_elbow, r_wrist_jc, l_wrist_jc)] #doctest: +NORMALIZE_WHITESPACE
    [array([[  0.44,   -0.84,   -0.31,  793.32],
            [  0.69,    0.1 ,    0.72,  451.29],
            [ -0.57,   -0.53,    0.62, 1084.43],
            [  0.  ,    0.  ,    0.  ,    1.  ]]), 
     array([[ -0.47,   -0.79,   -0.4 , -272.45],
            [  0.72,   -0.07,   -0.7 ,  485.8 ],
            [  0.52,   -0.61,    0.6 , 1091.36],
            [  0.  ,    0.  ,    0.  ,    1.  ]])]
    """
    # Bring Elbow joint center, axes and Wrist Joint Center for calculating Radius Axes
    r_elbow, l_elbow, r_wrist_jc, l_wrist_jc = map(np.asarray, [r_elbow, l_elbow, r_wrist_jc, l_wrist_jc])

    rejc = r_elbow[:3, 3]
    lejc = l_elbow[:3, 3]

    r_elbow_flex = r_elbow[1, :3]
    l_elbow_flex = l_elbow[1, :3]

    rwjc = r_wrist_jc[:3, 3]
    lwjc = l_wrist_jc[:3, 3]

    # this is the axis of radius
    # right
    y_axis = r_elbow_flex
    y_axis = y_axis/np.linalg.norm(y_axis)

    z_axis = np.subtract(rejc, rwjc)
    z_axis = z_axis/np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis/np.linalg.norm(x_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis/np.linalg.norm(z_axis)

    r_axis = np.zeros((4, 4))
    r_axis[3, 3] = 1.0
    r_axis[0, :3] = x_axis
    r_axis[1, :3] = y_axis
    r_axis[2, :3] = z_axis
    r_axis[:3, 3] = rwjc

    # left
    y_axis = l_elbow_flex
    y_axis = y_axis/np.linalg.norm(y_axis)

    z_axis = np.subtract(lejc, lwjc)
    z_axis = z_axis/np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis/np.linalg.norm(x_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis/np.linalg.norm(z_axis)

    l_axis = np.zeros((4, 4))
    l_axis[3, 3] = 1.0
    l_axis[0, :3] = x_axis
    l_axis[1, :3] = y_axis
    l_axis[2, :3] = z_axis
    l_axis[:3, 3] = lwjc

    return np.asarray([r_axis, l_axis])


def calc_axis_hand(rwra, rwrb, lwra, lwrb, rfin, lfin, r_wrist_jc, l_wrist_jc, r_hand_thickness, l_hand_thickness):
    r"""Calculate the hand joint center and axis.

    Takes in markers that correspond to (x, y, z) positions of the current
    frame, the right and left wrist joint centers, and the right and 
    left hand thickness.

    Markers used: RWRA, RWRB, LWRA, LWRB, RFIN, LFIN

    Subject Measurement values used: RightHandThickness, LeftHandThickness

    Parameters
    ----------
    rwra : array
        1x3 RWRA marker
    rwrb : array
        1x3 RWRB marker
    lwra : array
        1x3 LWRA marker
    lwrb : array
        1x3 LWRB marker
    rfin : array
        1x3 RFIN marker
    lfin : array
        1x3 LFIN marker
    r_wrist_jc : array
        4x4 affine matrix representing the right wrist joint center
    l_wrist_jc : array
        4x4 affine matrix representing the left wrist joint center
    r_hand_thickness : float
        The thickness of the right hand
    l_hand_thickness : float
        The thickness of the left hand

    Returns
    -------
    [r_axis, l_axis] : array
        An array of two 4x4 affine matrices representing the
        right and left hand axes and origins.

    Notes
    -----
    :math:`r_{delta} = (\frac{r\_hand\_thickness}{2.0} + 7.0) \hspace{1cm} l_{delta} = (\frac{l\_hand\_thickness}{2.0} + 7.0)`

    :math:`\textbf{m}_{RHND} = JC(\textbf{m}_{RWRI}, \textbf{m}_{RWJC}, \textbf{m}_{RFIN}, r_{delta})`

    :math:`\textbf{m}_{LHND} = JC(\textbf{m}_{LWRI}, \textbf{m}_{LWJC}, \textbf{m}_{LFIN}, r_{delta})`

    .. math::

        \begin{matrix}
            o_{L} = \frac{\textbf{m}_{LWRA} + \textbf{m}_{LWRB}}{2} & o_{R} = \frac{\textbf{m}_{RWRA} + \textbf{m}_{RWRB}}{2} \\
            \hat{z}_{L} = \textbf{m}_{LWJC} - \textbf{m}_{LHND} & \hat{z}_{R} = \textbf{m}_{RWJC} - \textbf{m}_{RHND} \\
            \hat{y}_{L} = \textbf{m}_{LWRI} - \textbf{m}_{LWRA} & \hat{y}_{R} = \textbf{m}_{RWRA} - \textbf{m}_{RWRI} \\
            \hat{x}_{L} = \hat{y}_{L} \times \hat{z}_{L} & \hat{x}_{R} = \hat{y}_{R} \times \hat{z}_{R} \\
            \hat{y}_{L} = \hat{z}_{L} \times \hat{x}_{L} & \hat{y}_{R} = \hat{z}_{R} \times \hat{x}_{R}
        \end{matrix}

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import calc_axis_hand
    >>> np.set_printoptions(suppress=True)
    >>> rwra = np.array([ 776.51, 495.68, 1108.38])
    >>> rwrb = np.array([ 830.90, 436.75, 1119.11])
    >>> lwra = np.array([-249.28, 525.32, 1117.09])
    >>> lwrb = np.array([-311.77, 477.22, 1125.16])
    >>> rfin = np.array([ 863.71, 524.44, 1074.54])
    >>> lfin = np.array([-326.65, 558.34, 1091.04])
    >>> r_wrist_jc = np.array([[ 793.77, 450.44, 1084.12,  793.32],
    ...                        [ 794.01, 451.38, 1085.15,  451.29],
    ...                        [ 792.75, 450.76, 1085.05, 1084.43],
    ...                        [   0.,     0.,      0.,      1.]])
    >>> l_wrist_jc = np.array([[-272.92, 485.01, 1090.96, -272.45],
    ...                        [-271.74, 485.72, 1090.67,  485.8],
    ...                        [-271.94, 485.19, 1091.96, 1091.36],
    ...                        [   0.,     0.,      0.,      1.]])
    >>> r_hand_thickness = 34.0
    >>> l_hand_thickness = 34.0
    >>> [np.around(arr, 2) for arr in calc_axis_hand(rwra, rwrb, lwra, lwrb, rfin, lfin, r_wrist_jc, l_wrist_jc, r_hand_thickness, l_hand_thickness)] #doctest: +NORMALIZE_WHITESPACE
    [array([[  0.15,  0.31,  0.94,  859.8 ],
            [ -0.73,  0.68, -0.11,  517.27],
            [ -0.67, -0.67,  0.33, 1051.97],
            [  0.  ,  0.  ,  0.  ,    1.  ]]), 
     array([[ -0.09,  0.27,  0.96, -324.52],
            [ -0.8 , -0.59,  0.1 ,  551.89],
            [  0.6 , -0.76,  0.27, 1068.02],
            [  0.  ,  0.  ,  0.  ,    1.  ]])]
    """
    r_wrist_jc, l_wrist_jc, rwra, rwrb, lwra, lwrb, rfin, lfin = map(np.asarray, [r_wrist_jc, l_wrist_jc, rwra, rwrb, lwra, lwrb, rfin, lfin])

    rwri = (rwra + rwrb) / 2.0
    lwri = (lwra + lwrb) / 2.0

    rwjc = r_wrist_jc[:3, 3]
    lwjc = l_wrist_jc[:3, 3]

    mm = 7.0

    r_delta = (r_hand_thickness/2.0 + mm)
    l_delta = (l_hand_thickness/2.0 + mm)

    lhnd = calc_joint_center(lwri, lwjc, lfin, l_delta)
    rhnd = calc_joint_center(rwri, rwjc, rfin, r_delta)

    # Left
    z_axis = lwjc - lhnd
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = np.divide(z_axis, z_axis_div)

    y_axis = lwri - lwra
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = np.divide(y_axis, y_axis_div)

    x_axis = np.cross(y_axis, z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = np.divide(x_axis, x_axis_div)

    y_axis = np.cross(z_axis, x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = np.divide(y_axis, y_axis_div)

    l_axis = np.zeros((4, 4))
    l_axis[3, 3] = 1.0
    l_axis[0, :3] = x_axis
    l_axis[1, :3] = y_axis
    l_axis[2, :3] = z_axis
    l_axis[:3, 3] = lhnd

    # Right
    z_axis = rwjc - rhnd
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = np.divide(z_axis, z_axis_div)

    y_axis = rwra - rwri
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = np.divide(y_axis, y_axis_div)

    x_axis = np.cross(y_axis, z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = np.divide(x_axis, x_axis_div)

    y_axis = np.cross(z_axis, x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = np.divide(y_axis, y_axis_div)

    r_axis = np.zeros((4, 4))
    r_axis[3, 3] = 1.0
    r_axis[0, :3] = x_axis
    r_axis[1, :3] = y_axis
    r_axis[2, :3] = z_axis
    r_axis[:3, 3] = rhnd

    return np.asarray([r_axis, l_axis])


def calc_joint_center(p_a, p_b, p_c, delta):
    r"""Calculate the Joint Center.

    This function is based on the physical markers p_a, p_b, p_c
    and the resulting joint center are all on the same plane.

    Parameters
    ----------
    p_a : array
        (x, y, z) position of marker a
    p_b : array 
        (x, y, z) position of marker b
    p_c : array
        (x, y, z) position of marker c
    delta : float
        The length from marker to joint center, retrieved from subject measurement file

    Returns
    -------
    joint_center : array
        (x, y, z) position of the joint center

    Notes
    -----
    :math:`vec_{1} = p\_a-p\_c, \ vec_{2} = (p\_b-p\_c), \ vec_{3} = vec_{1} \times vec_{2}`

    :math:`mid = \frac{(p\_b+p\_c)}{2.0}`

    :math:`length = (p\_b - mid)`

    :math:`\theta = \arccos(\frac{delta}{vec_{2}})`

    :math:`\alpha = \cos(\theta*2), \ \beta = \sin(\theta*2)`

    :math:`u_x, u_y, u_z = vec_{3}`

    .. math::

        rot =
        \begin{bmatrix}
            \alpha+u_x^2*(1-\alpha) & u_x*u_y*(1.0-\alpha)-u_z*\beta & u_x*u_z*(1.0-\alpha)+u_y*\beta \\
            u_y*u_x*(1.0-\alpha+u_z*\beta & \alpha+u_y^2.0*(1.0-\alpha) & u_y*u_z*(1.0-\alpha)-u_x*\beta \\
            u_z*u_x*(1.0-\alpha)-u_y*\beta & u_z*u_y*(1.0-\alpha)+u_x*\beta & \alpha+u_z**2.0*(1.0-\alpha) \\
        \end{bmatrix}

    :math:`r\_vec = rot * vec_2`

    :math:`r\_vec = r\_vec * \frac{length}{norm(r\_vec)}`

    :math:`joint\_center = r\_vec + mid`

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import calc_joint_center
    >>> p_a = np.array([468.14, 325.09, 673.12])
    >>> p_b = np.array([355.90, 365.38, 940.69])
    >>> p_c = np.array([452.35, 329.06, 524.77])
    >>> delta = 59.5
    >>> calc_joint_center(p_a, p_b, p_c, delta).round(2)
    array([396.25, 347.92, 518.63])
    """

    # make the two vector using 3 markers, which is on the same plane.

    p_a, p_b, p_c = map(np.asarray, [p_a, p_b, p_c])

    vec_1 = p_a - p_c
    vec_2 = p_b - p_c

    # vec_3 is cross vector of vec_1, vec_2, and then it normalized.
    vec_3 = np.cross(vec_1, vec_2)
    vec_3_div = np.linalg.norm(vec_3)
    vec_3 = vec_3 / vec_3_div

    mid = (p_b + p_c) / 2.0
    length = np.subtract(p_b, mid)
    length = np.linalg.norm(length)

    theta = math.acos(delta/np.linalg.norm(vec_2))

    alpha = math.cos(theta*2)
    beta = math.sin(theta*2)

    u_x, u_y, u_z = vec_3

    # This rotation matrix is called Rodriques' rotation formula.
    # In order to make a plane, at least 3 number of markers is required which
    # means three physical markers on the segment can make a plane.
    # then the orthogonal vector of the plane will be rotating axis.
    # joint center is determined by rotating the one vector of plane around rotating axis.

    rot = np.matrix([ 
        [alpha+u_x**2.0*(1.0-alpha),   u_x*u_y*(1.0-alpha) - u_z*beta, u_x*u_z*(1.0-alpha)+u_y*beta],
        [u_y*u_x*(1.0-alpha)+u_z*beta, alpha+u_y**2.0 * (1.0-alpha),   u_y*u_z*(1.0-alpha)-u_x*beta],
        [u_z*u_x*(1.0-alpha)-u_y*beta, u_z*u_y*(1.0-alpha) + u_x*beta, alpha+u_z**2.0*(1.0-alpha)]
    ])

    r_vec = rot * (np.matrix(vec_2).transpose())
    r_vec = r_vec * length/np.linalg.norm(r_vec)

    r_vec = np.asarray(r_vec)[:3, 0]
    joint_center = r_vec + mid

    return joint_center


def calc_angle_head(axis_p, axis_d):
    r"""Head angle calculation.

    Takes in two axes and returns the head rotation, 
    flexion, and abduction angles in degrees.

    Uses the inverse Euler rotation matrix in YXZ order.

    Parameters
    ----------
    axis_p : array
        4x4 affine matrix representing the position of the proximal axis.
    axis_d : array
        4x4 affine matrix representing the position of the distal axis.

    Returns
    -------
    angle : array
        1x3 array representing the head rotation, flexion, and abduction angles in degrees

    Notes
    -----
    :math:`\alpha = \arctan2{(-(axisD_{z} \cdot axisP_{x}), axisD_{z} \cdot axisP_{z})}`

    :math:`\beta = \arctan2{((axisD_{z} \cdot axisP_{y}), \sqrt{(axisD_{x} \cdot axisP_{y})^2 + (axisD_{y} \cdot axisP_{y})^2}})`

    :math:`\gamma = \arctan2{(-(axisD_{x} \cdot axisP_{y}), axisD_{y} \cdot axisP_{y})}`

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import calc_angle_head
    >>> axis_p = np.array([[ 0.04,  0.99,  0.06, 512.34],
    ...                    [ 0.99, -0.04, -0.05, 471.15],
    ...                    [-0.05,  0.07, -0.99, 124.14],
    ...                    [ 0.,    0.,    0.,     1.]])
    >>> axis_d = np.array([[-0.18, -0.98, -0.02, 842.14],
    ...                    [ 0.71, -0.11, -0.69, 985.38],
    ...                    [ 0.67, -0.14,  0.72, 412.87],
    ...                    [ 0.,    0.,    0.,     1.]])
    >>> np.around(calc_angle_head(axis_p, axis_d), 2)
    array([ 185.18,  -39.99, -190.54])
    """

    axis_p = np.asarray(axis_p)
    axis_d = np.asarray(axis_d)

    axis_p = axis_p[:3, :3]
    axis_d = axis_d[:3, :3]

    ang = (np.dot(-1 * axis_d[2], axis_p[1]))
    alpha = np.nan
    if -1 <= ang <= 1:
        alpha = np.arcsin(ang)

    # Beta is the flexion angle, alpha is the abduction angle, gamma is the rotation angle

    beta = np.arctan2(np.dot(axis_d[2], axis_p[1]),
                      np.sqrt((np.dot(axis_d[0], axis_p[1])) ** 2 
                            + (np.dot(axis_d[1], axis_p[1]) ** 2)))

    alpha = np.arctan2(-1 * (np.dot(axis_d[2], axis_p[0])), 
                            (np.dot(axis_d[2], axis_p[2])))

    gamma = np.arctan2(-1 * (np.dot(axis_d[0], axis_p[1])), 
                            (np.dot(axis_d[1], axis_p[1])))

    alpha = 180.0 * alpha / pi
    beta =  180.0 * beta / pi
    gamma = 180.0 * gamma / pi

    beta *= -1

    if alpha < 0:
        alpha *= -1
    elif 0 < alpha < 180:
        alpha = 180 + (180 - alpha)

    if gamma > 90.0:
        if gamma > 120:
            gamma = (gamma - 180) * -1
        else:
            gamma = (gamma + 180) * -1
    else:
        if gamma < 0:
            gamma = (gamma + 180) * -1
        else:
            gamma = (gamma * -1) - 180.0

    angle = [alpha, beta, gamma]

    return np.asarray(angle)


def calc_angle_shoulder(axis_thorax, axis_hum_right, axis_hum_left):
    r"""Shoulder angle calculation.

    Takes in the thorax and elbow axes and returns the right and 
    left shoulder rotation, flexion, and abduction angles in degrees.

    Parameters
    ----------
    axis_thorax : array
        4x4 affine matrix representing the position of the thorax axis
    axis_hum_right : array
        4x4 affine matrix representing the position of the right elbow axis
    axis_hum_left : array
        4x4 affine matrix representing the position of the left elbow axis

    Returns
    -------
    angles : array
        2x3 array representing the right and left elbow rotation, flexion,
        and abduction angles in degrees

    Notes
    -----
    :math:`\alpha_{right} = \arcsin{(axis\_hum\_right_{z} \cdot axis\_thorax_{x})}`

    :math:`\beta_{right} = \arctan2{(-(axis\_hum\_right_{z} \cdot axis\_thorax_{y}), axis\_hum\_right_{z} \cdot axis\_thorax_{z})}`

    :math:`\gamma_{right} = \arctan2{(-(axis\_hum\_right_{y} \cdot axis\_thorax_{x}), axis\_hum\_right_{x} \cdot axis\_thorax_{x})}`

    :math:`\alpha_{left} = \arcsin{(axis\_hum\_left_{z} \cdot axis\_thorax_{x})}`

    :math:`\beta_{left} = \arctan2{(-(axis\_hum\_left_{z} \cdot axis\_thorax_{y}), axis\_hum\_left_{z} \cdot axis\_thorax_{z})}`

    :math:`\gamma_{left} = \arctan2{(-(axis\_hum\_left_{y} \cdot axis\_thorax_{x}), axis\_hum\_left_{x} \cdot axis\_thorax_{x})}`


    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)
    >>> from .pyCGM import calc_angle_shoulder
    >>> axis_thorax = np.array([[ 0.04,  0.99,  0.06, 214.14],
    ...                         [ 0.99, -0.04, -0.05,  32.14],
    ...                         [-0.05,  0.07, -0.99, 452.89],
    ...                         [ 0.,    0.,    0.,     1.]])
    >>> axis_hum_right = np.array([[-0.97, -0.16, 0.19, -971.69],
    ...                            [ 0.18, -0.98, 0.11, -216.63],
    ...                            [ 0.16,  0.14, 0.98,  966.89],
    ...                            [ 0.,    0.,   0.,      1.]])
    >>> axis_hum_left = np.array([[ -0.97,  0.17,  0.15, -952.27],
    ...                            [-0.20, -0.95, -0.20,  235.83],
    ...                            [ 0.11, -0.23,  0.96,  954.59],
    ...                            [ 0.,    0.,    0.,      1.]])
    >>> np.around(calc_angle_shoulder(axis_thorax, axis_hum_right, axis_hum_left), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 11.76, -173.88, 100.99],
           [ -9.54, -175.88,  81.79]])
    """

    # beta is flexion / extension
    # gamma is adduction / abduction
    # alpha is internal / external rotation

    axis_thorax, axis_hum_right, axis_hum_left = map(np.asarray, [axis_thorax, axis_hum_right, axis_hum_left])

    axis_thorax = axis_thorax[:3, :3]
    axis_hum_right = axis_hum_right[:3, :3]
    axis_hum_left = axis_hum_left[:3, :3]

    # Right shoulder angle
    alpha = np.arcsin(np.dot(axis_hum_right[2], axis_thorax[0]))

    beta = np.arctan2(-1 * (np.dot(axis_hum_right[2], axis_thorax[1])), 
                           (np.dot(axis_hum_right[2], axis_thorax[2])))

    gamma = np.arctan2(-1 * (np.dot(axis_hum_right[1], axis_thorax[0])), 
                            (np.dot(axis_hum_right[0], axis_thorax[0])))

    right_angle = [180.0 * alpha / pi, 180.0 * beta / pi, 180.0 * gamma / pi]

    # Left shoulder angle
    alpha = np.arcsin(np.dot(axis_hum_left[2], axis_thorax[0]))

    beta = np.arctan2(-1 * (np.dot(axis_hum_left[2], axis_thorax[1])), 
                           (np.dot(axis_hum_left[2], axis_thorax[2])))

    gamma = np.arctan2(-1 * (np.dot(axis_hum_left[1], axis_thorax[0])),
                            (np.dot(axis_hum_left[0], axis_thorax[0])))

    left_angle = [180.0 * alpha / pi, 180.0 * beta / pi, 180.0 * gamma / pi]

    angles = np.array([right_angle, left_angle])

    return angles


def calc_angle_spine(axis_pelvis, axis_thorax):
    r"""Spine angle calculation.

    Takes in the pelvis and thorax axes and returns the spine rotation, 
    flexion, and abduction angles in degrees.

    Uses the inverse Euler rotation matrix in YXZ order.

    Parameters
    ----------
    axis_pelvis : array
        4x4 affine matrix representing the position of the pelvis axis.
    axis_thorax : array
        4x4 affine matrix representing the position of the thorax axis.

    Returns
    -------
    angle : array
        1x3 array representing the spine rotation, flexion, and abduction angles in degrees

    Notes
    -----
        :math:`\alpha = \arcsin{(axis\_d_{y} \cdot axis\_p_{z})}`

        :math:`\gamma = \arcsin{(-(axis\_d_{y} \cdot axis\_p_{x}) / \cos{\alpha})}`

        :math:`\beta = \arcsin{(-(axis\_d_{x} \cdot axis\_p_{z}) / \cos{\alpha})}`

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import calc_angle_spine
    >>> axis_pelvis = [[ 0.04,  0.99,  0.06, 749.24],
    ...                [ 0.99, -0.04, -0.05, 321.12],
    ...                [-0.05,  0.07, -0.99, 145.12],
    ...                [ 0.,    0.,    0.,     1.]]
    >>> axis_thorax = [[-0.18, -0.98, -0.02, 541.68],
    ...                [ 0.71, -0.11, -0.69, 112.48],
    ...                [ 0.67, -0.14,  0.72, 155.77],
    ...                [ 0.,    0.,    0.,     1.]]
    >>> np.around(calc_angle_spine(axis_pelvis, axis_thorax), 2) 
    array([ 2.97,  9.13, 39.78])
    """
    # Calculation for the spine angle.

    axis_pelvis = np.asarray(axis_pelvis)
    p_x = axis_pelvis[0, :3]
    p_y = axis_pelvis[1, :3]
    p_z = axis_pelvis[2, :3]

    axis_thorax = np.asarray(axis_thorax)
    t_x = axis_thorax[0, :3]
    t_y = axis_thorax[1, :3]
    t_z = axis_thorax[2, :3]

    alpha = np.arcsin(np.dot(t_y, p_z))
    gamma = np.arcsin((-1 * np.dot(t_y, p_x)) / np.cos(alpha))
    beta  = np.arcsin((-1 * np.dot(t_x, p_z)) / np.cos(alpha))

    angle = [180.0 * beta / pi, 180.0 * gamma / pi, 180.0 * alpha / pi]

    return np.asarray(angle)
    

def calc_angle(axis_p, axis_d):
    r"""Normal angle calculation.

    Takes in two axes and returns the rotation, flexion,
    and abduction angles in degrees.

    Uses the inverse Euler rotation matrix in YXZ order.

    Parameters
    ----------
    axis_p : array
        4x4 affine matrix representing the position of the proximal axis.
    axis_d : array
        4x4 affine matrix representing the position of the distal axis.

    Returns
    -------
    angle : array
        1x3 array representing the rotation, flexion, and abduction angles in degrees

    Notes
    -----
    As we use arcsin we have to care about if the angle is in area between -pi/2 to pi/2

    :math:`\alpha = \arcsin{(-axis\_d_{z} \cdot axis\_p_{y})}`

    If alpha is between -pi/2 and pi/2

    :math:`\beta = \arctan2{((axis\_d_{z} \cdot axis\_p_{x}), axis\_d_{z} \cdot axis\_p_{z})}`

    :math:`\gamma = \arctan2{((axis\_d_{y} \cdot axis\_p_{y}), axis\_d_{x} \cdot axis\_p_{y})}`

    Otherwise

    :math:`\beta = \arctan2{(-(axis\_d_{z} \cdot axis\_p_{x}), axis\_d_{z} \cdot axis\_p_{z})}`

    :math:`\gamma = \arctan2{(-(axis\_d_{y} \cdot axis\_p_{y}), axis\_d_{x} \cdot axis\_p_{y})}`

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import calc_angle
    >>> axis_p = [[ 0.04,  0.99,  0.06,  429.67],
    ...           [ 0.99, -0.04, -0.05,  275.15],
    ...           [-0.05,  0.07, -0.99, 1452.95],
    ...           [ 0.,    0.,    0.,      1.]]
    >>> axis_d = [[-0.18, -0.98, -0.02,   64.09],
    ...           [ 0.71, -0.11, -0.69,  275.83],
    ...           [ 0.67, -0.14,  0.72, 1463.78],
    ...           [ 0.,    0.,    0.,      1.]]
    >>> np.around(calc_angle(axis_p, axis_d), 2)
    array([-174.82,  -39.26,  100.54])
    """
    # Angle calculation is in Y-X-Z order

    axis_p = np.asarray(axis_p)
    p_x = axis_p[0, :3]
    p_y = axis_p[1, :3]
    p_z = axis_p[2, :3]

    axis_d = np.asarray(axis_d)
    d_x = axis_d[0, :3]
    d_y = axis_d[1, :3]
    d_z = axis_d[2, :3]

    ang = np.dot(-1 * d_z, p_y)

    alpha = np.nan
    if -1 <= ang <= 1:
        alpha = np.arcsin(ang)

    # Beta is the flexion angle, alpha is the abduction angle, gamma is the rotation angle
    # Check if the abduction angle is in the area between -pi/2 and pi/2
    if -1.57079633 < alpha < 1.57079633:
        beta = np.arctan2(np.dot(d_z, p_x),
                          np.dot(d_z, p_z))

        gamma = np.arctan2(np.dot(d_y, p_y),
                           np.dot(d_x, p_y))
    else:
        beta = np.arctan2(-1 * (np.dot(d_z, p_x)),
                                np.dot(d_z, p_z))

        gamma = np.arctan2(-1 * (np.dot(d_y, p_y)),
                                 np.dot(d_x, p_y))

    angle = [180.0 * beta / pi, 180.0 * alpha / pi, 180.0 * gamma / pi]

    return angle


def matrixmult (A, B):
    """Matrix multiplication.

    This function returns the product of a matrix multiplication given two matrices.

    Let the dimension of the matrix A be: m by n,
    let the dimension of the matrix B be: p by q,
    multiplication will only possible if n = p,
    creating a matrix of m by q size.

    Parameters
    ----------
    A : list
        First matrix, in a 2D array format.
    B : list
        Second matrix, in a 2D array format.

    Returns
    -------
    C : list
        The product of the matrix multiplication.

    Examples
    --------
    >>> from .pyCGM import matrixmult
    >>> A = [[11,12,13],[14,15,16]]
    >>> B = [[1,2],[3,4],[5,6]]
    >>> matrixmult(A, B)
    [[112, 148], [139, 184]]
    """

    C = [[0 for row in range(len(A))] for col in range(len(B[0]))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k]*B[k][j]
    return C


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
    r_xyz : array
        The product of the matrix multiplication.

    Notes
    -----
    :math:`r_x = [ [1,0,0], [0, \cos(x), -sin(x)], [0, sin(x), cos(x)] ]`
    :math:`r_y = [ [cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)] ]`
    :math:`r_z = [ [cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1] ]`
    :math:`r_{xy} = r_x * r_y`
    :math:`r_{xyz} = r_{xy} * r_z`

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import rotmat
    >>> x = 0.5
    >>> y = 0.3
    >>> z = 0.8
    >>> np.around(rotmat(x, y, z), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  , -0.01,  0.01],
           [ 0.01,  1.  , -0.01],
           [-0.01,  0.01,  1.  ]])
    >>> x = 0.5
    >>> np.around(rotmat(x), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1. ,  0.  ,  0.  ],
           [ 0. ,  1.  , -0.01],
           [ 0. ,  0.01,  1.  ]])
    >>> x = 1
    >>> y = 1
    >>> np.around(rotmat(x,y), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  ,  0.  ,  0.02],
           [ 0.  ,  1.  , -0.02],
           [-0.02,  0.02,  1.  ]])
    """

    x, y, z = math.radians(x), math.radians(y), math.radians(z)
    r_x = [[1, 0, 0], 
           [0, math.cos(x), math.sin(x) * -1], 
           [0, math.sin(x), math.cos(x)]]

    r_y = [[math.cos(y), 0, math.sin(y)],
           [0, 1, 0],
           [math.sin(y)*-1, 0, math.cos(y)]]

    r_z = [[math.cos(z), math.sin(z)*-1, 0],
           [math.sin(z), math.cos(z), 0],
           [0, 0, 1]]

    r_xy = np.matmul(r_x, r_y)
    r_xyz = np.matmul(r_xy, r_z)

    return r_xyz


def JointAngleCalc(frame,vsk):
    """ Joint Angle Calculation function.

    Calculates the Joint angles of plugingait and stores the data in array
    Stores:
    RPel_angle = []
    LPel_angle = []
    RHip_angle = []
    LHip_angle = []
    RKnee_angle = []
    LKnee_angle = []
    RAnkle_angle = []
    LAnkle_angle = []

    Joint Axis store like below form

    The axis is in the form [[origin], [axis]]
    Origin defines the position of axis and axis is the direction vector of
    x, y, z axis attached to the origin

    If it is just single one (Pelvis, Hip, Head, Thorax)

        Axis = [[origin_x, origin_y, origin_z],[[Xaxis_x,Xaxis_y,Xaxis_z],
                                                [Yaxis_x,Yaxis_y,Yaxis_z],
                                                [Zaxis_x,Zaxis_y,Zaxis_z]]]

    If it has both of Right and Left ( knee, angle, foot, clavicle, humerus, radius, hand)

        Axis = [[[R_origin_x,R_origin_y,R_origin_z],
                [L_origin_x,L_origin_y,L_origin_z]],[[[R_Xaxis_x,R_Xaxis_y,R_Xaxis_z],
                                                    [R_Yaxis_x,R_Yaxis_y,R_Yaxis_z],
                                                    [R_Zaxis_x,R_Zaxis_y,R_Zaxis_z]],
                                                    [[L_Xaxis_x,L_Xaxis_y,L_Xaxis_z],
                                                    [L_Yaxis_x,L_Yaxis_y,L_Yaxis_z],
                                                    [L_Zaxis_x,L_Zaxis_y,L_Zaxis_z]]]]

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.
    vsk : dict
        A dictionary containing subject measurements.

    Returns
    -------
    r, jc : tuple
        Returns a tuple containing an array that holds the result of all the joint calculations,
        followed by a dictionary for joint center marker positions.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import JointAngleCalc
    >>> from .pycgmIO import loadC3D, loadVSK
    >>> from .pycgmStatic import getStatic
    >>> from .pyCGM_Helpers import getfilenames
    >>> import os
    >>> fileNames=getfilenames(2)
    >>> c3dFile = fileNames[1]
    >>> vskFile = fileNames[2]
    >>> result = loadC3D(c3dFile)
    >>> data = result[0]
    >>> frame = result[0][0]
    >>> vskData = loadVSK(vskFile, False)
    >>> vsk = getStatic(data,vskData,flat_foot=False)
    >>> results = JointAngleCalc(frame, vsk)[1]
    >>> np.around(results['Pelvis'], 2)
    array([ 246.15,  353.26, 1031.71])
    >>> np.around(results['Thorax'], 2)
    array([ 250.56,  303.23, 1461.17])
    >>> np.around(results['Head'], 2)
    array([ 244.9 ,  325.06, 1730.16])
    >>> np.around(results['RHand'], 2)
    array([ 770.93,  591.05, 1079.05])
    """

    #First Calculate Pelvis
    pelvis_axis = calc_axis_pelvis(frame['RASI'] if 'RASI' in frame else None,
                                   frame['LASI'] if 'LASI' in frame else None,
                                   frame['RPSI'] if 'RPSI' in frame else None,
                                   frame['LPSI'] if 'LPSI' in frame else None,
                                   frame['SACR'] if 'SACR' in frame else None)

    pelvis_axes = pelvis_axis[:3, :3]
    pelvis_jc   = pelvis_axis[:3, 3]

    #need to update this based on the file
    global_axes = vsk['GCS']

    global_pelvis_angle = calc_angle(global_axes,pelvis_axes)
    pelx = global_pelvis_angle[0]
    pely = global_pelvis_angle[1]
    pelz = global_pelvis_angle[2]

    hip_jc = calc_joint_center_hip(pelvis_axis, 
                                   vsk["MeanLegLength"], 
                                   vsk["R_AsisToTrocanterMeasure"],
                                   vsk["L_AsisToTrocanterMeasure"],
                                   vsk["InterAsisDistance"])
    
    r_hip_jc = hip_jc[0]
    l_hip_jc = hip_jc[1]

    hip_axis = calc_axis_hip(r_hip_jc, l_hip_jc, pelvis_axis)
    hip_jc   = hip_axis[:3, 3]
    hip_axes = hip_axis[:3, :3]

    axis_knee = calc_axis_knee(frame['RTHI'] if 'RTHI' in frame else None,
                               frame['LTHI'] if 'LTHI' in frame else None,
                               frame['RKNE'] if 'RKNE' in frame else None,
                               frame['LKNE'] if 'LKNE' in frame else None,
                               r_hip_jc,
                               l_hip_jc,
                               vsk['RightKneeWidth'],
                               vsk['LeftKneeWidth'])

    knee_jc     = [axis_knee[0][:3, 3], axis_knee[1][:3, 3]]
    r_knee_jc   = knee_jc[0]
    l_knee_jc   = knee_jc[1]
    r_knee_axes = axis_knee[0][:3, :3]
    l_knee_axes = axis_knee[1][:3, :3]

    r_pelvis_knee_angle = calc_angle(hip_axis, r_knee_axes)
    l_pelvis_knee_angle = calc_angle(hip_axis, l_knee_axes)

    rhipx = r_pelvis_knee_angle[0] * -1
    rhipy = r_pelvis_knee_angle[1]
    rhipz = r_pelvis_knee_angle[2] * -1 + 90

    lhipx = l_pelvis_knee_angle[0] * -1
    lhipy = l_pelvis_knee_angle[1] * -1
    lhipz = l_pelvis_knee_angle[2] - 90

    ankle_axis = calc_axis_ankle(frame['RTIB'] if 'RTIB' in frame else None,
                                 frame['LTIB'] if 'LTIB' in frame else None,
                                 frame['RANK'] if 'RANK' in frame else None,
                                 frame['LANK'] if 'LANK' in frame else None,
                                 r_knee_jc,
                                 l_knee_jc,
                                 vsk['RightAnkleWidth'],
                                 vsk['LeftAnkleWidth'],
                                 vsk['RightTibialTorsion'],
                                 vsk['LeftTibialTorsion'])

    ankle_jc     = [ankle_axis[0][:3, 3], ankle_axis[1][:3, 3]]
    r_ankle_jc   = ankle_jc[0]
    l_ankle_jc   = ankle_jc[1]
    r_ankle_axes = ankle_axis[0][:3, :3]
    l_ankle_axes = ankle_axis[1][:3, :3]

    r_knee_ankle_angle = calc_angle(r_knee_axes, r_ankle_axes)
    l_knee_ankle_angle = calc_angle(l_knee_axes, l_ankle_axes)

    rkneex=r_knee_ankle_angle[0]
    rkneey=r_knee_ankle_angle[1]
    rkneez=r_knee_ankle_angle[2] * -1 + 90

    lkneex=l_knee_ankle_angle[0]
    lkneey=l_knee_ankle_angle[1] * -1
    lkneez=l_knee_ankle_angle[2] - 90


    # ANKLE ANGLE
    offset = 0
    foot_axis = calc_axis_foot(frame['RTOE'] if 'RTOE' in frame else None,
                               frame['LTOE'] if 'LTOE' in frame else None,
                               ankle_axis[0],
                               ankle_axis[1],
                               vsk['RightStaticRotOff'],
                               vsk['LeftStaticRotOff'],
                               vsk['RightStaticPlantFlex'],
                               vsk['LeftStaticPlantFlex'])

    foot_jc     = [foot_axis[0][:3, 3], foot_axis[1][:3, 3]]
    r_foot_jc   = foot_jc[0]
    r_foot_axes = foot_axis[0][:3, :3]
    l_foot_jc   = foot_jc[1]
    l_foot_axes = foot_axis[1][:3, :3]

    rhee = frame['RHEE']
    lhee = frame['LHEE']

    r_ankle_foot_angle = calc_angle(r_ankle_axes,r_foot_axes)
    l_ankle_foot_angle = calc_angle(l_ankle_axes,l_foot_axes)

    ranklex = r_ankle_foot_angle[0] * (-1) - 90
    rankley = r_ankle_foot_angle[2] * (-1) + 90
    ranklez = r_ankle_foot_angle[1]

    lanklex = l_ankle_foot_angle[0] * (-1) - 90
    lankley = l_ankle_foot_angle[2] - 90
    lanklez = l_ankle_foot_angle[1] * (-1)

    # ABSOLUTE FOOT ANGLE
    r_global_foot_angle = calc_angle(global_axes, r_foot_axes)
    l_global_foot_angle = calc_angle(global_axes, l_foot_axes)

    rfootx = r_global_foot_angle[0]
    rfooty = r_global_foot_angle[2] - 90
    rfootz = r_global_foot_angle[1]
    lfootx = l_global_foot_angle[0]
    lfooty = (l_global_foot_angle[2] - 90) * -1
    lfootz = l_global_foot_angle[1] * -1

    #First Calculate HEAD

    head_axis = calc_axis_head(frame['LFHD'] if 'LFHD' in frame else None,
                               frame['RFHD'] if 'RFHD' in frame else None,
                               frame['LBHD'] if 'LBHD' in frame else None,
                               frame['RBHD'] if 'RBHD' in frame else None,
                               vsk['HeadOffset'])
    head_jc   = head_axis[:3, 3]
    head_axes = head_axis[:3, :3]

    lfhd = frame['LFHD']
    rfhd = frame['RFHD']
    lbhd = frame['LBHD']
    rbhd = frame['RBHD']

    head_front = np.array((lfhd + rfhd) / 2)
    head_back  = np.array((lbhd + rbhd) / 2)

    #Global_axis_form = [[0,1,0],[-1,0,0],[0,0,1]]
    global_center = [0,0,0]
    global_axis = vsk['GCS']
    global_axes = np.vstack([np.subtract(global_axis[0],global_center),
                             np.subtract(global_axis[1],global_center),
                             np.subtract(global_axis[2],global_center)])

    global_head_angle = calc_angle_head(global_axes, head_axes)
    headx=(global_head_angle[0] * -1)# + 24.8

    if headx < -180:
        headx = headx + 360
    heady = global_head_angle[1] * -1
    headz = global_head_angle[2]#+180
    if headz < -180:
        headz = headz - 360

    # Calculate THORAX
    thorax_axis = calc_axis_thorax(frame['CLAV'] if 'CLAV' in frame else None,
                                   frame['C7'] if 'C7' in frame else None,
                                   frame['STRN'] if 'STRN' in frame else None,
                                   frame['T10'] if 'T10' in frame else None)

    thorax_jc   = thorax_axis[:3, 3]
    thorax_axes = thorax_axis[:3, :3]

    global_axis   = [[0,1,0],[-1,0,0],[0,0,1]]
    global_center = [0,0,0]
    global_axis   = rotmat(x=0,y=0,z=180) #this needs to be fixed for the global rotation

    global_axes = np.vstack([np.subtract(global_axis[0],global_center),
                             np.subtract(global_axis[1],global_center),
                             np.subtract(global_axis[2],global_center)])


    global_thorax_angle = calc_angle(global_axes,thorax_axes)

    if global_thorax_angle[0] > 0:
        global_thorax_angle[0] = global_thorax_angle[0] - 180

    elif global_thorax_angle[0] < 0:
        global_thorax_angle[0] = global_thorax_angle[0] + 180

    thox = global_thorax_angle[0]
    thoy = global_thorax_angle[1]
    thoz = global_thorax_angle[2] + 90

    # Calculate NECK

    head_thorax_angle = calc_angle_head(head_axes,thorax_axes)

    neckx = (head_thorax_angle[0] - 180) * -1 # - 24.9
    necky = head_thorax_angle[1]
    neckz = head_thorax_angle[2] * -1

    c7   = frame['C7']#quick fix to calculate CoM
    clav = frame['CLAV']
    strn = frame['STRN']
    t10  = frame['T10']

    # Calculate SPINE

    pel_tho_angle = calc_angle_spine(pelvis_axes,thorax_axes)

    spix = pel_tho_angle[0]
    spiy = pel_tho_angle[2] * -1
    spiz = pel_tho_angle[1]

    # Calculate SHOULDER

    wand = calc_marker_wand(frame['RSHO'] if 'RSHO' in frame else None,
                            frame['LSHO'] if 'LSHO' in frame else None,
                            thorax_axis)

    shoulder_jc = calc_joint_center_shoulder(frame['RSHO'] if 'RSHO' in frame else None,
                                             frame['LSHO'] if 'LSHO' in frame else None,
                                             thorax_axis,
                                             wand[0],
                                             wand[1],
                                             vsk['RightShoulderOffset'],
                                             vsk['LeftShoulderOffset'])

    shoulder_axis = calc_axis_shoulder(thorax_axis,
                                       shoulder_jc[0],
                                       shoulder_jc[1],
                                       wand[0],
                                       wand[1])

    r_shoulder_jc   = shoulder_axis[0][:3, 3]
    l_shoulder_jc   = shoulder_axis[1][:3, 3]
    r_shoulder_axes = shoulder_axis[0][:3, :3]
    l_shoulder_axes = shoulder_axis[1][:3, :3]

    elbow_axis = calc_axis_elbow(frame['RELB'] if 'RELB' in frame else None,
                                 frame['LELB'] if 'LELB' in frame else None,
                                 frame['RWRA'] if 'RWRA' in frame else None,
                                 frame['RWRB'] if 'RWRB' in frame else None,
                                 frame['LWRA'] if 'LWRA' in frame else None,
                                 frame['LWRB'] if 'LWRB' in frame else None,
                                 shoulder_axis[0],
                                 shoulder_axis[1],
                                 vsk['RightElbowWidth'],
                                 vsk['LeftElbowWidth'],
                                 vsk['RightWristWidth'],
                                 vsk['LeftWristWidth'],
                                 7.0)

    r_elbow_jc   = elbow_axis[0][:3, 3]
    l_elbow_jc   = elbow_axis[1][:3, 3]
    r_elbow_axes = elbow_axis[0][:3, :3]
    l_elbow_axes = elbow_axis[1][:3, :3]

    r_thorax_shoulder_angle, l_thorax_shoulder_angle = calc_angle_shoulder(thorax_axis,
                                                                           elbow_axis[0],
                                                                           elbow_axis[1])

    if r_thorax_shoulder_angle[2] < 0:
        r_thorax_shoulder_angle[2] = r_thorax_shoulder_angle[2] + 180
    elif r_thorax_shoulder_angle[2] > 0:
        r_thorax_shoulder_angle[2] = r_thorax_shoulder_angle[2] - 180

    if r_thorax_shoulder_angle[1] > 0:
        r_thorax_shoulder_angle[1] = r_thorax_shoulder_angle[1] - 180
    elif r_thorax_shoulder_angle[1] < 0:
        r_thorax_shoulder_angle[1] = r_thorax_shoulder_angle[1] * -1 - 180

    if l_thorax_shoulder_angle[1] < 0:
        l_thorax_shoulder_angle[1] = l_thorax_shoulder_angle[1]  + 180
    elif l_thorax_shoulder_angle[1] > 0:
        l_thorax_shoulder_angle[1] = l_thorax_shoulder_angle[1] - 180

    rshox = r_thorax_shoulder_angle[0] * -1
    rshoy = r_thorax_shoulder_angle[1] * -1
    rshoz = r_thorax_shoulder_angle[2]
    lshox = l_thorax_shoulder_angle[0] * -1
    lshoy = l_thorax_shoulder_angle[1]
    lshoz = (l_thorax_shoulder_angle[2] - 180) * -1

    if lshoz > 180:
        lshoz = lshoz - 360

    # Calculate ELBOW

    wrist_axis = calc_axis_wrist(elbow_axis[0],
                                 elbow_axis[1],
                                 elbow_axis[2],
                                 elbow_axis[3])

    r_wrist_jc   = wrist_axis[0][:3, 3]
    l_wrist_jc   = wrist_axis[1][:3, 3]
    r_wrist_axes = wrist_axis[0][:3, :3]
    l_wrist_axes = wrist_axis[1][:3, :3]

    r_humerus_radius_angle = calc_angle(r_elbow_axes, r_wrist_axes)
    l_humerus_radius_angle = calc_angle(l_elbow_axes, l_wrist_axes)

    relbx = r_humerus_radius_angle[0]
    relby = r_humerus_radius_angle[1]
    relbz = r_humerus_radius_angle[2] - 90.0
    lelbx = l_humerus_radius_angle[0]
    lelby = l_humerus_radius_angle[1]
    lelbz = l_humerus_radius_angle[2] - 90.0

    # Calculate WRIST
    hand_jc = calc_axis_hand(frame['RWRA'] if 'RWRA' in frame else None,
                             frame['RWRB'] if 'RWRB' in frame else None,
                             frame['LWRA'] if 'LWRA' in frame else None,
                             frame['LWRB'] if 'LWRB' in frame else None,
                             frame['RFIN'] if 'RFIN' in frame else None,
                             frame['LFIN'] if 'LFIN' in frame else None,
                             wrist_axis[0],
                             wrist_axis[1],
                             vsk['RightHandThickness'],
                             vsk['LeftHandThickness'])

    r_hand_jc   = hand_jc[0][:3, 3]
    l_hand_jc   = hand_jc[1][:3, 3]
    r_hand_axes = hand_jc[0][:3, :3]
    l_hand_axes = hand_jc[1][:3, :3]

    r_radius_hand_angle = calc_angle(r_wrist_axes, r_hand_axes)
    l_radius_hand_angle = calc_angle(l_wrist_axes, l_hand_axes)

    rwrtx = r_radius_hand_angle[0]
    rwrty = r_radius_hand_angle[1]
    rwrtz = r_radius_hand_angle[2] * -1 + 90
    lwrtx = l_radius_hand_angle[0]
    lwrty = l_radius_hand_angle[1] * -1
    lwrtz = l_radius_hand_angle[2] - 90

    if lwrtz < -180:
        lwrtz = lwrtz + 360

    # make each axis as same format to store

    # Pelvis
    # origin
    pel_origin = pelvis_jc
    pel_ox, pel_oy, pel_oz = pel_origin
    
	# xaxis
    pel_x_axis = pelvis_axes[0] + pelvis_jc
    pel_xx, pel_xy, pel_xz = pel_x_axis
    
	# yaxis
    pel_y_axis = pelvis_axes[1] + pelvis_jc
    pel_yx, pel_yy, pel_yz = pel_y_axis
    
	# zaxis
    pel_z_axis = pelvis_axes[2] + pelvis_jc
    pel_zx, pel_zy, pel_zz = pel_z_axis

    # Hip
    # origin
    hip_origin = hip_jc
    hip_ox, hip_oy, hip_oz = hip_origin
    
	# xaxis
    hip_x_axis = hip_axes[0] + hip_jc
    hip_xx, hip_xy, hip_xz = hip_x_axis
    
	# yaxis
    hip_y_axis = hip_axes[1] + hip_jc
    hip_yx, hip_yy, hip_yz = hip_y_axis
    
	# zaxis
    hip_z_axis = hip_axes[2] + hip_jc
    hip_zx, hip_zy, hip_zz = hip_z_axis

    # R KNEE
    # origin
    rknee_origin = r_knee_jc
    rknee_ox, rknee_oy, rknee_oz = rknee_origin

    
	# xaxis
    rknee_x_axis = r_knee_axes[0] + r_knee_jc
    rknee_xx, rknee_xy, rknee_xz = rknee_x_axis
    
	# yaxis
    rknee_y_axis = r_knee_axes[1] + r_knee_jc
    rknee_yx, rknee_yy, rknee_yz = rknee_y_axis
    
	# zaxis
    rknee_z_axis = r_knee_axes[2] + r_knee_jc
    rknee_zx, rknee_zy, rknee_zz = rknee_z_axis

    # L KNEE
    # origin
    lknee_origin = l_knee_jc
    lknee_ox, lknee_oy, lknee_oz = lknee_origin
    
	# xaxis
    lknee_x_axis = l_knee_axes[0] + l_knee_jc
    lknee_xx, lknee_xy, lknee_xz = lknee_x_axis
    
	# yaxis
    lknee_y_axis = l_knee_axes[1] + l_knee_jc
    lknee_yx, lknee_yy, lknee_yz = lknee_y_axis
    
	# zaxis
    lknee_z_axis = l_knee_axes[2] + l_knee_jc
    lknee_zx, lknee_zy, lknee_zz = lknee_z_axis

    # R ANKLE
    # origin
    rank_origin = r_ankle_jc
    rank_ox, rank_oy, rank_oz = rank_origin
    
	# xaxis
    rank_x_axis = r_ankle_axes[0] + r_ankle_jc
    rank_xx, rank_xy, rank_xz = rank_x_axis
    
	# yaxis
    rank_y_axis = r_ankle_axes[1] + r_ankle_jc
    rank_yx, rank_yy, rank_yz = rank_y_axis
    
	# zaxis
    rank_z_axis = r_ankle_axes[2] + r_ankle_jc
    rank_zx, rank_zy, rank_zz = rank_z_axis

    # L ANKLE
    # origin
    lank_origin = l_ankle_jc
    lank_ox, lank_oy, lank_oz = lank_origin
    
	# xaxis
    lank_x_axis = l_ankle_axes[0] + l_ankle_jc
    lank_xx, lank_xy, lank_xz = lank_x_axis
    
	# yaxis
    lank_y_axis = l_ankle_axes[1] + l_ankle_jc
    lank_yx, lank_yy, lank_yz = lank_y_axis
    
	# zaxis
    lank_z_axis = l_ankle_axes[2] + l_ankle_jc
    lank_zx, lank_zy, lank_zz = lank_z_axis

    # R FOOT
    # origin
    rfoot_origin = r_foot_jc
    rfoot_ox, rfoot_oy, rfoot_oz = rfoot_origin
    
	# xaxis
    rfoot_x_axis = r_foot_axes[0] + r_foot_jc
    rfoot_xx, rfoot_xy, rfoot_xz = rfoot_x_axis
    
	# yaxis
    rfoot_y_axis = r_foot_axes[1] + r_foot_jc
    rfoot_yx, rfoot_yy, rfoot_yz = rfoot_y_axis
    
	# zaxis
    rfoot_z_axis = r_foot_axes[2] + r_foot_jc
    rfoot_zx, rfoot_zy, rfoot_zz = rfoot_z_axis

    # L FOOT
    # origin
    lfoot_origin = l_foot_jc
    lfoot_ox, lfoot_oy, lfoot_oz = lfoot_origin
    
	# xaxis
    lfoot_x_axis = l_foot_axes[0] + l_foot_jc
    lfoot_xx, lfoot_xy, lfoot_xz = lfoot_x_axis
    
	# yaxis
    lfoot_y_axis = l_foot_axes[1] + l_foot_jc
    lfoot_yx, lfoot_yy, lfoot_yz = lfoot_y_axis
    
	# zaxis
    lfoot_z_axis = l_foot_axes[2] + l_foot_jc
    lfoot_zx, lfoot_zy, lfoot_zz = lfoot_z_axis

    # HEAD
    # origin
    head_origin = head_jc
    head_ox, head_oy, head_oz = head_origin
    
	# xaxis
    head_x_axis = head_axes[0] + head_jc
    head_xx, head_xy, head_xz = head_x_axis
    
	# yaxis
    head_y_axis = head_axes[1] + head_jc
    head_yx, head_yy, head_yz = head_y_axis
    
	# zaxis
    head_z_axis = head_axes[2] + head_jc
    head_zx, head_zy, head_zz = head_z_axis

    # THORAX
    # origin
    tho_origin = thorax_jc
    tho_ox, tho_oy, tho_oz = tho_origin
    
	# xaxis
    tho_x_axis = thorax_axes[0] + thorax_jc
    tho_xx, tho_xy, tho_xz = tho_x_axis
    
	# yaxis
    tho_y_axis = thorax_axes[1] + thorax_jc
    tho_yx, tho_yy, tho_yz = tho_y_axis
    
	# zaxis
    tho_z_axis = thorax_axes[2] + thorax_jc
    tho_zx, tho_zy, tho_zz = tho_z_axis

    # R CLAVICLE
    # origin
    rclav_origin = r_shoulder_jc
    rclav_ox, rclav_oy, rclav_oz = rclav_origin
    
	# xaxis
    rclav_x_axis = r_shoulder_axes[0] + r_shoulder_jc
    rclav_xx, rclav_xy, rclav_xz = rclav_x_axis
    
	# yaxis
    rclav_y_axis = r_shoulder_axes[1] + r_shoulder_jc
    rclav_yx, rclav_yy, rclav_yz = rclav_y_axis
    
	# zaxis
    rclav_z_axis = r_shoulder_axes[2] + r_shoulder_jc
    rclav_zx, rclav_zy, rclav_zz = rclav_z_axis

    # L CLAVICLE
    # origin
    lclav_origin = l_shoulder_jc
    lclav_ox, lclav_oy, lclav_oz = lclav_origin
    
	# xaxis
    lclav_x_axis = l_shoulder_axes[0] + l_shoulder_jc
    lclav_xx, lclav_xy, lclav_xz = lclav_x_axis
    
	# yaxis
    lclav_y_axis = l_shoulder_axes[1] + l_shoulder_jc
    lclav_yx, lclav_yy, lclav_yz = lclav_y_axis
    
	# zaxis
    lclav_z_axis = l_shoulder_axes[2] + l_shoulder_jc
    lclav_zx, lclav_zy, lclav_zz = lclav_z_axis

    # R HUMERUS
    # origin
    rhum_origin = r_elbow_jc
    rhum_ox, rhum_oy, rhum_oz = rhum_origin
    
	# xaxis
    rhum_x_axis = r_elbow_axes[0] + r_elbow_jc
    rhum_xx, rhum_xy, rhum_xz = rhum_x_axis
    
	# yaxis
    rhum_y_axis = r_elbow_axes[1] + r_elbow_jc
    rhum_yx, rhum_yy, rhum_yz = rhum_y_axis
    
	# zaxis
    rhum_z_axis = r_elbow_axes[2] + r_elbow_jc
    rhum_zx, rhum_zy, rhum_zz = rhum_z_axis

    # L HUMERUS
    # origin
    lhum_origin = l_elbow_jc
    lhum_ox, lhum_oy, lhum_oz = lhum_origin
    
	# xaxis
    lhum_x_axis = l_elbow_axes[0] + l_elbow_jc
    lhum_xx, lhum_xy, lhum_xz = lhum_x_axis
    
	# yaxis
    lhum_y_axis = l_elbow_axes[1] + l_elbow_jc
    lhum_yx, lhum_yy, lhum_yz = lhum_y_axis
    
	# zaxis
    lhum_z_axis = l_elbow_axes[2] + l_elbow_jc
    lhum_zx, lhum_zy, lhum_zz = lhum_z_axis

    # R RADIUS
    # origin
    rrad_origin = r_wrist_jc
    rrad_ox, rrad_oy, rrad_oz = rrad_origin
    
	# xaxis
    rrad_x_axis = r_wrist_axes[0] + r_wrist_jc
    rrad_xx, rrad_xy, rrad_xz = rrad_x_axis
    
	# yaxis
    rrad_y_axis = r_wrist_axes[1] + r_wrist_jc
    rrad_yx, rrad_yy, rrad_yz = rrad_y_axis
    
	# zaxis
    rrad_z_axis = r_wrist_axes[2] + r_wrist_jc
    rrad_zx, rrad_zy, rrad_zz = rrad_z_axis

    # L RADIUS
    # origin
    lrad_origin = l_wrist_jc
    lrad_ox, lrad_oy, lrad_oz = lrad_origin
    
	# xaxis
    lrad_x_axis = l_wrist_axes[0] + l_wrist_jc
    lrad_xx, lrad_xy, lrad_xz = lrad_x_axis
    
	# yaxis
    lrad_y_axis = l_wrist_axes[1] + l_wrist_jc
    lrad_yx, lrad_yy, lrad_yz = lrad_y_axis
    
	# zaxis
    lrad_z_axis = l_wrist_axes[2] + l_wrist_jc
    lrad_zx, lrad_zy, lrad_zz = lrad_z_axis

    # R HAND
    # origin
    rhand_origin = r_hand_jc
    rhand_ox, rhand_oy, rhand_oz = rhand_origin
    
	# xaxis
    rhand_x_axis= r_hand_axes[0] + r_hand_jc
    rhand_xx, rhand_xy, rhand_xz = rhand_x_axis
    
	# yaxis
    rhand_y_axis= r_hand_axes[1] + r_hand_jc
    rhand_yx, rhand_yy, rhand_yz = rhand_y_axis
    
	# zaxis
    rhand_z_axis= r_hand_axes[2] + r_hand_jc
    rhand_zx, rhand_zy, rhand_zz = rhand_z_axis

    # L HAND
    # origin
    lhand_origin = l_hand_jc
    lhand_ox, lhand_oy, lhand_oz = lhand_origin
    
	# xaxis
    lhand_x_axis = l_hand_axes[0] + l_hand_jc
    lhand_xx, lhand_xy, lhand_xz = lhand_x_axis
    
	# yaxis
    lhand_y_axis = l_hand_axes[1] + l_hand_jc
    lhand_yx, lhand_yy, lhand_yz = lhand_y_axis
    
	# zaxis
    lhand_z_axis = l_hand_axes[2] + l_hand_jc
    lhand_zx, lhand_zy, lhand_zz = lhand_z_axis
    #-----------------------------------------------------

    #Store everything in an array to send back to results of process

    r=[
    pelx,    pely,    pelz, 
    rhipx,   rhipy,   rhipz, 
    lhipx,   lhipy,   lhipz, 
    rkneex,  rkneey,  rkneez, 
    lkneex,  lkneey,  lkneez, 
    ranklex, rankley, ranklez, 
    lanklex, lankley, lanklez, 
    rfootx,  rfooty,  rfootz, 
    lfootx,  lfooty,  lfootz, 
    headx,   heady,   headz, 
    thox,    thoy,    thoz, 
    neckx,   necky,   neckz, 
    spix,    spiy,    spiz, 
    rshox,   rshoy,   rshoz, 
    lshox,   lshoy,   lshoz, 
    relbx,   relby,   relbz, 
    lelbx,   lelby,   lelbz, 
    rwrtx,   rwrty,   rwrtz, 
    lwrtx,   lwrty,   lwrtz, 
    pel_ox,   pel_oy,   pel_oz,   pel_xx,   pel_xy,   pel_xz,   pel_yx,   pel_yy,   pel_yz,   pel_zx,   pel_zy,   pel_zz, 
    hip_ox,   hip_oy,   hip_oz,   hip_xx,   hip_xy,   hip_xz,   hip_yx,   hip_yy,   hip_yz,   hip_zx,   hip_zy,   hip_zz, 
    rknee_ox, rknee_oy, rknee_oz, rknee_xx, rknee_xy, rknee_xz, rknee_yx, rknee_yy, rknee_yz, rknee_zx, rknee_zy, rknee_zz, 
    lknee_ox, lknee_oy, lknee_oz, lknee_xx, lknee_xy, lknee_xz, lknee_yx, lknee_yy, lknee_yz, lknee_zx, lknee_zy, lknee_zz, 
    rank_ox,  rank_oy,  rank_oz,  rank_xx,  rank_xy,  rank_xz,  rank_yx,  rank_yy,  rank_yz,  rank_zx,  rank_zy,  rank_zz, 
    lank_ox,  lank_oy,  lank_oz,  lank_xx,  lank_xy,  lank_xz,  lank_yx,  lank_yy,  lank_yz,  lank_zx,  lank_zy,  lank_zz, 
    rfoot_ox, rfoot_oy, rfoot_oz, rfoot_xx, rfoot_xy, rfoot_xz, rfoot_yx, rfoot_yy, rfoot_yz, rfoot_zx, rfoot_zy, rfoot_zz, 
    lfoot_ox, lfoot_oy, lfoot_oz, lfoot_xx, lfoot_xy, lfoot_xz, lfoot_yx, lfoot_yy, lfoot_yz, lfoot_zx, lfoot_zy, lfoot_zz, 
    head_ox,  head_oy,  head_oz,  head_xx,  head_xy,  head_xz,  head_yx,  head_yy,  head_yz,  head_zx,  head_zy,  head_zz, 
    tho_ox,   tho_oy,   tho_oz,   tho_xx,   tho_xy,   tho_xz,   tho_yx,   tho_yy,   tho_yz,   tho_zx,   tho_zy,   tho_zz, 
    rclav_ox, rclav_oy, rclav_oz, rclav_xx, rclav_xy, rclav_xz, rclav_yx, rclav_yy, rclav_yz, rclav_zx, rclav_zy, rclav_zz, 
    lclav_ox, lclav_oy, lclav_oz, lclav_xx, lclav_xy, lclav_xz, lclav_yx, lclav_yy, lclav_yz, lclav_zx, lclav_zy, lclav_zz, 
    rhum_ox,  rhum_oy,  rhum_oz,  rhum_xx,  rhum_xy,  rhum_xz,  rhum_yx,  rhum_yy,  rhum_yz,  rhum_zx,  rhum_zy,  rhum_zz, 
    lhum_ox,  lhum_oy,  lhum_oz,  lhum_xx,  lhum_xy,  lhum_xz,  lhum_yx,  lhum_yy,  lhum_yz,  lhum_zx,  lhum_zy,  lhum_zz, 
    rrad_ox,  rrad_oy,  rrad_oz,  rrad_xx,  rrad_xy,  rrad_xz,  rrad_yx,  rrad_yy,  rrad_yz,  rrad_zx,  rrad_zy,  rrad_zz, 
    lrad_ox,  lrad_oy,  lrad_oz,  lrad_xx,  lrad_xy,  lrad_xz,  lrad_yx,  lrad_yy,  lrad_yz,  lrad_zx,  lrad_zy,  lrad_zz, 
    rhand_ox, rhand_oy, rhand_oz, rhand_xx, rhand_xy, rhand_xz, rhand_yx, rhand_yy, rhand_yz, rhand_zx, rhand_zy, rhand_zz, 
    lhand_ox, lhand_oy, lhand_oz, lhand_xx, lhand_xy, lhand_xz, lhand_yx, lhand_yy, lhand_yz, lhand_zx, lhand_zy, lhand_zz
    ]

    r=np.array(r,dtype=np.float64)


    #Put temporary dictionary for joint centers to return for now, then modify later
    jc = {}
    jc['Pelvis_axis'] = pelvis_axis
    jc['Thorax_axis'] = thorax_axis

    jc['Pelvis'] = pelvis_jc
    jc['RHip']   = r_hip_jc
    jc['LHip']   = l_hip_jc
    jc['RKnee']  = r_knee_jc
    jc['LKnee']  = l_knee_jc
    jc['RAnkle'] = r_ankle_jc
    jc['LAnkle'] = l_ankle_jc
    jc['RFoot']  = r_foot_jc
    jc['LFoot']  = l_foot_jc

    jc['RHEE'] = rhee
    jc['LHEE'] = lhee

    jc['C7']   = c7
    jc['CLAV'] = clav
    jc['STRN'] = strn
    jc['T10']  = t10


    jc['Front_Head'] = head_front
    jc['Back_Head']  = head_back

    jc['Head']   = head_jc
    jc['Thorax'] = thorax_jc

    jc['RShoulder'] = r_shoulder_jc
    jc['LShoulder'] = l_shoulder_jc
    jc['RHumerus']  = r_elbow_jc
    jc['LHumerus']  = l_elbow_jc
    jc['RRadius']   = r_wrist_jc
    jc['LRadius']   = l_wrist_jc
    jc['RHand']     = r_hand_jc
    jc['LHand']     = l_hand_jc

    return r,jc