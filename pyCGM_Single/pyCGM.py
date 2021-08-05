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

def calc_pelvis_axis(rasi, lasi, rpsi, lpsi, sacr=None):
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
    >>> from .pyCGM import calc_pelvis_axis
    >>> rasi = np.array([ 395.36,  428.09, 1036.82])
    >>> lasi = np.array([ 183.18,  422.78, 1033.07])
    >>> rpsi = np.array([ 341.41,  246.72, 1055.99])
    >>> lpsi = np.array([ 255.79,  241.42, 1057.30])
    >>> [arr.round(2) for arr in calc_pelvis_axis(rasi, lasi, rpsi, lpsi, None)] # doctest: +NORMALIZE_WHITESPACE
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

def calc_hip_joint_center(pelvis, subject):
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
    >>> from .pyCGM import calc_hip_joint_center
    >>> vsk = {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.51,
    ...        'L_AsisToTrocanterMeasure': 72.51, 'InterAsisDistance': 215.90}
    >>> pelvis_axis = np.array([
    ...     [0.14, 0.98, -0.11, 251.60],
    ...     [-0.99, 0.13, -0.02, 391.74],
    ...     [0, 0.1, 0.99, 1032.89],
    ...     [0, 0, 0, 1]
    ... ])
    >>> np.around(calc_hip_joint_center(pelvis_axis,vsk), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[307.36, 323.83, 938.72],
           [181.71, 340.33, 936.18]])
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

    hip_jc = np.array([right_hip_jc, left_hip_jc])

    return hip_jc


def calc_hip_axis(r_hip_jc, l_hip_jc, pelvis_axis):
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
    >>> from .pyCGM import calc_hip_axis
    >>> r_hip_jc = [182.57, 339.43, 935.52]
    >>> l_hip_jc = [308.38, 322.80, 937.98]
    >>> pelvis_axis = np.array([
    ...     [0.14, 0.98, -0.11, 251.60],
    ...     [-0.99, 0.13, -0.02, 391.74],
    ...     [0, 0.1, 0.99, 1032.89],
    ...     [0, 0, 0, 1]
    ... ])
    >>> [np.around(arr, 2) for arr in calc_hip_axis(l_hip_jc,r_hip_jc, pelvis_axis)] #doctest: +NORMALIZE_WHITESPACE
    [array([  0.14,   0.98,  -0.11, 245.48]),
    array([ -0.99,   0.13,  -0.02, 331.12]),
    array([  0.  ,   0.1 ,   0.99, 936.75]), 
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

def calc_knee_axis(rthi, lthi, rkne, lkne, r_hip_jc, l_hip_jc, rkne_width, lkne_width):
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
    >>> [arr.round(2) for arr in calc_knee_axis(rthi, lthi, rkne, lkne, l_hip_jc, r_hip_jc, rkne_width, lkne_width)] #doctest: +NORMALIZE_WHITESPACE
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

    # Determine the position of kneeJointCenter using findJointC function
    R = findJointC(rthi, r_hip_jc, rkne, R_delta)
    L = findJointC(lthi, l_hip_jc, lkne, L_delta)

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


def calc_ankle_axis(rtib, ltib, rank, lank, r_knee_JC, l_knee_JC, rank_width, lank_width, rtib_torsion, ltib_torsion):
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
    >>> [np.around(arr, 2) for arr in calc_ankle_axis(rtib, ltib, rank, lank, knee_JC[0], knee_JC[1], rank_width, lank_width, rtib_torsion, ltib_torsion)] #doctest: +NORMALIZE_WHITESPACE
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

    # Determine the position of ankleJointCenter using findJointC function
    R = findJointC(rtib, r_knee_JC, rank, R_delta)
    L = findJointC(ltib, l_knee_JC, lank, L_delta)

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

def calc_foot_axis(rtoe, ltoe, r_ankle_axis, l_ankle_axis, r_static_rot_off, l_static_rot_off, r_static_plant_flex, l_static_plant_flex):
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
    >>> from .pyCGM import calc_foot_axis
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
    >>> [np.around(arr, 2) for arr in calc_foot_axis(rtoe, ltoe, r_ankle_axis, l_ankle_axis, r_static_rot_off, l_static_rot_off, r_static_plant_flex, l_static_plant_flex)] #doctest: +NORMALIZE_WHITESPACE
    [array([[  0.02,   0.03,   1.  , 442.81],
            [ -0.94,   0.34,   0.01, 381.62],
            [ -0.34,  -0.94,   0.04,  42.66],
            [  0.  ,   0.  ,   0.  ,   1.  ]]), 
     array([[  0.13,   0.07,   0.99,  39.43],
            [ -0.94,  -0.31,   0.14, 382.44],
            [  0.31,  -0.95,   0.02,  41.78],
            [  0.  ,   0.  ,   0.  ,   1.  ]])]

    """

    # REQUIRE JOINT CENTER & AXIS
    # KNEE JOINT CENTER
    # ANKLE JOINT CENTER
    # ANKLE FLEXION AXIS

    ankle_JC_R = r_ankle_axis[:3, 3]
    ankle_JC_L = l_ankle_axis[:3, 3]
    ankle_flexion_R = r_ankle_axis[1, :3] + ankle_JC_R
    ankle_flexion_L = l_ankle_axis[1, :3] + ankle_JC_L

    # Toe axis's origin is marker position of TOE
    R = rtoe
    L = ltoe

    # HERE IS THE INCORRECT AXIS

    # the first setting, the foot axis show foot uncorrected anatomical axis and static_info is None
    ankle_JC_R = [ankle_JC_R[0], ankle_JC_R[1], ankle_JC_R[2]]
    ankle_JC_L = [ankle_JC_L[0], ankle_JC_L[1], ankle_JC_L[2]]

    # Right

    # z axis is from TOE marker to AJC. and normalized it.
    R_axis_z = [ankle_JC_R[0]-rtoe[0],
                ankle_JC_R[1]-rtoe[1], ankle_JC_R[2]-rtoe[2]]
    R_axis_z_div = np.linalg.norm(R_axis_z)
    R_axis_z = [R_axis_z[0]/R_axis_z_div, R_axis_z[1] /
                R_axis_z_div, R_axis_z[2]/R_axis_z_div]

    # bring the flexion axis of ankle axes from AnkleJointCenter function. and normalized it.
    y_flex_R = ankle_flexion_R - ankle_JC_R
    y_flex_R_div = np.linalg.norm(y_flex_R)
    y_flex_R = y_flex_R/y_flex_R_div

    # x axis is calculated as a cross product of z axis and ankle flexion axis.
    R_axis_x = np.cross(y_flex_R, R_axis_z)
    R_axis_x_div = np.linalg.norm(R_axis_x)
    R_axis_x = [R_axis_x[0]/R_axis_x_div, R_axis_x[1] /
                R_axis_x_div, R_axis_x[2]/R_axis_x_div]

    # y axis is then perpendicularly calculated from z axis and x axis. and normalized.
    R_axis_y = np.cross(R_axis_z, R_axis_x)
    R_axis_y_div = np.linalg.norm(R_axis_y)
    R_axis_y = [R_axis_y[0]/R_axis_y_div, R_axis_y[1] /
                R_axis_y_div, R_axis_y[2]/R_axis_y_div]

    R_foot_axis = [R_axis_x, R_axis_y, R_axis_z]

    # Left

    # z axis is from TOE marker to AJC. and normalized it.
    L_axis_z = [ankle_JC_L[0]-ltoe[0],
                ankle_JC_L[1]-ltoe[1], ankle_JC_L[2]-ltoe[2]]
    L_axis_z_div = np.linalg.norm(L_axis_z)
    L_axis_z = [L_axis_z[0]/L_axis_z_div, L_axis_z[1] /
                L_axis_z_div, L_axis_z[2]/L_axis_z_div]

    # bring the flexion axis of ankle axes from AnkleJointCenter function. and normalized it.
    y_flex_L = [ankle_flexion_L[0]-ankle_JC_L[0], ankle_flexion_L[1] -
                ankle_JC_L[1], ankle_flexion_L[2]-ankle_JC_L[2]]
    y_flex_L_div = np.linalg.norm(y_flex_L)
    y_flex_L = [y_flex_L[0]/y_flex_L_div, y_flex_L[1] /
                y_flex_L_div, y_flex_L[2]/y_flex_L_div]

    # x axis is calculated as a cross product of z axis and ankle flexion axis.
    L_axis_x = np.cross(y_flex_L, L_axis_z)
    L_axis_x_div = np.linalg.norm(L_axis_x)
    L_axis_x = [L_axis_x[0]/L_axis_x_div, L_axis_x[1] /
                L_axis_x_div, L_axis_x[2]/L_axis_x_div]

    # y axis is then perpendicularly calculated from z axis and x axis. and normalized.
    L_axis_y = np.cross(L_axis_z, L_axis_x)
    L_axis_y_div = np.linalg.norm(L_axis_y)
    L_axis_y = [L_axis_y[0]/L_axis_y_div, L_axis_y[1] /
                L_axis_y_div, L_axis_y[2]/L_axis_y_div]

    L_foot_axis = [L_axis_x, L_axis_y, L_axis_z]

    foot_axis = [R_foot_axis, L_foot_axis]

    # Apply static offset angle to the incorrect foot axes

    # static offset angle are taken from static_info variable in radians.
    R_alpha = r_static_rot_off
    R_beta = r_static_plant_flex
    #R_gamma = static_info[0][2]
    L_alpha = l_static_rot_off
    L_beta = l_static_plant_flex
    #L_gamma = static_info[1][2]

    R_alpha = np.around(math.degrees(R_alpha), decimals=5)
    R_beta = np.around(math.degrees(R_beta), decimals=5)
    #R_gamma = np.around(math.degrees(static_info[0][2]),decimals=5)
    L_alpha = np.around(math.degrees(L_alpha), decimals=5)
    L_beta = np.around(math.degrees(L_beta), decimals=5)
    #L_gamma = np.around(math.degrees(static_info[1][2]),decimals=5)

    R_alpha = -math.radians(R_alpha)
    R_beta = math.radians(R_beta)
    #R_gamma = 0
    L_alpha = math.radians(L_alpha)
    L_beta = math.radians(L_beta)
    #L_gamma = 0

    R_axis = [[(R_foot_axis[0][0]), (R_foot_axis[0][1]), (R_foot_axis[0][2])],
              [(R_foot_axis[1][0]), (R_foot_axis[1][1]), (R_foot_axis[1][2])],
              [(R_foot_axis[2][0]), (R_foot_axis[2][1]), (R_foot_axis[2][2])]]

    L_axis = [[(L_foot_axis[0][0]), (L_foot_axis[0][1]), (L_foot_axis[0][2])],
              [(L_foot_axis[1][0]), (L_foot_axis[1][1]), (L_foot_axis[1][2])],
              [(L_foot_axis[2][0]), (L_foot_axis[2][1]), (L_foot_axis[2][2])]]

    # rotate incorrect foot axis around y axis first.

    # right
    R_rotmat = [[(math.cos(R_beta)*R_axis[0][0]+math.sin(R_beta)*R_axis[2][0]),
                (math.cos(R_beta)*R_axis[0][1] +
                 math.sin(R_beta)*R_axis[2][1]),
                (math.cos(R_beta)*R_axis[0][2]+math.sin(R_beta)*R_axis[2][2])],
                [R_axis[1][0], R_axis[1][1], R_axis[1][2]],
                [(-1*math.sin(R_beta)*R_axis[0][0]+math.cos(R_beta)*R_axis[2][0]),
                (-1*math.sin(R_beta)*R_axis[0]
                 [1]+math.cos(R_beta)*R_axis[2][1]),
                (-1*math.sin(R_beta)*R_axis[0][2]+math.cos(R_beta)*R_axis[2][2])]]
    # left
    L_rotmat = [[(math.cos(L_beta)*L_axis[0][0]+math.sin(L_beta)*L_axis[2][0]),
                (math.cos(L_beta)*L_axis[0][1] +
                 math.sin(L_beta)*L_axis[2][1]),
                (math.cos(L_beta)*L_axis[0][2]+math.sin(L_beta)*L_axis[2][2])],
                [L_axis[1][0], L_axis[1][1], L_axis[1][2]],
                [(-1*math.sin(L_beta)*L_axis[0][0]+math.cos(L_beta)*L_axis[2][0]),
                (-1*math.sin(L_beta)*L_axis[0]
                 [1]+math.cos(L_beta)*L_axis[2][1]),
                (-1*math.sin(L_beta)*L_axis[0][2]+math.cos(L_beta)*L_axis[2][2])]]

    # rotate incorrect foot axis around x axis next.

    # right
    R_rotmat = np.array([[R_rotmat[0][0], R_rotmat[0][1], R_rotmat[0][2]],
                         [(math.cos(R_alpha)*R_rotmat[1][0]-math.sin(R_alpha)*R_rotmat[2][0]),
                          (math.cos(R_alpha)*R_rotmat[1][1] -
                           math.sin(R_alpha)*R_rotmat[2][1]),
                          (math.cos(R_alpha)*R_rotmat[1][2]-math.sin(R_alpha)*R_rotmat[2][2])],
                         [(math.sin(R_alpha)*R_rotmat[1][0]+math.cos(R_alpha)*R_rotmat[2][0]),
                          (math.sin(R_alpha)*R_rotmat[1][1] +
                             math.cos(R_alpha)*R_rotmat[2][1]),
                          (math.sin(R_alpha)*R_rotmat[1][2]+math.cos(R_alpha)*R_rotmat[2][2])]])

    # left
    L_rotmat = np.asarray([[L_rotmat[0][0], L_rotmat[0][1], L_rotmat[0][2]],
                           [(math.cos(L_alpha)*L_rotmat[1][0]-math.sin(L_alpha)*L_rotmat[2][0]),
                            (math.cos(L_alpha)*L_rotmat[1][1] -
                             math.sin(L_alpha)*L_rotmat[2][1]),
                            (math.cos(L_alpha)*L_rotmat[1][2]-math.sin(L_alpha)*L_rotmat[2][2])],
                           [(math.sin(L_alpha)*L_rotmat[1][0]+math.cos(L_alpha)*L_rotmat[2][0]),
                            (math.sin(L_alpha)*L_rotmat[1][1] +
                               math.cos(L_alpha)*L_rotmat[2][1]),
                            (math.sin(L_alpha)*L_rotmat[1][2]+math.cos(L_alpha)*L_rotmat[2][2])]])

    # Bring each x,y,z axis from rotation axes
    R_axis_x = R_rotmat[0]
    R_axis_y = R_rotmat[1]
    R_axis_z = R_rotmat[2]
    L_axis_x = L_rotmat[0]
    L_axis_y = L_rotmat[1]
    L_axis_z = L_rotmat[2]

    # Attach each axis to the origin

    r_foot_axis = np.zeros((4, 4))
    r_foot_axis[3, 3] = 1.0
    r_foot_axis[:3, :3] = R_rotmat
    r_foot_axis[:3, 3] = R

    l_foot_axis = np.zeros((4, 4))
    l_foot_axis[3, 3] = 1.0
    l_foot_axis[:3, :3] = L_rotmat
    l_foot_axis[:3, 3] = L

    foot_axis = np.array([r_foot_axis, l_foot_axis])

    return foot_axis


# Upperbody Coordinate System

def calc_head_axis(lfhd, rfhd, lbhd, rbhd, head_offset):
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
    >>> from .pyCGM import calc_head_axis
    >>> head_offset = 0.25
    >>> rfhd = np.array([325.82, 402.55, 1722.49])
    >>> lfhd = np.array([184.55, 409.68, 1721.34])
    >>> rbhd = np.array([304.39, 242.91, 1694.97])
    >>> lbhd = np.array([197.86, 251.28, 1696.90])
    >>> [np.around(arr, 2) for arr in calc_head_axis(lfhd, rfhd, lbhd, rbhd, head_offset)] #doctest: +NORMALIZE_WHITESPACE
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


def calc_thorax_axis(clav, c7, strn, t10):
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
    >>> from .pyCGM import calc_thorax_axis
    >>> c7 = np.array([256.78, 371.28, 1459.70])
    >>> t10 = np.array([228.64, 192.32, 1279.64])
    >>> clav = np.array([256.78, 371.28, 1459.70])
    >>> strn = np.array([251.67, 414.10, 1292.08])
    >>> [np.around(arr, 2) for arr in calc_thorax_axis(clav, c7, strn, t10)] #doctest: +NORMALIZE_WHITESPACE
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

def calc_wand_marker(rsho, lsho, thorax_axis):
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
    >>> from .pyCGM import calc_wand_marker
    >>> rsho = np.array([428.88, 270.55, 1500.73])
    >>> lsho = np.array([68.24, 269.01, 1510.10])
    >>> thorax_axis = np.array([[ 0.07,  0.93, -0.37,  256.27], 
    ...                        [  0.99, -0.1 , -0.06,  364.8 ], 
    ...                        [ -0.09, -0.36, -0.93, 1462.29], 
    ...                        [  0.,    0.,    0.,      1.]])
    >>> [np.around(arr, 2) for arr in calc_wand_marker(rsho, lsho, thorax_axis)] #doctest: +NORMALIZE_WHITESPACE
    [array([ 256.78,  365.61, 1462.  ]), 
     array([ 255.79,  365.67, 1462.16])]
    """

    thorax_origin = thorax_axis[:3, 3]

    axis_x_vec = thorax_axis[0, :3] - thorax_origin
    axis_x_vec /= np.linalg.norm(axis_x_vec)

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


def calc_shoulder_joint_center(rsho, lsho, thorax_axis, r_wand, l_wand, r_sho_off, l_sho_off):
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
    >>> from .pyCGM import calc_shoulder_joint_center
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
    >>> [np.around(arr, 2) for arr in calc_shoulder_joint_center(rsho, lsho, thorax_axis, r_wand, l_wand, r_sho_off, l_sho_off)] #doctest: +NORMALIZE_WHITESPACE
    [array([[   1.  ,    0.  ,    0.  ,  419.62],
            [   0.  ,    1.  ,    0.  ,  293.35],
            [   0.  ,    0.  ,    1.  , 1540.77],
            [   0.  ,    0.  ,    0.  ,    1.  ]]),
     array([[   1.  ,    0.  ,    0.  ,   79.26],
            [   0.  ,    1.  ,    0.  ,  290.54],
            [   0.  ,    0.  ,    1.  , 1550.4 ],
            [   0.  ,    0.  ,    0.  ,    1.  ]])]
    """

    thorax_origin = thorax_axis[:3, 3]

    # Get Subject Measurement Values
    mm = 7.0
    R_delta = (r_sho_off + mm)
    L_delta = (l_sho_off + mm)

    # REQUIRED MARKERS:
    # RSHO
    # LSHO

    R_Sho_JC = findJointC(r_wand, thorax_origin, rsho, R_delta)
    L_Sho_JC = findJointC(l_wand, thorax_origin, lsho, L_delta)

    r_sho_jc = np.identity(4)
    r_sho_jc[:3, 3] = R_Sho_JC

    l_sho_jc = np.identity(4)
    l_sho_jc[:3, 3] = L_Sho_JC

    shoulder_JC = np.array([r_sho_jc, l_sho_jc])

    return shoulder_JC

def calc_shoulder_axis(thorax_axis, r_sho_jc, l_sho_jc, r_wand, l_wand):
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
    >>> from .pyCGM import calc_shoulder_axis
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
    >>> [np.around(arr, 2) for arr in calc_shoulder_axis(thorax, r_sho_jc, l_sho_jc, wand[0], wand[1])] #doctest: +NORMALIZE_WHITESPACE
    [array([[  -0.51,   -0.79,    0.33,  419.62],
            [  -0.2 ,    0.49,    0.85,  293.35],
            [  -0.84,    0.37,   -0.4 , 1540.77],
            [   0.  ,    0.  ,    0.  ,    1.  ]]),
     array([[   0.49,   -0.82,    0.3 ,   79.26],
            [  -0.23,   -0.46,   -0.86,  290.54],
            [   0.84,    0.35,   -0.42, 1550.4 ],
            [   0.  ,    0.  ,    0.  ,    1.  ]])]
    """

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

def calc_elbow_axis(relb, lelb, rwra, rwrb, lwra, lwrb, r_shoulder_jc, l_shoulder_jc, r_elbow_width, l_elbow_width, r_wrist_width, l_wrist_width, mm):
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
        >>> from .pyCGM import calc_elbow_axis
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
        >>> [np.around(arr, 2) for arr in calc_elbow_axis(relb, lelb, rwra, rwrb, lwra, lwrb, shoulder_jc[0], shoulder_jc[1], 74.0, 74.0, 55.0, 55.0, 7.0)] #doctest: +NORMALIZE_WHITESPACE
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

        r_elbow_width *= -1
        r_delta = (r_elbow_width/2.0)-mm
        l_delta = (l_elbow_width/2.0)+mm

        rwri = [(rwra[0]+rwrb[0])/2.0, (rwra[1]+rwrb[1]) /
                2.0, (rwra[2]+rwrb[2])/2.0]
        lwri = [(lwra[0]+lwrb[0])/2.0, (lwra[1]+lwrb[1]) /
                2.0, (lwra[2]+lwrb[2])/2.0]

        rsjc = r_shoulder_jc[:3, 3]
        lsjc = l_shoulder_jc[:3, 3]

        # make the construction vector for finding the elbow joint center
        r_con_1 = np.subtract(rsjc, relb)
        r_con_1_div = np.linalg.norm(r_con_1)
        r_con_1 = [r_con_1[0]/r_con_1_div, r_con_1[1] /
                   r_con_1_div, r_con_1[2]/r_con_1_div]

        r_con_2 = np.subtract(rwri, relb)
        r_con_2_div = np.linalg.norm(r_con_2)
        r_con_2 = [r_con_2[0]/r_con_2_div, r_con_2[1] /
                   r_con_2_div, r_con_2[2]/r_con_2_div]

        r_cons_vec = np.cross(r_con_1, r_con_2)
        r_cons_vec_div = np.linalg.norm(r_cons_vec)
        r_cons_vec = [r_cons_vec[0]/r_cons_vec_div, r_cons_vec[1] /
                      r_cons_vec_div, r_cons_vec[2]/r_cons_vec_div]

        r_cons_vec = [r_cons_vec[0]*500+relb[0], r_cons_vec[1]
                      * 500+relb[1], r_cons_vec[2]*500+relb[2]]

        l_con_1 = np.subtract(lsjc, lelb)
        l_con_1_div = np.linalg.norm(l_con_1)
        l_con_1 = [l_con_1[0]/l_con_1_div, l_con_1[1] /
                   l_con_1_div, l_con_1[2]/l_con_1_div]

        l_con_2 = np.subtract(lwri, lelb)
        l_con_2_div = np.linalg.norm(l_con_2)
        l_con_2 = [l_con_2[0]/l_con_2_div, l_con_2[1] /
                   l_con_2_div, l_con_2[2]/l_con_2_div]

        l_cons_vec = np.cross(l_con_1, l_con_2)
        l_cons_vec_div = np.linalg.norm(l_cons_vec)

        l_cons_vec = [l_cons_vec[0]/l_cons_vec_div, l_cons_vec[1] /
                      l_cons_vec_div, l_cons_vec[2]/l_cons_vec_div]

        l_cons_vec = [l_cons_vec[0]*500+lelb[0], l_cons_vec[1]
                      * 500+lelb[1], l_cons_vec[2]*500+lelb[2]]

        rejc = findJointC(r_cons_vec, rsjc, relb, r_delta)
        lejc = findJointC(l_cons_vec, lsjc, lelb, l_delta)

        # this is radius axis for humerus
        # right
        x_axis = np.subtract(rwra, rwrb)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        z_axis = np.subtract(rejc, rwri)
        z_axis_div = np.linalg.norm(z_axis)
        z_axis = [z_axis[0]/z_axis_div, z_axis[1] /
                  z_axis_div, z_axis[2]/z_axis_div]

        y_axis = np.cross(z_axis, x_axis)
        y_axis_div = np.linalg.norm(y_axis)
        y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
                  y_axis_div, y_axis[2]/y_axis_div]

        x_axis = np.cross(y_axis, z_axis)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        r_radius = [x_axis, y_axis, z_axis]

        # left
        x_axis = np.subtract(lwra, lwrb)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        z_axis = np.subtract(lejc, lwri)
        z_axis_div = np.linalg.norm(z_axis)
        z_axis = [z_axis[0]/z_axis_div, z_axis[1] /
                  z_axis_div, z_axis[2]/z_axis_div]

        y_axis = np.cross(z_axis, x_axis)
        y_axis_div = np.linalg.norm(y_axis)
        y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
                  y_axis_div, y_axis[2]/y_axis_div]

        x_axis = np.cross(y_axis, z_axis)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        l_radius = [x_axis, y_axis, z_axis]

        # calculate wrist joint center for humerus
        r_wrist_width = (r_wrist_width/2.0 + mm)
        l_wrist_width = (l_wrist_width/2.0 + mm)

        rwjc = [rwri[0]+r_wrist_width*r_radius[1][0], rwri[1] +
                r_wrist_width*r_radius[1][1], rwri[2]+r_wrist_width*r_radius[1][2]]
        lwjc = [lwri[0]-l_wrist_width*l_radius[1][0], lwri[1] -
                l_wrist_width*l_radius[1][1], lwri[2]-l_wrist_width*l_radius[1][2]]

        # recombine the humerus axis
        # right
        z_axis = np.subtract(rsjc, rejc)
        z_axis_div = np.linalg.norm(z_axis)
        z_axis = [z_axis[0]/z_axis_div, z_axis[1] /
                  z_axis_div, z_axis[2]/z_axis_div]

        x_axis = np.subtract(rwjc, rejc)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        y_axis = np.cross(x_axis, z_axis)
        y_axis_div = np.linalg.norm(y_axis)
        y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
                  y_axis_div, y_axis[2]/y_axis_div]

        x_axis = np.cross(y_axis, z_axis)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        r_axis = np.zeros((4, 4))
        r_axis[3, 3] = 1.0
        r_axis[0, :3] = x_axis
        r_axis[1, :3] = y_axis
        r_axis[2, :3] = z_axis
        r_axis[:3, 3] = rejc

        # left
        z_axis = np.subtract(lsjc, lejc)
        z_axis_div = np.linalg.norm(z_axis)
        z_axis = [z_axis[0]/z_axis_div, z_axis[1] /
                  z_axis_div, z_axis[2]/z_axis_div]

        x_axis = np.subtract(lwjc, lejc)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        y_axis = np.cross(x_axis, z_axis)
        y_axis_div = np.linalg.norm(y_axis)
        y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
                  y_axis_div, y_axis[2]/y_axis_div]

        x_axis = np.cross(y_axis, z_axis)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

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

def calc_wrist_axis(r_elbow, l_elbow, r_wrist_jc, l_wrist_jc):
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
    >>> from .pyCGM import calc_wrist_axis
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
    >>> [np.around(arr, 2) for arr in calc_wrist_axis(r_elbow, l_elbow, r_wrist_jc, l_wrist_jc)] #doctest: +NORMALIZE_WHITESPACE
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


def calc_hand_axis(rwra, rwrb, lwra, lwrb, rfin, lfin, r_wrist_jc, l_wrist_jc, r_hand_thickness, l_hand_thickness):
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
    >>> from .pyCGM import calc_hand_axis
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
    >>> [np.around(arr, 2) for arr in calc_hand_axis(rwra, rwrb, lwra, lwrb, rfin, lfin, r_wrist_jc, l_wrist_jc, r_hand_thickness, l_hand_thickness)] #doctest: +NORMALIZE_WHITESPACE
    [array([[  0.15,  0.31,  0.94,  859.8 ],
            [ -0.73,  0.68, -0.11,  517.27],
            [ -0.67, -0.67,  0.33, 1051.97],
            [  0.  ,  0.  ,  0.  ,    1.  ]]), 
     array([[ -0.09,  0.27,  0.96, -324.52],
            [ -0.8 , -0.59,  0.1 ,  551.89],
            [  0.6 , -0.76,  0.27, 1068.02],
            [  0.  ,  0.  ,  0.  ,    1.  ]])]
    """

    rwri = [(rwra[0]+rwrb[0])/2.0, (rwra[1]+rwrb[1]) /
            2.0, (rwra[2]+rwrb[2])/2.0]
    lwri = [(lwra[0]+lwrb[0])/2.0, (lwra[1]+lwrb[1]) /
            2.0, (lwra[2]+lwrb[2])/2.0]

    rwjc = r_wrist_jc[:3, 3]
    lwjc = l_wrist_jc[:3, 3]

    mm = 7.0

    r_delta = (r_hand_thickness/2.0 + mm)
    l_delta = (l_hand_thickness/2.0 + mm)

    lhnd = findJointC(lwri, lwjc, lfin, l_delta)
    rhnd = findJointC(rwri, rwjc, rfin, r_delta)

    # Left
    z_axis = [lwjc[0]-lhnd[0], lwjc[1]-lhnd[1], lwjc[2]-lhnd[2]]
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div, z_axis[1] /
              z_axis_div, z_axis[2]/z_axis_div]

    y_axis = [lwri[0]-lwra[0], lwri[1]-lwra[1], lwri[2]-lwra[2]]
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
              y_axis_div, y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis, z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
              x_axis_div, x_axis[2]/x_axis_div]

    y_axis = np.cross(z_axis, x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
              y_axis_div, y_axis[2]/y_axis_div]

    l_axis = np.zeros((4, 4))
    l_axis[3, 3] = 1.0
    l_axis[0, :3] = x_axis
    l_axis[1, :3] = y_axis
    l_axis[2, :3] = z_axis
    l_axis[:3, 3] = lhnd

    # Right
    z_axis = [rwjc[0]-rhnd[0], rwjc[1]-rhnd[1], rwjc[2]-rhnd[2]]
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div, z_axis[1] /
              z_axis_div, z_axis[2]/z_axis_div]

    y_axis = [rwra[0]-rwri[0], rwra[1]-rwri[1], rwra[2]-rwri[2]]
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
              y_axis_div, y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis, z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
              x_axis_div, x_axis[2]/x_axis_div]

    y_axis = np.cross(z_axis, x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
              y_axis_div, y_axis[2]/y_axis_div]

    r_axis = np.zeros((4, 4))
    r_axis[3, 3] = 1.0
    r_axis[0, :3] = x_axis
    r_axis[1, :3] = y_axis
    r_axis[2, :3] = z_axis
    r_axis[:3, 3] = rhnd

    return np.asarray([r_axis, l_axis])


def findJointC(a, b, c, delta):
    """Calculate the Joint Center.

    This function is based on physical markers, a,b,c and joint center which
    will be calulcated in this function are all in the same plane.

    Parameters
    ----------
    a,b,c : list
        Three markers x,y,z position of a, b, c.
    delta : float
        The length from marker to joint center, retrieved from subject measurement file.

    Returns
    -------
    mr : array
        Returns the Joint C x, y, z positions in a 1x3 array.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import findJointC
    >>> a = [468.14, 325.09, 673.12]
    >>> b = [355.90, 365.38, 940.69]
    >>> c = [452.35, 329.06, 524.77]
    >>> delta = 59.5
    >>> findJointC(a,b,c,delta).round(2)
    array([396.25, 347.92, 518.63])
    """
    # make the two vector using 3 markers, which is on the same plane.
    v1 = (a[0]-c[0],a[1]-c[1],a[2]-c[2])
    v2 = (b[0]-c[0],b[1]-c[1],b[2]-c[2])

    # v3 is cross vector of v1, v2
    # and then it normalized.
    # v3 = cross(v1,v2)
    v3 = [v1[1]*v2[2] - v1[2]*v2[1],v1[2]*v2[0] - v1[0]*v2[2],v1[0]*v2[1] - v1[1]*v2[0]]
    v3_div = norm2d(v3)
    v3 = [v3[0]/v3_div,v3[1]/v3_div,v3[2]/v3_div]

    m = [(b[0]+c[0])/2.0,(b[1]+c[1])/2.0,(b[2]+c[2])/2.0]
    length = np.subtract(b,m)
    length = norm2d(length)

    theta = math.acos(delta/norm2d(v2))

    cs = math.cos(theta*2)
    sn = math.sin(theta*2)

    ux = v3[0]
    uy = v3[1]
    uz = v3[2]

    # this rotation matrix is called Rodriques' rotation formula.
    # In order to make a plane, at least 3 number of markers is required which means three physical markers on the segment can make a plane.
    # then the orthogonal vector of the plane will be rotating axis.
    # joint center is determined by rotating the one vector of plane around rotating axis.

    rot = np.matrix([[cs+ux**2.0*(1.0-cs),ux*uy*(1.0-cs)-uz*sn,ux*uz*(1.0-cs)+uy*sn],
                    [uy*ux*(1.0-cs)+uz*sn,cs+uy**2.0*(1.0-cs),uy*uz*(1.0-cs)-ux*sn],
                    [uz*ux*(1.0-cs)-uy*sn,uz*uy*(1.0-cs)+ux*sn,cs+uz**2.0*(1.0-cs)]])
    r = rot*(np.matrix(v2).transpose())
    r = r* length/np.linalg.norm(r)

    r = [r[0,0],r[1,0],r[2,0]]
    mr = np.array([r[0]+m[0],r[1]+m[1],r[2]+m[2]])

    return mr

def cross(a, b):
    """Cross Product.

    Given vectors a and b, calculate the cross product.

    Parameters
    ----------
    a : list
        First 3D vector.
    b : list
        Second 3D vector.

    Returns
    -------
    c : list
        The cross product of vector a and vector b.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import cross
    >>> a = [6.25, 7.91, 18.63]
    >>> b = [3.49, 4.42, 19.23]
    >>> np.around(cross(a, b), 2)
    array([ 6.976e+01, -5.517e+01,  2.000e-02])
    """
    c = [a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]]

    return c

def getPelangle(axisP,axisD):
    """Pelvis angle calculation.

    This function takes in two axes and returns three angles and uses the
    inverse Euler rotation matrix in YXZ order.

    Returns the angles in degrees.

    Parameters
    ----------
    axisP : list
        Shows the unit vector of axisP, the position of the proximal axis.
    axisD : list
        Shows the unit vector of axisD, the position of the distal axis.

    Returns
    -------
    angle : list
        Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import getPelangle
    >>> axisP = [[ 0.04, 0.99, 0.06],
    ...        [ 0.99, -0.04, -0.05],
    ...       [-0.05,  0.07, -0.99]]
    >>> axisD = [[-0.18, -0.98, -0.02],
    ...        [ 0.71, -0.11, -0.69],
    ...        [ 0.67, -0.14, 0.72 ]]
    >>> np.around(getPelangle(axisP,axisD), 2)
    array([-174.82,   39.82,  -10.54])
    """
    # this is the angle calculation which order is Y-X-Z

    # alpha is abdcution angle.
    # beta is flextion angle
    # gamma is rotation angle

    beta = np.arctan2(((axisD[2][0]*axisP[1][0])+(axisD[2][1]*axisP[1][1])+(axisD[2][2]*axisP[1][2])),
                        np.sqrt(pow(axisD[2][0]*axisP[0][0]+axisD[2][1]*axisP[0][1]+axisD[2][2]*axisP[0][2],2)+pow((axisD[2][0]*axisP[2][0]+axisD[2][1]*axisP[2][1]+axisD[2][2]*axisP[2][2]),2)))

    alpha = np.arctan2(((axisD[2][0]*axisP[0][0])+(axisD[2][1]*axisP[0][1])+(axisD[2][2]*axisP[0][2])),((axisD[2][0]*axisP[2][0])+(axisD[2][1]*axisP[2][1])+(axisD[2][2]*axisP[2][2])))
    gamma = np.arctan2(((axisD[0][0]*axisP[1][0])+(axisD[0][1]*axisP[1][1])+(axisD[0][2]*axisP[1][2])),((axisD[1][0]*axisP[1][0])+(axisD[1][1]*axisP[1][1])+(axisD[1][2]*axisP[1][2])))

    alpha = 180.0 * alpha/ pi
    beta = 180.0 * beta/ pi
    gamma = 180.0 * gamma/ pi
    angle = [alpha, beta, gamma]

    return angle

def getHeadangle(axisP,axisD):
    """Head angle calculation function.

    This function takes in two axes and returns three angles and uses the
    inverse Euler rotation matrix in YXZ order.

    Returns the angles in degrees.

    Parameters
    ----------
    axisP : list
        Shows the unit vector of axisP, the position of the proximal axis.
    axisD : list
        Shows the unit vector of axisD, the position of the distal axis.

    Returns
    -------
    angle : list
        Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import getHeadangle
    >>> axisP = [[ 0.04, 0.99, 0.06],
    ...        [ 0.99, -0.04, -0.05],
    ...       [-0.05,  0.07, -0.99]]
    >>> axisD = [[-0.18, -0.98, -0.02],
    ...        [ 0.71, -0.11, -0.69],
    ...        [ 0.67, -0.14, 0.72 ]]
    >>> np.around(getHeadangle(axisP,axisD), 2)
    array([ 185.18,  -39.99, -190.54])
    """
    # this is the angle calculation which order is Y-X-Z

    # alpha is abdcution angle.

    ang=((-1*axisD[2][0]*axisP[1][0])+(-1*axisD[2][1]*axisP[1][1])+(-1*axisD[2][2]*axisP[1][2]))
    alpha = np.nan
    if -1<=ang<=1:
        alpha = np.arcsin(ang)

    # check the abduction angle is in the area between -pi/2 and pi/2
    # beta is flextion angle
    # gamma is rotation angle

    beta = np.arctan2(((axisD[2][0]*axisP[1][0])+(axisD[2][1]*axisP[1][1])+(axisD[2][2]*axisP[1][2])),
                        np.sqrt(pow(axisD[0][0]*axisP[1][0]+axisD[0][1]*axisP[1][1]+axisD[0][2]*axisP[1][2],2)+pow((axisD[1][0]*axisP[1][0]+axisD[1][1]*axisP[1][1]+axisD[1][2]*axisP[1][2]),2)))

    alpha = np.arctan2(-1*((axisD[2][0]*axisP[0][0])+(axisD[2][1]*axisP[0][1])+(axisD[2][2]*axisP[0][2])),((axisD[2][0]*axisP[2][0])+(axisD[2][1]*axisP[2][1])+(axisD[2][2]*axisP[2][2])))
    gamma = np.arctan2(-1*((axisD[0][0]*axisP[1][0])+(axisD[0][1]*axisP[1][1])+(axisD[0][2]*axisP[1][2])),((axisD[1][0]*axisP[1][0])+(axisD[1][1]*axisP[1][1])+(axisD[1][2]*axisP[1][2])))

    alpha = 180.0 * alpha/ pi
    beta = 180.0 * beta/ pi
    gamma = 180.0 * gamma/ pi

    beta = -1*beta

    if alpha <0:
        alpha = alpha *-1

    else:
        if 0<alpha < 180:

            alpha = 180+(180-alpha)

    if gamma > 90.0:
        if gamma >120:
            gamma =  (gamma - 180)*-1
        else:
            gamma = (gamma + 180)*-1

    else:
        if gamma <0:
            gamma = (gamma + 180)*-1
        else:
            gamma = (gamma*-1)-180.0

    angle = [alpha, beta, gamma]

    return angle

def getangle_sho(axisP,axisD):
    """Shoulder angle calculation.

    This function takes in two axes and returns three angles and uses the
    inverse Euler rotation matrix in YXZ order.

    Returns the angles in degrees.

    Parameters
    ----------
    axisP : list
        Shows the unit vector of axisP, the position of the proximal axis.
    axisD : list
        Shows the unit vector of axisD, the position of the distal axis.

    Returns
    -------
    angle : list
        Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import getangle_sho
    >>> axisP = [[ 0.04, 0.99, 0.06],
    ...        [ 0.99, -0.04, -0.05],
    ...       [-0.05,  0.07, -0.99]]
    >>> axisD = [[-0.18, -0.98, -0.02],
    ...        [ 0.71, -0.11, -0.69],
    ...        [ 0.67, -0.14, 0.72 ]]
    >>> np.around(getangle_sho(axisP,axisD), 2)
    array([  -3.93, -140.07,  172.9 ])
    """

    # beta is flexion /extension
    # gamma is adduction / abduction
    # alpha is internal / external rotation

    # this is shoulder angle calculation
    alpha = np.arcsin(((axisD[2][0]*axisP[0][0])+(axisD[2][1]*axisP[0][1])+(axisD[2][2]*axisP[0][2])))
    beta = np.arctan2(-1*((axisD[2][0]*axisP[1][0])+(axisD[2][1]*axisP[1][1])+(axisD[2][2]*axisP[1][2])) , ((axisD[2][0]*axisP[2][0])+(axisD[2][1]*axisP[2][1])+(axisD[2][2]*axisP[2][2])))
    gamma = np.arctan2(-1*((axisD[1][0]*axisP[0][0])+(axisD[1][1]*axisP[0][1])+(axisD[1][2]*axisP[0][2])) , ((axisD[0][0]*axisP[0][0])+(axisD[0][1]*axisP[0][1])+(axisD[0][2]*axisP[0][2])))

    angle = [180.0 * alpha/ pi, 180.0 *beta/ pi, 180.0 * gamma/ pi]

    return angle

def getangle_spi(axisP,axisD):
    """Spine angle calculation.

    This function takes in two axes and returns three angles and uses the
    inverse Euler rotation matrix in YXZ order.

    Returns the angles in degrees.

    Parameters
    ----------
    axisP : list
        Shows the unit vector of axisP, the position of the proximal axis.
    axisD : list
        Shows the unit vector of axisD, the position of the distal axis.

    Returns
    -------
    angle : list
        Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import getangle_spi
    >>> axisP = [[ 0.04,   0.99,  0.06],
    ...        [ 0.99, -0.04, -0.05],
    ...        [-0.05,  0.07, -0.99]]
    >>> axisD = [[-0.18, -0.98,-0.02],
    ...        [ 0.71, -0.11,  -0.69],
    ...        [ 0.67, -0.14,   0.72 ]]
    >>> np.around(getangle_spi(axisP,axisD), 2)
    array([ 2.97,  9.13, 39.78])
    """
    # this angle calculation is for spine angle.

    alpha = np.arcsin(((axisD[1][0]*axisP[2][0])+(axisD[1][1]*axisP[2][1])+(axisD[1][2]*axisP[2][2])))
    gamma = np.arcsin(((-1*axisD[1][0]*axisP[0][0])+(-1*axisD[1][1]*axisP[0][1])+(-1*axisD[1][2]*axisP[0][2])) / np.cos(alpha))
    beta = np.arcsin(((-1*axisD[0][0]*axisP[2][0])+(-1*axisD[0][1]*axisP[2][1])+(-1*axisD[0][2]*axisP[2][2])) / np.cos(alpha))

    angle = [180.0 * beta/ pi, 180.0 *gamma/ pi, 180.0 * alpha/ pi]

    return angle

def getangle(axisP,axisD):
    """Normal angle calculation.

    This function takes in two axes and returns three angles and uses the
    inverse Euler rotation matrix in YXZ order.

    Returns the angles in degrees.

    As we use arc sin we have to care about if the angle is in area between -pi/2 to pi/2

    Parameters
    ----------
    axisP : list
        Shows the unit vector of axisP, the position of the proximal axis.
    axisD : list
        Shows the unit vector of axisD, the position of the distal axis.

    Returns
    -------
    angle : list
        Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import getangle
    >>> axisP = [[ 0.04,   0.99,  0.06],
    ...         [ 0.99, -0.04, -0.05],
    ...         [-0.05,  0.07, -0.99]]
    >>> axisD = [[-0.18, -0.98, -0.02],
    ...         [ 0.71, -0.11,  -0.69],
    ...         [ 0.67, -0.14,   0.72 ]]
    >>> np.around(getangle(axisP,axisD), 2)
    array([-174.82,  -39.26,  100.54])
    """
    # this is the angle calculation which order is Y-X-Z

    # alpha is abdcution angle.

    ang=((-1*axisD[2][0]*axisP[1][0])+(-1*axisD[2][1]*axisP[1][1])+(-1*axisD[2][2]*axisP[1][2]))
    alpha = np.nan
    if -1<=ang<=1:
#       alpha = np.arcsin(ang)
        alpha = np.arcsin(ang)

    # check the abduction angle is in the area between -pi/2 and pi/2
    # beta is flextion angle
    # gamma is rotation angle

    if -1.57079633<alpha<1.57079633:

        beta = np.arctan2(((axisD[2][0]*axisP[0][0])+(axisD[2][1]*axisP[0][1])+(axisD[2][2]*axisP[0][2])) , ((axisD[2][0]*axisP[2][0])+(axisD[2][1]*axisP[2][1])+(axisD[2][2]*axisP[2][2])))
        gamma = np.arctan2(((axisD[1][0]*axisP[1][0])+(axisD[1][1]*axisP[1][1])+(axisD[1][2]*axisP[1][2])) , ((axisD[0][0]*axisP[1][0])+(axisD[0][1]*axisP[1][1])+(axisD[0][2]*axisP[1][2])))

    else:
        beta = np.arctan2(-1*((axisD[2][0]*axisP[0][0])+(axisD[2][1]*axisP[0][1])+(axisD[2][2]*axisP[0][2])) , ((axisD[2][0]*axisP[2][0])+(axisD[2][1]*axisP[2][1])+(axisD[2][2]*axisP[2][2])))
        gamma = np.arctan2(-1*((axisD[1][0]*axisP[1][0])+(axisD[1][1]*axisP[1][1])+(axisD[1][2]*axisP[1][2])) , ((axisD[0][0]*axisP[1][0])+(axisD[0][1]*axisP[1][1])+(axisD[0][2]*axisP[1][2])))

    angle = [180.0 * beta/ pi, 180.0 *alpha/ pi, 180.0 * gamma / pi ]

    return angle

def norm2d(v):
    """2D Vector normalization.

    This function calculates the normalization of a 3-dimensional vector.

    Parameters
    ----------
    v : list
        A 3D vector.

    Returns
    -------
    float
        The normalization of the vector as a float.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import norm2d
    >>> v = [105.14, 101.89, 326.77]
    >>> np.around(norm2d(v), 2)
    358.07
    """
    try:
        return sqrt((v[0]*v[0]+v[1]*v[1]+v[2]*v[2]))
    except:
        return np.nan

def norm3d(v):
    """3D Vector normalization.

    This function calculates the normalization of a 3-dimensional vector.

    Parameters
    ----------
    v : list
        A 3D vector.

    Returns
    -------
    list
        The normalization of the vector returned as a float in an array.

    Examples
    --------
    >>> from .pyCGM import norm3d
    >>> v = [125.44, 143.94, 213.49]
    >>> np.around(norm3d(v), 2)
    286.41
    """
    try:
        return np.asarray(sqrt((v[0]*v[0]+v[1]*v[1]+v[2]*v[2])))
    except:
        return np.nan

def normDiv(v):
    """Normalized divison.

    This function calculates the normalization division of a 3-dimensional vector.

    Parameters
    ----------
    v : list
        A 3D vector.

    Returns
    -------
    array
        The divison normalization of the vector returned as a float in an array.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import normDiv
    >>> v = [1.44, 1.94, 2.49]
    >>> np.around(normDiv(v), 2)
    array([0.12, 0.16, 0.21])
    """
    try:
        vec = sqrt((v[0]*v[0]+v[1]*v[1]+v[2]*v[2]))
        v = [v[0]/vec,v[1]/vec,v[2]/vec]
    except:
        vec = np.nan

    return [v[0]/vec,v[1]/vec,v[2]/vec]

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

def rotmat(x=0,y=0,z=0):
    """Rotation Matrix.

    This function creates and returns a rotation matrix.

    Parameters
    ----------
    x,y,z : float, optional
        Angle, which will be converted to radians, in
        each respective axis to describe the rotations.
        The default is 0 for each unspecified angle.

    Returns
    -------
    Rxyz : list
        The product of the matrix multiplication.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import rotmat
    >>> x = 0.5
    >>> y = 0.3
    >>> z = 0.8
    >>> np.around(rotmat(x,y,z), 2) #doctest: +NORMALIZE_WHITESPACE
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
    x = math.radians(x)
    y = math.radians(y)
    z = math.radians(z)
    Rx = [ [1,0,0],[0,math.cos(x),math.sin(x)*-1],[0,math.sin(x),math.cos(x)] ]
    Ry = [ [math.cos(y),0,math.sin(y)],[0,1,0],[math.sin(y)*-1,0,math.cos(y)] ]
    Rz = [ [math.cos(z),math.sin(z)*-1,0],[math.sin(z),math.cos(z),0],[0,0,1] ]
    Rxy = matrixmult(Rx,Ry)
    Rxyz = matrixmult(Rxy,Rz)

    Ryx = matrixmult(Ry,Rx)
    Ryxz = matrixmult(Ryx,Rz)

    return Rxyz


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

    # THIS IS FOOT PROGRESS ANGLE
    rfoot_prox,rfoot_proy,rfoot_proz,lfoot_prox,lfoot_proy,lfoot_proz = [None]*6

    #First Calculate Pelvis
    axis_pelvis = calc_pelvis_axis(frame['RASI'] if 'RASI' in frame else None,
                                   frame['LASI'] if 'LASI' in frame else None,
                                   frame['RPSI'] if 'RPSI' in frame else None,
                                   frame['LPSI'] if 'LPSI' in frame else None,
                                   frame['SACR'] if 'SACR' in frame else None)

    kin_Pelvis_axis = axis_pelvis

    kin_Pelvis_JC = axis_pelvis[:3, 3] #quick fix for storing JC

    #change to same format
    pelvis_vectors = axis_pelvis[:3, :3]
    pelvis_origin = axis_pelvis[:3, 3]

    #need to update this based on the file
    global_Axis = vsk['GCS']

    #make the array which will be the input of findangle function
    pelvis_Axis_mod = pelvis_vectors

    global_pelvis_angle = getangle(global_Axis,pelvis_Axis_mod)

    pelx=global_pelvis_angle[0]
    pely=global_pelvis_angle[1]
    pelz=global_pelvis_angle[2]

    # and then find hip JC
    hip_JC = calc_hip_joint_center(axis_pelvis, vsk)

    kin_L_Hip_JC = hip_JC[0] #quick fix for storing JC
    kin_R_Hip_JC = hip_JC[1] #quick fix for storing JC

    hip_axis = calc_hip_axis(hip_JC[0],hip_JC[1],axis_pelvis)

    axis_knee = calc_knee_axis(frame['RTHI'] if 'RTHI' in frame else None,
                               frame['LTHI'] if 'LTHI' in frame else None,
                               frame['RKNE'] if 'RKNE' in frame else None,
                               frame['LKNE'] if 'LKNE' in frame else None,
                               hip_JC[0],
                               hip_JC[1],
                               vsk['RightKneeWidth'],
                               vsk['LeftKneeWidth'])


    knee_JC = [axis_knee[0][:3, 3], axis_knee[1][:3, 3]] #quick fix for storing JC

    kin_R_Knee_JC = knee_JC[0]
    kin_L_Knee_JC = knee_JC[1]

    #change to same format
    Hip_center_form = hip_axis[:3, 3]
    Hip_axis_form = hip_axis[:3, :3] + Hip_center_form
    R_Knee_center_form = knee_JC[0]
    R_Knee_axis_form = axis_knee[0][:3, :3] + R_Knee_center_form
    L_Knee_center_form = knee_JC[1]
    L_Knee_axis_form = axis_knee[1][:3, :3] + L_Knee_center_form

    #make the array which will be the input of findangle function
    hip_Axis = np.vstack([np.subtract(Hip_axis_form[0],Hip_center_form),
                          np.subtract(Hip_axis_form[1],Hip_center_form),
                          np.subtract(Hip_axis_form[2],Hip_center_form)])

    R_knee_Axis = np.vstack([np.subtract(R_Knee_axis_form[0],R_Knee_center_form),
                           np.subtract(R_Knee_axis_form[1],R_Knee_center_form),
                           np.subtract(R_Knee_axis_form[2],R_Knee_center_form)])

    L_knee_Axis = np.vstack([np.subtract(L_Knee_axis_form[0],L_Knee_center_form),
                           np.subtract(L_Knee_axis_form[1],L_Knee_center_form),
                           np.subtract(L_Knee_axis_form[2],L_Knee_center_form)])

    R_pelvis_knee_angle = getangle(hip_Axis,R_knee_Axis)
    L_pelvis_knee_angle = getangle(hip_Axis,L_knee_Axis)

    rhipx=R_pelvis_knee_angle[0]*-1
    rhipy=R_pelvis_knee_angle[1]
    rhipz=R_pelvis_knee_angle[2]*-1+90

    lhipx=L_pelvis_knee_angle[0]*-1
    lhipy=L_pelvis_knee_angle[1]*-1
    lhipz=L_pelvis_knee_angle[2]-90

    axis_ankle = calc_ankle_axis(frame['RTIB'] if 'RTIB' in frame else None,
                                 frame['LTIB'] if 'LTIB' in frame else None,
                                 frame['RANK'] if 'RANK' in frame else None,
                                 frame['LANK'] if 'LANK' in frame else None,
                                 R_Knee_center_form,
                                 L_Knee_center_form,
                                 vsk['RightAnkleWidth'],
                                 vsk['LeftAnkleWidth'],
                                 vsk['RightTibialTorsion'],
                                 vsk['LeftTibialTorsion'])

    ankle_JC = [axis_ankle[0][:3, 3], axis_ankle[1][:3, 3]] #quick fix for storing JC
    kin_R_Ankle_JC = ankle_JC[0] 
    kin_L_Ankle_JC = ankle_JC[1]

    #change to same format

    R_Ankle_center_form = ankle_JC[0]
    R_Ankle_axis_form = axis_ankle[0][:3, :3] + R_Ankle_center_form
    L_Ankle_center_form = ankle_JC[1]
    L_Ankle_axis_form = axis_ankle[1][:3, :3] + L_Ankle_center_form


    #make the array which will be the input of findangle function
    # In case of knee axis I mentioned it before as R_knee_Axis and L_knee_Axis
    R_ankle_Axis = np.vstack([np.subtract(R_Ankle_axis_form[0],R_Ankle_center_form),
                              np.subtract(R_Ankle_axis_form[1],R_Ankle_center_form),
                              np.subtract(R_Ankle_axis_form[2],R_Ankle_center_form)])

    L_ankle_Axis = np.vstack([np.subtract(L_Ankle_axis_form[0],L_Ankle_center_form),
                              np.subtract(L_Ankle_axis_form[1],L_Ankle_center_form),
                              np.subtract(L_Ankle_axis_form[2],L_Ankle_center_form)])

    R_knee_ankle_angle = getangle(R_knee_Axis,R_ankle_Axis)
    L_knee_ankle_angle = getangle(L_knee_Axis,L_ankle_Axis)

    rkneex=R_knee_ankle_angle[0]
    rkneey=R_knee_ankle_angle[1]
    rkneez=R_knee_ankle_angle[2]*-1+90


    lkneex=L_knee_ankle_angle[0]
    lkneey=L_knee_ankle_angle[1]*-1
    lkneez=L_knee_ankle_angle[2] - 90


    # ANKLE ANGLE

    offset = 0
    axis_foot = calc_foot_axis(frame['RTOE'] if 'RTOE' in frame else None,
                               frame['LTOE'] if 'LTOE' in frame else None,
                               axis_ankle[0],
                               axis_ankle[1],
                               vsk['RightStaticRotOff'],
                               vsk['LeftStaticRotOff'],
                               vsk['RightStaticPlantFlex'],
                               vsk['LeftStaticPlantFlex'])

    foot_JC = [axis_foot[0][:3, 3], axis_foot[1][:3, 3]] #quick fix for storing JC
    kin_R_Foot_JC = foot_JC[0] 
    kin_L_Foot_JC = foot_JC[1]

    kin_RHEE = frame['RHEE']
    kin_LHEE = frame['LHEE']

    R_Foot_center_form = foot_JC[0]
    R_Foot_axis_form = axis_foot[0][:3, :3] + R_Foot_center_form
    L_Foot_center_form = foot_JC[1]
    L_Foot_axis_form = axis_foot[1][:3, :3] + L_Foot_center_form


    R_foot_Axis = np.vstack([np.subtract(R_Foot_axis_form[0],R_Foot_center_form),
                             np.subtract(R_Foot_axis_form[1],R_Foot_center_form),
                             np.subtract(R_Foot_axis_form[2],R_Foot_center_form)])

    L_foot_Axis = np.vstack([np.subtract(L_Foot_axis_form[0],L_Foot_center_form),
                             np.subtract(L_Foot_axis_form[1],L_Foot_center_form),
                             np.subtract(L_Foot_axis_form[2],L_Foot_center_form)])


    R_ankle_foot_angle = getangle(R_ankle_Axis,R_foot_Axis)
    L_ankle_foot_angle = getangle(L_ankle_Axis,L_foot_Axis)

    ranklex=R_ankle_foot_angle[0]*(-1)-90
    rankley=R_ankle_foot_angle[2]*(-1)+90
    ranklez=R_ankle_foot_angle[1]

    lanklex=L_ankle_foot_angle[0]*(-1)-90
    lankley=L_ankle_foot_angle[2]-90
    lanklez=L_ankle_foot_angle[1]*(-1)

    # ABSOLUTE FOOT ANGLE


    R_global_foot_angle = getangle(global_Axis,R_foot_Axis)
    L_global_foot_angle = getangle(global_Axis,L_foot_Axis)

    rfootx=R_global_foot_angle[0]
    rfooty=R_global_foot_angle[2]-90
    rfootz=R_global_foot_angle[1]
    lfootx=L_global_foot_angle[0]
    lfooty=(L_global_foot_angle[2]-90)*-1
    lfootz=L_global_foot_angle[1]*-1

    #First Calculate HEAD

    axis_head = calc_head_axis(frame['LFHD'] if 'LFHD' in frame else None,
                               frame['RFHD'] if 'RFHD' in frame else None,
                               frame['LBHD'] if 'LBHD' in frame else None,
                               frame['RBHD'] if 'RBHD' in frame else None,
                               vsk['HeadOffset'])

    kin_Head_JC = axis_head[:3, 3] #quick fix for storing JC

    LFHD = frame['LFHD'] #as above
    RFHD = frame['RFHD']
    LBHD = frame['LBHD']
    RBHD = frame['RBHD']

    kin_Head_Front = np.array((LFHD+RFHD)/2)
    kin_Head_Back = np.array((LBHD+RBHD)/2)

    #change to same format
    Head_center_form = axis_head[:3, 3]
    Head_axis_form = axis_head[:3, :3] + Head_center_form
    #Global_axis_form = [[0,1,0],[-1,0,0],[0,0,1]]
    Global_center_form = [0,0,0]

    #***********************************************************
    Global_axis_form = vsk['GCS']
    #Global_axis_form = rotmat(x=0,y=0,z=180) #this is some weird fix to global axis

    #make the array which will be the input of findangle function
    head_Axis_mod = np.vstack([np.subtract(Head_axis_form[0],Head_center_form),
                             np.subtract(Head_axis_form[1],Head_center_form),
                             np.subtract(Head_axis_form[2],Head_center_form)])

    global_Axis = np.vstack([np.subtract(Global_axis_form[0],Global_center_form),
                             np.subtract(Global_axis_form[1],Global_center_form),
                             np.subtract(Global_axis_form[2],Global_center_form)])

    global_head_angle = getHeadangle(global_Axis,head_Axis_mod)

    headx=(global_head_angle[0]*-1)# + 24.8

    if headx <-180:
        headx = headx+360
    heady=global_head_angle[1]*-1
    headz=global_head_angle[2]#+180
    if headz <-180:
        headz = headz-360


    # Calculate THORAX

    thorax_axis = calc_thorax_axis(frame['CLAV'] if 'CLAV' in frame else None,
                                   frame['C7'] if 'C7' in frame else None,
                                   frame['STRN'] if 'STRN' in frame else None,
                                   frame['T10'] if 'T10' in frame else None)

    kin_Thorax_JC = thorax_axis[:3, 3] #quick fix for storing JC
    kin_Thorax_axis = thorax_axis[:3, :3]

    # Change to same format
    Thorax_center_form = thorax_axis[:3, 3]
    Thorax_axis_form = thorax_axis[:3, :3]
    thorax_axis[:3, :3] += thorax_axis[:3, 3]

    Global_axis_form = [[0,1,0],[-1,0,0],[0,0,1]]
    Global_center_form = [0,0,0]
    #*******************************************************
    Global_axis_form = rotmat(x=0,y=0,z=180) #this needs to be fixed for the global rotation

    #make the array which will be the input of findangle function
    thorax_Axis_mod = np.vstack([np.subtract(Thorax_axis_form[0],Thorax_center_form),
                                np.subtract(Thorax_axis_form[1],Thorax_center_form),
                                np.subtract(Thorax_axis_form[2],Thorax_center_form)])

    global_Axis = np.vstack([np.subtract(Global_axis_form[0],Global_center_form),
                             np.subtract(Global_axis_form[1],Global_center_form),
                             np.subtract(Global_axis_form[2],Global_center_form)])


    global_thorax_angle = getangle(global_Axis,thorax_Axis_mod)

    if global_thorax_angle[0] > 0:
        global_thorax_angle[0] = global_thorax_angle[0] - 180

    elif global_thorax_angle[0] < 0:
        global_thorax_angle[0] = global_thorax_angle[0] + 180

    thox=global_thorax_angle[0]
    thoy=global_thorax_angle[1]
    thoz=global_thorax_angle[2]+90

    # Calculate NECK

    head_thorax_angle = getHeadangle(head_Axis_mod,thorax_Axis_mod)

    neckx=(head_thorax_angle[0]-180)*-1# - 24.9
    necky=head_thorax_angle[1]
    neckz=head_thorax_angle[2]*-1

    kin_C7 = frame['C7']#quick fix to calculate CoM
    kin_CLAV = frame['CLAV']
    kin_STRN = frame['STRN']
    kin_T10 = frame['T10']

    # Calculate SPINE

    pel_tho_angle = getangle_spi(pelvis_Axis_mod,thorax_Axis_mod)

    spix=pel_tho_angle[0]
    spiy=pel_tho_angle[2]*-1
    spiz=pel_tho_angle[1]

    # Calculate SHOULDER

    wand = calc_wand_marker(frame['RSHO'] if 'RSHO' in frame else None,
                            frame['LSHO'] if 'LSHO' in frame else None,
                            thorax_axis)

    shoulder_JC = calc_shoulder_joint_center(frame['RSHO'] if 'RSHO' in frame else None,
                                             frame['LSHO'] if 'LSHO' in frame else None,
                                             thorax_axis,
                                             wand[0],
                                             wand[1],
                                             vsk['RightShoulderOffset'],
                                             vsk['LeftShoulderOffset'])


    kin_R_Shoulder_JC = shoulder_JC[0] #quick fix for storing JC
    kin_L_Shoulder_JC = shoulder_JC[1] #quick fix for storing JC

    axis_shoulder = calc_shoulder_axis(thorax_axis,
                                       shoulder_JC[0],
                                       shoulder_JC[1],
                                       wand[0],
                                       wand[1])

    axis_elbow = calc_elbow_axis(frame['RELB'] if 'RELB' in frame else None,
                                 frame['LELB'] if 'LELB' in frame else None,
                                 frame['RWRA'] if 'RWRA' in frame else None,
                                 frame['RWRB'] if 'RWRB' in frame else None,
                                 frame['LWRA'] if 'LWRA' in frame else None,
                                 frame['LWRB'] if 'LWRB' in frame else None,
                                 axis_shoulder[0],
                                 axis_shoulder[1],
                                 vsk['RightElbowWidth'],
                                 vsk['LeftElbowWidth'],
                                 vsk['RightWristWidth'],
                                 vsk['LeftWristWidth'],
                                 7.0)

    kin_R_Humerus_JC = axis_elbow[0][:3, 3] #quick fix for storing JC
    kin_L_Humerus_JC = axis_elbow[1][:3, 3] #quick fix for storing JC

    # Change to same format
    R_Clavicle_center_form = axis_shoulder[0][:3, 3]
    L_Clavicle_center_form = axis_shoulder[1][:3, 3]
    R_Clavicle_axis_form = axis_shoulder[0][:3, :3] + R_Clavicle_center_form
    L_Clavicle_axis_form = axis_shoulder[1][:3, :3] + L_Clavicle_center_form

    # Change to same format
    R_Humerus_center_form = axis_elbow[0][:3, 3]
    L_Humerus_center_form = axis_elbow[1][:3, 3]
    R_Humerus_axis_form = axis_elbow[0][:3, :3] + R_Humerus_center_form
    L_Humerus_axis_form = axis_elbow[1][:3, :3] + L_Humerus_center_form

    # make the array which will be the input of findangle function
    R_humerus_Axis_mod = np.vstack([np.subtract(R_Humerus_axis_form[0],R_Humerus_center_form),
                                   np.subtract(R_Humerus_axis_form[1],R_Humerus_center_form),
                                   np.subtract(R_Humerus_axis_form[2],R_Humerus_center_form)])
    L_humerus_Axis_mod = np.vstack([np.subtract(L_Humerus_axis_form[0],L_Humerus_center_form),
                                    np.subtract(L_Humerus_axis_form[1],L_Humerus_center_form),
                                    np.subtract(L_Humerus_axis_form[2],L_Humerus_center_form)])

    R_thorax_shoulder_angle = getangle_sho(thorax_Axis_mod,R_humerus_Axis_mod)
    L_thorax_shoulder_angle = getangle_sho(thorax_Axis_mod,L_humerus_Axis_mod)

    if R_thorax_shoulder_angle[2] < 0:
        R_thorax_shoulder_angle[2]=R_thorax_shoulder_angle[2]+180
    elif R_thorax_shoulder_angle[2] >0:
        R_thorax_shoulder_angle[2] = R_thorax_shoulder_angle[2]-180

    if R_thorax_shoulder_angle[1] > 0:
        R_thorax_shoulder_angle[1] = R_thorax_shoulder_angle[1]-180
    elif R_thorax_shoulder_angle[1] <0:
        R_thorax_shoulder_angle[1] = R_thorax_shoulder_angle[1]*-1-180

    if L_thorax_shoulder_angle[1] < 0:
        L_thorax_shoulder_angle[1]=L_thorax_shoulder_angle[1]+180
    elif L_thorax_shoulder_angle[1] >0:
        L_thorax_shoulder_angle[1] = L_thorax_shoulder_angle[1]-180



    rshox=R_thorax_shoulder_angle[0]*-1
    rshoy=R_thorax_shoulder_angle[1]*-1
    rshoz=R_thorax_shoulder_angle[2]
    lshox=L_thorax_shoulder_angle[0]*-1
    lshoy=L_thorax_shoulder_angle[1]
    lshoz=(L_thorax_shoulder_angle[2]-180)*-1

    if lshoz >180:
        lshoz = lshoz - 360

    # Calculate ELBOW

    axis_wrist = calc_wrist_axis(axis_elbow[0],
                                 axis_elbow[1],
                                 axis_elbow[2],
                                 axis_elbow[3])

    kin_R_Radius_JC = axis_wrist[0][:3, 3] #quick fix for storing JC
    kin_L_Radius_JC = axis_wrist[1][:3, 3] #quick fix for storing JC


    # Change to same format
    R_Radius_center_form = axis_wrist[0][:3, 3]
    L_Radius_center_form = axis_wrist[1][:3, 3]
    R_Radius_axis_form = axis_wrist[0][:3, :3] + R_Radius_center_form
    L_Radius_axis_form = axis_wrist[1][:3, :3] + L_Radius_center_form

    # make the array which will be the input of findangle function
    R_radius_Axis_mod = np.vstack([np.subtract(R_Radius_axis_form[0],R_Radius_center_form),
                                    np.subtract(R_Radius_axis_form[1],R_Radius_center_form),
                                    np.subtract(R_Radius_axis_form[2],R_Radius_center_form)])
    L_radius_Axis_mod = np.vstack([np.subtract(L_Radius_axis_form[0],L_Radius_center_form),
                                    np.subtract(L_Radius_axis_form[1],L_Radius_center_form),
                                    np.subtract(L_Radius_axis_form[2],L_Radius_center_form)])

    R_humerus_radius_angle = getangle(R_humerus_Axis_mod,R_radius_Axis_mod)
    L_humerus_radius_angle = getangle(L_humerus_Axis_mod,L_radius_Axis_mod)

    relbx=R_humerus_radius_angle[0]
    relby=R_humerus_radius_angle[1]
    relbz=R_humerus_radius_angle[2]-90.0
    lelbx=L_humerus_radius_angle[0]
    lelby=L_humerus_radius_angle[1]
    lelbz=L_humerus_radius_angle[2]-90.0

    # Calculate WRIST
    hand_JC = calc_hand_axis(frame['RWRA'] if 'RWRA' in frame else None,
                             frame['RWRB'] if 'RWRB' in frame else None,
                             frame['LWRA'] if 'LWRA' in frame else None,
                             frame['LWRB'] if 'LWRB' in frame else None,
                             frame['RFIN'] if 'RFIN' in frame else None,
                             frame['LFIN'] if 'LFIN' in frame else None,
                             axis_wrist[0],
                             axis_wrist[1],
                             vsk['RightHandThickness'],
                             vsk['LeftHandThickness'])

    kin_R_Hand_JC = hand_JC[0][:3, 3] #quick fix for storing JC
    kin_L_Hand_JC = hand_JC[1][:3, 3] #quick fix for storing JC


    # Change to same format

    R_Hand_center_form = hand_JC[0][:3, 3]
    L_Hand_center_form = hand_JC[1][:3, 3]
    R_Hand_axis_form = hand_JC[0][:3, :3] + R_Hand_center_form
    L_Hand_axis_form = hand_JC[1][:3, :3] + L_Hand_center_form

    # make the array which will be the input of findangle function
    R_hand_Axis_mod = np.vstack([np.subtract(R_Hand_axis_form[0],R_Hand_center_form),
                                np.subtract(R_Hand_axis_form[1],R_Hand_center_form),
                                np.subtract(R_Hand_axis_form[2],R_Hand_center_form)])
    L_hand_Axis_mod = np.vstack([np.subtract(L_Hand_axis_form[0],L_Hand_center_form),
                                np.subtract(L_Hand_axis_form[1],L_Hand_center_form),
                                np.subtract(L_Hand_axis_form[2],L_Hand_center_form)])

    R_radius_hand_angle = getangle(R_radius_Axis_mod,R_hand_Axis_mod)
    L_radius_hand_angle = getangle(L_radius_Axis_mod,L_hand_Axis_mod)

    rwrtx=R_radius_hand_angle[0]
    rwrty=R_radius_hand_angle[1]
    rwrtz=R_radius_hand_angle[2]*-1 + 90
    lwrtx=L_radius_hand_angle[0]
    lwrty=L_radius_hand_angle[1]*-1
    lwrtz=L_radius_hand_angle[2]-90

    if lwrtz < -180:
        lwrtz = lwrtz + 360


    # make each axis as same format to store

    # Pelvis
        # origin
    pel_origin = pelvis_origin
    pel_ox=pel_origin[0]
    pel_oy=pel_origin[1]
    pel_oz=pel_origin[2]
        # xaxis
    pel_x_axis = pelvis_vectors[0] + pelvis_origin
    pel_xx=pel_x_axis[0]
    pel_xy=pel_x_axis[1]
    pel_xz=pel_x_axis[2]
        # yaxis
    pel_y_axis = pelvis_vectors[1] + pelvis_origin
    pel_yx=pel_y_axis[0]
    pel_yy=pel_y_axis[1]
    pel_yz=pel_y_axis[2]
        # zaxis
    pel_z_axis = pelvis_vectors[2] + pelvis_origin
    pel_zx=pel_z_axis[0]
    pel_zy=pel_z_axis[1]
    pel_zz=pel_z_axis[2]

    # Hip
        # origin
    hip_origin = Hip_center_form
    hip_ox=hip_origin[0]
    hip_oy=hip_origin[1]
    hip_oz=hip_origin[2]
        # xaxis
    hip_x_axis = Hip_axis_form[0]
    hip_xx=hip_x_axis[0]
    hip_xy=hip_x_axis[1]
    hip_xz=hip_x_axis[2]
        # yaxis
    hip_y_axis = Hip_axis_form[1]
    hip_yx=hip_y_axis[0]
    hip_yy=hip_y_axis[1]
    hip_yz=hip_y_axis[2]
        # zaxis
    hip_z_axis = Hip_axis_form[2]
    hip_zx=hip_z_axis[0]
    hip_zy=hip_z_axis[1]
    hip_zz=hip_z_axis[2]

    # R KNEE
        # origin
    rknee_origin = R_Knee_center_form
    rknee_ox=rknee_origin[0]
    rknee_oy=rknee_origin[1]
    rknee_oz=rknee_origin[2]

        # xaxis
    rknee_x_axis = R_Knee_axis_form[0]
    rknee_xx=rknee_x_axis[0]
    rknee_xy=rknee_x_axis[1]
    rknee_xz=rknee_x_axis[2]
        # yaxis
    rknee_y_axis = R_Knee_axis_form[1]
    rknee_yx=rknee_y_axis[0]
    rknee_yy=rknee_y_axis[1]
    rknee_yz=rknee_y_axis[2]
        # zaxis
    rknee_z_axis = R_Knee_axis_form[2]
    rknee_zx=rknee_z_axis[0]
    rknee_zy=rknee_z_axis[1]
    rknee_zz=rknee_z_axis[2]

    # L KNEE
        # origin
    lknee_origin = L_Knee_center_form
    lknee_ox=lknee_origin[0]
    lknee_oy=lknee_origin[1]
    lknee_oz=lknee_origin[2]
        # xaxis
    lknee_x_axis = L_Knee_axis_form[0]
    lknee_xx=lknee_x_axis[0]
    lknee_xy=lknee_x_axis[1]
    lknee_xz=lknee_x_axis[2]
        # yaxis
    lknee_y_axis = L_Knee_axis_form[1]
    lknee_yx=lknee_y_axis[0]
    lknee_yy=lknee_y_axis[1]
    lknee_yz=lknee_y_axis[2]
        # zaxis
    lknee_z_axis = L_Knee_axis_form[2]
    lknee_zx=lknee_z_axis[0]
    lknee_zy=lknee_z_axis[1]
    lknee_zz=lknee_z_axis[2]

    # R ANKLE
        # origin
    rank_origin = R_Ankle_center_form
    rank_ox=rank_origin[0]
    rank_oy=rank_origin[1]
    rank_oz=rank_origin[2]
        # xaxis
    rank_x_axis = R_Ankle_axis_form[0]
    rank_xx=rank_x_axis[0]
    rank_xy=rank_x_axis[1]
    rank_xz=rank_x_axis[2]
        # yaxis
    rank_y_axis = R_Ankle_axis_form[1]
    rank_yx=rank_y_axis[0]
    rank_yy=rank_y_axis[1]
    rank_yz=rank_y_axis[2]
        # zaxis
    rank_z_axis = R_Ankle_axis_form[2]
    rank_zx=rank_z_axis[0]
    rank_zy=rank_z_axis[1]
    rank_zz=rank_z_axis[2]

    # L ANKLE
        # origin
    lank_origin = L_Ankle_center_form
    lank_ox=lank_origin[0]
    lank_oy=lank_origin[1]
    lank_oz=lank_origin[2]
        # xaxis
    lank_x_axis = L_Ankle_axis_form[0]
    lank_xx=lank_x_axis[0]
    lank_xy=lank_x_axis[1]
    lank_xz=lank_x_axis[2]
        # yaxis
    lank_y_axis = L_Ankle_axis_form[1]
    lank_yx=lank_y_axis[0]
    lank_yy=lank_y_axis[1]
    lank_yz=lank_y_axis[2]
        # zaxis
    lank_z_axis = L_Ankle_axis_form[2]
    lank_zx=lank_z_axis[0]
    lank_zy=lank_z_axis[1]
    lank_zz=lank_z_axis[2]

    # R FOOT
        # origin
    rfoot_origin = R_Foot_center_form
    rfoot_ox=rfoot_origin[0]
    rfoot_oy=rfoot_origin[1]
    rfoot_oz=rfoot_origin[2]
        # xaxis
    rfoot_x_axis = R_Foot_axis_form[0]
    rfoot_xx=rfoot_x_axis[0]
    rfoot_xy=rfoot_x_axis[1]
    rfoot_xz=rfoot_x_axis[2]
        # yaxis
    rfoot_y_axis = R_Foot_axis_form[1]
    rfoot_yx=rfoot_y_axis[0]
    rfoot_yy=rfoot_y_axis[1]
    rfoot_yz=rfoot_y_axis[2]
        # zaxis
    rfoot_z_axis = R_Foot_axis_form[2]
    rfoot_zx=rfoot_z_axis[0]
    rfoot_zy=rfoot_z_axis[1]
    rfoot_zz=rfoot_z_axis[2]

    # L FOOT
        # origin
    lfoot_origin = L_Foot_center_form
    lfoot_ox=lfoot_origin[0]
    lfoot_oy=lfoot_origin[1]
    lfoot_oz=lfoot_origin[2]
        # xaxis
    lfoot_x_axis = L_Foot_axis_form[0]
    lfoot_xx=lfoot_x_axis[0]
    lfoot_xy=lfoot_x_axis[1]
    lfoot_xz=lfoot_x_axis[2]
        # yaxis
    lfoot_y_axis = L_Foot_axis_form[1]
    lfoot_yx=lfoot_y_axis[0]
    lfoot_yy=lfoot_y_axis[1]
    lfoot_yz=lfoot_y_axis[2]
        # zaxis
    lfoot_z_axis = L_Foot_axis_form[2]
    lfoot_zx=lfoot_z_axis[0]
    lfoot_zy=lfoot_z_axis[1]
    lfoot_zz=lfoot_z_axis[2]

    # HEAD
        # origin
    head_origin = Head_center_form
    head_ox=head_origin[0]
    head_oy=head_origin[1]
    head_oz=head_origin[2]
        # xaxis
    head_x_axis = Head_axis_form[0]
    head_xx=head_x_axis[0]
    head_xy=head_x_axis[1]
    head_xz=head_x_axis[2]
        # yaxis
    head_y_axis = Head_axis_form[1]
    head_yx=head_y_axis[0]
    head_yy=head_y_axis[1]
    head_yz=head_y_axis[2]
        # zaxis
    head_z_axis = Head_axis_form[2]
    head_zx=head_z_axis[0]
    head_zy=head_z_axis[1]
    head_zz=head_z_axis[2]

    # THORAX
        # origin
    tho_origin = Thorax_center_form
    tho_ox=tho_origin[0]
    tho_oy=tho_origin[1]
    tho_oz=tho_origin[2]
        # xaxis
    tho_x_axis = Thorax_axis_form[0]
    tho_xx=tho_x_axis[0]
    tho_xy=tho_x_axis[1]
    tho_xz=tho_x_axis[2]
        # yaxis
    tho_y_axis = Thorax_axis_form[1]
    tho_yx=tho_y_axis[0]
    tho_yy=tho_y_axis[1]
    tho_yz=tho_y_axis[2]
        # zaxis
    tho_z_axis = Thorax_axis_form[2]
    tho_zx=tho_z_axis[0]
    tho_zy=tho_z_axis[1]
    tho_zz=tho_z_axis[2]

    # R CLAVICLE
        # origin
    rclav_origin = R_Clavicle_center_form
    rclav_ox=rclav_origin[0]
    rclav_oy=rclav_origin[1]
    rclav_oz=rclav_origin[2]
        # xaxis
    rclav_x_axis = R_Clavicle_axis_form[0]
    rclav_xx=rclav_x_axis[0]
    rclav_xy=rclav_x_axis[1]
    rclav_xz=rclav_x_axis[2]
        # yaxis
    rclav_y_axis = R_Clavicle_axis_form[1]
    rclav_yx=rclav_y_axis[0]
    rclav_yy=rclav_y_axis[1]
    rclav_yz=rclav_y_axis[2]
        # zaxis
    rclav_z_axis = R_Clavicle_axis_form[2]
    rclav_zx=rclav_z_axis[0]
    rclav_zy=rclav_z_axis[1]
    rclav_zz=rclav_z_axis[2]

    # L CLAVICLE
        # origin
    lclav_origin = L_Clavicle_center_form
    lclav_ox=lclav_origin[0]
    lclav_oy=lclav_origin[1]
    lclav_oz=lclav_origin[2]
        # xaxis
    lclav_x_axis = L_Clavicle_axis_form[0]
    lclav_xx=lclav_x_axis[0]
    lclav_xy=lclav_x_axis[1]
    lclav_xz=lclav_x_axis[2]
        # yaxis
    lclav_y_axis = L_Clavicle_axis_form[1]
    lclav_yx=lclav_y_axis[0]
    lclav_yy=lclav_y_axis[1]
    lclav_yz=lclav_y_axis[2]
        # zaxis
    lclav_z_axis = L_Clavicle_axis_form[2]
    lclav_zx=lclav_z_axis[0]
    lclav_zy=lclav_z_axis[1]
    lclav_zz=lclav_z_axis[2]

    # R HUMERUS
        # origin
    rhum_origin = R_Humerus_center_form
    rhum_ox=rhum_origin[0]
    rhum_oy=rhum_origin[1]
    rhum_oz=rhum_origin[2]
        # xaxis
    rhum_x_axis = R_Humerus_axis_form[0]
    rhum_xx=rhum_x_axis[0]
    rhum_xy=rhum_x_axis[1]
    rhum_xz=rhum_x_axis[2]
        # yaxis
    rhum_y_axis = R_Humerus_axis_form[1]
    rhum_yx=rhum_y_axis[0]
    rhum_yy=rhum_y_axis[1]
    rhum_yz=rhum_y_axis[2]
        # zaxis
    rhum_z_axis = R_Humerus_axis_form[2]
    rhum_zx=rhum_z_axis[0]
    rhum_zy=rhum_z_axis[1]
    rhum_zz=rhum_z_axis[2]

    # L HUMERUS
        # origin
    lhum_origin = L_Humerus_center_form
    lhum_ox=lhum_origin[0]
    lhum_oy=lhum_origin[1]
    lhum_oz=lhum_origin[2]
        # xaxis
    lhum_x_axis = L_Humerus_axis_form[0]
    lhum_xx=lhum_x_axis[0]
    lhum_xy=lhum_x_axis[1]
    lhum_xz=lhum_x_axis[2]
        # yaxis
    lhum_y_axis = L_Humerus_axis_form[1]
    lhum_yx=lhum_y_axis[0]
    lhum_yy=lhum_y_axis[1]
    lhum_yz=lhum_y_axis[2]
        # zaxis
    lhum_z_axis = L_Humerus_axis_form[2]
    lhum_zx=lhum_z_axis[0]
    lhum_zy=lhum_z_axis[1]
    lhum_zz=lhum_z_axis[2]

    # R RADIUS
        # origin
    rrad_origin = R_Radius_center_form
    rrad_ox=rrad_origin[0]
    rrad_oy=rrad_origin[1]
    rrad_oz=rrad_origin[2]
        # xaxis
    rrad_x_axis = R_Radius_axis_form[0]
    rrad_xx=rrad_x_axis[0]
    rrad_xy=rrad_x_axis[1]
    rrad_xz=rrad_x_axis[2]
        # yaxis
    rrad_y_axis = R_Radius_axis_form[1]
    rrad_yx=rrad_y_axis[0]
    rrad_yy=rrad_y_axis[1]
    rrad_yz=rrad_y_axis[2]
        # zaxis
    rrad_z_axis = R_Radius_axis_form[2]
    rrad_zx=rrad_z_axis[0]
    rrad_zy=rrad_z_axis[1]
    rrad_zz=rrad_z_axis[2]

    # L RADIUS
        # origin
    lrad_origin = L_Radius_center_form
    lrad_ox=lrad_origin[0]
    lrad_oy=lrad_origin[1]
    lrad_oz=lrad_origin[2]
        # xaxis
    lrad_x_axis = L_Radius_axis_form[0]
    lrad_xx=lrad_x_axis[0]
    lrad_xy=lrad_x_axis[1]
    lrad_xz=lrad_x_axis[2]
        # yaxis
    lrad_y_axis = L_Radius_axis_form[1]
    lrad_yx=lrad_y_axis[0]
    lrad_yy=lrad_y_axis[1]
    lrad_yz=lrad_y_axis[2]
        # zaxis
    lrad_z_axis = L_Radius_axis_form[2]
    lrad_zx=lrad_z_axis[0]
    lrad_zy=lrad_z_axis[1]
    lrad_zz=lrad_z_axis[2]

    # R HAND
        # origin
    rhand_origin = R_Hand_center_form
    rhand_ox=rhand_origin[0]
    rhand_oy=rhand_origin[1]
    rhand_oz=rhand_origin[2]
        # xaxis
    rhand_x_axis= R_Hand_axis_form[0]
    rhand_xx=rhand_x_axis[0]
    rhand_xy=rhand_x_axis[1]
    rhand_xz=rhand_x_axis[2]
        # yaxis
    rhand_y_axis= R_Hand_axis_form[1]
    rhand_yx=rhand_y_axis[0]
    rhand_yy=rhand_y_axis[1]
    rhand_yz=rhand_y_axis[2]
        # zaxis
    rhand_z_axis= R_Hand_axis_form[2]
    rhand_zx=rhand_z_axis[0]
    rhand_zy=rhand_z_axis[1]
    rhand_zz=rhand_z_axis[2]

    # L HAND
        # origin
    lhand_origin = L_Hand_center_form
    lhand_ox=lhand_origin[0]
    lhand_oy=lhand_origin[1]
    lhand_oz=lhand_origin[2]
        # xaxis
    lhand_x_axis = L_Hand_axis_form[0]
    lhand_xx=lhand_x_axis[0]
    lhand_xy=lhand_x_axis[1]
    lhand_xz=lhand_x_axis[2]
        # yaxis
    lhand_y_axis = L_Hand_axis_form[1]
    lhand_yx=lhand_y_axis[0]
    lhand_yy=lhand_y_axis[1]
    lhand_yz=lhand_y_axis[2]
        # zaxis
    lhand_z_axis = L_Hand_axis_form[2]
    lhand_zx=lhand_z_axis[0]
    lhand_zy=lhand_z_axis[1]
    lhand_zz=lhand_z_axis[2]
    #-----------------------------------------------------

    #Store everything in an array to send back to results of process

    r=[
    pelx,pely,pelz,
    rhipx,rhipy,rhipz,
    lhipx,lhipy,lhipz,
    rkneex,rkneey,rkneez,
    lkneex,lkneey,lkneez,
    ranklex,rankley,ranklez,
    lanklex,lankley,lanklez,
    rfootx,rfooty,rfootz,
    lfootx,lfooty,lfootz,
    headx,heady,headz,
    thox,thoy,thoz,
    neckx,necky,neckz,
    spix,spiy,spiz,
    rshox,rshoy,rshoz,
    lshox,lshoy,lshoz,
    relbx,relby,relbz,
    lelbx,lelby,lelbz,
    rwrtx,rwrty,rwrtz,
    lwrtx,lwrty,lwrtz,
    pel_ox,pel_oy,pel_oz,pel_xx,pel_xy,pel_xz,pel_yx,pel_yy,pel_yz,pel_zx,pel_zy,pel_zz,
    hip_ox,hip_oy,hip_oz,hip_xx,hip_xy,hip_xz,hip_yx,hip_yy,hip_yz,hip_zx,hip_zy,hip_zz,
    rknee_ox,rknee_oy,rknee_oz,rknee_xx,rknee_xy,rknee_xz,rknee_yx,rknee_yy,rknee_yz,rknee_zx,rknee_zy,rknee_zz,
    lknee_ox,lknee_oy,lknee_oz,lknee_xx,lknee_xy,lknee_xz,lknee_yx,lknee_yy,lknee_yz,lknee_zx,lknee_zy,lknee_zz,
    rank_ox,rank_oy,rank_oz,rank_xx,rank_xy,rank_xz,rank_yx,rank_yy,rank_yz,rank_zx,rank_zy,rank_zz,
    lank_ox,lank_oy,lank_oz,lank_xx,lank_xy,lank_xz,lank_yx,lank_yy,lank_yz,lank_zx,lank_zy,lank_zz,
    rfoot_ox,rfoot_oy,rfoot_oz,rfoot_xx,rfoot_xy,rfoot_xz,rfoot_yx,rfoot_yy,rfoot_yz,rfoot_zx,rfoot_zy,rfoot_zz,
    lfoot_ox,lfoot_oy,lfoot_oz,lfoot_xx,lfoot_xy,lfoot_xz,lfoot_yx,lfoot_yy,lfoot_yz,lfoot_zx,lfoot_zy,lfoot_zz,
    head_ox,head_oy,head_oz,head_xx,head_xy,head_xz,head_yx,head_yy,head_yz,head_zx,head_zy,head_zz,
    tho_ox,tho_oy,tho_oz,tho_xx,tho_xy,tho_xz,tho_yx,tho_yy,tho_yz,tho_zx,tho_zy,tho_zz,
    rclav_ox,rclav_oy,rclav_oz,rclav_xx,rclav_xy,rclav_xz,rclav_yx,rclav_yy,rclav_yz,rclav_zx,rclav_zy,rclav_zz,
    lclav_ox,lclav_oy,lclav_oz,lclav_xx,lclav_xy,lclav_xz,lclav_yx,lclav_yy,lclav_yz,lclav_zx,lclav_zy,lclav_zz,
    rhum_ox,rhum_oy,rhum_oz,rhum_xx,rhum_xy,rhum_xz,rhum_yx,rhum_yy,rhum_yz,rhum_zx,rhum_zy,rhum_zz,
    lhum_ox,lhum_oy,lhum_oz,lhum_xx,lhum_xy,lhum_xz,lhum_yx,lhum_yy,lhum_yz,lhum_zx,lhum_zy,lhum_zz,
    rrad_ox,rrad_oy,rrad_oz,rrad_xx,rrad_xy,rrad_xz,rrad_yx,rrad_yy,rrad_yz,rrad_zx,rrad_zy,rrad_zz,
    lrad_ox,lrad_oy,lrad_oz,lrad_xx,lrad_xy,lrad_xz,lrad_yx,lrad_yy,lrad_yz,lrad_zx,lrad_zy,lrad_zz,
    rhand_ox,rhand_oy,rhand_oz,rhand_xx,rhand_xy,rhand_xz,rhand_yx,rhand_yy,rhand_yz,rhand_zx,rhand_zy,rhand_zz,
    lhand_ox,lhand_oy,lhand_oz,lhand_xx,lhand_xy,lhand_xz,lhand_yx,lhand_yy,lhand_yz,lhand_zx,lhand_zy,lhand_zz
    ]

    r=np.array(r,dtype=np.float64)


    #Put temporary dictionary for joint centers to return for now, then modify later
    jc = {}
    jc['Pelvis_axis'] = kin_Pelvis_axis
    jc['Thorax_axis'] = kin_Thorax_axis

    jc['Pelvis'] = kin_Pelvis_JC
    jc['RHip'] = kin_R_Hip_JC
    jc['LHip'] = kin_L_Hip_JC
    jc['RKnee'] = kin_R_Knee_JC
    jc['LKnee'] = kin_L_Knee_JC
    jc['RAnkle'] = kin_R_Ankle_JC
    jc['LAnkle'] = kin_L_Ankle_JC
    jc['RFoot'] = kin_R_Foot_JC
    jc['LFoot'] = kin_L_Foot_JC

    jc['RHEE'] = kin_RHEE
    jc['LHEE'] = kin_LHEE

    jc['C7'] = kin_C7
    jc['CLAV'] = kin_CLAV
    jc['STRN'] = kin_STRN
    jc['T10'] = kin_T10


    jc['Front_Head'] = kin_Head_Front
    jc['Back_Head'] = kin_Head_Back

    jc['Head'] = kin_Head_JC
    jc['Thorax'] = kin_Thorax_JC

    jc['RShoulder'] = kin_R_Shoulder_JC
    jc['LShoulder'] = kin_L_Shoulder_JC
    jc['RHumerus'] = kin_R_Humerus_JC
    jc['LHumerus'] = kin_L_Humerus_JC
    jc['RRadius'] = kin_R_Radius_JC
    jc['LRadius'] = kin_L_Radius_JC
    jc['RHand'] = kin_R_Hand_JC
    jc['LHand'] = kin_L_Hand_JC

    return r,jc
