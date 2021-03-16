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
import sys
import os
from math import *
import math
import numpy as np
from .pycgmIO import *

# Lowerbody Coordinate System
def pelvisJointCenter(frame):
    """Make the Pelvis Axis.

    Takes in a dictionary of x,y,z positions and marker names, as well as an index
    Calculates the pelvis joint center and axis and returns both.

    Markers used: RASI, LASI, RPSI, LPSI
    Other landmarks used: origin, sacrum

    Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure [1]_ and then normalized.
    Pelvis Y_axis: LASI-RASI x,y,z positions, then normalized.
    Pelvis Z_axis: Cross product of x_axis and y_axis.

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.

    Returns
    -------
    pelvis : array
        Returns an array that contains the pelvis origin in a 1x3 array of xyz values,
        which is then followed by a 4x1x3 array composed of the pelvis x, y, z
        axis components, and the sacrum x,y,z position.

    References
    ----------
    .. [1] M. P. Kadaba, H. K. Ramakrishnan, and M. E. Wootten, “Measurement of
            lower extremity kinematics during level walking,” J. Orthop. Res.,
            vol. 8, no. 3, pp. 383–392, May 1990, doi: 10.1002/jor.1100080310.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import pelvisJointCenter
    >>> frame = {'RASI': np.array([ 395.36532593,  428.09790039, 1036.82763672]),
    ...          'LASI': np.array([ 183.18504333,  422.78927612, 1033.07299805]),
    ...          'RPSI': np.array([ 341.41815186,  246.72117615, 1055.99145508]),
    ...          'LPSI': np.array([ 255.79994202,  241.42199707, 1057.30065918]) }
    >>> pelvisJointCenter(frame) #doctest: +NORMALIZE_WHITESPACE
    [array([ 289.27518463,  425.44358826, 1034.95031739]),
    array([[ 289.25243803,  426.43632163, 1034.8321521 ],
    [ 288.27565385,  425.41858059, 1034.93263018],
    [ 289.25467091,  425.56129577, 1035.94315379]]),
    array([ 298.60904694,  244.07158661, 1056.64605713])]

    >>> frame = {'RASI': np.array([ 395.36532593,  428.09790039, 1036.82763672]),
    ...          'LASI': np.array([ 183.18504333,  422.78927612, 1033.07299805]),
    ...          'SACR': np.array([ 294.60904694,  242.07158661, 1049.64605713]) }
    >>> pelvisJointCenter(frame) #doctest: +NORMALIZE_WHITESPACE
    [array([ 289.27518463,  425.44358826, 1034.95031739]),
    array([[ 289.25166321,  426.44012508, 1034.87056085],
    [ 288.27565385,  425.41858059, 1034.93263018],
    [ 289.25556415,  425.52289134, 1035.94697483]]),
    array([ 294.60904694,  242.07158661, 1049.64605713])]
    """
    # Get the Pelvis Joint Centre

    #REQUIRED MARKERS:
    # RASI
    # LASI
    # RPSI
    # LPSI

    RASI = frame['RASI']
    LASI = frame['LASI']

    try:
        RPSI = frame['RPSI']
        LPSI = frame['LPSI']
        #  If no sacrum, mean of posterior markers is used as the sacrum
        sacrum = (RPSI+LPSI)/2.0
    except:
        pass #going to use sacrum marker

    #  If no sacrum, mean of posterior markers is used as the sacrum
    if 'SACR' in frame:
        sacrum = frame['SACR']

    # REQUIRED LANDMARKS:
    # origin
    # sacrum

    # Origin is Midpoint between RASI and LASI
    origin = (RASI+LASI)/2.0

    # This calculate the each axis
    # beta1,2,3 is arbitrary name to help calculate.
    beta1 = origin-sacrum
    beta2 = LASI-RASI

    # Y_axis is normalized beta2
    y_axis = beta2/norm3d(beta2)

    # X_axis computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990)
    # and then normalized.
    beta3_cal = np.dot(beta1,y_axis)
    beta3_cal2 = beta3_cal*y_axis
    beta3 = beta1-beta3_cal2
    x_axis = beta3/norm3d(beta3)

    # Z-axis is cross product of x_axis and y_axis.
    z_axis = cross(x_axis,y_axis)

     # Add the origin back to the vector
    y_axis = y_axis+origin
    z_axis = z_axis+origin
    x_axis = x_axis+origin

    pelvis_axis = np.asarray([x_axis,y_axis,z_axis])

    pelvis = [origin,pelvis_axis,sacrum] #probably don't need to return sacrum

    return pelvis


def hipJointCenter(frame, pel_origin, pel_x, pel_y, pel_z, vsk=None):
    u"""Calculate the hip joint center.

    Takes in a dictionary of x,y,z positions and marker names, as well as an
    index. Calculates the hip joint center and returns the hip joint center.

    Other landmarks used: origin, sacrum

    Subject Measurement values used: MeanLegLength, R_AsisToTrocanterMeasure,
    InterAsisDistance, L_AsisToTrocanterMeasure

    Hip Joint Center: Computed using Hip Joint Center Calculation [1]_.

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.
    pel_origin : array
        An array of pelvis origin, (pel_x, pel_y, pel_z) each x, y, z position.
    pel_x, pel_y, pel_z : int
        Respective axes of the pelvis.
    vsk : dict, optional
        A dictionary containing subject measurements.

    Returns
    -------
    hip_JC : array
        Returns a 2x3 array that contains two 1x3 arrays
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
    >>> from .pyCGM import hipJointCenter
    >>> frame = None
    >>> vsk = {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512,
    ...        'L_AsisToTrocanterMeasure': 72.512, 'InterAsisDistance': 215.908996582031}
    >>> pel_origin = [ 251.60830688, 391.74131775, 1032.89349365]
    >>> pel_x = [251.74063624, 392.72694721, 1032.78850073]
    >>> pel_y = [250.61711554, 391.87232862, 1032.8741063]
    >>> pel_z = [251.60295336, 391.84795134, 1033.88777762]
    >>> hipJointCenter(frame,pel_origin,pel_x,pel_y,pel_z,vsk)
    array([[182.57097799, 339.43231799, 935.52900136],
           [308.38050352, 322.80342433, 937.98979092]])

    """
    #Get Global Values

    # Requires
    # pelvis axis

    pel_origin=np.asarray(pel_origin)
    pel_x=np.asarray(pel_x)
    pel_y=np.asarray(pel_y)
    pel_z=np.asarray(pel_z)

    # Model's eigen value
    #
    # LegLength
    # MeanLegLength
    # mm (marker radius)
    # interAsisMeasure

    #Set the variables needed to calculate the joint angle
    #Half of marker size
    mm = 7.0

    MeanLegLength = vsk['MeanLegLength']
    R_AsisToTrocanterMeasure = vsk['R_AsisToTrocanterMeasure']
    L_AsisToTrocanterMeasure = vsk['L_AsisToTrocanterMeasure']
    interAsisMeasure = vsk['InterAsisDistance']
    C = ( MeanLegLength * 0.115 ) - 15.3
    theta = 0.500000178813934
    beta = 0.314000427722931
    aa = interAsisMeasure/2.0
    S = -1

    # Hip Joint Center Calculation (ref. Davis_1991)

    # Left: Calculate the distance to translate along the pelvis axis
    L_Xh = (-L_AsisToTrocanterMeasure - mm) * cos(beta) + C * cos(theta) * sin(beta)
    L_Yh = S*(C*sin(theta)- aa)
    L_Zh = (-L_AsisToTrocanterMeasure - mm) * sin(beta) - C * cos(theta) * cos(beta)

    # Right:  Calculate the distance to translate along the pelvis axis
    R_Xh = (-R_AsisToTrocanterMeasure - mm) * cos(beta) + C * cos(theta) * sin(beta)
    R_Yh = (C*sin(theta)- aa)
    R_Zh = (-R_AsisToTrocanterMeasure - mm) * sin(beta) - C * cos(theta) * cos(beta)

    # get the unit pelvis axis
    pelvis_xaxis = pel_x-pel_origin
    pelvis_yaxis = pel_y-pel_origin
    pelvis_zaxis = pel_z-pel_origin

    # multiply the distance to the unit pelvis axis
    L_hipJCx = pelvis_xaxis*L_Xh
    L_hipJCy = pelvis_yaxis*L_Yh
    L_hipJCz = pelvis_zaxis*L_Zh
    L_hipJC = np.asarray([  L_hipJCx[0]+L_hipJCy[0]+L_hipJCz[0],
                            L_hipJCx[1]+L_hipJCy[1]+L_hipJCz[1],
                            L_hipJCx[2]+L_hipJCy[2]+L_hipJCz[2]])

    R_hipJCx = pelvis_xaxis*R_Xh
    R_hipJCy = pelvis_yaxis*R_Yh
    R_hipJCz = pelvis_zaxis*R_Zh
    R_hipJC = np.asarray([  R_hipJCx[0]+R_hipJCy[0]+R_hipJCz[0],
                            R_hipJCx[1]+R_hipJCy[1]+R_hipJCz[1],
                            R_hipJCx[2]+R_hipJCy[2]+R_hipJCz[2]])

    L_hipJC = L_hipJC+pel_origin
    R_hipJC = R_hipJC+pel_origin

    hip_JC = np.asarray([L_hipJC,R_hipJC])

    return hip_JC


def hipAxisCenter(l_hip_jc, r_hip_jc, pelvis_axis):
    """Calculate the hip joint axis.

    Takes in a hip joint center of x,y,z positions and pelvis axis, and
    calculates the hip axis and returns the hip joint origin and axis.

    Hip center axis: Computed by taking the mean at each x,y,z axis of the
    left and right hip joint center.

    Hip axis: Computed by getting the summation of the pelvis and hip center
    axes.

    Parameters
    ----------
    l_hip_jc, r_hip_jc : array
        left and right hip joint center with x, y, z position in an array.
    pelvis_axis : array
        An array of pelvis origin and axis. The first element is an
        1x3 array containing the x, y, z axis of the origin.
        The second elemnt is a 3x3 containing 3 arrays of
        x, y, z coordinates of the individual pelvis axis.

    Returns
    -------
    hipaxis_center, axis : array
        Returns an array that contains the hip axis center in a
        1x3 array of xyz values, which is then followed by a
        3x2x3 array composed of the hip axis center
        x, y, and z axis components. The xyz axis components are
        2x3 arrays consisting of the axis center in the first
        dimension and the direction of the axis in the second dimension.


    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import hipAxisCenter
    >>> r_hip_jc = [182.57097863, 339.43231855, 935.529000126]
    >>> l_hip_jc = [308.38050472, 322.80342417, 937.98979061]
    >>> pelvis_axis = [np.array([251.60830688, 391.74131775, 1032.89349365]),
    ...                np.array([[251.74063624, 392.72694721, 1032.78850073],
    ...                    [250.61711554, 391.87232862, 1032.8741063],
    ...                    [251.60295336, 391.84795134, 1033.88777762]]),
    ...                np.array([231.57849121, 210.25262451, 1052.24969482])]
    >>> [np.around(arr,8) for arr in hipAxisCenter(l_hip_jc,r_hip_jc,pelvis_axis)] #doctest: +NORMALIZE_WHITESPACE
    [array([245.47574168, 331.11787136, 936.75939537]),
    array([[245.60807104, 332.10350082, 936.65440245],
           [244.48455034, 331.24888223, 936.74000802],
           [245.47038816, 331.22450495, 937.75367934]])]
    """

    # Get shared hip axis, it is inbetween the two hip joint centers
    hipaxis_center = [(r_hip_jc[0]+l_hip_jc[0])/2.0,(r_hip_jc[1]+l_hip_jc[1])/2.0,(r_hip_jc[2]+l_hip_jc[2])/2.0]

    #convert pelvis_axis to x,y,z axis to use more easy
    pelvis_x_axis = np.subtract(pelvis_axis[1][0],pelvis_axis[0])
    pelvis_y_axis = np.subtract(pelvis_axis[1][1],pelvis_axis[0])
    pelvis_z_axis = np.subtract(pelvis_axis[1][2],pelvis_axis[0])

    #Translate pelvis axis to shared hip centre
    # Add the origin back to the vector
    y_axis = [pelvis_y_axis[0]+hipaxis_center[0],pelvis_y_axis[1]+hipaxis_center[1],pelvis_y_axis[2]+hipaxis_center[2]]
    z_axis = [pelvis_z_axis[0]+hipaxis_center[0],pelvis_z_axis[1]+hipaxis_center[1],pelvis_z_axis[2]+hipaxis_center[2]]
    x_axis = [pelvis_x_axis[0]+hipaxis_center[0],pelvis_x_axis[1]+hipaxis_center[1],pelvis_x_axis[2]+hipaxis_center[2]]

    axis = [x_axis,y_axis,z_axis]

    return [hipaxis_center,axis]


def kneeJointCenter(frame, hip_JC, delta, vsk=None):
    """Calculate the knee joint center and axis.

    Takes in a dictionary of marker names to x, y, z positions, the hip axis
    and pelvis axis. Calculates the knee joint axis and returns the knee origin
    and axis.

    Markers used: RTHI, LTHI, RKNE, LKNE, hip_JC
    Subject Measurement values used: RightKneeWidth, LeftKneeWidth

    Knee joint center: Computed using Knee Axis Calculation [1]_.

    Parameters
    ----------
    frame : dict
        dictionaries of marker lists.
    hip_JC : array
        An array of hip_JC containing the x,y,z axes marker positions of the
        hip joint center.
    delta : float, optional
        The length from marker to joint center, retrieved from subject
        measurement file.
    vsk : dict, optional
        A dictionary containing subject measurements.

    Returns
    -------
    R, L, axis : array
        Returns an array that contains the knee axis center in a 1x3 array of
        xyz values, which is then followed by a 2x3x3
        array composed of the knee axis center x, y, and z axis components. The
        xyz axis components are 2x3 arrays consisting of the
        axis center in the first dimension and the direction of the axis in the
        second dimension.

    References
    ----------
    .. [1] Baker, R. (2013). Measuring walking : a handbook of clinical gait
            analysis. Mac Keith Press.

    Modifies
    --------
    delta is changed suitably to knee.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import kneeJointCenter
    >>> vsk = { 'RightKneeWidth' : 105.0, 'LeftKneeWidth' : 105.0 }
    >>> frame = { 'RTHI': np.array([426.50338745, 262.65310669, 673.66247559]),
    ...           'LTHI': np.array([51.93867874, 320.01849365, 723.03186035]),
    ...           'RKNE': np.array([416.98687744, 266.22558594, 524.04089355]),
    ...           'LKNE': np.array([84.62355804, 286.69122314, 529.39819336])}
    >>> hip_JC = [[182.57097863, 339.43231855, 935.52900126],
    ...         [309.38050472, 32280342417, 937.98979061]]
    >>> delta = 0
    >>> kneeJointCenter(frame,hip_JC,delta,vsk) #doctest: +NORMALIZE_WHITESPACE
    [array([413.21007973, 266.22558784, 464.66088466]),
    array([143.55478579, 279.90370346, 524.78408753]),
    array([[[414.20806312, 266.22558785, 464.59740907],
    [413.14660414, 266.22558786, 463.66290127],
    [413.21007973, 267.22558784, 464.66088468]],
    [[143.65611281, 280.88685896, 524.63197541],
    [142.56434499, 280.01777942, 524.86163553],
    [143.64837987, 280.0465038 , 525.76940383]]])]
    """



    #Get Global Values
    mm = 7.0
    R_kneeWidth = vsk['RightKneeWidth']
    L_kneeWidth = vsk['LeftKneeWidth']
    R_delta = (R_kneeWidth/2.0)+mm
    L_delta = (L_kneeWidth/2.0)+mm

    #REQUIRED MARKERS:
    # RTHI
    # LTHI
    # RKNE
    # LKNE
    # hip_JC

    RTHI = frame['RTHI']
    LTHI = frame['LTHI']
    RKNE = frame['RKNE']
    LKNE = frame['LKNE']

    R_hip_JC = hip_JC[1]
    L_hip_JC = hip_JC[0]

     # Determine the position of kneeJointCenter using findJointC function
    R = findJointC(RTHI,R_hip_JC,RKNE,R_delta)
    L = findJointC(LTHI,L_hip_JC,LKNE,L_delta)

    # Knee Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)
    #Right axis calculation

    thi_kne_R = RTHI-RKNE

    # Z axis is Thigh bone calculated by the hipJC and  kneeJC
    # the axis is then normalized
    axis_z = R_hip_JC-R

    # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
    # and calculated by each point's vector cross vector.
    # the axis is then normalized.
    # axis_x = cross(axis_z,thi_kne_R)
    axis_x = cross(axis_z,RKNE-R_hip_JC)

    # Y axis is determined by cross product of axis_z and axis_x.
    # the axis is then normalized.
    axis_y = cross(axis_z,axis_x)

    Raxis = np.asarray([axis_x,axis_y,axis_z])

    #Left axis calculation

    thi_kne_L = LTHI-LKNE

    # Z axis is Thigh bone calculated by the hipJC and  kneeJC
    # the axis is then normalized
    axis_z = L_hip_JC-L

    # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
    # and calculated by each point's vector cross vector.
    # the axis is then normalized.
    # axis_x = cross(thi_kne_L,axis_z)
    #using hipjc instead of thigh marker
    axis_x = cross(LKNE-L_hip_JC,axis_z)

    # Y axis is determined by cross product of axis_z and axis_x.
    # the axis is then normalized.
    axis_y = cross(axis_z,axis_x)

    Laxis = np.asarray([axis_x,axis_y,axis_z])

    # Clear the name of axis and then nomalize it.
    R_knee_x_axis = Raxis[0]
    R_knee_x_axis = R_knee_x_axis/norm3d(R_knee_x_axis)
    R_knee_y_axis = Raxis[1]
    R_knee_y_axis = R_knee_y_axis/norm3d(R_knee_y_axis)
    R_knee_z_axis = Raxis[2]
    R_knee_z_axis = R_knee_z_axis/norm3d(R_knee_z_axis)
    L_knee_x_axis = Laxis[0]
    L_knee_x_axis = L_knee_x_axis/norm3d(L_knee_x_axis)
    L_knee_y_axis = Laxis[1]
    L_knee_y_axis = L_knee_y_axis/norm3d(L_knee_y_axis)
    L_knee_z_axis = Laxis[2]
    L_knee_z_axis = L_knee_z_axis/norm3d(L_knee_z_axis)

    #Put both axis in array
    # Add the origin back to the vector
    y_axis = R_knee_y_axis+R
    z_axis = R_knee_z_axis+R
    x_axis = R_knee_x_axis+R
    Raxis = np.asarray([x_axis,y_axis,z_axis])

    # Add the origin back to the vector
    y_axis = L_knee_y_axis+L
    z_axis = L_knee_z_axis+L
    x_axis = L_knee_x_axis+L
    Laxis = np.asarray([x_axis,y_axis,z_axis])

    axis = np.asarray([Raxis,Laxis])

    return [R,L,axis]

def ankleJointCenter(frame,knee_JC,delta,vsk=None):
    """Calculate the ankle joint center and axis.

    Takes in a dictionary of marker names to x, y, z positions and the knee
    joint center.
    Calculates the ankle joint axis and returns the ankle origin and axis

    Markers used: tib_R, tib_L, ank_R, ank_L, knee_JC
    Subject Measurement values used: RightKneeWidth, LeftKneeWidth

    Ankle Axis: Computed using Ankle Axis Calculation [1]_.

    Parameters
    ----------
    frame : dict
        dictionaries of marker lists.
    knee_JC : array
        An array of knee_JC each x,y,z position.
    delta : float
        The length from marker to joint center, retrieved from subject measurement file
    vsk : dict, optional
        A dictionary containing subject measurements.

    Returns
    -------
    R, L, axis : array
        Returns an array that contains the ankle axis origin in a 1x3 array of xyz values,
        which is then followed by a 3x2x3 array composed of the ankle origin, x, y, and z
        axis components. The xyz axis components are 2x3 arrays consisting of the origin
        in the first dimension and the direction of the axis in the second dimension.

    References
    ----------
    .. [1] Baker, R. (2013). Measuring walking : a handbook of clinical gait
            analysis. Mac Keith Press.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import ankleJointCenter
    >>> vsk = { 'RightAnkleWidth' : 70.0, 'LeftAnkleWidth' : 70.0,
    ...         'RightTibialTorsion': 0.0, 'LeftTibialTorsion' : 0.0}
    >>> frame = { 'RTIB': np.array([433.97537231, 211.93408203, 273.3008728]),
    ...           'LTIB': np.array([50.04016495, 235.90718079, 364.32226562]),
    ...           'RANK': np.array([422.77005005, 217.74053955, 92.86152649]),
    ...           'LANK': np.array([58.57380676, 208.54806519, 86.16953278]) }
    >>> knee_JC = [np.array([364.17774614, 292.17051722, 515.19181496]),
    ...           np.array([143.55478579, 279.90370346, 524.78408753]),
    ...           np.array([[[364.64959153, 293.06758353, 515.18513093],
    ...           [363.29019771, 292.60656648, 515.04309095],
    ...           [364.04724541, 292.24216264, 516.18067112]],
    ...           [[143.65611282, 280.88685896, 524.63197541],
    ...           [142.56434499, 280.01777943, 524.86163553],
    ...            [143.64837987, 280.04650381, 525.76940383]]])]
    >>> delta = 0
    >>> ankleJointCenter(frame,knee_JC,delta,vsk) #doctest: +NORMALIZE_WHITESPACE
    [array([393.76181609, 247.67829633,  87.73775041]),
    array([ 98.74901939, 219.46930221,  80.63068161]),
    [[array([394.48171575, 248.37201349,  87.715368  ]),
    array([393.07114385, 248.39110006,  87.61575574]),
    array([393.69314056, 247.78157916,  88.73002876])],
    [array([ 98.47494966, 220.42553804,  80.52821783]),
    array([ 97.79246671, 219.20927276,  80.76255902]),
    array([ 98.84848169, 219.60345781,  81.61663776])]]]
    """

    #Get Global Values
    R_ankleWidth = vsk['RightAnkleWidth']
    L_ankleWidth = vsk['LeftAnkleWidth']
    R_torsion = vsk['RightTibialTorsion']
    L_torsion = vsk['LeftTibialTorsion']
    mm = 7.0
    R_delta = ((R_ankleWidth)/2.0)+mm
    L_delta = ((L_ankleWidth)/2.0)+mm

    #REQUIRED MARKERS:
    # tib_R
    # tib_L
    # ank_R
    # ank_L
    # knee_JC

    tib_R = frame['RTIB']
    tib_L = frame['LTIB']
    ank_R = frame['RANK']
    ank_L = frame['LANK']

    knee_JC_R = knee_JC[0]
    knee_JC_L = knee_JC[1]

    # This is Torsioned Tibia and this describe the ankle angles
    # Tibial frontal plane being defined by ANK,TIB and KJC

    # Determine the position of ankleJointCenter using findJointC function
    R = findJointC(tib_R, knee_JC_R, ank_R, R_delta)
    L = findJointC(tib_L, knee_JC_L, ank_L, L_delta)

    # Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)
        #Right axis calculation

    # Z axis is shank bone calculated by the ankleJC and  kneeJC
    axis_z = knee_JC_R-R

    # X axis is perpendicular to the points plane which is determined by ANK,TIB and KJC markers.
    # and calculated by each point's vector cross vector.
    # tib_ank_R vector is making a tibia plane to be assumed as rigid segment.
    tib_ank_R = tib_R-ank_R
    axis_x = cross(axis_z,tib_ank_R)

    # Y axis is determined by cross product of axis_z and axis_x.
    axis_y = cross(axis_z,axis_x)

    Raxis = [axis_x,axis_y,axis_z]

        #Left axis calculation

    # Z axis is shank bone calculated by the ankleJC and  kneeJC
    axis_z = knee_JC_L-L

    # X axis is perpendicular to the points plane which is determined by ANK,TIB and KJC markers.
    # and calculated by each point's vector cross vector.
    # tib_ank_L vector is making a tibia plane to be assumed as rigid segment.
    tib_ank_L = tib_L-ank_L
    axis_x = cross(tib_ank_L,axis_z)

    # Y axis is determined by cross product of axis_z and axis_x.
    axis_y = cross(axis_z,axis_x)

    Laxis = [axis_x,axis_y,axis_z]

    # Clear the name of axis and then normalize it.
    R_ankle_x_axis = Raxis[0]
    R_ankle_x_axis_div = norm2d(R_ankle_x_axis)
    R_ankle_x_axis = [R_ankle_x_axis[0]/R_ankle_x_axis_div,R_ankle_x_axis[1]/R_ankle_x_axis_div,R_ankle_x_axis[2]/R_ankle_x_axis_div]

    R_ankle_y_axis = Raxis[1]
    R_ankle_y_axis_div = norm2d(R_ankle_y_axis)
    R_ankle_y_axis = [R_ankle_y_axis[0]/R_ankle_y_axis_div,R_ankle_y_axis[1]/R_ankle_y_axis_div,R_ankle_y_axis[2]/R_ankle_y_axis_div]

    R_ankle_z_axis = Raxis[2]
    R_ankle_z_axis_div = norm2d(R_ankle_z_axis)
    R_ankle_z_axis = [R_ankle_z_axis[0]/R_ankle_z_axis_div,R_ankle_z_axis[1]/R_ankle_z_axis_div,R_ankle_z_axis[2]/R_ankle_z_axis_div]

    L_ankle_x_axis = Laxis[0]
    L_ankle_x_axis_div = norm2d(L_ankle_x_axis)
    L_ankle_x_axis = [L_ankle_x_axis[0]/L_ankle_x_axis_div,L_ankle_x_axis[1]/L_ankle_x_axis_div,L_ankle_x_axis[2]/L_ankle_x_axis_div]

    L_ankle_y_axis = Laxis[1]
    L_ankle_y_axis_div = norm2d(L_ankle_y_axis)
    L_ankle_y_axis = [L_ankle_y_axis[0]/L_ankle_y_axis_div,L_ankle_y_axis[1]/L_ankle_y_axis_div,L_ankle_y_axis[2]/L_ankle_y_axis_div]

    L_ankle_z_axis = Laxis[2]
    L_ankle_z_axis_div = norm2d(L_ankle_z_axis)
    L_ankle_z_axis = [L_ankle_z_axis[0]/L_ankle_z_axis_div,L_ankle_z_axis[1]/L_ankle_z_axis_div,L_ankle_z_axis[2]/L_ankle_z_axis_div]


    #Put both axis in array
    Raxis = [R_ankle_x_axis,R_ankle_y_axis,R_ankle_z_axis]
    Laxis = [L_ankle_x_axis,L_ankle_y_axis,L_ankle_z_axis]

    # Rotate the axes about the tibia torsion.
    R_torsion = np.radians(R_torsion)
    L_torsion = np.radians(L_torsion)

    Raxis = [[math.cos(R_torsion)*Raxis[0][0]-math.sin(R_torsion)*Raxis[1][0],
            math.cos(R_torsion)*Raxis[0][1]-math.sin(R_torsion)*Raxis[1][1],
            math.cos(R_torsion)*Raxis[0][2]-math.sin(R_torsion)*Raxis[1][2]],
            [math.sin(R_torsion)*Raxis[0][0]+math.cos(R_torsion)*Raxis[1][0],
            math.sin(R_torsion)*Raxis[0][1]+math.cos(R_torsion)*Raxis[1][1],
            math.sin(R_torsion)*Raxis[0][2]+math.cos(R_torsion)*Raxis[1][2]],
            [Raxis[2][0],Raxis[2][1],Raxis[2][2]]]

    Laxis = [[math.cos(L_torsion)*Laxis[0][0]-math.sin(L_torsion)*Laxis[1][0],
            math.cos(L_torsion)*Laxis[0][1]-math.sin(L_torsion)*Laxis[1][1],
            math.cos(L_torsion)*Laxis[0][2]-math.sin(L_torsion)*Laxis[1][2]],
            [math.sin(L_torsion)*Laxis[0][0]+math.cos(L_torsion)*Laxis[1][0],
            math.sin(L_torsion)*Laxis[0][1]+math.cos(L_torsion)*Laxis[1][1],
            math.sin(L_torsion)*Laxis[0][2]+math.cos(L_torsion)*Laxis[1][2]],
            [Laxis[2][0],Laxis[2][1],Laxis[2][2]]]

    # Add the origin back to the vector
    x_axis = Raxis[0]+R
    y_axis = Raxis[1]+R
    z_axis = Raxis[2]+R
    Raxis = [x_axis,y_axis,z_axis]

    x_axis = Laxis[0]+L
    y_axis = Laxis[1]+L
    z_axis = Laxis[2]+L
    Laxis = [x_axis,y_axis,z_axis]

    # Both of axis in array.
    axis = [Raxis,Laxis]

    return [R,L,axis]

def footJointCenter(frame,vsk,ankle_JC,knee_JC,delta):
    """Calculate the foot joint center and axis.

    Takes in a dictionary of marker names to x, y, z positions, subject
    measurements from the vsk file and the ankle joint center.
    Calculate the foot joint axis by rotating incorrect foot joint axes about
    offset angle.

    In case of foot joint center, we've already make 2 kinds of axis for static offset angle.
    and then, Call this static offset angle as an input of this function for dynamic trial.

    Special Cases:

    (anatomical uncorrect foot axis)
    if foot flat is checked, make the reference markers instead of HEE marker which height is as same as TOE marker's height.
    elif foot flat is not checked, use the HEE marker for making Z axis.

    Markers used: RTOE, LTOE
    Other landmarks used: ANKLE_FLEXION_AXIS
    Subject Measurement values used: RightStaticRotOff, RightStaticPlantFlex, LeftStaticRotOff, LeftStaticPlantFlex

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.
    vsk : dict
        A dictionary containing subject measurements.
    ankle_JC : array
        An array of ankle_JC containing the x,y,z axes marker positions of the ankle joint center.
    knee_JC : array
        An array of knee_JC containing the x,y,z axes marker positions of the knee joint center.
    delta
        The length from marker to joint center, retrieved from subject measurement file.

    Returns
    -------
    R, L, foot_axis : array
        Returns an array that contains the foot axis center in a 1x3 array of xyz values,
        which is then followed by a 2x3x3 array composed of the foot axis center x, y, and z
        axis components. The xyz axis components are 2x3 arrays consisting of the axis center
        in the first dimension and the direction of the axis in the second dimension.
        This function also saves the static offset angle in a global variable.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import footJointCenter
    >>> vsk = { 'RightStaticRotOff' : 0.015683497632642047, 'LeftStaticRotOff': 0.009402910292403012,
    ...         'RightStaticPlantFlex' : 0.2702417907002758, 'LeftStaticPlantFlex': 0.20251085737834015}
    >>> frame = { 'RHEE': np.array([374.01257324, 181.57929993, 49.50960922]),
    ...           'LHEE': np.array([105.30126953, 180.2130127, 47.15660858]),
    ...           'RTOE': np.array([442.81997681, 381.62280273, 42.66047668]),
    ...           'LTOE': np.array([39.43652725, 382.44522095, 41.78911591])}
    >>> knee_JC = [np.array([364.17774614, 292.17051722, 515.19181496]),
    ...           np.array([143.55478579, 279.90370346, 524.78408753]),
    ...           np.array([[[364.64959153, 293.06758353, 515.18513093],
    ...           [363.29019771, 292.60656648, 515.04309095],
    ...           [364.04724541, 292.24216264, 516.18067112]],
    ...           [[143.65611282, 280.88685896, 524.63197541],
    ...           [142.56434499, 280.01777943, 524.86163553],
    ...           [143.64837987, 280.04650381, 525.76940383]]])]
    >>> ankle_JC = [np.array([393.76181608, 247.67829633, 87.73775041]),
    ...            np.array([98.74901939, 219.46930221, 80.6306816]),
    ...            [[np.array([394.4817575, 248.37201348, 87.715368]),
    ...            np.array([393.07114384, 248.39110006, 87.61575574]),
    ...            np.array([393.69314056, 247.78157916, 88.73002876])],
    ...            [np.array([98.47494966, 220.42553803, 80.52821783]),
    ...            np.array([97.79246671, 219.20927275, 80.76255901]),
    ...            np.array([98.84848169, 219.60345781, 81.61663775])]]]
    >>> delta = 0
    >>> [np.around(arr,8) for arr in footJointCenter(frame,vsk,ankle_JC,knee_JC,delta)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.81997681, 381.62280273,  42.66047668]),
    array([ 39.43652725, 382.44522095,  41.78911591]),
    array([[[442.84624127, 381.6513024 ,  43.65972537],
            [441.87735057, 381.9563035 ,  42.67574106],
            [442.48716163, 380.68048378,  42.69610043]],
           [[ 39.56652626, 382.50901001,  42.77857597],
            [ 38.49313328, 382.14606841,  41.93234851],
            [ 39.74166341, 381.4931502 ,  41.81040459]]])]
    """

      #REQUIRED MARKERS:
      # RTOE
      # LTOE

    TOE_R = frame["RTOE"]
    TOE_L = frame["LTOE"]

    #REQUIRE JOINT CENTER & AXIS
    #KNEE JOINT CENTER
    #ANKLE JOINT CENTER
    #ANKLE FLEXION AXIS

    ankle_JC_R = ankle_JC[0]
    ankle_JC_L = ankle_JC[1]
    ankle_flexion_R = ankle_JC[2][0][1]
    ankle_flexion_L = ankle_JC[2][1][1]

    # Toe axis's origin is marker position of TOE
    R = TOE_R
    L = TOE_L

    # HERE IS THE INCORRECT AXIS

    # the first setting, the foot axis show foot uncorrected anatomical axis and static_info is None
    ankle_JC_R = [ankle_JC_R[0],ankle_JC_R[1],ankle_JC_R[2]]
    ankle_JC_L = [ankle_JC_L[0],ankle_JC_L[1],ankle_JC_L[2]]

    # Right

    # z axis is from TOE marker to AJC. and normalized it.
    R_axis_z = [ankle_JC_R[0]-TOE_R[0],ankle_JC_R[1]-TOE_R[1],ankle_JC_R[2]-TOE_R[2]]
    R_axis_z_div = norm2d(R_axis_z)
    R_axis_z = [R_axis_z[0]/R_axis_z_div,R_axis_z[1]/R_axis_z_div,R_axis_z[2]/R_axis_z_div]

    # bring the flexion axis of ankle axes from AnkleJointCenter function. and normalized it.
    y_flex_R = [ankle_flexion_R[0]-ankle_JC_R[0],ankle_flexion_R[1]-ankle_JC_R[1],ankle_flexion_R[2]-ankle_JC_R[2]]
    y_flex_R_div = norm2d(y_flex_R)
    y_flex_R = [y_flex_R[0]/y_flex_R_div,y_flex_R[1]/y_flex_R_div,y_flex_R[2]/y_flex_R_div]

    # x axis is calculated as a cross product of z axis and ankle flexion axis.
    R_axis_x = cross(y_flex_R,R_axis_z)
    R_axis_x_div = norm2d(R_axis_x)
    R_axis_x = [R_axis_x[0]/R_axis_x_div,R_axis_x[1]/R_axis_x_div,R_axis_x[2]/R_axis_x_div]

    # y axis is then perpendicularly calculated from z axis and x axis. and normalized.
    R_axis_y = cross(R_axis_z,R_axis_x)
    R_axis_y_div = norm2d(R_axis_y)
    R_axis_y = [R_axis_y[0]/R_axis_y_div,R_axis_y[1]/R_axis_y_div,R_axis_y[2]/R_axis_y_div]

    R_foot_axis = [R_axis_x,R_axis_y,R_axis_z]

    # Left

    # z axis is from TOE marker to AJC. and normalized it.
    L_axis_z = [ankle_JC_L[0]-TOE_L[0],ankle_JC_L[1]-TOE_L[1],ankle_JC_L[2]-TOE_L[2]]
    L_axis_z_div = norm2d(L_axis_z)
    L_axis_z = [L_axis_z[0]/L_axis_z_div,L_axis_z[1]/L_axis_z_div,L_axis_z[2]/L_axis_z_div]

    # bring the flexion axis of ankle axes from AnkleJointCenter function. and normalized it.
    y_flex_L = [ankle_flexion_L[0]-ankle_JC_L[0],ankle_flexion_L[1]-ankle_JC_L[1],ankle_flexion_L[2]-ankle_JC_L[2]]
    y_flex_L_div = norm2d(y_flex_L)
    y_flex_L = [y_flex_L[0]/y_flex_L_div,y_flex_L[1]/y_flex_L_div,y_flex_L[2]/y_flex_L_div]

    # x axis is calculated as a cross product of z axis and ankle flexion axis.
    L_axis_x = cross(y_flex_L,L_axis_z)
    L_axis_x_div = norm2d(L_axis_x)
    L_axis_x = [L_axis_x[0]/L_axis_x_div,L_axis_x[1]/L_axis_x_div,L_axis_x[2]/L_axis_x_div]

    # y axis is then perpendicularly calculated from z axis and x axis. and normalized.
    L_axis_y = cross(L_axis_z,L_axis_x)
    L_axis_y_div = norm2d(L_axis_y)
    L_axis_y = [L_axis_y[0]/L_axis_y_div,L_axis_y[1]/L_axis_y_div,L_axis_y[2]/L_axis_y_div]

    L_foot_axis = [L_axis_x,L_axis_y,L_axis_z]

    foot_axis = [R_foot_axis,L_foot_axis]

    # Apply static offset angle to the incorrect foot axes

    # static offset angle are taken from static_info variable in radians.
    R_alpha = vsk['RightStaticRotOff']
    R_beta = vsk['RightStaticPlantFlex']
    #R_gamma = static_info[0][2]
    L_alpha = vsk['LeftStaticRotOff']
    L_beta = vsk['LeftStaticPlantFlex']
    #L_gamma = static_info[1][2]

    R_alpha = np.around(math.degrees(R_alpha),decimals=5)
    R_beta = np.around(math.degrees(R_beta),decimals=5)
    #R_gamma = np.around(math.degrees(static_info[0][2]),decimals=5)
    L_alpha = np.around(math.degrees(L_alpha),decimals=5)
    L_beta = np.around(math.degrees(L_beta),decimals=5)
    #L_gamma = np.around(math.degrees(static_info[1][2]),decimals=5)

    R_alpha = -math.radians(R_alpha)
    R_beta = math.radians(R_beta)
    #R_gamma = 0
    L_alpha = math.radians(L_alpha)
    L_beta = math.radians(L_beta)
    #L_gamma = 0

    R_axis = [[(R_foot_axis[0][0]),(R_foot_axis[0][1]),(R_foot_axis[0][2])],
              [(R_foot_axis[1][0]),(R_foot_axis[1][1]),(R_foot_axis[1][2])],
              [(R_foot_axis[2][0]),(R_foot_axis[2][1]),(R_foot_axis[2][2])]]

    L_axis = [[(L_foot_axis[0][0]),(L_foot_axis[0][1]),(L_foot_axis[0][2])],
              [(L_foot_axis[1][0]),(L_foot_axis[1][1]),(L_foot_axis[1][2])],
              [(L_foot_axis[2][0]),(L_foot_axis[2][1]),(L_foot_axis[2][2])]]

    # rotate incorrect foot axis around y axis first.

    # right
    R_rotmat = [[(math.cos(R_beta)*R_axis[0][0]+math.sin(R_beta)*R_axis[2][0]),
                (math.cos(R_beta)*R_axis[0][1]+math.sin(R_beta)*R_axis[2][1]),
                (math.cos(R_beta)*R_axis[0][2]+math.sin(R_beta)*R_axis[2][2])],
                [R_axis[1][0],R_axis[1][1],R_axis[1][2]],
                [(-1*math.sin(R_beta)*R_axis[0][0]+math.cos(R_beta)*R_axis[2][0]),
                (-1*math.sin(R_beta)*R_axis[0][1]+math.cos(R_beta)*R_axis[2][1]),
                (-1*math.sin(R_beta)*R_axis[0][2]+math.cos(R_beta)*R_axis[2][2])]]
    # left
    L_rotmat = [[(math.cos(L_beta)*L_axis[0][0]+math.sin(L_beta)*L_axis[2][0]),
                (math.cos(L_beta)*L_axis[0][1]+math.sin(L_beta)*L_axis[2][1]),
                (math.cos(L_beta)*L_axis[0][2]+math.sin(L_beta)*L_axis[2][2])],
                [L_axis[1][0],L_axis[1][1],L_axis[1][2]],
                [(-1*math.sin(L_beta)*L_axis[0][0]+math.cos(L_beta)*L_axis[2][0]),
                (-1*math.sin(L_beta)*L_axis[0][1]+math.cos(L_beta)*L_axis[2][1]),
                (-1*math.sin(L_beta)*L_axis[0][2]+math.cos(L_beta)*L_axis[2][2])]]

    # rotate incorrect foot axis around x axis next.

    # right
    R_rotmat = [[R_rotmat[0][0],R_rotmat[0][1],R_rotmat[0][2]],
                [(math.cos(R_alpha)*R_rotmat[1][0]-math.sin(R_alpha)*R_rotmat[2][0]),
                (math.cos(R_alpha)*R_rotmat[1][1]-math.sin(R_alpha)*R_rotmat[2][1]),
                (math.cos(R_alpha)*R_rotmat[1][2]-math.sin(R_alpha)*R_rotmat[2][2])],
                [(math.sin(R_alpha)*R_rotmat[1][0]+math.cos(R_alpha)*R_rotmat[2][0]),
                (math.sin(R_alpha)*R_rotmat[1][1]+math.cos(R_alpha)*R_rotmat[2][1]),
                (math.sin(R_alpha)*R_rotmat[1][2]+math.cos(R_alpha)*R_rotmat[2][2])]]

    # left
    L_rotmat = [[L_rotmat[0][0],L_rotmat[0][1],L_rotmat[0][2]],
                [(math.cos(L_alpha)*L_rotmat[1][0]-math.sin(L_alpha)*L_rotmat[2][0]),
                (math.cos(L_alpha)*L_rotmat[1][1]-math.sin(L_alpha)*L_rotmat[2][1]),
                (math.cos(L_alpha)*L_rotmat[1][2]-math.sin(L_alpha)*L_rotmat[2][2])],
                [(math.sin(L_alpha)*L_rotmat[1][0]+math.cos(L_alpha)*L_rotmat[2][0]),
                (math.sin(L_alpha)*L_rotmat[1][1]+math.cos(L_alpha)*L_rotmat[2][1]),
                (math.sin(L_alpha)*L_rotmat[1][2]+math.cos(L_alpha)*L_rotmat[2][2])]]

    # Bring each x,y,z axis from rotation axes
    R_axis_x = R_rotmat[0]
    R_axis_y = R_rotmat[1]
    R_axis_z = R_rotmat[2]
    L_axis_x = L_rotmat[0]
    L_axis_y = L_rotmat[1]
    L_axis_z = L_rotmat[2]

    # Attach each axis to the origin
    R_axis_x = [R_axis_x[0]+R[0],R_axis_x[1]+R[1],R_axis_x[2]+R[2]]
    R_axis_y = [R_axis_y[0]+R[0],R_axis_y[1]+R[1],R_axis_y[2]+R[2]]
    R_axis_z = [R_axis_z[0]+R[0],R_axis_z[1]+R[1],R_axis_z[2]+R[2]]

    R_foot_axis = [R_axis_x,R_axis_y,R_axis_z]

    L_axis_x = [L_axis_x[0]+L[0],L_axis_x[1]+L[1],L_axis_x[2]+L[2]]
    L_axis_y = [L_axis_y[0]+L[0],L_axis_y[1]+L[1],L_axis_y[2]+L[2]]
    L_axis_z = [L_axis_z[0]+L[0],L_axis_z[1]+L[1],L_axis_z[2]+L[2]]

    L_foot_axis = [L_axis_x,L_axis_y,L_axis_z]

    foot_axis = [R_foot_axis,L_foot_axis]

    return [R,L,foot_axis]


# Upperbody Coordinate System

def headJC(frame,vsk=None):
    """Calculate the head joint axis function.

    Takes in a dictionary of marker names to x, y, z positions.
    Calculates the head joint center and returns the head joint center and axis.

    Markers used: LFHD, RFHD, LBHD, RBHD
    Subject Measurement values used: HeadOffset

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.
    vsk : dict, optional
        A dictionary containing subject measurements.

    Returns
    -------
    head_axis, origin : array
        Returns an array containing a 1x3x3 array containing the x, y, z axis
        components of the head joint center, and a 1x3 array containing the
        head origin x, y, z position.


    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import headJC
    >>> vsk = { 'HeadOffset': 0.2571990469310653 }
    >>> frame = {'RFHD': np.array([325.82983398, 402.55450439, 1722.49816895]),
    ...          'LFHD': np.array([184.55158997, 409.68713379, 1721.34289551]),
    ...          'RBHD': np.array([304.39898682, 242.91339111, 1694.97497559]),
    ...          'LBHD': np.array([197.8621521, 251.28889465, 1696.90197754])}
    >>> [np.around(arr,8) for arr in headJC(frame,vsk)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 255.21685583,  407.11593888, 1721.82538439],
           [ 254.19105385,  406.14680918, 1721.91767712],
           [ 255.1903437 ,  406.21600904, 1722.91599129]]),
    array([ 255.19071198,  406.12081909, 1721.92053223])]
    """

    #Get Global Values
    head_off = vsk['HeadOffset']
    head_off = -1*head_off

    #Get the marker positions used for joint calculation
    LFHD = frame['LFHD']
    RFHD = frame['RFHD']
    LBHD = frame['LBHD']
    RBHD = frame['RBHD']

    #get the midpoints of the head to define the sides
    front = [(LFHD[0]+RFHD[0])/2.0, (LFHD[1]+RFHD[1])/2.0,(LFHD[2]+RFHD[2])/2.0]
    back = [(LBHD[0]+RBHD[0])/2.0, (LBHD[1]+RBHD[1])/2.0,(LBHD[2]+RBHD[2])/2.0]
    left = [(LFHD[0]+LBHD[0])/2.0, (LFHD[1]+LBHD[1])/2.0,(LFHD[2]+LBHD[2])/2.0]
    right = [(RFHD[0]+RBHD[0])/2.0, (RFHD[1]+RBHD[1])/2.0,(RFHD[2]+RBHD[2])/2.0]
    origin = front

    #Get the vectors from the sides with primary x axis facing front
    #First get the x direction
    x_vec = [front[0]-back[0],front[1]-back[1],front[2]-back[2]]
    x_vec_div = norm2d(x_vec)
    x_vec = [x_vec[0]/x_vec_div,x_vec[1]/x_vec_div,x_vec[2]/x_vec_div]

    #get the direction of the y axis
    y_vec = [left[0]-right[0],left[1]-right[1],left[2]-right[2]]
    y_vec_div = norm2d(y_vec)
    y_vec = [y_vec[0]/y_vec_div,y_vec[1]/y_vec_div,y_vec[2]/y_vec_div]

    # get z axis by cross-product of x axis and y axis.
    z_vec = cross(x_vec,y_vec)
    z_vec_div = norm2d(z_vec)
    z_vec = [z_vec[0]/z_vec_div,z_vec[1]/z_vec_div,z_vec[2]/z_vec_div]

    # make sure all x,y,z axis is orthogonal each other by cross-product
    y_vec = cross(z_vec,x_vec)
    y_vec_div = norm2d(y_vec)
    y_vec = [y_vec[0]/y_vec_div,y_vec[1]/y_vec_div,y_vec[2]/y_vec_div]
    x_vec = cross(y_vec,z_vec)
    x_vec_div = norm2d(x_vec)
    x_vec = [x_vec[0]/x_vec_div,x_vec[1]/x_vec_div,x_vec[2]/x_vec_div]

    # rotate the head axis around y axis about head offset angle.
    x_vec_rot = [x_vec[0]*math.cos(head_off)+z_vec[0]*math.sin(head_off),
            x_vec[1]*math.cos(head_off)+z_vec[1]*math.sin(head_off),
            x_vec[2]*math.cos(head_off)+z_vec[2]*math.sin(head_off)]
    y_vec_rot = [y_vec[0],y_vec[1],y_vec[2]]
    z_vec_rot = [x_vec[0]*-1*math.sin(head_off)+z_vec[0]*math.cos(head_off),
            x_vec[1]*-1*math.sin(head_off)+z_vec[1]*math.cos(head_off),
            x_vec[2]*-1*math.sin(head_off)+z_vec[2]*math.cos(head_off)]

    #Add the origin back to the vector to get it in the right position
    x_axis = [x_vec_rot[0]+origin[0],x_vec_rot[1]+origin[1],x_vec_rot[2]+origin[2]]
    y_axis = [y_vec_rot[0]+origin[0],y_vec_rot[1]+origin[1],y_vec_rot[2]+origin[2]]
    z_axis = [z_vec_rot[0]+origin[0],z_vec_rot[1]+origin[1],z_vec_rot[2]+origin[2]]

    head_axis =[x_axis,y_axis,z_axis]

    #Return the three axis and origin
    return [head_axis,origin]

def thoraxJC(frame):
    """Calculate the thorax joint axis function.

    Takes in a dictionary of marker names to x, y, z positions.
    Calculates the thorax joint center and returns the thorax joint center and axis.

    Markers used: CLAV, C7, STRN, T10

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.

    Returns
    -------
    thorax_axis, origin : array
        Returns an array which contains a 2x3 array representing the right thorax joint center 1x3
        and the left thorax joint center 1x3, which is then followed by a 6x3 array representing the
        right thorax x, y, z axis components 3x3 followed by the the left thorax x, y, z axis components 3x3.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import thoraxJC
    >>> frame = {'C7': np.array([256.78051758, 371.28042603, 1459.70300293]),
    ...          'T10': np.array([228.64323425, 192.32041931, 1279.6418457]),
    ...          'CLAV': np.array([256.78051758, 371.28042603, 1459.70300293]),
    ...          'STRN': np.array([251.67492676, 414.10391235, 1292.08508301])}
    >>> [np.around(arr,8) for arr in thoraxJC(frame)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 256.34546332,  365.72239585, 1461.92089119],
           [ 257.26637166,  364.696025  , 1462.23472346],
           [ 256.18427318,  364.43288984, 1461.36304534]]),
    array([ 256.27295428,  364.79605749, 1462.29053923])]
    """

    #Set or get a marker size as mm
    marker_size = (14.0) /2.0

    #Get the marker positions used for joint calculation
    CLAV = frame['CLAV']
    C7 = frame['C7']
    STRN = frame['STRN']
    T10 = frame['T10']

    #Temporary origin since the origin will be moved at the end
    origin = CLAV

    #Get the midpoints of the upper and lower sections, as well as the front and back sections
    upper = [(CLAV[0]+C7[0])/2.0,(CLAV[1]+C7[1])/2.0,(CLAV[2]+C7[2])/2.0]
    lower = [(STRN[0]+T10[0])/2.0,(STRN[1]+T10[1])/2.0,(STRN[2]+T10[2])/2.0]
    front = [(CLAV[0]+STRN[0])/2.0,(CLAV[1]+STRN[1])/2.0,(CLAV[2]+STRN[2])/2.0]
    back = [(T10[0]+C7[0])/2.0,(T10[1]+C7[1])/2.0,(T10[2]+C7[2])/2.0]



    C7_CLAV = [C7[0]-CLAV[0],C7[1]-CLAV[1],C7[2]-CLAV[2]]
    C7_CLAV = C7_CLAV/norm3d(C7_CLAV)

    #Get the direction of the primary axis Z (facing down)
    z_direc = [lower[0]-upper[0],lower[1]-upper[1],lower[2]-upper[2]]
    z_vec = z_direc/norm3d(z_direc)

    #The secondary axis X is from back to front
    x_direc = [front[0]-back[0],front[1]-back[1],front[2]-back[2]]
    x_vec = x_direc/norm3d(x_direc)

    # make sure all the axes are orthogonal each othe by cross-product
    y_direc = cross(z_vec,x_vec)
    y_vec = y_direc/norm3d(y_direc)
    x_direc = cross(y_vec,z_vec)
    x_vec = x_direc/norm3d(x_direc)
    z_direc = cross(x_vec,y_vec)
    z_vec = z_direc/norm3d(z_direc)

    # move the axes about offset along the x axis.
    offset = [x_vec[0]*marker_size,x_vec[1]*marker_size,x_vec[2]*marker_size]

    #Add the CLAV back to the vector to get it in the right position before translating it
    origin = [CLAV[0]-offset[0],CLAV[1]-offset[1],CLAV[2]-offset[2]]

    # Attach all the axes to the origin.
    x_axis = [x_vec[0]+origin[0],x_vec[1]+origin[1],x_vec[2]+origin[2]]
    y_axis = [y_vec[0]+origin[0],y_vec[1]+origin[1],y_vec[2]+origin[2]]
    z_axis = [z_vec[0]+origin[0],z_vec[1]+origin[1],z_vec[2]+origin[2]]

    thorax_axis = [x_axis,y_axis,z_axis]

    return [thorax_axis,origin]

def findwandmarker(frame,thorax):
    """Calculate the wand marker function.

    Takes in a dictionary of marker names to x, y, z positions and the thorax axis.
    Calculates the wand marker for calculating the clavicle.

    Markers used: RSHO, LSHO

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.
    thorax : array
        The x,y,z position of the thorax.

    Returns
    -------
    wand : array
        Returns wand marker position for calculating knee joint center later.
        The wand marker position is returned as a 2x3 array containing the
        right wand marker x,y,z positions 1x3 followed by the left
        wand marker x,y,z positions 1x3.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import findwandmarker
    >>> frame = {'RSHO': np.array([428.88496562, 270.552948, 1500.73010254]),
    ...          'LSHO': np.array([68.24668121, 269.01049805, 1510.1072998])}
    >>> thorax = [[[256.23991128535846, 365.30496976939753, 1459.662169500559],
    ...          [257.1435863244796, 364.21960599061947, 1459.5889787129829],
    ...          [256.08430536580352, 354.32180498523223, 1458.6575930699294]],
    ...          [256.14981023656401, 364.30906039339868, 1459.6553639290375]]
    >>> [np.around(arr,8) for arr in findwandmarker(frame,thorax)]
    [array([ 255.92550246,  364.32269503, 1460.6297869 ]), array([ 256.42380097,  364.27770361, 1460.61658494])]
    """
    thorax_origin = thorax[1]

    tho_axis_x = thorax[0][0]


    #REQUIRED MARKERS:
    # RSHO
    # LSHO

    RSHO = frame['RSHO']
    LSHO = frame['LSHO']

    # Calculate for getting a wand marker

    # bring x axis from thorax axis
    axis_x_vec = [tho_axis_x[0]-thorax_origin[0],tho_axis_x[1]-thorax_origin[1],tho_axis_x[2]-thorax_origin[2]]
    axis_x_vec = axis_x_vec/norm3d(axis_x_vec)

    RSHO_vec = [RSHO[0]-thorax_origin[0],RSHO[1]-thorax_origin[1],RSHO[2]-thorax_origin[2]]
    LSHO_vec = [LSHO[0]-thorax_origin[0],LSHO[1]-thorax_origin[1],LSHO[2]-thorax_origin[2]]
    RSHO_vec = RSHO_vec/norm3d(RSHO_vec)
    LSHO_vec = LSHO_vec/norm3d(LSHO_vec)

    R_wand = cross(RSHO_vec,axis_x_vec)
    R_wand = R_wand/norm3d(R_wand)
    R_wand = [thorax_origin[0]+R_wand[0],
            thorax_origin[1]+R_wand[1],
            thorax_origin[2]+R_wand[2]]

    L_wand = cross(axis_x_vec,LSHO_vec)
    L_wand = L_wand/norm3d(L_wand)
    L_wand = [thorax_origin[0]+L_wand[0],
            thorax_origin[1]+L_wand[1],
            thorax_origin[2]+L_wand[2]]
    wand = [R_wand,L_wand]

    return wand

def findshoulderJC(frame,thorax,wand,vsk=None):
    """Calculate the Shoulder joint center function.

    Takes in a dictionary of marker names to x, y, z positions and the thorax
    axis and wand marker.
    Calculate each shoulder joint center and returns it.

    Markers used: RSHO, LSHO
    Subject Measurement values used: RightShoulderOffset, LeftShoulderOffset

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.
    thorax : array
        Array containing several x,y,z markers for the thorax.
    wand : array
        Array containing two x,y,z markers for wand.
    vsk : dict, optional
        A dictionary containing subject measurements.

    Returns
    -------
    Sho_JC : array
        Returns a 2x3 array representing the right shoulder joint
        center x, y, z, marker positions 1x3 followed by the left
        shoulder joint center x, y, z, marker positions 1x3.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import findshoulderJC
    >>> vsk = { 'RightShoulderOffset' : 40.0, 'LeftShoulderOffset' : 40.0 }
    >>> frame = {'RSHO': np.array([428.88496562, 270.552948, 1500.73010254]),
    ...          'LSHO': np.array([68.24668121, 269.01049805, 1510.1072998])}
    >>> thorax = [[[256.23991128535846, 365.30496976939753, 1459.662169500559],
    ...        [257.1435863244796, 364.21960599061947, 1459.5889787129829],
    ...        [256.08430536580352, 354.32180498523223, 1458.6575930699294]],
    ...        [256.14981023656401, 364.30906039339868, 1459.6553639290375]]
    >>> wand = [[255.92550222678443, 364.32269504976051, 1460.6297868417887],
    ...        [256.42380097331767, 364.27770361353487, 1460.6165849382387]]
    >>> findshoulderJC(frame,thorax,wand,vsk)
    [array([ 429.66971693,  275.06718208, 1453.95397769]), array([  64.51952733,  274.93442161, 1463.63133339])]
    """

    thorax_origin = thorax[1]


    #Get Subject Measurement Values
    R_shoulderoffset = vsk['RightShoulderOffset']
    L_shoulderoffset = vsk['LeftShoulderOffset']
    mm = 7.0
    R_delta =( R_shoulderoffset + mm )
    L_delta =( L_shoulderoffset + mm )


    #REQUIRED MARKERS:
    # RSHO
    # LSHO
    RSHO = frame['RSHO']
    LSHO = frame['LSHO']

    # Calculate the shoulder joint center first.
    R_wand = wand[0]
    L_wand = wand[1]

    R_Sho_JC = findJointC(R_wand,thorax_origin,RSHO,R_delta)
    L_Sho_JC = findJointC(L_wand,thorax_origin,LSHO,L_delta)
    Sho_JC = [R_Sho_JC,L_Sho_JC]

    return Sho_JC

def shoulderAxisCalc(frame,thorax,shoulderJC,wand):
    """Calculate the Shoulder joint axis ( Clavicle) function.

    Takes in the thorax axis, wand marker and shoulder joint center.
    Calculate each shoulder joint axis and returns it.

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.
    thorax : array
        The x,y,z position of the thorax.
            thorax = [[R_thorax joint center x,y,z position],
                        [L_thorax_joint center x,y,z position],
                        [[R_thorax x axis x,y,z position],
                        [R_thorax,y axis x,y,z position],
                        [R_thorax z axis x,y,z position]]]
    shoulderJC : array
        The x,y,z position of the shoulder joint center.
            shoulderJC = [[R shoulder joint center x,y,z position],
                        [L shoulder joint center x,y,z position]]
    wand : array
        The x,y,z position of the wand.
            wand = [[R wand x,y,z, position],
                    [L wand x,y,z position]]

    Returns
    -------
    shoulderJC, axis : array
        Returns the Shoulder joint center and axis in three array
            shoulder_JC = [[[[R_shoulder x axis, x,y,z position],
                        [R_shoulder y axis, x,y,z position],
                        [R_shoulder z axis, x,y,z position]],
                        [[L_shoulder x axis, x,y,z position],
                        [L_shoulder y axis, x,y,z position],
                        [L_shoulder z axis, x,y,z position]]],
                        [R_shoulderJC_x, R_shoulderJC_y, R_shoulderJC_z],
                        [L_shoulderJC_x,L_shoulderJC_y,L_shoulderJC_z]]

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import shoulderAxisCalc
    >>> frame = None
    >>> thorax = [[[256.23991128535846, 365.30496976939753, 1459.662169500559],
    ...          [257.1435863244796, 364.21960599061947, 1459.5889787129829],
    ...          [256.08430536580352, 354.32180498523223, 1458.6575930699294]],
    ...          [256.14981023656401, 364.30906039339868, 1459.6553639290375]]
    >>> shoulderJC = [np.array([429.66951995, 275.06718615, 1453.953978131]),
    ...              np.array([64.51952734, 274.93442161, 1463.6313334])]
    >>> wand = [[255.92550222678443, 364.32269504976051, 1460.6297868417887],
    ...        [256.42380097331767, 364.27770361353487, 1460.6165849382387]]
    >>> [np.around(arr,8) for arr in shoulderAxisCalc(frame,thorax,shoulderJC,wand)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 429.66951995,  275.06718615, 1453.95397813],
           [  64.51952734,  274.93442161, 1463.6313334 ]]),
    array([[[ 430.12731331,  275.95136619, 1454.04698829],
            [ 429.68621685,  275.16323377, 1452.95874144],
            [ 428.78061813,  275.52435188, 1453.98318503]],
           [[  64.10400325,  275.83192827, 1463.77905455],
            [  64.59882849,  274.80838068, 1464.62018375],
            [  65.42564602,  275.3570272 , 1463.61253313]]])]
    """


    thorax_origin = thorax[1]

    R_shoulderJC = shoulderJC[0]
    L_shoulderJC = shoulderJC[1]

    R_wand = wand[0]
    L_wand = wand[1]
    R_wand_direc = [R_wand[0]-thorax_origin[0],R_wand[1]-thorax_origin[1],R_wand[2]-thorax_origin[2]]
    L_wand_direc = [L_wand[0]-thorax_origin[0],L_wand[1]-thorax_origin[1],L_wand[2]-thorax_origin[2]]
    R_wand_direc = R_wand_direc/norm3d(R_wand_direc)
    L_wand_direc = L_wand_direc/norm3d(L_wand_direc)

    # Right

    #Get the direction of the primary axis Z,X,Y
    z_direc = [(thorax_origin[0]-R_shoulderJC[0]),
            (thorax_origin[1]-R_shoulderJC[1]),
            (thorax_origin[2]-R_shoulderJC[2])]
    z_direc = z_direc/norm3d(z_direc)
    y_direc = [R_wand_direc[0]*-1,R_wand_direc[1]*-1,R_wand_direc[2]*-1]
    x_direc = cross(y_direc,z_direc)
    x_direc = x_direc/norm3d(x_direc)
    y_direc = cross(z_direc,x_direc)
    y_direc = y_direc/norm3d(y_direc)

    # backwards to account for marker size
    x_axis = [x_direc[0]+R_shoulderJC[0],x_direc[1]+R_shoulderJC[1],x_direc[2]+R_shoulderJC[2]]
    y_axis = [y_direc[0]+R_shoulderJC[0],y_direc[1]+R_shoulderJC[1],y_direc[2]+R_shoulderJC[2]]
    z_axis = [z_direc[0]+R_shoulderJC[0],z_direc[1]+R_shoulderJC[1],z_direc[2]+R_shoulderJC[2]]

    R_axis = [x_axis,y_axis,z_axis]

    # Left

    #Get the direction of the primary axis Z,X,Y
    z_direc = [(thorax_origin[0]-L_shoulderJC[0]),
            (thorax_origin[1]-L_shoulderJC[1]),
            (thorax_origin[2]-L_shoulderJC[2])]
    z_direc = z_direc/norm3d(z_direc)
    y_direc = L_wand_direc
    x_direc = cross(y_direc,z_direc)
    x_direc = x_direc/norm3d(x_direc)
    y_direc = cross(z_direc,x_direc)
    y_direc = y_direc/norm3d(y_direc)

    # backwards to account for marker size
    x_axis = [x_direc[0]+L_shoulderJC[0],x_direc[1]+L_shoulderJC[1],x_direc[2]+L_shoulderJC[2]]
    y_axis = [y_direc[0]+L_shoulderJC[0],y_direc[1]+L_shoulderJC[1],y_direc[2]+L_shoulderJC[2]]
    z_axis = [z_direc[0]+L_shoulderJC[0],z_direc[1]+L_shoulderJC[1],z_direc[2]+L_shoulderJC[2]]

    L_axis = [x_axis,y_axis,z_axis]

    axis = [R_axis,L_axis]

    return [shoulderJC,axis]

def elbowJointCenter(frame,thorax,shoulderJC,wand,vsk=None):
    """Calculate the Elbow joint axis ( Humerus) function.

    Takes in a dictionary of marker names to x, y, z positions, the thorax
    axis, wand marker and shoulder joint center.

    Calculates each elbow joint axis.

    Markers used: RSHO, LSHO, RELB, LELB, RWRA ,RWRB, LWRA, LWRB
    Subject Measurement values used: RightElbowWidth, LeftElbowWidth

    Parameters
    ----------
    frame
        Dictionaries of marker lists.
    thorax : array
        The x,y,z position of the thorax.
    shoulderJC : array
        The x,y,z position of the shoulder joint center.
    wand : array
        The x,y,z position of the wand.
    vsk : dict, optional
        A dictionary containing subject measurements.

    Returns
    -------
    origin, axis, wrist_O : array
        Returns an array containing a 2x3 array containing the right
        elbow x, y, z marker positions 1x3, and the left elbow x, y,
        z marker positions 1x3, which is followed by a 2x3x3 array containing
        right elbow x, y, z axis components (1x3x3) followed by the left x, y, z axis
        components (1x3x3) which is then followed by the right wrist joint center
        x, y, z marker positions 1x3, and the left wrist joint center x, y, z marker positions 1x3.


    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import elbowJointCenter
    >>> frame = {'RSHO': np.array([428.88496562, 270.552948, 1500.73010254]),
    ...          'LSHO': np.array([68.24668121, 269.01049805, 1510.1072998]),
    ...          'RELB': np.array([658.90338135, 326.07580566, 1285.28515625]),
    ...          'LELB': np.array([-156.32162476, 335.2593313, 1287.39916992]),
    ...          'RWRA': np.array([776.51898193,495.68103027, 1108.38464355]),
    ...          'RWRB': np.array([830.9072876, 436.75341797, 1119.11901855]),
    ...          'LWRA': np.array([-249.28146362, 525.32977295, 1117.09057617]),
    ...          'LWRB': np.array([-311.77532959, 477.22512817, 1125.1619873])}
    >>> thorax = [[[256.23991128535846, 365.30496976939753, 1459.662169500559],
    ...        [257.1435863244796, 364.21960599061947, 1459.5889787129829],
    ...        [256.08430536580352, 354.32180498523223, 1458.6575930699294]],
    ...        [256.14981023656401, 364.30906039339868, 1459.6553639290375]]
    >>> shoulderJC = [np.array([429.66951995, 275.06718615, 1453.953978131]),
    ...            np.array([64.51952734, 274.93442161, 1463.6313334])]
    >>> wand = [[255.92550222678443, 364.32269504976051, 1460.6297868417887],
    ...        [256.42380097331767, 364.27770361353487, 1460.6165849382387]]
    >>> vsk = { 'RightElbowWidth': 74.0, 'LeftElbowWidth': 74.0,
    ...         'RightWristWidth': 55.0, 'LeftWristWidth': 55.0}
    >>> [np.around(arr,8) for arr in elbowJointCenter(frame,thorax,shoulderJC,wand,vsk)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 633.66707588,  304.95542115, 1256.07799541],
           [-129.16966701,  316.86794653, 1258.06440971]]),
    array([[[ 633.81070139,  303.96579005, 1256.07658507],
            [ 634.35247992,  305.05386589, 1256.79947301],
            [ 632.95321804,  304.8508319 , 1256.77043175]],
           [[-129.32406616,  315.88151182, 1258.00866516],
            [-128.45131692,  316.79460332, 1257.37260488],
            [-128.4913352 ,  316.72108835, 1258.78433931]]]),
    array([[ 793.32814303,  451.29134788, 1084.4325513 ],
           [-272.45939135,  485.80149026, 1091.36664789]])]
    """

    RSHO = frame['RSHO']
    LSHO = frame['LSHO']
    RELB = frame['RELB']
    LELB = frame['LELB']
    RWRA = frame['RWRA']
    RWRB = frame['RWRB']
    LWRA = frame['LWRA']
    LWRB = frame['LWRB']


    R_elbowwidth = vsk['RightElbowWidth']
    L_elbowwidth = vsk['LeftElbowWidth']
    R_elbowwidth = R_elbowwidth * -1
    L_elbowwidth = L_elbowwidth
    mm = 7.0
    R_delta =( (R_elbowwidth/2.0)-mm )
    L_delta =( (L_elbowwidth/2.0)+mm )


    RWRI = [(RWRA[0]+RWRB[0])/2.0,(RWRA[1]+RWRB[1])/2.0,(RWRA[2]+RWRB[2])/2.0]
    LWRI = [(LWRA[0]+LWRB[0])/2.0,(LWRA[1]+LWRB[1])/2.0,(LWRA[2]+LWRB[2])/2.0]

    # make humerus axis
    tho_y_axis = np.subtract(thorax[0][1],thorax[1])

    R_sho_mod = [(RSHO[0]-R_delta*tho_y_axis[0]-RELB[0]),
                (RSHO[1]-R_delta*tho_y_axis[1]-RELB[1]),
                (RSHO[2]-R_delta*tho_y_axis[2]-RELB[2])]
    L_sho_mod = [(LSHO[0]+L_delta*tho_y_axis[0]-LELB[0]),
                (LSHO[1]+L_delta*tho_y_axis[1]-LELB[1]),
                (LSHO[2]+L_delta*tho_y_axis[2]-LELB[2])]

    # right axis
    z_axis = R_sho_mod
    z_axis_div = norm2d(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

        # this is reference axis
    x_axis = np.subtract(RWRI,RELB)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = cross(z_axis,x_axis)
    y_axis_div = norm2d(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = cross(y_axis,z_axis)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    R_axis = [x_axis,y_axis,z_axis]

    # left axis
    z_axis = np.subtract(L_sho_mod,LELB)
    z_axis_div = norm2d(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

        # this is reference axis
    x_axis = L_sho_mod
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = cross(z_axis,x_axis)
    y_axis_div = norm2d(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = cross(y_axis,z_axis)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    L_axis = [x_axis,y_axis,z_axis]

    RSJC = shoulderJC[0]
    LSJC = shoulderJC[1]

    # make the construction vector for finding Elbow joint center
    R_con_1 = np.subtract(RSJC,RELB)
    R_con_1_div = norm2d(R_con_1)
    R_con_1 = [R_con_1[0]/R_con_1_div,R_con_1[1]/R_con_1_div,R_con_1[2]/R_con_1_div]

    R_con_2 = np.subtract(RWRI,RELB)
    R_con_2_div = norm2d(R_con_2)
    R_con_2 = [R_con_2[0]/R_con_2_div,R_con_2[1]/R_con_2_div,R_con_2[2]/R_con_2_div]

    R_cons_vec = cross(R_con_1,R_con_2)
    R_cons_vec_div = norm2d(R_cons_vec)
    R_cons_vec = [R_cons_vec[0]/R_cons_vec_div,R_cons_vec[1]/R_cons_vec_div,R_cons_vec[2]/R_cons_vec_div]

    R_cons_vec = [R_cons_vec[0]*500+RELB[0],R_cons_vec[1]*500+RELB[1],R_cons_vec[2]*500+RELB[2]]

    L_con_1 = np.subtract(LSJC,LELB)
    L_con_1_div = norm2d(L_con_1)
    L_con_1 = [L_con_1[0]/L_con_1_div,L_con_1[1]/L_con_1_div,L_con_1[2]/L_con_1_div]

    L_con_2 = np.subtract(LWRI,LELB)
    L_con_2_div = norm2d(L_con_2)
    L_con_2 = [L_con_2[0]/L_con_2_div,L_con_2[1]/L_con_2_div,L_con_2[2]/L_con_2_div]

    L_cons_vec = cross(L_con_1,L_con_2)
    L_cons_vec_div = norm2d(L_cons_vec)

    L_cons_vec = [L_cons_vec[0]/L_cons_vec_div,L_cons_vec[1]/L_cons_vec_div,L_cons_vec[2]/L_cons_vec_div]

    L_cons_vec = [L_cons_vec[0]*500+LELB[0],L_cons_vec[1]*500+LELB[1],L_cons_vec[2]*500+LELB[2]]

    REJC = findJointC(R_cons_vec,RSJC,RELB,R_delta)
    LEJC = findJointC(L_cons_vec,LSJC,LELB,L_delta)


    # this is radius axis for humerus

        # right
    x_axis = np.subtract(RWRA,RWRB)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    z_axis = np.subtract(REJC,RWRI)
    z_axis_div = norm2d(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    y_axis = cross(z_axis,x_axis)
    y_axis_div = norm2d(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = cross(y_axis,z_axis)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    R_radius = [x_axis,y_axis,z_axis]

        # left
    x_axis = np.subtract(LWRA,LWRB)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    z_axis = np.subtract(LEJC,LWRI)
    z_axis_div = norm2d(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    y_axis = cross(z_axis,x_axis)
    y_axis_div = norm2d(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = cross(y_axis,z_axis)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    L_radius = [x_axis,y_axis,z_axis]

    # calculate wrist joint center for humerus
    R_wristThickness = vsk['RightWristWidth']
    L_wristThickness = vsk['LeftWristWidth']
    R_wristThickness = (R_wristThickness / 2.0 + mm )
    L_wristThickness = (L_wristThickness / 2.0 + mm )

    RWJC = [RWRI[0]+R_wristThickness*R_radius[1][0],RWRI[1]+R_wristThickness*R_radius[1][1],RWRI[2]+R_wristThickness*R_radius[1][2]]
    LWJC = [LWRI[0]-L_wristThickness*L_radius[1][0],LWRI[1]-L_wristThickness*L_radius[1][1],LWRI[2]-L_wristThickness*L_radius[1][2]]

    # recombine the humerus axis

        #right

    z_axis = np.subtract(RSJC,REJC)
    z_axis_div = norm2d(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    x_axis = np.subtract(RWJC,REJC)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = cross(x_axis,z_axis)
    y_axis_div = norm2d(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = cross(y_axis,z_axis)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    # attach each calulcated elbow axis to elbow joint center.
    x_axis = [x_axis[0]+REJC[0],x_axis[1]+REJC[1],x_axis[2]+REJC[2]]
    y_axis = [y_axis[0]+REJC[0],y_axis[1]+REJC[1],y_axis[2]+REJC[2]]
    z_axis = [z_axis[0]+REJC[0],z_axis[1]+REJC[1],z_axis[2]+REJC[2]]

    R_axis = [x_axis,y_axis,z_axis]

        # left

    z_axis = np.subtract(LSJC,LEJC)
    z_axis_div = norm2d(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    x_axis = np.subtract(LWJC,LEJC)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = cross(x_axis,z_axis)
    y_axis_div = norm2d(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = cross(y_axis,z_axis)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    # attach each calulcated elbow axis to elbow joint center.
    x_axis = [x_axis[0]+LEJC[0],x_axis[1]+LEJC[1],x_axis[2]+LEJC[2]]
    y_axis = [y_axis[0]+LEJC[0],y_axis[1]+LEJC[1],y_axis[2]+LEJC[2]]
    z_axis = [z_axis[0]+LEJC[0],z_axis[1]+LEJC[1],z_axis[2]+LEJC[2]]

    L_axis = [x_axis,y_axis,z_axis]

    axis = [R_axis,L_axis]

    origin = [REJC,LEJC]
    wrist_O = [RWJC,LWJC]

    return [origin,axis,wrist_O]

def wristJointCenter(frame,shoulderJC,wand,elbowJC):
    """Calculate the Wrist joint axis ( Radius) function.

    Takes in the elbow axis to calculate each wrist joint axis and returns it.

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.
    shoulderJC : array
        The x,y,z position of the shoulder joint center.
    elbowJC : array
        The x,y,z position of the elbow joint center.
    wand : array
        The x,y,z position of the wand.

    Returns
    --------
    origin, axis : array
        Returns the Shoulder joint center and axis in three array
            return = [[R_wrist_JC_x, R_wrist_JC_y, R_wrist_JC_z],
                        [L_wrist_JC_x,L_wrist_JC_y,L_wrist_JC_z],
                        [[[R_wrist x axis, x,y,z position],
                        [R_wrist y axis, x,y,z position],
                        [R_wrist z axis, x,y,z position]],
                        [[L_wrist x axis, x,y,z position],
                        [L_wrist y axis, x,y,z position],
                        [L_wrist z axis, x,y,z position]]]]


    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import wristJointCenter
    >>> frame = {'RSHO': np.array([428.88496562, 270.552948, 1500.73010254]),
    ...          'LSHO': np.array([68.24668121, 269.01049805, 1510.1072998]),
    ...          'RELB': np.array([658.90338135, 326.07580566, 1285.28515625]),
    ...          'LELB': np.array([-156.32162476, 335.2593313, 1287.39916992]),
    ...          'RWRA': np.array([776.51898193,495.68103027, 1108.38464355]),
    ...          'RWRB': np.array([830.9072876, 436.75341797, 1119.11901855]),
    ...          'LWRA': np.array([-249.28146362, 525.32977295, 1117.09057617]),
    ...          'LWRB': np.array([-311.77532959, 477.22512817, 1125.1619873])}
    >>> wand = [[255.92550222678443, 364.32269504976051, 1460.6297868417887],
    ...        [256.42380097331767, 364.27770361353487, 1460.6165849382387]]
    >>> shoulderJC = [np.array([429.66951995, 275.06718615, 1453.953978131]),
    ...               np.array([64.51952734, 274.93442161, 1463.6313334])]
    >>> elbowJC = [[np.array([633.66707587, 304.95542115, 1256.07799541]),
    ...           np.array([-129.1695218, 316.8671644, 1258.06440717])],
    ...           [[[633.81070138699954, 303.96579004975194, 1256.07658506845],
    ...           [634.35247991784638, 305.05386589332528, 1256.7994730142241],
    ...           [632.95321803901493, 304.85083190737765, 1256.7704317504911]],
    ...           [[-129.32391792749493, 315.88072913249465, 1258.0086629318362],
    ...           [-128.45117135279025, 316.79382333592832, 1257.37260287807],
    ...           [-128.49119037560905, 316.7203088419364, 1258.783373067024]]],
    ...           [[793.32814303250677, 451.29134788252043, 1084.4325513020426],
    ...           [-272.4594189740742, 485.80152210947699, 1091.3666238350822]]]
    >>> [np.around(arr,8) for arr in wristJointCenter(frame,shoulderJC,wand,elbowJC)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 793.32814303,  451.29134788, 1084.4325513 ],
           [-272.45941897,  485.80152211, 1091.36662384]]),
    array([[[ 793.77133728,  450.44879187, 1084.12648231],
            [ 794.01354708,  451.38979263, 1085.1540289 ],
            [ 792.75038863,  450.76181223, 1085.05367274]],
           [[-272.92507295,  485.01202419, 1090.9667996 ],
            [-271.74106833,  485.72818103, 1090.67481935],
            [-271.94256432,  485.19216661, 1091.96791174]]])]
    """
    # Bring Elbow joint center, axes and Wrist Joint Center for calculating Radius Axes

    REJC = elbowJC[0][0]
    LEJC = elbowJC[0][1]

    R_elbow_axis = elbowJC[1][0]
    L_elbow_axis = elbowJC[1][1]

    R_elbow_flex = [R_elbow_axis[1][0]-REJC[0],R_elbow_axis[1][1]-REJC[1],R_elbow_axis[1][2]-REJC[2]]
    L_elbow_flex = [L_elbow_axis[1][0]-LEJC[0],L_elbow_axis[1][1]-LEJC[1],L_elbow_axis[1][2]-LEJC[2]]

    RWJC = elbowJC[2][0]
    LWJC = elbowJC[2][1]

    # this is the axis of radius

        # right
    y_axis = R_elbow_flex
    y_axis = y_axis/ norm3d(y_axis)

    z_axis = np.subtract(REJC,RWJC)
    z_axis = z_axis/ norm3d(z_axis)

    x_axis = cross(y_axis,z_axis)
    x_axis = x_axis/ norm3d(x_axis)

    z_axis = cross(x_axis,y_axis)
    z_axis = z_axis/ norm3d(z_axis)

    # Attach all the axes to wrist joint center.
    x_axis = [x_axis[0]+RWJC[0],x_axis[1]+RWJC[1],x_axis[2]+RWJC[2]]
    y_axis = [y_axis[0]+RWJC[0],y_axis[1]+RWJC[1],y_axis[2]+RWJC[2]]
    z_axis = [z_axis[0]+RWJC[0],z_axis[1]+RWJC[1],z_axis[2]+RWJC[2]]

    R_axis = [x_axis,y_axis,z_axis]

        # left

    y_axis = L_elbow_flex
    y_axis = y_axis/ norm3d(y_axis)

    z_axis = np.subtract(LEJC,LWJC)
    z_axis = z_axis/ norm3d(z_axis)

    x_axis = cross(y_axis,z_axis)
    x_axis = x_axis/ norm3d(x_axis)

    z_axis = cross(x_axis,y_axis)
    z_axis = z_axis/ norm3d(z_axis)

    # Attach all the axes to wrist joint center.
    x_axis = [x_axis[0]+LWJC[0],x_axis[1]+LWJC[1],x_axis[2]+LWJC[2]]
    y_axis = [y_axis[0]+LWJC[0],y_axis[1]+LWJC[1],y_axis[2]+LWJC[2]]
    z_axis = [z_axis[0]+LWJC[0],z_axis[1]+LWJC[1],z_axis[2]+LWJC[2]]

    L_axis = [x_axis,y_axis,z_axis]

    origin = [RWJC,LWJC]

    axis = [R_axis,L_axis]

    return [origin,axis]

def handJointCenter(frame,elbowJC,wristJC,vsk=None):
    """Calculate the Hand joint axis (Hand).

    Takes in a dictionary of marker names to x, y, z positions, wrist axis
    subject measurements.
    Calculate each Hand joint axis and returns it.

    Markers used: RWRA, RWRB, LWRA, LWRB, RFIN, LFIN
    Subject Measurement values used: RightHandThickness, LeftHandThickness

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.
    elbowJC : array, optional
        The x,y,z position of the elbow joint center.
    wristJC : array
        The x,y,z position of the wrist joint center.
    vsk : dict, optional
        A dictionary containing subject measurements.

    Returns
    -------
    origin, axis : array
        Returns an array containing an array representing the right hand joint center
        x, y, z marker positions 1x3, followed by an array containing the
        left hand joint center x, y, z marker positions 1x3, followed by a 2x3x3 array
        containing the right hand joint center x, y, z axis components (1x3x3),
        followed by the left hand joint center x, y, z axis components (1x3x3).

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import handJointCenter
    >>> frame = {'RWRA': np.array([776.51898193,495.68103027, 1108.38464355]),
    ...          'RWRB': np.array([830.9072876, 436.75341797, 1119.11901855]),
    ...          'LWRA': np.array([-249.28146362, 525.32977295, 1117.09057617]),
    ...          'LWRB': np.array([-311.77532959, 477.22512817, 1125.1619873]),
    ...          'RFIN': np.array([863.71374512, 524.4475708, 1074.54248047]),
    ...          'LFIN': np.array([-326.65890503, 558.34338379, 1091.04284668])}
    >>> elbowJC = [[np.array([633.66707587, 304.95542115, 1256.07799541]),
    ...            np.array([-129.1695218, 316.8671644, 1258.06440717])],
    ...           [[[633.81070138699954, 303.96579004975194, 1256.07658506845],
    ...                [634.35247991784638, 305.05386589332528, 1256.7994730142241],
    ...                [632.95321803901493, 304.85083190737765, 1256.7704317504911]],
    ...                [[-129.32391792749493, 315.88072913249465, 1258.0086629318362],
    ...                [-128.45117135279025, 316.79382333592832, 1257.37260287807],
    ...                [-128.49119037560905, 316.7203088419364, 1258.783373067024]]],
    ...                [[793.32814303250677, 451.29134788252043, 1084.4325513020426],
    ...                [-272.4594189740742, 485.80152210947699, 1091.3666238350822]]]
    >>> wristJC = [[[793.32814303250677, 451.29134788252043, 1084.4325513020426],
    ...            [-272.4594189740742, 485.80152210947699, 1091.3666238350822]],
    ...           [[[793.77133727961598, 450.44879187190122, 1084.1264823093322],
    ...            [794.01354707689597, 451.38979262469761, 1085.1540289034019],
    ...            [792.7503886251119, 450761812234714, 1085.0536727414069]],
    ...            [[-272.9250728167512, 485.01202418036871, 1090.9667994752267],
    ...            [-271.74106814470946, 485.72818104689361, 1090.6748195459295],
    ...            [-271.94256446383838, 485.1921666233502, 1091.967911874857]]]]
    >>> vsk = { 'RightHandThickness': 34.0, 'LeftHandThickness': 34.0}
    >>> [np.around(arr,8) for arr in handJointCenter(frame,elbowJC,wristJC,vsk)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 859.80614366,  517.28239823, 1051.97278945],
           [-324.53477798,  551.88744289, 1068.02526837]]),
    array([[[ 859.95675979,  517.59241232, 1052.9115152 ],
            [ 859.07975674,  517.96120459, 1051.86516062],
            [ 859.1355642 ,  516.61673075, 1052.30021881]],
           [[-324.61994077,  552.15893309, 1068.9839343 ],
            [-325.33293185,  551.29292486, 1068.12272964],
            [-323.93837401,  551.13058004, 1068.29259013]]])]
    """


    RWRA = frame['RWRA']
    RWRB = frame['RWRB']
    LWRA = frame['LWRA']
    LWRB = frame['LWRB']
    RFIN = frame['RFIN']
    LFIN = frame['LFIN']

    RWRI = [(RWRA[0]+RWRB[0])/2.0,(RWRA[1]+RWRB[1])/2.0,(RWRA[2]+RWRB[2])/2.0]
    LWRI = [(LWRA[0]+LWRB[0])/2.0,(LWRA[1]+LWRB[1])/2.0,(LWRA[2]+LWRB[2])/2.0]

    LWJC = wristJC[0][1]
    RWJC = wristJC[0][0]

    mm = 7.0
    R_handThickness = vsk['RightHandThickness']
    L_handThickness = vsk['LeftHandThickness']

    R_delta =( R_handThickness/2.0 + mm )
    L_delta =( L_handThickness/2.0 + mm )

    LHND = findJointC(LWRI,LWJC,LFIN,L_delta)
    RHND = findJointC(RWRI,RWJC,RFIN,R_delta)

        # Left
    z_axis = [LWJC[0]-LHND[0],LWJC[1]-LHND[1],LWJC[2]-LHND[2]]
    z_axis_div = norm2d(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    y_axis = [LWRI[0]-LWRA[0],LWRI[1]-LWRA[1],LWRI[2]-LWRA[2]]
    y_axis_div = norm2d(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = cross(y_axis,z_axis)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = cross(z_axis,x_axis)
    y_axis_div = norm2d(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    L_axis = [x_axis,y_axis,z_axis]

        # Right
    z_axis = [RWJC[0]-RHND[0],RWJC[1]-RHND[1],RWJC[2]-RHND[2]]
    z_axis_div = norm2d(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    y_axis = [RWRA[0]-RWRI[0],RWRA[1]-RWRI[1],RWRA[2]-RWRI[2]]
    y_axis_div = norm2d(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = cross(y_axis,z_axis)
    x_axis_div = norm2d(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = cross(z_axis,x_axis)
    y_axis_div = norm2d(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    R_axis = [x_axis,y_axis,z_axis]

    R_origin = RHND
    L_origin = LHND

    # Attach it to the origin.
    L_axis = [[L_axis[0][0]+L_origin[0],L_axis[0][1]+L_origin[1],L_axis[0][2]+L_origin[2]],
            [L_axis[1][0]+L_origin[0],L_axis[1][1]+L_origin[1],L_axis[1][2]+L_origin[2]],
            [L_axis[2][0]+L_origin[0],L_axis[2][1]+L_origin[1],L_axis[2][2]+L_origin[2]]]
    R_axis = [[R_axis[0][0]+R_origin[0],R_axis[0][1]+R_origin[1],R_axis[0][2]+R_origin[2]],
            [R_axis[1][0]+R_origin[0],R_axis[1][1]+R_origin[1],R_axis[1][2]+R_origin[2]],
            [R_axis[2][0]+R_origin[0],R_axis[2][1]+R_origin[1],R_axis[2][2]+R_origin[2]]]

    origin = [R_origin, L_origin]

    axis = [R_axis, L_axis]

    return [origin,axis]

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
    >>> a = [468.14532471, 325.09780884, 673.12591553]
    >>> b = [355.90861996, 365.38260964, 940.6974861]
    >>> c = [452.35180664, 329.0609436, 524.77893066]
    >>> delta = 59.5
    >>> findJointC(a,b,c,delta)
    array([396.25286248, 347.91367254, 518.63620527])
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
    >>> a = [6.25286248, 7.91367254, 18.63620527]
    >>> b = [3.49290439, 4.42038315, 19.23948238]
    >>> np.around(cross(a, b),8)
    array([ 6.98757956e+01, -5.52073543e+01, -1.65361000e-03])
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
    -------
    >>> import numpy as np
    >>> from .pyCGM import getPelangle
    >>> axisP = [[ 0.0464229, 0.99648672, 0.06970743],
    ...        [ 0.99734011, -0.04231089, -0.05935067],
    ...       [-0.05619277,  0.07227725, -0.99580037]]
    >>> axisD = [[-0.18067218, -0.98329158, -0.02225371],
    ...        [ 0.71383942, -0.1155303, -0.69071415],
    ...        [ 0.67660243, -0.1406784, 0.7227854 ]]
    >>> np.around(getPelangle(axisP,axisD),8)
    array([-175.65183483,   39.63221918,  -10.2668477 ])
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
    >>> axisP = [[ 0.0464229, 0.99648672, 0.06970743],
    ...        [ 0.99734011, -0.04231089, -0.05935067],
    ...       [-0.05619277,  0.07227725, -0.99580037]]
    >>> axisD = [[-0.18067218, -0.98329158, -0.02225371],
    ...        [ 0.71383942, -0.1155303, -0.69071415],
    ...        [ 0.67660243, -0.1406784, 0.7227854 ]]
    >>> np.around(getHeadangle(axisP,axisD),8)
    array([ 184.34816517,  -39.63221894, -190.2668477 ])
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
    >>> axisP = [[ 0.0464229, 0.99648672, 0.06970743],
    ...        [ 0.99734011, -0.04231089, -0.05935067],
    ...       [-0.05619277,  0.07227725, -0.99580037]]
    >>> axisD = [[-0.18067218, -0.98329158, -0.02225371],
    ...        [ 0.71383942, -0.1155303, -0.69071415],
    ...        [ 0.67660243, -0.1406784, 0.7227854 ]]
    >>> np.around(getangle_sho(axisP,axisD),8)
    array([  -3.3474503 , -140.28662977,  172.50982168])
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
    >>> axisP = [[ 0.0464229,   0.99648672,  0.06970743],
    ...        [ 0.99734011, -0.04231089, -0.05935067],
    ...        [-0.05619277,  0.07227725, -0.99580037]]
    >>> axisD = [[-0.18067218, -0.98329158,-0.02225371],
    ...        [ 0.71383942, -0.1155303,  -0.69071415],
    ...        [ 0.67660243, -0.1406784,   0.7227854 ]]
    >>> np.around(getangle_spi(axisP,axisD),8)
    array([ 2.8891964 ,  9.7438295 , 39.74341087])
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
    >>> axisP = [[ 0.0464229,   0.99648672,  0.06970743],
    ...         [ 0.99734011, -0.04231089, -0.05935067],
    ...         [-0.05619277,  0.07227725, -0.99580037]]
    >>> axisD = [[-0.18067218, -0.98329158, -0.02225371],
    ...         [ 0.71383942, -0.1155303,  -0.69071415],
    ...         [ 0.67660243, -0.1406784,   0.7227854 ]]
    >>> np.around(getangle(axisP,axisD),8)
    array([-175.65183483,  -39.6322192 ,  100.2668477 ])
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
    >>> v = [105.141121037153, 101.890788777524, 326.7710280245359]
    >>> np.around(norm2d(v),8)
    358.07218955
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
    >>> v = [125.44928201, 143.94301493, 213.49204956]
    >>> norm3d(v)
    array(286.4192192)
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
    >>> v = [1.44928201, 1.94301493, 2.49204956]
    >>> np.around(normDiv(v),8)
    array([0.11991376, 0.16076527, 0.20619246])
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
    >>> np.around(rotmat(x,y,z),8)
    array([[ 0.99988882, -0.01396199,  0.00523596],
           [ 0.01400734,  0.99986381, -0.00872642],
           [-0.00511341,  0.00879879,  0.99994822]])
    >>> x = 0.5
    >>> np.around(rotmat(x),8)
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.99996192, -0.00872654],
           [ 0.        ,  0.00872654,  0.99996192]])
    >>> x = 1
    >>> y = 1
    >>> np.around(rotmat(x,y),8)
    array([[ 9.9984770e-01,  0.0000000e+00,  1.7452410e-02],
           [ 3.0459000e-04,  9.9984770e-01, -1.7449750e-02],
           [-1.7449750e-02,  1.7452410e-02,  9.9969541e-01]])
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
    Stores
        RPel_angle = []
        LPel_angle = []
        RHip_angle = []
        LHip_angle = []
        RKnee_angle = []
        LKnee_angle = []
        RAnkle_angle = []
        LAnkle_angle = []

    Joint Axis store like below form

    The axis form is in the form [[origin],[axis]]
    Origin defines the position of axis and axis is the unit vector of each
    axis attached to the origin

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
    >>> np.around(results['Pelvis'],8)
    array([ 246.152565  ,  353.26243591, 1031.71362305])
    >>> np.around(results['Thorax'],8)
    array([ 250.56159618,  303.23273922, 1461.17230698])
    >>> np.around(results['Head'],8)
    array([ 244.89547729,  325.05789185, 1730.1619873 ])
    >>> np.around(results['RHand'],8)
    array([ 770.93339376,  591.04557736, 1079.04817118])
    """

    # THIS IS FOOT PROGRESS ANGLE
    rfoot_prox,rfoot_proy,rfoot_proz,lfoot_prox,lfoot_proy,lfoot_proz = [None]*6

    #First Calculate Pelvis
    pelvis_axis = pelvisJointCenter(frame)

    kin_Pelvis_axis = pelvis_axis

    kin_Pelvis_JC = pelvis_axis[0] #quick fix for storing JC

    #change to same format
    Pelvis_vectors = pelvis_axis[1]
    Pelvis_origin = pelvis_axis[0]

    #need to update this based on the file
    global_Axis = vsk['GCS']

    #make the array which will be the input of findangle function
    pelvis_Axis_mod = np.vstack([np.subtract(Pelvis_vectors[0],Pelvis_origin),
                            np.subtract(Pelvis_vectors[1],Pelvis_origin),
                            np.subtract(Pelvis_vectors[2],Pelvis_origin)])


    global_pelvis_angle = getangle(global_Axis,pelvis_Axis_mod)

    pelx=global_pelvis_angle[0]
    pely=global_pelvis_angle[1]
    pelz=global_pelvis_angle[2]

    # and then find hip JC
    hip_JC = hipJointCenter(frame,pelvis_axis[0],pelvis_axis[1][0],pelvis_axis[1][1],pelvis_axis[1][2],vsk=vsk)

    kin_L_Hip_JC = hip_JC[0] #quick fix for storing JC
    kin_R_Hip_JC = hip_JC[1] #quick fix for storing JC

    hip_axis = hipAxisCenter(hip_JC[0],hip_JC[1],pelvis_axis)

    knee_JC = kneeJointCenter(frame,hip_JC,0,vsk=vsk)

    kin_R_Knee_JC = knee_JC[0] #quick fix for storing JC
    kin_L_Knee_JC = knee_JC[1] #quick fix for storing JC

    #change to same format
    Hip_axis_form = hip_axis[1]
    Hip_center_form = hip_axis[0]
    R_Knee_axis_form = knee_JC[2][0]
    R_Knee_center_form = knee_JC[0]
    L_Knee_axis_form = knee_JC[2][1]
    L_Knee_center_form = knee_JC[1]

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

    ankle_JC = ankleJointCenter(frame,knee_JC,0,vsk=vsk)

    kin_R_Ankle_JC = ankle_JC[0] #quick fix for storing JC
    kin_L_Ankle_JC = ankle_JC[1] #quick fix for storing JC

    #change to same format

    R_Ankle_axis_form = ankle_JC[2][0]
    R_Ankle_center_form = ankle_JC[0]
    L_Ankle_axis_form = ankle_JC[2][1]
    L_Ankle_center_form = ankle_JC[1]


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
    foot_JC = footJointCenter(frame,vsk,ankle_JC,knee_JC,offset)


    kin_R_Foot_JC = foot_JC[0] #quick fix for storing JC
    kin_L_Foot_JC = foot_JC[1] #quick fix for storing JC

    kin_RHEE = frame['RHEE']
    kin_LHEE = frame['LHEE']

    # Change to same format
    R_Foot_axis_form = foot_JC[2][0]
    R_Foot_center_form = foot_JC[0]
    L_Foot_axis_form = foot_JC[2][1]
    L_Foot_center_form = foot_JC[1]

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

    head_axis = headJC(frame,vsk=vsk)

    kin_Head_JC = head_axis[1] #quick fix for storing JC

    LFHD = frame['LFHD'] #as above
    RFHD = frame['RFHD']
    LBHD = frame['LBHD']
    RBHD = frame['RBHD']

    kin_Head_Front = np.array((LFHD+RFHD)/2)
    kin_Head_Back = np.array((LBHD+RBHD)/2)

    #change to same format
    Head_axis_form = head_axis[0]
    Head_center_form = head_axis[1]
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

    thorax_axis = thoraxJC(frame)

    kin_Thorax_JC = thorax_axis[1] #quick fix for storing JC
    kin_Thorax_axis = thorax_axis

    # Change to same format
    Thorax_axis_form = thorax_axis[0]
    Thorax_center_form = thorax_axis[1]
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

    wand = findwandmarker(frame,thorax_axis)
    shoulder_JC = findshoulderJC(frame,thorax_axis,wand,vsk=vsk)

    kin_R_Shoulder_JC = shoulder_JC[0] #quick fix for storing JC
    kin_L_Shoulder_JC = shoulder_JC[1] #quick fix for storing JC

    shoulder_axis = shoulderAxisCalc(frame,thorax_axis,shoulder_JC,wand)
    humerus_JC = elbowJointCenter(frame,thorax_axis,shoulder_JC,wand,vsk=vsk)

    kin_R_Humerus_JC = humerus_JC[0][0] #quick fix for storing JC
    kin_L_Humerus_JC = humerus_JC[0][1] #quick fix for storing JC

    # Change to same format
    R_Clavicle_axis_form = shoulder_axis[1][0]
    L_Clavicle_axis_form = shoulder_axis[1][1]
    R_Clavicle_center_form = shoulder_axis[0][0]
    L_Clavicle_center_form = shoulder_axis[0][1]

    # Change to same format
    R_Humerus_axis_form = humerus_JC[1][0]
    L_Humerus_axis_form = humerus_JC[1][1]
    R_Humerus_center_form = humerus_JC[0][0]
    L_Humerus_center_form = humerus_JC[0][1]

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

    radius_JC = wristJointCenter(frame,shoulder_JC,wand,humerus_JC)

    kin_R_Radius_JC = radius_JC[0][0] #quick fix for storing JC
    kin_L_Radius_JC = radius_JC[0][1] #quick fix for storing JC


    # Change to same format
    R_Radius_axis_form = radius_JC[1][0]
    L_Radius_axis_form = radius_JC[1][1]
    R_Radius_center_form = radius_JC[0][0]
    L_Radius_center_form = radius_JC[0][1]

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
    hand_JC = handJointCenter(frame,humerus_JC,radius_JC,vsk=vsk)

    kin_R_Hand_JC = hand_JC[0][0] #quick fix for storing JC
    kin_L_Hand_JC = hand_JC[0][1] #quick fix for storing JC


    # Change to same format

    R_Hand_axis_form = hand_JC[1][0]
    L_Hand_axis_form = hand_JC[1][1]
    R_Hand_center_form = hand_JC[0][0]
    L_Hand_center_form = hand_JC[0][1]

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
    pel_origin = Pelvis_origin
    pel_ox=pel_origin[0]
    pel_oy=pel_origin[1]
    pel_oz=pel_origin[2]
        # xaxis
    pel_x_axis = Pelvis_vectors[0]
    pel_xx=pel_x_axis[0]
    pel_xy=pel_x_axis[1]
    pel_xz=pel_x_axis[2]
        # yaxis
    pel_y_axis = Pelvis_vectors[1]
    pel_yx=pel_y_axis[0]
    pel_yy=pel_y_axis[1]
    pel_yz=pel_y_axis[2]
        # zaxis
    pel_z_axis = Pelvis_vectors[2]
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