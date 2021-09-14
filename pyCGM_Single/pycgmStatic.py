# -*- coding: utf-8 -*-
"""
This file provides helper functions for static calculations.

Created on Tue Jul 28 16:55:25 2015

@author: cadop
"""
import numpy as np
from math import sin, cos, acos, sqrt, radians

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
    array([[1., 0.  ,  0.  ],
           [0., 1.  , -0.01],
           [0., 0.01,  1.  ]])
    >>> x = 1
    >>> y = 1
    >>> np.around(rotmat(x,y), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  ,  0.  ,  0.02],
           [ 0.  ,  1.  , -0.02],
           [-0.02,  0.02,  1.  ]])
    """

    x, y, z = radians(x), radians(y), radians(z)
    r_x = [[1, 0, 0], 
           [0, cos(x), sin(x) * -1], 
           [0, sin(x), cos(x)]]

    r_y = [[cos(y), 0, sin(y)],
           [0, 1, 0],
           [sin(y)*-1, 0, cos(y)]]

    r_z = [[cos(z), sin(z)*-1, 0],
           [sin(z), cos(z), 0],
           [0, 0, 1]]

    r_xy = np.matmul(r_x, r_y)
    r_xyz = np.matmul(r_xy, r_z)

    return r_xyz


def get_dist(p0, p1):
    """Get Distance

    This function calculates the distance between two 3-D positions.

    Parameters
    ----------
    p0 : array
        Position of first (x, y, z) coordinate.
    p1 : array
        Position of second (x, y, z) coordinate.

    Returns
    -------
    distance : float
        The distance between positions p0 and p1.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import get_dist
    >>> p0 = [0,1,2]
    >>> p1 = [1,2,3]
    >>> np.around(get_dist(p0,p1), 2)
    1.73
    >>> p0 = np.array([991.45, 741.95, 321.36])
    >>> p1 = np.array([117.09, 142.24, 481.95])
    >>> np.around(get_dist(p0,p1), 2)
    1072.36
    """
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)

    distance = np.linalg.norm(p0 - p1)
    return distance

def getStatic(motionData,vsk,flat_foot=False,GCS=None):
    """ Get Static Offset function

    Calculate the static offset angle values and return the values in radians

    Parameters
    ----------
    motionData : dict
        Dictionary of marker lists.
    vsk : dict, optional
        Dictionary of various attributes of the skeleton.
    flat_foot : boolean, optional
        A boolean indicating if the feet are flat or not.
        The default value is False.
    GCS : array, optional
        An array containing the Global Coordinate System.
        If not provided, the default will be set to: [[1, 0, 0], [0, 1, 0], [0, 0, 1]].

    Returns
    -------
    calSM : dict
        Dictionary containing various marker lists of offsets.

    Examples
    --------
    >>> from .pycgmIO import loadC3D, loadVSK
    >>> from .pycgmStatic import getStatic
    >>> import os
    >>> from .pyCGM_Helpers import getfilenames
    >>> fileNames=getfilenames(2)
    >>> c3dFile = fileNames[1]
    >>> vskFile = fileNames[2]
    >>> result = loadC3D(c3dFile)
    >>> data = result[0]
    >>> vskData = loadVSK(vskFile, False)
    >>> result = getStatic(data,vskData,flat_foot=False)
    >>> result['Bodymass']
    75.0
    >>> result['RightKneeWidth']
    105.0
    >>> result['LeftTibialTorsion']
    0.0
    """
    static_offset = []
    head_offset = []
    IAD = []
    calSM = {}
    LeftLegLength = vsk['LeftLegLength']
    RightLegLength = vsk['RightLegLength']
    calSM['MeanLegLength'] = (LeftLegLength+RightLegLength)/2.0
    calSM['Bodymass'] = vsk['Bodymass']

    #Define the global coordinate system
    if GCS == None: calSM['GCS'] = [[1,0,0],[0,1,0],[0,0,1]]

    if vsk['LeftAsisTrocanterDistance'] != 0 and vsk['RightAsisTrocanterDistance'] != 0:
        calSM['L_AsisToTrocanterMeasure'] = vsk['LeftAsisTrocanterDistance']
        calSM['R_AsisToTrocanterMeasure'] = vsk['RightAsisTrocanterDistance']
    else:
        calSM['R_AsisToTrocanterMeasure'] = ( 0.1288 * RightLegLength ) - 48.56
        calSM['L_AsisToTrocanterMeasure'] = ( 0.1288 * LeftLegLength ) - 48.56

    if vsk['InterAsisDistance'] != 0:
        calSM['InterAsisDistance'] = vsk['InterAsisDistance']
    else:
        for frame in motionData:
            iadCalc = calc_IAD(frame["RASI"] if "RASI" in frame else None,
                               frame["LASI"] if "LASI" in frame else None) 
            IAD.append(iadCalc)
        InterAsisDistance = np.average(IAD)
        calSM['InterAsisDistance'] = InterAsisDistance

    try:
        calSM['RightKneeWidth'] = vsk['RightKneeWidth']
        calSM['LeftKneeWidth'] = vsk['LeftKneeWidth']

    except:
        #no knee width
        calSM['RightKneeWidth'] = 0
        calSM['LeftKneeWidth'] = 0

    if calSM['RightKneeWidth'] == 0:
        if 'RMKN' in list(motionData[0].keys()):
            #medial knee markers are available
            Rwidth = []
            Lwidth = []
            #average each frame
            for frame in motionData:
                RMKN = frame['RMKN']
                LMKN = frame['LMKN']

                RKNE = frame['RKNE']
                LKNE = frame['LKNE']

                Rdst = get_dist(RKNE,RMKN)
                Ldst = get_dist(LKNE,LMKN)
                Rwidth.append(Rdst)
                Lwidth.append(Ldst)

            calSM['RightKneeWidth'] = sum(Rwidth)/len(Rwidth)
            calSM['LeftKneeWidth'] = sum(Lwidth)/len(Lwidth)
    try:
        calSM['RightAnkleWidth'] = vsk['RightAnkleWidth']
        calSM['LeftAnkleWidth'] = vsk['LeftAnkleWidth']

    except:
        #no knee width
        calSM['RightAnkleWidth'] = 0
        calSM['LeftAnkleWidth'] = 0

    if calSM['RightAnkleWidth'] == 0:
        if 'RMKN' in list(motionData[0].keys()):
            #medial knee markers are available
            Rwidth = []
            Lwidth = []
            #average each frame
            for frame in motionData:
                RMMA = frame['RMMA']
                LMMA = frame['LMMA']

                RANK = frame['RANK']
                LANK = frame['LANK']

                Rdst = get_dist(RMMA,RANK)
                Ldst = get_dist(LMMA,LANK)
                Rwidth.append(Rdst)
                Lwidth.append(Ldst)

            calSM['RightAnkleWidth'] = sum(Rwidth)/len(Rwidth)
            calSM['LeftAnkleWidth'] = sum(Lwidth)/len(Lwidth)

    #calSM['RightKneeWidth'] = vsk['RightKneeWidth']
    #calSM['LeftKneeWidth'] = vsk['LeftKneeWidth']

    #calSM['RightAnkleWidth'] = vsk['RightAnkleWidth']
    #calSM['LeftAnkleWidth'] = vsk['LeftAnkleWidth']

    calSM['RightTibialTorsion'] = vsk['RightTibialTorsion']
    calSM['LeftTibialTorsion'] =vsk['LeftTibialTorsion']

    calSM['RightShoulderOffset'] = vsk['RightShoulderOffset']
    calSM['LeftShoulderOffset'] = vsk['LeftShoulderOffset']

    calSM['RightElbowWidth'] = vsk['RightElbowWidth']
    calSM['LeftElbowWidth'] = vsk['LeftElbowWidth']
    calSM['RightWristWidth'] = vsk['RightWristWidth']
    calSM['LeftWristWidth'] = vsk['LeftWristWidth']

    calSM['RightHandThickness'] = vsk['RightHandThickness']
    calSM['LeftHandThickness'] = vsk['LeftHandThickness']

    for frame in motionData:
        ankle_axis = calc_axis_ankle(frame['RTIB'] if 'RTIB' in frame else None,
                                     frame['LTIB'] if 'LTIB' in frame else None,
                                     frame['RANK'] if 'RANK' in frame else None,
                                     frame['LANK'] if 'LANK' in frame else None,
                                     knee_axis[0][:3, 3],
                                     knee_axis[1][:3, 3],
                                     vsk['RightAnkleWidth'],
                                     vsk['LeftAnkleWidth'],
                                     vsk['RightTibialTorsion'],
                                     vsk['LeftTibialTorsion'])
        pelvis_axis = calc_axis_pelvis(frame['RASI'] if 'RASI' in frame else None,
                                       frame['LASI'] if 'LASI' in frame else None,
                                       frame['RPSI'] if 'RPSI' in frame else None,
                                       frame['LPSI'] if 'LPSI' in frame else None,
                                       frame['SACR'] if 'SACR' in frame else None)
        hip_JC = calc_joint_center_hip(pelvis_axis, calSM)
        knee_axis = calc_axis_knee(frame['RTHI'] if 'RTHI' in frame else None,
                                   frame['LTHI'] if 'LTHI' in frame else None,
                                   frame['RKNE'] if 'RKNE' in frame else None,
                                   frame['LKNE'] if 'LKNE' in frame else None,
                                   hip_JC[0],
                                   hip_JC[1],
                                   vsk['RightKneeWidth'],
                                   vsk['LeftKneeWidth'])
        ankle_JC = ankleJointCenter(frame,knee_JC,0,vsk=calSM)
        angle = staticCalculation(frame,ankle_JC,knee_JC,flat_foot,calSM)
        head = calc_axis_head(frame['LFHD'] if 'LFHD' in frame else None,
                              frame['RFHD'] if 'RFHD' in frame else None,
                              frame['LBHD'] if 'LBHD' in frame else None,
                              frame['RBHD'] if 'RBHD' in frame else None)
        headangle = calc_static_head(head)

        static_offset.append(angle)
        head_offset.append(headangle)


    static=np.average(static_offset,axis=0)
    staticHead=np.average(head_offset)

    calSM['RightStaticRotOff'] = static[0][0]*-1
    calSM['RightStaticPlantFlex'] = static[0][1]
    calSM['LeftStaticRotOff'] = static[1][0]
    calSM['LeftStaticPlantFlex'] = static[1][1]
    calSM['HeadOffset'] = staticHead

    return calSM

def average(lst):
    """Average Calculation function

    Calculates the average of the values in a given lst or array.

    Parameters
    ----------
    lst : list
        List or array of values.

    Returns
    -------
    avg : float
        The mean of the list.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import average
    >>> lst = [1,2,3,4,5]
    >>> average(lst)
    3.0
    >>> lst = np.array([93.82, 248.96, 782.62])
    >>> np.around(average(lst), 2)
    375.13
    """
    avg = sum(lst) / len(lst)
    return avg

def calc_IAD(rasi, lasi):
    """Inter ASIS Distance (IAD) Calculation

    Calculates the Inter ASIS Distance.
    Markers used: RASI, LASI

    Parameters
    ----------
    rasi: array
        1x3 RASI marker
    lasi: array
        1x3 LASI marker

    Returns
    -------
    IAD : float
        The Inter ASIS Distance

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import calc_IAD
    >>> lasi = np.array([ 183.19,  422.79, 1033.07])
    >>> rasi = np.array([ 395.37,  428.1, 1036.83])
    >>> np.around(calc_IAD(rasi, lasi), 2)
    212.28
    """
    rasi = np.asarray(rasi)
    lasi = np.asarray(lasi)

    iad = np.linalg.norm(rasi - lasi)

    return iad

def calc_static_head(head_axis):
    """Static Head Calculation

    This function converts the head axis to a numpy array,
    and then calculates the offset of the head using the calc_head_offset function.

    Parameters
    ----------
    head_axis : array
        4x4 affine matrix containing the head (x, y, z) axes and origin

    Returns
    -------
    offset : float
        The head offset angle for static calibration.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import calc_static_head
    >>> head_axis = np.array([[ 0.75,    0.6 ,    0.28,   99.58],
    ...                       [-0.61,    0.79,   -0.03,   82.79],
    ...                       [-0.23,   -0.15,    0.96, 1483.8 ],
    ...                       [ 0.  ,    0.  ,    0.  ,    0.  ]])
    >>> np.around(calc_static_head(head_axis), 2)
    0.28
    """

    head_axis = np.asarray(head_axis)

    global_axis = [[ 0, 1, 0, 0],
                   [-1, 0, 0, 0],
                   [ 0, 0, 1, 0],
                   [ 0, 0, 0, 0]]

    offset = calc_head_offset(global_axis, head_axis)

    return offset

def calc_head_offset(axisP, axisD):
    """Head Offset Calculation

    Calculate head offset angle for static calibration.
    This function is only called in static trial.
    Output will be used later in the dynamic trial.

    Parameters
    ----------
    axisP : array
        4x4 affine matrix representing the position of the proximal axis.
    axisD : array
        4x4 affine matrix representing the position of the distal axis.

    Returns
    -------
    angle : float
        The beta angle of the head offset.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import calc_head_offset
    >>> axisP = np.array([[0.96, 0.81, 0.82, 0],
    ...                   [0.24, 0.72, 0.38, 0],
    ...                   [0.98, 0.21, 0.68, 0],
    ...                   [0,    0,    0,    1]])
    >>> axisD = np.array([[0.21, 0.25, 0.94, 0],
    ...                   [0.8,  0.45, 0.91, 0],
    ...                   [0.17, 0.67, 0.85, 0],
    ...                   [0,    0,    0,    1]])
    >>> np.around(calc_head_offset(axisP,axisD), 2)
    0.95
    """
    axisP = np.asarray(axisP)[:3, :3]
    axisD = np.asarray(axisD)[:3, :3]

    
    axisPi = np.linalg.inv(axisP)

    # rotation matrix is in order XYZ
    M = matrixmult(axisD, axisPi)

    # get y angle from rotation matrix using inverse trigonometry.
    getB= M[0][2] / M[2][2]

    beta = np.arctan(getB)

    angle = beta

    return angle

def staticCalculation(frame,ankle_JC,knee_JC,flat_foot,vsk=None):
    """Calculate the Static angle function

    Takes in anatomically uncorrected axis and anatomically correct axis.
    Corrects the axis depending on flat-footedness.

    Calculates the offset angle between those two axes.

    It is rotated from uncorrected axis in YXZ order.

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.
    ankle_JC : array
        An array containing the x,y,z axes marker positions of the ankle joint center.
    knee_JC : array
        An array containing the x,y,z axes marker positions of the knee joint center.
    flat_foot : boolean
        A boolean indicating if the feet are flat or not.
    vsk : dict, optional
        A dictionary containing subject measurements from a VSK file.

    Returns
    -------
    angle : list
        Returns the offset angle represented by a 2x3x3 list.
        The array contains the right flexion, abduction, rotation angles (1x3x3)
        followed by the left flexion, abduction, rotation angles (1x3x3).

    Notes
    -----
    The correct axis changes depending on the flat foot option.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import staticCalculation
    >>> frame = {'RTOE': np.array([427.95, 437.1,  41.77]),
    ...          'LTOE': np.array([175.79, 379.5,  42.61]),
    ...          'RHEE': np.array([406.46, 227.56,  48.76]),
    ...          'LHEE': np.array([223.6, 173.43,  47.93])}
    >>> ankle_JC = [np.array([393.76, 247.68, 87.74]),
    ...            np.array([98.75, 219.47, 80.63]),
    ...            [[np.array([394.48, 248.37, 87.72]),
    ...            np.array([393.07, 248.39, 87.62]),
    ...            np.array([393.69, 247.78, 88.73])],
    ...            [np.array([98.47, 220.43, 80.53]),
    ...            np.array([97.79, 219.21, 80.76]),
    ...            np.array([98.85, 219.60, 81.62])]]]
    >>> knee_JC = [np.array([364.18, 292.17, 515.19]),
    ...           np.array([143.55, 279.90, 524.78]),
    ...           np.array([[[364.65, 293.07, 515.19],
    ...           [363.29, 292.61, 515.04],
    ...           [364.05, 292.24, 516.18]],
    ...           [[143.66, 280.89, 524.63],
    ...           [142.56, 280.02, 524.86],
    ...           [143.65, 280.05, 525.77]]])]
    >>> flat_foot = True
    >>> vsk = { 'RightSoleDelta': 0.45,'LeftSoleDelta': 0.45 }
    >>> np.around(staticCalculation(frame,ankle_JC,knee_JC,flat_foot,vsk), 2)
    array([[-0.08,  0.23, -0.66],
           [-0.67,  0.22, -0.3 ]])
    >>> flat_foot = False # Using the same variables and switching the flat_foot flag.
    >>> np.around(staticCalculation(frame,ankle_JC,knee_JC,flat_foot,vsk), 2)
    array([[-0.08,  0.2 , -0.15],
           [-0.67,  0.19,  0.12]])
    """

    # Get the each axis from the function.
    uncorrect_foot = uncorrect_footaxis(frame,ankle_JC)

    #change the reference uncorrect foot axis to same format
    RF_center1_R_form = uncorrect_foot[0]
    RF_center1_L_form = uncorrect_foot[1]
    RF_axis1_R_form = uncorrect_foot[2][0]
    RF_axis1_L_form = uncorrect_foot[2][1]

    #make the array which will be the input of findangle function
    RF1_R_Axis = np.vstack([np.subtract(RF_axis1_R_form[0],RF_center1_R_form),
                            np.subtract(RF_axis1_R_form[1],RF_center1_R_form),
                            np.subtract(RF_axis1_R_form[2],RF_center1_R_form)])
    RF1_L_Axis = np.vstack([np.subtract(RF_axis1_L_form[0],RF_center1_L_form),
                            np.subtract(RF_axis1_L_form[1],RF_center1_L_form),
                            np.subtract(RF_axis1_L_form[2],RF_center1_L_form)])

    # Check if it is flat foot or not.
    if flat_foot == False:
        RF_axis2 = calc_axis_nonflatfoot(frame["RTOE"] if "RTOE" in frame else None,
                                         frame["LTOE"] if "LTOE" in frame else None,
                                         frame["RHEE"] if "RHEE" in frame else None,
                                         frame["LHEE"] if "LHEE" in frame else None,
                                         ankle_axis)
        RF_center2_R_form = RF_axis2[0]
        RF_center2_L_form = RF_axis2[1]
        RF_axis2_R_form = RF_axis2[2][0]
        RF_axis2_L_form = RF_axis2[2][1]
        # make the array to same format for calculating angle.
        RF2_R_Axis = np.vstack([np.subtract(RF_axis2_R_form[0],RF_center2_R_form),
                            np.subtract(RF_axis2_R_form[1],RF_center2_R_form),
                            np.subtract(RF_axis2_R_form[2],RF_center2_R_form)])
        RF2_L_Axis = np.vstack([np.subtract(RF_axis2_L_form[0],RF_center2_L_form),
                            np.subtract(RF_axis2_L_form[1],RF_center2_L_form),
                            np.subtract(RF_axis2_L_form[2],RF_center2_L_form)])

        R_AnkleFlex_angle = getankleangle(RF1_R_Axis,RF2_R_Axis)
        L_AnkleFlex_angle = getankleangle(RF1_L_Axis,RF2_L_Axis)

    elif flat_foot == True:
        RF_axis3 = calc_axis_flatfoot(frame["RTOE"] if "RTOE" in frame else None,
                                      frame["LTOE"] if "LTOE" in frame else None, 
                                      frame["RHEE"] if "RHEE" in frame else None, 
                                      frame["LHEE"] if "LHEE" in frame else None, 
                                      ankle_JC,
                                      vsk["RightSoleDelta"],
                                      vsk["LeftSoleDelta"])
        RF_center3_R_form = RF_axis3[0]
        RF_center3_L_form = RF_axis3[1]
        RF_axis3_R_form = RF_axis3[2][0]
        RF_axis3_L_form = RF_axis3[2][1]
        # make the array to same format for calculating angle.
        RF3_R_Axis = np.vstack([np.subtract(RF_axis3_R_form[0],RF_center3_R_form),
                            np.subtract(RF_axis3_R_form[1],RF_center3_R_form),
                            np.subtract(RF_axis3_R_form[2],RF_center3_R_form)])
        RF3_L_Axis = np.vstack([np.subtract(RF_axis3_L_form[0],RF_center3_L_form),
                            np.subtract(RF_axis3_L_form[1],RF_center3_L_form),
                            np.subtract(RF_axis3_L_form[2],RF_center3_L_form)])

        R_AnkleFlex_angle = getankleangle(RF1_R_Axis,RF3_R_Axis)
        L_AnkleFlex_angle = getankleangle(RF1_L_Axis,RF3_L_Axis)

    angle = [R_AnkleFlex_angle,L_AnkleFlex_angle]

    return angle

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
    >>> from .pycgmStatic import calc_axis_pelvis
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

def calc_joint_center_hip(pelvis, subject):
    u"""Calculate the right and left hip joint center.

    Takes in a 4x4 affine matrix of pelvis axis and subject measurements
    dictionary. Calculates and returns the right and left hip joint centers.

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
    >>> from .pycgmStatic import calc_joint_center_hip
    >>> vsk = {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.51,
    ...        'L_AsisToTrocanterMeasure': 72.51, 'InterAsisDistance': 215.90}
    >>> pelvis_axis = np.array([
    ...     [ 0.14, 0.98, -0.11,  251.60],
    ...     [-0.99, 0.13, -0.02,  391.74],
    ...     [ 0,    0.1,   0.99, 1032.89],
    ...     [ 0,    0,     0,       1   ]])
    >>> np.around(calc_joint_center_hip(pelvis_axis,vsk), 2) #doctest: +NORMALIZE_WHITESPACE
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
        cos(beta) + C * cos(theta) * sin(beta)
    L_Yh = S*(C*sin(theta) - aa)
    L_Zh = (-left_asis_to_trochanter - mm) * \
        sin(beta) - C * cos(theta) * cos(beta)

    # Right:  Calculate the distance to translate along the pelvis axis
    R_Xh = (-right_asis_to_trochanter - mm) * \
        cos(beta) + C * cos(theta) * sin(beta)
    R_Yh = (C*sin(theta) - aa)
    R_Zh = (-right_asis_to_trochanter - mm) * \
        sin(beta) - C * cos(theta) * cos(beta)

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

    Raxis = [[cos(rtib_torsion)*Raxis[0][0]-sin(rtib_torsion)*Raxis[1][0],
              cos(rtib_torsion)*Raxis[0][1] -
              sin(rtib_torsion)*Raxis[1][1],
              cos(rtib_torsion)*Raxis[0][2]-sin(rtib_torsion)*Raxis[1][2]],
             [sin(rtib_torsion)*Raxis[0][0]+cos(rtib_torsion)*Raxis[1][0],
             sin(rtib_torsion)*Raxis[0][1] +
              cos(rtib_torsion)*Raxis[1][1],
             sin(rtib_torsion)*Raxis[0][2]+cos(rtib_torsion)*Raxis[1][2]],
             [Raxis[2][0], Raxis[2][1], Raxis[2][2]]]

    Laxis = [[cos(ltib_torsion)*Laxis[0][0]-sin(ltib_torsion)*Laxis[1][0],
              cos(ltib_torsion)*Laxis[0][1] -
              sin(ltib_torsion)*Laxis[1][1],
              cos(ltib_torsion)*Laxis[0][2]-sin(ltib_torsion)*Laxis[1][2]],
             [sin(ltib_torsion)*Laxis[0][0]+cos(ltib_torsion)*Laxis[1][0],
             sin(ltib_torsion)*Laxis[0][1] +
              cos(ltib_torsion)*Laxis[1][1],
             sin(ltib_torsion)*Laxis[0][2]+cos(ltib_torsion)*Laxis[1][2]],
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


def calc_axis_head(lfhd, rfhd, lbhd, rbhd):
    """Calculate the head joint center and axis.

    Takes in markers that correspond to (x, y, z) positions of the current
    frame, and the head offset. 

    Calculates the head joint center and axis.

    Markers used: LFHD, RFHD, LBHD, RBHD

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

    Returns
    -------
    head_axis : array
        4x4 affine matrix with head (x, y, z) axes and origin.


    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)
    >>> from .pycgmStatic import calc_axis_head
    >>> rfhd = np.array([325.82, 402.55, 1722.49])
    >>> lfhd = np.array([184.55, 409.68, 1721.34])
    >>> rbhd = np.array([304.39, 242.91, 1694.97])
    >>> lbhd = np.array([197.86, 251.28, 1696.90])
    >>> [np.around(arr, 2) for arr in calc_axis_head(lfhd, rfhd, lbhd, rbhd)] #doctest: +NORMALIZE_WHITESPACE
    [array([  0.03,   0.99,  0.16,  255.18]),
     array([ -1.  ,   0.03, -0.  ,  406.12]),
     array([ -0.01,  -0.16,  0.99, 1721.92]),
       array([0.,     0.,    0.,      1.])]
    """

    lfhd, rfhd, lbhd, rbhd = map(np.array, [lfhd, rfhd, lbhd, rbhd])

    # get the midpoints of the head to define the sides
    front = (lfhd + rfhd) / 2.0
    back  = (lbhd + rbhd) / 2.0
    left  = (lfhd + lbhd) / 2.0
    right = (rfhd + rbhd) / 2.0

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

    # Create the return matrix
    head_axis = np.zeros((4, 4))
    head_axis[3, 3] = 1.0
    head_axis[0, :3] = x_axis
    head_axis[1, :3] = y_axis
    head_axis[2, :3] = z_axis
    head_axis[:3, 3] = front

    return head_axis

def calc_axis_uncorrect_foot(rtoe, ltoe, r_ankle_axis, l_ankle_axis):
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
    >>> from .pycgmStatic import calc_axis_uncorrect_foot
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
    >>> [np.around(arr, 2) for arr in calc_axis_uncorrect_foot(rtoe, ltoe, r_ankle_axis, l_ankle_axis)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 0.12,   0.28,   0.95, 442.81],
           [ -0.94,   0.35,   0.01, 381.62],
           [ -0.33,  -0.89,   0.3 ,  42.66],
           [  0.  ,   0.  ,   0.  ,   1.  ]]), 
    array([[  0.06,   0.25,   0.97,  39.43],
           [ -0.94,  -0.31,   0.14, 382.44],
           [  0.33,  -0.92,   0.22,  41.78],
           [  0.  ,   0.  ,   0.  ,   1.  ]])]
    """

    # REQUIRE JOINT CENTER & AXIS
    # KNEE JOINT CENTER
    # ANKLE JOINT CENTER
    # ANKLE FLEXION AXIS
    r_ankle_axis = np.asarray(r_ankle_axis)
    l_ankle_axis = np.asarray(l_ankle_axis)

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


    r_axis = np.identity(4)
    r_axis[0, :3] = R_foot_axis[0]
    r_axis[1, :3] = R_foot_axis[1]
    r_axis[2, :3] = R_foot_axis[2]
    r_axis[:3, 3] = R

    l_axis = np.identity(4)
    l_axis[0, :3] = L_foot_axis[0]
    l_axis[1, :3] = L_foot_axis[1]
    l_axis[2, :3] = L_foot_axis[2]
    l_axis[:3, 3] = L

    foot_axis = [r_axis, l_axis]

    return foot_axis

    R_sole_delta = vsk['RightSoleDelta']
    L_sole_delta = vsk['LeftSoleDelta']

def calc_axis_flatfoot(rtoe, ltoe, rhee, lhee, ankle_axis, r_sole_delta=0, l_sole_delta=0):
    """Calculate the anatomically correct foot joint center and axis for a flat foot.

    Takes in the RTOE, LTOE, RHEE and LHEE marker positions
    as well as the ankle axes. Calculates the anatomically
    correct foot axis for flat feet.

    Markers used: RTOE, LTOE, RHEE, LHEE

    Subject Measurement values used:
        RightSoleDelta

        LeftSoleDelta

    Parameters
    ----------
    rtoe : array
        1x3 RTOE marker
    ltoe : array
        1x3 LTOE marker
    rhee : array
        1x3 RHEE marker
    lhee : array
        1x3 LHEE marker
    ankle_axis : array
        array of two 4x4 affine matrices representing the right and left ankle axes and origins
    r_sole_delta : float, optional
        The right sole delta from the subject measurement file
    l_sole_delta : float, optional
        The left sole delta from the subject measurement file

    Returns
    -------
    axis : array
        An array of two 4x4 affine matrices representing the right and left flat foot
        axes and origins

    Notes
    -----
    If the subject wears a shoe, sole_delta is applied. then axes are changed following sole_delta.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)
    >>> from .pycgmStatic import calc_axis_flatfoot
    >>> rhee = [374.01, 181.58, 49.51]
    >>> lhee = [105.30, 180.21, 47.16]
    >>> rtoe = [442.82, 381.62, 42.66]
    >>> ltoe = [39.44, 382.45, 41.79]
    >>> ankle_axis = np.array([[[ 0.72,   0.69,  -0.02, 393.76],
    ...                        [ -0.69,   0.71,  -0.12, 247.68],
    ...                        [ -0.07,   0.1 ,   0.99,  87.74],
    ...                        [  0.  ,   0.  ,   0.  ,   1.  ]],
    ...                       [[ -0.27,   0.96,  -0.1 ,  98.75],
    ...                        [ -0.96,  -0.26,   0.13, 219.47],
    ...                        [  0.1 ,   0.13,   0.99,  80.63],
    ...                        [  0.  ,   0.  ,   0.  ,   1.  ]]])
    >>> r_sole_delta = 0.45
    >>> l_sole_delta = 0.45
    >>> [np.around(arr, 2) for arr in calc_axis_flatfoot(rtoe, ltoe, rhee, lhee, ankle_axis, r_sole_delta, l_sole_delta)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ -0.51,   0.18,   0.84, 442.82],
            [ -0.79,   0.27,  -0.54, 381.62],
            [ -0.33,  -0.95,   0.  ,  42.66],
            [  0.  ,   0.  ,   0.  ,   1.  ]]), 
     array([[ -0.29,  -0.09,   0.95,  39.44],
            [ -0.91,  -0.29,  -0.31, 382.45],
            [  0.31,  -0.95,   0.  ,  41.79],
            [  0.  ,   0.  ,   0.  ,   1.  ]])]
    """
    #REQUIRED MARKERS:
    # RTOE
    # LTOE
    # RHEE
    # LHEE
    # ankle_axis

    rtoe, ltoe, rhee, lhee = map(np.asarray, [rtoe, ltoe, rhee, lhee])

    ankle_jc_right = ankle_axis[0][:3, 3]
    ankle_jc_left = ankle_axis[1][:3, 3]
    ankle_flexion_right = ankle_axis[0][1, :3] + ankle_jc_right
    ankle_flexion_left = ankle_axis[1][1, :3] + ankle_jc_left

    # Toe axis's origin is marker position of TOE
    right_origin = rtoe
    left_origin = ltoe

    ankle_jc_right = [ankle_jc_right[0], ankle_jc_right[1], ankle_jc_right[2] + r_sole_delta]
    ankle_jc_left = [ankle_jc_left[0], ankle_jc_left[1], ankle_jc_left[2] + l_sole_delta]
    #ankle_jc_right[2] += r_sole_delta
    #ankle_jc_left[2] += l_sole_delta

    # Calculate the z axis
    right_axis_z = ankle_jc_right - rtoe
    right_axis_z = np.divide(right_axis_z, np.linalg.norm(right_axis_z))

    # For foot flat, Z axis pointing same height of TOE marker from TOE to AJC
    heel_to_toe = rhee - rtoe
    heel_to_toe[2] = 0
    heel_to_toe = np.divide(heel_to_toe, np.linalg.norm(heel_to_toe))
    A = np.cross(heel_to_toe,right_axis_z)
    A = A/np.linalg.norm(A)
    B = np.cross(A,heel_to_toe)
    B = B/np.linalg.norm(B)
    C = np.cross(B,A)
    right_axis_z = C/np.linalg.norm(C)

    # Bring flexion axis from ankle axis
    right_y_flex = ankle_flexion_right - ankle_jc_right
    right_y_flex = np.divide(right_y_flex, np.linalg.norm(right_y_flex))

    # Calculate each x,y,z axis of foot using np.cross-product and make sure x,y,z axis is orthogonal each other.
    right_axis_x = np.cross(right_y_flex,right_axis_z)
    right_axis_x = np.divide(right_axis_x, np.linalg.norm(right_axis_x))

    right_axis_y = np.cross(right_axis_z,right_axis_x)
    right_axis_y = np.divide(right_axis_y, np.linalg.norm(right_axis_y))

    right_axis_z = np.cross(right_axis_x,right_axis_y)
    right_axis_z = np.divide(right_axis_z, np.linalg.norm(right_axis_z))

    right_foot_axis = [right_axis_x,right_axis_y,right_axis_z]

    # Left

    # Calculate the z axis of foot flat.
    left_axis_z = ankle_jc_left - ltoe
    left_axis_z = np.divide(left_axis_z, np.linalg.norm(left_axis_z))

    # For foot flat, Z axis pointing same height of TOE marker from TOE to AJC
    heel_to_toe = lhee - ltoe
    heel_to_toe[2] = 0
    heel_to_toe = np.divide(heel_to_toe, np.linalg.norm(heel_to_toe))
    A = np.cross(heel_to_toe,left_axis_z)
    A = A/np.linalg.norm(A)
    B = np.cross(A,heel_to_toe)
    B = B/np.linalg.norm(B)
    C = np.cross(B,A)
    left_axis_z = C/np.linalg.norm(C)

    # Bring flexion axis from ankle axis
    left_y_flex = ankle_flexion_left - ankle_jc_left
    left_y_flex = np.divide(left_y_flex, np.linalg.norm(left_y_flex))

    # Calculate each x,y,z axis of foot using np.cross-product and make sure (x, y, z) axes are orthogonal to each other
    left_axis_x = np.cross(left_y_flex,left_axis_z)
    left_axis_x = np.divide(left_axis_x, np.linalg.norm(left_axis_x))

    left_axis_y = np.cross(left_axis_z,left_axis_x)
    left_axis_y = np.divide(left_axis_y, np.linalg.norm(left_axis_y))

    left_axis_z = np.cross(left_axis_x,left_axis_y)
    left_axis_z = np.divide(left_axis_z, np.linalg.norm(left_axis_z))

    left_foot_axis = [left_axis_x,left_axis_y,left_axis_z]

    right_axis = np.identity(4)
    right_axis[:3, :3] = right_foot_axis
    right_axis[:3, 3] = right_origin

    left_axis = np.identity(4)
    left_axis[:3, :3] = left_foot_axis
    left_axis[:3, 3] = left_origin

    axis = [right_axis, left_axis]

    return axis

def calc_axis_nonflatfoot(rtoe, ltoe, rhee, lhee, ankle_axis):
    """Calculate the anatomically correct foot joint center and axis for a non-flat foot.

    Takes in the RTOE, LTOE, RHEE and LHEE marker positions
    as well as the ankle axes. Calculates the anatomically
    correct foot axis for non-flat feet.

    Markers used: RTOE, LTOE, RHEE, LHEE

    Parameters
    ----------
    rtoe : array
        1x3 RTOE marker
    ltoe : array
        1x3 LTOE marker
    rhee : array
        1x3 RHEE marker
    lhee : array
        1x3 LHEE marker
    ankle_axis : array
        array of two 4x4 affine matrices representing the right and left ankle axes and origins

    Returns
    -------
    axis : array
        An array of two 4x4 affine matrices representing the right and left non-flat foot
        axes and origins


    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)
    >>> from .pycgmStatic import calc_axis_nonflatfoot
    >>> rhee = [374.01, 181.58, 49.51]
    >>> lhee = [105.30, 180.21, 47.16]
    >>> rtoe = [442.82, 381.62, 42.66]
    >>> ltoe = [39.44, 382.45, 41.79]
    >>> ankle_axis = np.array([[[ 0.72,   0.69,  -0.02, 393.76],
    ...                        [ -0.69,   0.71,  -0.12, 247.68],
    ...                        [ -0.07,   0.1 ,   0.99,  87.74],
    ...                        [  0.  ,   0.  ,   0.  ,   1.  ]],
    ...                       [[ -0.27,   0.96,  -0.1 ,  98.75],
    ...                        [ -0.96,  -0.26,   0.13, 219.47],
    ...                        [  0.1 ,   0.13,   0.99,  80.63],
    ...                        [  0.  ,   0.  ,   0.  ,   1.  ]]])
    >>> [np.around(arr, 2) for arr in calc_axis_nonflatfoot(rtoe, ltoe, rhee, lhee, ankle_axis)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ -0.1 ,   0.07,   0.99, 442.82],
            [ -0.94,   0.32,  -0.12, 381.62],
            [ -0.33,  -0.95,   0.03,  42.66],
            [  0.  ,   0.  ,   0.  ,   1.  ]]), 
     array([[  0.12,   0.06,   0.99,  39.44],
            [ -0.94,  -0.3 ,   0.13, 382.45],
            [  0.31,  -0.95,   0.03,  41.79],
            [  0.  ,   0.  ,   0.  ,   1.  ]])]
    """

    #REQUIRED MARKERS:
    # RTOE
    # LTOE
    # RHEE
    # LHEE
    # ankle_axis

    rtoe, ltoe, rhee, lhee, ankle_axis = map(np.asarray, [rtoe, ltoe, rhee, lhee, ankle_axis])

    ankle_jc_right = ankle_axis[0][:3, 3]
    ankle_jc_left = ankle_axis[1][:3, 3]
    ankle_flexion_right = ankle_axis[0][1, :3] + ankle_jc_right
    ankle_flexion_left = ankle_axis[1][1, :3] + ankle_jc_left

    # Toe axis's origin is marker position of TOE
    right_origin = rtoe
    left_origin = ltoe

    # in case of non foot flat we just use the HEE marker
    right_axis_z = rhee - rtoe
    right_axis_z = right_axis_z/norm3d(right_axis_z)

    y_flex_R = ankle_flexion_right - ankle_jc_right
    y_flex_R = y_flex_R/norm3d(y_flex_R)

    right_axis_x = cross(y_flex_R,right_axis_z)
    right_axis_x = right_axis_x/norm3d(right_axis_x)

    right_axis_y = cross(right_axis_z,right_axis_x)
    right_axis_y = right_axis_y/norm3d(right_axis_y)

    right_foot_axis = [right_axis_x,right_axis_y,right_axis_z]

    # Left
    left_axis_z = lhee - ltoe
    left_axis_z = left_axis_z/norm3d(left_axis_z)

    y_flex_L = ankle_flexion_left - ankle_jc_left
    y_flex_L = y_flex_L/norm3d(y_flex_L)

    left_axis_x = cross(y_flex_L,left_axis_z)
    left_axis_x = left_axis_x/norm3d(left_axis_x)

    left_axis_y = cross(left_axis_z,left_axis_x)
    left_axis_y = left_axis_y/norm3d(left_axis_y)

    left_foot_axis = [left_axis_x,left_axis_y,left_axis_z]

    right_axis = np.identity(4)
    right_axis[:3, :3] = right_foot_axis
    right_axis[:3, 3] = right_origin

    left_axis = np.identity(4)
    left_axis[:3, :3] = left_foot_axis
    left_axis[:3, 3] = left_origin

    axis = [right_axis, left_axis]

    return axis

def calc_static_angle_ankle(axis_p, axis_d):
    """Static angle calculation function.

    Takes in two axes and returns the rotation, flexion,
    and abduction angles in degrees.
    Uses the inverse Euler rotation matrix in YXZ order.

    Since we use arcsin we must check if the angle is in area between -pi/2 and pi/2
    but because the static offset angle is less than pi/2, it doesn't matter.

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

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import calc_static_angle_ankle
    >>> axis_p = np.array([[ 0.59,  0.11,  0.16, 0],
    ...                    [-0.13, -0.10, -0.90, 0],
    ...                    [ 0.94, -0.05,  0.75, 0],
    ...                    [ 0,     0,     0,    0]])
    >>> axis_d = np.array([[ 0.17,  0.69, -0.37, 0],
    ...                    [ 0.14, -0.39,  0.94, 0],
    ...                    [-0.16, -0.53, -0.60, 0],
    ...                    [ 0,     0,     0,    0]])
    >>> np.around(calc_static_angle_ankle(axis_p, axis_d), 2)
    array([0.48, 1.  , 1.56])
    """
    # make inverse matrix of axis_p
    axis_p = np.asarray(axis_p)
    axis_d = np.asarray(axis_d)

    axis_p = axis_p[:3, :3]
    axis_d = axis_d[:3, :3]

    axis_p_inv = np.linalg.inv(axis_p)

    # M is multiply of axis_d and axis_p_inv
    M = matrixmult(axis_d, axis_p_inv)

    # This is the angle calculation in YXZ Euler angle
    a = M[2][1] / sqrt((M[2][0] * M[2][0]) + (M[2][2] * M[2][2]))
    b = -1 * M[2][0] / M[2][2]
    g = -1 * M[0][1] / M[1][1]

    gamma =np.arctan(g)
    alpha = np.arctan(a)
    beta = np.arctan(b)

    angle = [alpha, beta, gamma]
    return angle

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
    >>> from .pycgmStatic import calc_joint_center
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

    theta = acos(delta/np.linalg.norm(vec_2))

    alpha = cos(theta*2)
    beta = sin(theta*2)

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


def norm2d(v):
    """2D Vector normalization function

    This function calculates the normalization of a 3-dimensional vector.

    Parameters
    ----------
    v : list
        A 3-element list.

    Returns
    -------
    float
        The normalization of the vector as a float.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import norm2d
    >>> v = [50.0, 96.37, 264.85]
    >>> np.around(norm2d(v), 2)
    286.24
    """
    try:
        return sqrt((v[0]*v[0]+v[1]*v[1]+v[2]*v[2]))
    except:
        return np.nan

def norm3d(v):
    """3D Vector normalization function

    This function calculates the normalization of a 3-dimensional vector.

    Parameters
    ----------
    v : list
        A 3-element list.

    Returns
    -------
    array
        The normalization of the vector returned as a float in an array.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import norm3d
    >>> v = [124.98, 368.64, 18.43]
    >>> np.array(norm3d(v).round(2))
    array(389.69)
    """
    try:
        return np.asarray(sqrt((v[0]*v[0]+v[1]*v[1]+v[2]*v[2])))
    except:
        return np.nan

def normDiv(v):
    """Normalized divison function

    This function calculates the normalization division of a 3-dimensional vector.

    Parameters
    ----------
    v : list
        A 3-element list.

    Returns
    -------
    list
        The divison normalization of the vector returned as a float in a list.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import normDiv
    >>> v = [1.45, 1.94, 2.49]
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
    """Matrix multiplication function

    This function returns the product of a matrix multiplication given two matrices.

    Let the dimension of the matrix A be: m by n,
    let the dimension of the matrix B be: p by q,
    multiplication will only possible if n = p,
    thus creating a matrix of m by q size.

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
    >>> from .pycgmStatic import matrixmult
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

def cross(a, b):
    """Cross Product function

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
    >>> from .pycgmStatic import cross
    >>> a = [12.83, 61.25, 99.6]
    >>> b = [14.8, 61.72, 95.44]
    >>> np.around(cross(a, b), 2)
    array([-301.61,  249.58, -114.63])
    """
    c = [a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]]

    return c
