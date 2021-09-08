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
        pelvis_origin,pelvis_axis,sacrum = pelvisJointCenter(frame)
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
        RF_axis2 = rotaxis_nonfootflat(frame,ankle_JC)
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
        RF_axis3 = rotaxis_footflat(frame,ankle_JC,vsk=vsk)
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

def pelvisJointCenter(frame):
    """Make the Pelvis Axis function

    Takes in a dictionary of x,y,z positions and marker names, as well as an index.
    Calculates the pelvis joint center and axis and returns both.

    Markers used: RASI,LASI,RPSI,LPSI
    Other landmarks used: origin, sacrum

    Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990) and then normalized.
    Pelvis Y_axis: LASI-RASI x,y,z positions, then normalized.
    Pelvis Z_axis: Cross product of x_axis and y_axis.

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.

    Returns
    -------
    pelvis : list
        Returns a list that contains the pelvis origin in a 1x3 array of xyz values,
        a 4x1x3 array composed of the pelvis x, y, z axes components,
        and the sacrum x, y, z position.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import pelvisJointCenter
    >>> frame = {'RASI': np.array([ 395.37,  428.1, 1036.83]),
    ...          'LASI': np.array([ 183.19,  422.79, 1033.07]),
    ...          'RPSI': np.array([ 341.42,  246.72, 1055.99]),
    ...          'LPSI': np.array([ 255.8,  241.42, 1057.3]) }
    >>> [np.around(arr, 2) for arr in pelvisJointCenter(frame)] #doctest: +NORMALIZE_WHITESPACE
    [array([ 289.28,  425.45, 1034.95]), array([[ 289.26,  426.44, 1034.83],
       [ 288.28,  425.42, 1034.93],
       [ 289.26,  425.56, 1035.94]]), array([ 298.61,  244.07, 1056.64])]
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

    if 'SACR' in frame:
        sacrum = frame['SACR']


    # REQUIRED LANDMARKS:
    # origin
    # sacrum

    # Origin is Midpoint between RASI and LASI
    origin = (RASI+LASI)/2.0

    # print('Static calc Origin: ',origin)
    # print('Static calc RASI: ',RASI)
    # print('Static calc LASI: ',LASI)


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

    pelvis = [origin,pelvis_axis,sacrum]

    #print('Pelvis JC in static: ',pelvis)
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


def hipAxisCenter(l_hip_jc,r_hip_jc,pelvis_axis):
    """Calculate the hip joint axis function.

    Takes in a hip joint center of x,y,z positions as well as an index.
    and takes the hip joint center and pelvis origin/axis from previous functions.
    Calculates the hip axis and returns hip joint origin and axis.

    Hip center axis: mean at each x,y,z axis of the left and right hip joint center.
    Hip axis: summation of the pelvis and hip center axes.

    Parameters
    ----------
    l_hip_jc, r_hip_jc: array
        Array of R_hip_jc and L_hip_jc each x,y,z position.
    pelvis_axis : array
        An array of pelvis origin and axis. The axis is also composed of 3 arrays,
        each contain the x axis, y axis and z axis.

    Returns
    -------
    hipaxis_center, axis : list
        Returns a list that contains the hip axis center in a 1x3 list of xyz values,
        which is then followed by a 3x2x3 list composed of the hip axis center x, y, and z
        axis components. The xyz axis components are 2x3 lists consisting of the axis center
        in the first dimension and the direction of the axis in the second dimension.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import hipAxisCenter
    >>> r_hip_jc = [182.57, 339.43, 935.53]
    >>> l_hip_jc = [308.38, 322.80, 937.99]
    >>> pelvis_axis = [np.array([251.61, 391.74, 1032.89]),
    ...                np.array([[251.74, 392.73, 1032.79],
    ...                    [250.62, 391.87, 1032.87],
    ...                    [251.60, 391.85, 1033.89]]),
    ...                np.array([231.58, 210.25, 1052.25])]
    >>> [np.around(arr,8) for arr in hipAxisCenter(l_hip_jc,r_hip_jc,pelvis_axis)] #doctest: +NORMALIZE_WHITESPACE
    [array([245.475, 331.115, 936.76 ]),
    array([[245.605, 332.105, 936.66 ],
           [244.485, 331.245, 936.74 ],
           [245.465, 331.225, 937.76 ]])]
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


def ankleJointCenter(frame,knee_JC,delta,vsk=None):
    """Calculate the ankle joint center and axis function.

    Takes in a dictionary of xyz positions and marker names, an index
    and the knee axis.
    Calculates the ankle joint axis and returns the ankle origin and axis.

    Markers used: tib_R, tib_L, ank_R, ank_L, knee_JC
    Subject Measurement values used: RightKneeWidth, LeftKneeWidth

    Ankle Axis: Computed using Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013).

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.
    knee_JC : array
        An array of knee_JC each x,y,z position.
    delta : float
        The length from marker to joint center, retrieved from subject measurement file.
    vsk : dict, optional
        A dictionary containing subject measurements from a VSK file.

    Returns
    -------
    R, L, axis : list
        Returns a list that contains the ankle axis origin in 1x3 arrays of xyz values
        and a 3x2x3 list composed of the ankle origin, x, y, and z axis components. The
        xyz axis components are 2x3 lists consisting of the origin in the first
        dimension and the direction of the axis in the second dimension.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import ankleJointCenter
    >>> vsk = { 'RightAnkleWidth' : 70.0, 'LeftAnkleWidth' : 70.0,
    ...         'RightTibialTorsion': 0.0, 'LeftTibialTorsion' : 0.0}
    >>> frame = { 'RTIB': np.array([433.98, 211.93, 273.30]),
    ...           'LTIB': np.array([50.04, 235.91, 364.32]),
    ...           'RANK': np.array([422.77, 217.74, 92.86]),
    ...           'LANK': np.array([58.57, 208.55, 86.17]) }
    >>> knee_JC = [np.array([364.18, 292.17, 515.19]),
    ...           np.array([143.55, 279.90, 524.78]),
    ...           np.array([[[364.65, 293.07, 515.18],
    ...           [363.29, 292.61, 515.04],
    ...           [364.05, 292.24, 516.18]],
    ...           [[143.66, 280.89, 524.63],
    ...           [142.56, 280.02, 524.86],
    ...            [143.65, 280.05, 525.77]]])]
    >>> delta = 0
    >>> [np.around(arr, 2) for arr in ankleJointCenter(frame,knee_JC,delta,vsk)] #doctest: +NORMALIZE_WHITESPACE
    [array([393.76, 247.68,  87.74]), array([ 98.75, 219.47,  80.63]), array([[[394.48, 248.37,  87.71],
    [393.07, 248.39,  87.61],
    [393.69, 247.78,  88.73]],
    [[ 98.47, 220.43,  80.53],
    [ 97.79, 219.21,  80.76],
    [ 98.85, 219.6 ,  81.62]]])]
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

    Raxis = [[cos(R_torsion)*Raxis[0][0]-sin(R_torsion)*Raxis[1][0],
            cos(R_torsion)*Raxis[0][1]-sin(R_torsion)*Raxis[1][1],
            cos(R_torsion)*Raxis[0][2]-sin(R_torsion)*Raxis[1][2]],
            [sin(R_torsion)*Raxis[0][0]+cos(R_torsion)*Raxis[1][0],
            sin(R_torsion)*Raxis[0][1]+cos(R_torsion)*Raxis[1][1],
            sin(R_torsion)*Raxis[0][2]+cos(R_torsion)*Raxis[1][2]],
            [Raxis[2][0],Raxis[2][1],Raxis[2][2]]]

    Laxis = [[cos(L_torsion)*Laxis[0][0]-sin(L_torsion)*Laxis[1][0],
            cos(L_torsion)*Laxis[0][1]-sin(L_torsion)*Laxis[1][1],
            cos(L_torsion)*Laxis[0][2]-sin(L_torsion)*Laxis[1][2]],
            [sin(L_torsion)*Laxis[0][0]+cos(L_torsion)*Laxis[1][0],
            sin(L_torsion)*Laxis[0][1]+cos(L_torsion)*Laxis[1][1],
            sin(L_torsion)*Laxis[0][2]+cos(L_torsion)*Laxis[1][2]],
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

def footJointCenter(frame,static_info,ankle_JC,knee_JC,delta):
    """Calculate the foot joint center and axis function.

    Takes in a dictionary of xyz positions and marker names, the ankle axis and
    knee axis.
    Calculates the foot joint axis by rotating the incorrect foot joint axes
    about the offset angle.
    Returns the foot axis origin and axis.

    In the case of the foot joint center, we've already made 2 kinds of axes
    for the static offset angle and then, we call this static offset angle as an
    input of this function for thedynamic trial.

    Special Cases:

    (anatomically uncorrected foot axis)
    If flat foot, make the reference markers instead of HEE marker whose height
    is the same as TOE marker's height. Else use the HEE marker for making Z axis.

    Markers used: RTOE,LTOE,RHEE, LHEE
    Other landmarks used: ANKLE_FLEXION_AXIS
    Subject Measurement values used: RightStaticRotOff, RightStaticPlantFlex, LeftStaticRotOff, LeftStaticPlantFlex

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.
    static_info : array
        An array containing offset angles.
    ankle_JC : array
        An array of ankle_JC containing the x,y,z axes marker positions of the ankle joint center.
    knee_JC : array
        An array of ankle_JC containing the x,y,z axes marker positions of the knee joint center.
    delta
        The length from marker to joint center, retrieved from subject measurement file.

    Returns
    -------
    R, L, foot_axis : list
        Returns a list that contain the foot axis' (right and left) origin in 1x3 arrays
        of xyz values and a 2x3x3 list composed of the foot axis center x, y, and z
        axis components. The xyz axis components are 2x3 lists consisting of the axis center
        in the first dimension and the direction of the axis in the second dimension.
        This function also saves the static offset angle in a global variable.

    Notes
    -----
    Axis changes the following in the static info.

    You can set the static_info with the button and this will calculate the offset angles.
    The first setting, the foot axis shows the uncorrected foot anatomical reference axis(Z_axis point to the AJC from TOE).

    If you press the static_info button so if static_info is not None,
    then the static offset angles are applied to the reference axis.
    The reference axis is Z axis point to HEE from TOE

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import footJointCenter
    >>> frame = { 'RHEE': np.array([374.01, 181.58, 49.51]),
    ...           'LHEE': np.array([105.30, 180.21, 47.16]),
    ...           'RTOE': np.array([442.82, 381.62, 42.66]),
    ...           'LTOE': np.array([39.44, 382.45, 41.79])}
	>>> static_info = [[0.03, 0.15, 0],
	...               [0.01, 0.02, 0]]
    >>> knee_JC = [np.array([364.18, 292.17, 515.19]),
    ...           np.array([143.55, 279.90, 524.78]),
    ...           np.array([[[364.65, 293.07, 515.19],
    ...           [363.29, 292.61, 515.04],
    ...           [364.05, 292.24, 516.18]],
    ...           [[143.66, 280.89, 524.63],
    ...           [142.56, 280.02, 524.86],
    ...           [143.65, 280.05, 525.77]]])]
    >>> ankle_JC = [np.array([393.76, 247.68, 87.74]),
    ...            np.array([98.75, 219.47, 80.63]),
    ...            [[np.array([394.48, 248.37, 87.72]),
    ...            np.array([393.07, 248.39, 87.62]),
    ...            np.array([393.69, 247.78, 88.73])],
    ...            [np.array([98.47, 220.43, 80.53]),
    ...            np.array([97.79, 219.21, 80.76]),
    ...            np.array([98.85, 219.60, 81.62])]]]
    >>> delta = 0
    >>> [np.around(arr,2) for arr in footJointCenter(frame,static_info,ankle_JC,knee_JC,delta)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.82, 381.62,  42.66]), array([ 39.44, 382.45,  41.79]), array([[[442.89, 381.76,  43.65],
            [441.89, 382.  ,  42.67],
            [442.45, 380.7 ,  42.82]],
           [[ 39.51, 382.68,  42.76],
            [ 38.5 , 382.15,  41.93],
            [ 39.76, 381.53,  41.99]]])]
    """
    import math

      #REQUIRED MARKERS:
      # RTOE
      # LTOE
      # RHEE
      # LHEE

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
    R_alpha = static_info[0][0]*-1
    R_beta = static_info[0][1]
    #R_gamma = static_info[0][2]
    L_alpha = static_info[1][0]
    L_beta = static_info[1][1]
    #L_gamma = static_info[1][2]

    R_alpha = np.around(math.degrees(static_info[0][0]*-1),decimals=5)
    R_beta = np.around(math.degrees(static_info[0][1]),decimals=5)
    #R_gamma = np.around(math.degrees(static_info[0][2]),decimals=5)
    L_alpha = np.around(math.degrees(static_info[1][0]),decimals=5)
    L_beta = np.around(math.degrees(static_info[1][1]),decimals=5)
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

def uncorrect_footaxis(frame,ankle_JC):
    """Calculate the anatomically uncorrected foot joint center and axis function.

    Takes in a dictionary of xyz positions and marker names
    and takes the ankle axis.
    Calculate the anatomical uncorrect foot axis.

    Markers used: RTOE, LTOE

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.
    ankle_JC : array
        An array of ankle_JC each x,y,z position.

    Returns
    -------
    R, L, foot_axis : list
        Returns a list representing the incorrect foot joint center, the list contains two 1x3 arrays
        representing the foot axis origin x, y, z positions and a 3x2x3 list
        containing the foot axis center in the first dimension and the direction of the
        axis in the second dimension. This will be used for calculating static offset angle
        in static calibration.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import uncorrect_footaxis
    >>> frame = { 'RTOE': [442.82, 381.62, 42.66],
    ...           'LTOE': [39.44, 382.45, 41.79]}
    >>> ankle_JC = [np.array([393.76, 247.68, 87.74]),
    ...            np.array([98.75, 219.47, 80.63]),
    ...            [[np.array([394.48, 248.37, 87.72]),
    ...            np.array([393.07, 248.39, 87.62]),
    ...            np.array([393.69, 247.78, 88.73])],
    ...            [np.array([98.47, 220.43, 80.53]),
    ...            np.array([97.79, 219.21, 80.76]),
    ...            np.array([98.85, 219.60, 81.62])]]]
    >>> [np.around(arr, 2) for arr in uncorrect_footaxis(frame,ankle_JC)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.82, 381.62,  42.66]),
    array([ 39.44, 382.45,  41.79]),
    array([[[442.94, 381.9 ,  43.61],
            [441.88, 381.97,  42.68],
            [442.49, 380.72,  42.96]],
           [[ 39.5 , 382.7 ,  42.76],
            [ 38.5 , 382.14,  41.93],
            [ 39.77, 381.53,  42.01]]])]
    """

    #REQUIRED MARKERS:
    # RTOE
    # LTOE
    # ankle_JC
    TOE_R = frame['RTOE']
    TOE_L = frame['LTOE']

    ankle_JC_R = ankle_JC[0]
    ankle_JC_L = ankle_JC[1]
    ankle_flexion_R = ankle_JC[2][0][1]
    ankle_flexion_L = ankle_JC[2][1][1]

    # Foot axis's origin is marker position of TOE
    R = TOE_R
    L = TOE_L

    # z axis is from Toe to AJC and normalized.
    R_axis_z = [ankle_JC_R[0]-TOE_R[0],ankle_JC_R[1]-TOE_R[1],ankle_JC_R[2]-TOE_R[2]]
    R_axis_z_div = norm2d(R_axis_z)
    R_axis_z = [R_axis_z[0]/R_axis_z_div,R_axis_z[1]/R_axis_z_div,R_axis_z[2]/R_axis_z_div]

    # Bring y flexion axis from ankle axis.
    y_flex_R = [ankle_flexion_R[0]-ankle_JC_R[0],ankle_flexion_R[1]-ankle_JC_R[1],ankle_flexion_R[2]-ankle_JC_R[2]]
    y_flex_R_div = norm2d(y_flex_R)
    y_flex_R = [y_flex_R[0]/y_flex_R_div,y_flex_R[1]/y_flex_R_div,y_flex_R[2]/y_flex_R_div]

    # Calculate x axis by cross-product of ankle flexion axis and z axis.
    R_axis_x = cross(y_flex_R,R_axis_z)
    R_axis_x_div = norm2d(R_axis_x)
    R_axis_x = [R_axis_x[0]/R_axis_x_div,R_axis_x[1]/R_axis_x_div,R_axis_x[2]/R_axis_x_div]

    # Calculate y axis by cross-product of z axis and x axis.
    R_axis_y = cross(R_axis_z,R_axis_x)
    R_axis_y_div = norm2d(R_axis_y)
    R_axis_y = [R_axis_y[0]/R_axis_y_div,R_axis_y[1]/R_axis_y_div,R_axis_y[2]/R_axis_y_div]

    # Attach each axes to origin.
    R_axis_x = [R_axis_x[0]+R[0],R_axis_x[1]+R[1],R_axis_x[2]+R[2]]
    R_axis_y = [R_axis_y[0]+R[0],R_axis_y[1]+R[1],R_axis_y[2]+R[2]]
    R_axis_z = [R_axis_z[0]+R[0],R_axis_z[1]+R[1],R_axis_z[2]+R[2]]

    R_foot_axis = [R_axis_x,R_axis_y,R_axis_z]

    # Left

    # z axis is from Toe to AJC and normalized.
    L_axis_z = [ankle_JC_L[0]-TOE_L[0],ankle_JC_L[1]-TOE_L[1],ankle_JC_L[2]-TOE_L[2]]
    L_axis_z_div = norm2d(L_axis_z)
    L_axis_z = [L_axis_z[0]/L_axis_z_div,L_axis_z[1]/L_axis_z_div,L_axis_z[2]/L_axis_z_div]

    # Bring y flexion axis from ankle axis.
    y_flex_L = [ankle_flexion_L[0]-ankle_JC_L[0],ankle_flexion_L[1]-ankle_JC_L[1],ankle_flexion_L[2]-ankle_JC_L[2]]
    y_flex_L_div = norm2d(y_flex_L)
    y_flex_L = [y_flex_L[0]/y_flex_L_div,y_flex_L[1]/y_flex_L_div,y_flex_L[2]/y_flex_L_div]

    # Calculate x axis by cross-product of ankle flexion axis and z axis.
    L_axis_x = cross(y_flex_L,L_axis_z)
    L_axis_x_div = norm2d(L_axis_x)
    L_axis_x = [L_axis_x[0]/L_axis_x_div,L_axis_x[1]/L_axis_x_div,L_axis_x[2]/L_axis_x_div]

    # Calculate y axis by cross-product of z axis and x axis.
    L_axis_y = cross(L_axis_z,L_axis_x)
    L_axis_y_div = norm2d(L_axis_y)
    L_axis_y = [L_axis_y[0]/L_axis_y_div,L_axis_y[1]/L_axis_y_div,L_axis_y[2]/L_axis_y_div]

    # Attach each axis to origin.
    L_axis_x = [L_axis_x[0]+L[0],L_axis_x[1]+L[1],L_axis_x[2]+L[2]]
    L_axis_y = [L_axis_y[0]+L[0],L_axis_y[1]+L[1],L_axis_y[2]+L[2]]
    L_axis_z = [L_axis_z[0]+L[0],L_axis_z[1]+L[1],L_axis_z[2]+L[2]]

    L_foot_axis = [L_axis_x,L_axis_y,L_axis_z]

    foot_axis = [R_foot_axis,L_foot_axis]

    return [R,L,foot_axis]

def rotaxis_footflat(frame,ankle_JC,vsk=None):
    """Calculate the anatomically correct foot joint center and axis function for a flat foot.

    Takes in a dictionary of xyz positions and marker names
    and the ankle axis then Calculates the anatomically
    correct foot axis for a flat foot.

    Markers used: RTOE, LTOE, RHEE, LHEE

    Parameters
    ----------
    frame : array
        Dictionary of marker lists.
    ankle_JC : array
        An array of ankle_JC each x,y,z position.
    vsk : dict, optional
        A dictionary containing subject measurements from a VSK file.

    Returns
    -------
    R, L, foot_axis: list
        Returns a list representing the correct foot joint center for a flat foot,
        the list contains 2 1x3 arrays representing the foot axis origin x, y, z
        positions and a 3x2x3 list containing the foot axis center in the first
        dimension and the direction of the axis in the second dimension.

    Notes
    -----
    If the subject wears shoe, Soledelta is applied. then axes are changed following Soledelta.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import rotaxis_footflat
    >>> frame = { 'RHEE': [374.01, 181.58, 49.51],
    ...            'LHEE': [105.30, 180.21, 47.16],
    ...            'RTOE': [442.82, 381.62, 42.66],
    ...            'LTOE': [39.44, 382.45, 41.79]}
    >>> ankle_JC = [np.array([393.76, 247.68, 87.74]),
    ...            np.array([98.75, 219.47, 80.63]),
    ...            [[np.array([394.48, 248.37, 87.72]),
    ...            np.array([393.07, 248.39, 87.62]),
    ...           np.array([393.69, 247.78, 88.73])],
    ...            [np.array([98.48, 220.43, 80.53]),
    ...            np.array([97.79, 219.21, 80.76]),
    ...            np.array([98.85, 219.60, 81.62])]]]
    >>> vsk = { 'RightSoleDelta': 0.45, 'LeftSoleDelta': 0.45}
    >>> [np.around(arr, 2) for arr in rotaxis_footflat(frame,ankle_JC,vsk)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.82, 381.62,  42.66]),
    array([ 39.44, 382.45,  41.79]),
    array([[[442.31, 381.8 ,  43.5 ],
            [442.03, 381.89,  42.12],
            [442.49, 380.67,  42.66]],
           [[ 39.15, 382.36,  42.74],
            [ 38.53, 382.16,  41.48],
            [ 39.75, 381.5 ,  41.79]]])]
    """
    #Get Global Values

    R_sole_delta = vsk['RightSoleDelta']
    L_sole_delta = vsk['LeftSoleDelta']

    #REQUIRED MARKERS:
    # RTOE
    # LTOE
    # ankle_JC

    TOE_R = frame['RTOE']
    TOE_L = frame['LTOE']
    HEE_R = frame['RHEE']
    HEE_L = frame['LHEE']
    ankle_JC_R = ankle_JC[0]
    ankle_JC_L = ankle_JC[1]
    ankle_flexion_R = ankle_JC[2][0][1]
    ankle_flexion_L = ankle_JC[2][1][1]

    # Toe axis's origin is marker position of TOE
    R = TOE_R
    L = TOE_L

    ankle_JC_R = [ankle_JC_R[0],ankle_JC_R[1],ankle_JC_R[2]+R_sole_delta]
    ankle_JC_L = [ankle_JC_L[0],ankle_JC_L[1],ankle_JC_L[2]+L_sole_delta]

    # this is the way to calculate the z axis
    R_axis_z = [ankle_JC_R[0]-TOE_R[0],ankle_JC_R[1]-TOE_R[1],ankle_JC_R[2]-TOE_R[2]]
    R_axis_z = R_axis_z/norm3d(R_axis_z)

    # For foot flat, Z axis pointing same height of TOE marker from TOE to AJC
    hee2_toe =[HEE_R[0]-TOE_R[0],HEE_R[1]-TOE_R[1],TOE_R[2]-TOE_R[2]]
    hee2_toe = hee2_toe/norm3d(hee2_toe)
    A = cross(hee2_toe,R_axis_z)
    A = A/norm3d(A)
    B = cross(A,hee2_toe)
    B = B/norm3d(B)
    C = cross(B,A)
    R_axis_z = C/norm3d(C)

    # Bring flexion axis from ankle axis.
    y_flex_R = [ankle_flexion_R[0]-ankle_JC_R[0],ankle_flexion_R[1]-ankle_JC_R[1],ankle_flexion_R[2]-ankle_JC_R[2]]
    y_flex_R = y_flex_R/norm3d(y_flex_R)

    # Calculate each x,y,z axis of foot using cross-product and make sure x,y,z axis is orthogonal each other.
    R_axis_x = cross(y_flex_R,R_axis_z)
    R_axis_x = R_axis_x/norm3d(R_axis_x)

    R_axis_y = cross(R_axis_z,R_axis_x)
    R_axis_y = R_axis_y/norm3d(R_axis_y)

    R_axis_z = cross(R_axis_x,R_axis_y)
    R_axis_z = R_axis_z/norm3d(R_axis_z)

    # Attach each axis to origin.
    R_axis_x = [R_axis_x[0]+R[0],R_axis_x[1]+R[1],R_axis_x[2]+R[2]]
    R_axis_y = [R_axis_y[0]+R[0],R_axis_y[1]+R[1],R_axis_y[2]+R[2]]
    R_axis_z = [R_axis_z[0]+R[0],R_axis_z[1]+R[1],R_axis_z[2]+R[2]]

    R_foot_axis = [R_axis_x,R_axis_y,R_axis_z]

    # Left

    # this is the way to calculate the z axis of foot flat.
    L_axis_z = [ankle_JC_L[0]-TOE_L[0],ankle_JC_L[1]-TOE_L[1],ankle_JC_L[2]-TOE_L[2]]
    L_axis_z = L_axis_z/norm3d(L_axis_z)

    # For foot flat, Z axis pointing same height of TOE marker from TOE to AJC
    hee2_toe =[HEE_L[0]-TOE_L[0],HEE_L[1]-TOE_L[1],TOE_L[2]-TOE_L[2]]
    hee2_toe = hee2_toe/norm3d(hee2_toe)
    A = cross(hee2_toe,L_axis_z)
    A = A/norm3d(A)
    B = cross(A,hee2_toe)
    B = B/norm3d(B)
    C = cross(B,A)
    L_axis_z = C/norm3d(C)

    # Bring flexion axis from ankle axis.
    y_flex_L = [ankle_flexion_L[0]-ankle_JC_L[0],ankle_flexion_L[1]-ankle_JC_L[1],ankle_flexion_L[2]-ankle_JC_L[2]]
    y_flex_L = y_flex_L/norm3d(y_flex_L)

    # Calculate each x,y,z axis of foot using cross-product and make sure x,y,z axis is orthogonal each other.
    L_axis_x = cross(y_flex_L,L_axis_z)
    L_axis_x = L_axis_x/norm3d(L_axis_x)

    L_axis_y = cross(L_axis_z,L_axis_x)
    L_axis_y = L_axis_y/norm3d(L_axis_y)

    L_axis_z = cross(L_axis_x,L_axis_y)
    L_axis_z = L_axis_z/norm3d(L_axis_z)

    # Attach each axis to origin.
    L_axis_x = [L_axis_x[0]+L[0],L_axis_x[1]+L[1],L_axis_x[2]+L[2]]
    L_axis_y = [L_axis_y[0]+L[0],L_axis_y[1]+L[1],L_axis_y[2]+L[2]]
    L_axis_z = [L_axis_z[0]+L[0],L_axis_z[1]+L[1],L_axis_z[2]+L[2]]

    L_foot_axis = [L_axis_x,L_axis_y,L_axis_z]

    foot_axis = [R_foot_axis,L_foot_axis]

    return [R,L,foot_axis]

def rotaxis_nonfootflat(frame,ankle_JC):
    """Calculate the anatomically correct foot joint center and axis function for a non-flat foot.

    Takes in a dictionary of xyz positions & marker names
    and the ankle axis then calculates the anatomically
    correct foot axis for a non-flat foot.

    Markers used: RTOE, LTOE, RHEE, LHEE

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.
    ankle_JC : array
        An array of ankle_JC each x,y,z position.

    Returns
    -------
    R, L, foot_axis: list
        Returns a list representing the correct foot joint center for a non-flat foot,
        the list contains two 1x3 arrays representing the foot axis origin x, y, z
        positions and a 3x2x3 list containing the foot axis center in the first
        dimension and the direction of the axis in the second dimension.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import rotaxis_nonfootflat
    >>> frame = { 'RHEE': [374.01, 181.58, 49.51],
    ...            'LHEE': [105.30, 180.21, 47.16],
    ...            'RTOE': [442.82, 381.62, 42.66],
    ...            'LTOE': [39.44, 382.45, 41.79]}
    >>> ankle_JC = [np.array([393.76, 247.68, 87.74]),
    ...            np.array([98.75, 219.47, 80.63]),
    ...            [[np.array([394.48, 248.37, 87.72]),
    ...            np.array([393.07, 248.39, 87.62]),
    ...            np.array([393.69, 247.78, 88.73])],
    ...            [np.array([98.47, 220.43, 80.53]),
    ...            np.array([97.79, 219.21, 80.76]),
    ...            np.array([98.85, 219.60, 81.62])]]]
    >>> [np.around(arr, 2) for arr in rotaxis_nonfootflat(frame,ankle_JC)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.82, 381.62,  42.66]),
    array([ 39.44, 382.45,  41.79]),
    array([[[442.72, 381.69,  43.65],
            [441.88, 381.94,  42.54],
            [442.49, 380.67,  42.69]],
           [[ 39.56, 382.51,  42.78],
            [ 38.5 , 382.15,  41.92],
            [ 39.75, 381.5 ,  41.82]]])]
    """

    #REQUIRED MARKERS:
    # RTOE
    # LTOE
    # ankle_JC
    TOE_R = frame['RTOE']
    TOE_L = frame['LTOE']
    HEE_R = frame['RHEE']
    HEE_L = frame['LHEE']

    ankle_JC_R = ankle_JC[0]
    ankle_JC_L = ankle_JC[1]
    ankle_flexion_R = ankle_JC[2][0][1]
    ankle_flexion_L = ankle_JC[2][1][1]

    # Toe axis's origin is marker position of TOE
    R = TOE_R
    L = TOE_L

    ankle_JC_R = [ankle_JC_R[0],ankle_JC_R[1],ankle_JC_R[2]]
    ankle_JC_L = [ankle_JC_L[0],ankle_JC_L[1],ankle_JC_L[2]]

    # in case of non foot flat we just use the HEE marker
    R_axis_z = [HEE_R[0]-TOE_R[0],HEE_R[1]-TOE_R[1],HEE_R[2]-TOE_R[2]]
    R_axis_z = R_axis_z/norm3d(R_axis_z)

    y_flex_R = [ankle_flexion_R[0]-ankle_JC_R[0],ankle_flexion_R[1]-ankle_JC_R[1],ankle_flexion_R[2]-ankle_JC_R[2]]
    y_flex_R = y_flex_R/norm3d(y_flex_R)

    R_axis_x = cross(y_flex_R,R_axis_z)
    R_axis_x = R_axis_x/norm3d(R_axis_x)

    R_axis_y = cross(R_axis_z,R_axis_x)
    R_axis_y = R_axis_y/norm3d(R_axis_y)

    R_axis_x = [R_axis_x[0]+R[0],R_axis_x[1]+R[1],R_axis_x[2]+R[2]]
    R_axis_y = [R_axis_y[0]+R[0],R_axis_y[1]+R[1],R_axis_y[2]+R[2]]
    R_axis_z = [R_axis_z[0]+R[0],R_axis_z[1]+R[1],R_axis_z[2]+R[2]]

    R_foot_axis = [R_axis_x,R_axis_y,R_axis_z]

    # Left

    ankle_JC_R = [ankle_JC_R[0],ankle_JC_R[1],ankle_JC_R[2]]
    ankle_JC_L = [ankle_JC_L[0],ankle_JC_L[1],ankle_JC_L[2]]

    L_axis_z = [HEE_L[0]-TOE_L[0],HEE_L[1]-TOE_L[1],HEE_L[2]-TOE_L[2]]
    L_axis_z = L_axis_z/norm3d(L_axis_z)

    y_flex_L = [ankle_flexion_L[0]-ankle_JC_L[0],ankle_flexion_L[1]-ankle_JC_L[1],ankle_flexion_L[2]-ankle_JC_L[2]]
    y_flex_L = y_flex_L/norm3d(y_flex_L)

    L_axis_x = cross(y_flex_L,L_axis_z)
    L_axis_x = L_axis_x/norm3d(L_axis_x)

    L_axis_y = cross(L_axis_z,L_axis_x)
    L_axis_y = L_axis_y/norm3d(L_axis_y)

    L_axis_x = [L_axis_x[0]+L[0],L_axis_x[1]+L[1],L_axis_x[2]+L[2]]
    L_axis_y = [L_axis_y[0]+L[0],L_axis_y[1]+L[1],L_axis_y[2]+L[2]]
    L_axis_z = [L_axis_z[0]+L[0],L_axis_z[1]+L[1],L_axis_z[2]+L[2]]

    L_foot_axis = [L_axis_x,L_axis_y,L_axis_z]

    foot_axis = [R_foot_axis,L_foot_axis]

    return [R,L,foot_axis]

def getankleangle(axisP,axisD):
    """Static angle calculation function.

    This function takes in two axes and returns three angles.
    It uses an inverse Euler rotation matrix in YXZ order.
    The output shows the angle in degrees.

    Since we use arc sin we must check if the angle is in area between -pi/2 and pi/2
    but because the static offset angle under pi/2, it doesn't matter.

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
    >>> from .pycgmStatic import getankleangle
    >>> axisP = [[ 0.59, 0.11, 0.16],
    ...         [-0.13, -0.10, -0.90],
    ...         [0.94, -0.05, 0.75]]
    >>> axisD = [[0.17, 0.69, -0.37],
    ...         [0.14, -0.39, 0.94],
    ...         [-0.16, -0.53, -0.60]]
    >>> np.around(getankleangle(axisP,axisD), 2)
    array([0.48, 1.  , 1.56])
    """
    # make inverse matrix of axisP
    axisPi = np.linalg.inv(axisP)

    # M is multiply of axisD and axisPi
    M = matrixmult(axisD,axisPi)

    # This is the angle calculation in YXZ Euler angle
    getA = M[2][1] / sqrt((M[2][0]*M[2][0])+(M[2][2]*M[2][2]))
    getB = -1*M[2][0] / M[2][2]
    getG = -1*M[0][1] / M[1][1]

    gamma =np.arctan(getG)
    alpha = np.arctan(getA)
    beta = np.arctan(getB)

    angle = [alpha,beta,gamma]
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
