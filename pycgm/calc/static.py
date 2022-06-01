# -*- coding: utf-8 -*-
"""
This file provides helper functions for static calculations.

Created on Tue Jul 28 16:55:25 2015

@author: cadop
"""
import numpy as np
import numpy.lib.recfunctions as rfn

from . import dynamic


def getStatic(motionData, vsk, data, flat_foot=False, GCS=None):
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

    def get_markers(arr, names, points_only=True):
        if isinstance(names, str):
            names = [names]
        num_frames = arr[0][0].shape[0]

        if any(name not in arr[0].dtype.names for name in names):
            return None

        point = [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
        marker_dtype = [('frame', 'f8'), ('point', point)]
        rec = rfn.repack_fields(arr[names]).view(marker_dtype).reshape(len(names), int(num_frames))


        if points_only:
            rec = rec['point'][['x', 'y', 'z']]

        rec = rfn.structured_to_unstructured(rec)


        return rec

    pelvis_axis = dynamic.CalcAxes().calc_axis_pelvis(get_markers(data, 'RASI')[0],
                                                      get_markers(data, 'LASI')[0],
                                                      get_markers(data, 'RPSI')[0],
                                                      get_markers(data, 'LPSI')[0],
                                                      get_markers(data, 'SACR')[0] if 'SACR' in data.dtype.names else None)


    hip_jc = dynamic.CalcAxes().calc_joint_center_hip(pelvis_axis, 
                                                      calSM['MeanLegLength'],
                                                      calSM['R_AsisToTrocanterMeasure'],
                                                      calSM['L_AsisToTrocanterMeasure'],
                                                      calSM['InterAsisDistance'])

    knee_axis = dynamic.CalcAxes().calc_axis_knee(get_markers(data, 'RTHI')[0],
                                                  get_markers(data, 'LTHI')[0],
                                                  get_markers(data, 'RKNE')[0],
                                                  get_markers(data, 'LKNE')[0],
                                                  hip_jc[0],
                                                  hip_jc[1],
                                                  calSM['RightKneeWidth'],
                                                  calSM['LeftKneeWidth'])

    ankle_axis = dynamic.CalcAxes().calc_axis_ankle(get_markers(data, 'RTIB')[0],
                                                    get_markers(data, 'LTIB')[0],
                                                    get_markers(data, 'RANK')[0],
                                                    get_markers(data, 'LANK')[0],
                                                    knee_axis[0],
                                                    knee_axis[1],
                                                    calSM['RightAnkleWidth'],
                                                    calSM['LeftAnkleWidth'],
                                                    calSM['RightTibialTorsion'],
                                                    calSM['LeftTibialTorsion'])

    static_foot_offset = calc_foot_offset(get_markers(data, 'RTOE')[0],
                                          get_markers(data, 'LTOE')[0],
                                          get_markers(data, 'RHEE')[0],
                                          get_markers(data, 'LHEE')[0],
                                          ankle_axis,
                                          flat_foot,
                                          calSM['RightSoleDelta'] if 'RightSoleDelta' in calSM else 0,
                                          calSM['LeftSoleDelta'] if 'LeftSoleDelta' in calSM else 0)

    head = calc_axis_head(get_markers(data, 'LFHD')[0],
                          get_markers(data, 'RFHD')[0],
                          get_markers(data, 'LBHD')[0],
                          get_markers(data, 'RBHD')[0])

    head_offset = calc_static_head(head)

    right_static_rot_off    = np.average(static_foot_offset[0][0]) * -1
    right_static_plant_flex = np.average(static_foot_offset[0][1])
    left_static_rot_off     = np.average(static_foot_offset[1][0])
    left_static_plant_flex  = np.average(static_foot_offset[1][1])
    static_head=np.average(head_offset)

    calSM['RightStaticRotOff']    = right_static_rot_off
    calSM['RightStaticPlantFlex'] = right_static_plant_flex
    calSM['LeftStaticRotOff']     = left_static_rot_off
    calSM['LeftStaticPlantFlex']  = left_static_plant_flex
    calSM['HeadOffset']           = static_head

    return calSM


def get_dist(p0, p1):
    """Get Distance function

    This function calculates the distance between two 3-D positions.

    Parameters
    ----------
    p0 : array
        Position of first x, y, z coordinate.
    p1 : array
        Position of second x, y, z coordinate.

    Returns
    -------
    float
        The distance between positions p0 and p1.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import getDist
    >>> p0 = [0,1,2]
    >>> p1 = [1,2,3]
    >>> np.around(getDist(p0,p1), 2)
    1.73
    >>> p0 = np.array([991.45, 741.95, 321.36])
    >>> p1 = np.array([117.09, 142.24, 481.95])
    >>> np.around(getDist(p0,p1), 2)
    1072.36
    """
    return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)


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
    return np.linalg.norm(rasi - lasi, axis=1)


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

    head_axis = head_axis[:, :, :3].transpose(0,2,1)

    global_axis = np.array([[ 0, 1, 0],
                            [-1, 0, 0],
                            [ 0, 0, 1]])

    offset = calc_head_offset(global_axis, head_axis)

    return offset


def calc_head_offset(axis_p, axis_d):
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
    axis_p_inv = np.linalg.inv(axis_p)
    
    # Repeat GCS to match shape of vectorized head axis
    axis_p_inv_stack = np.repeat(axis_p_inv[None,...], axis_d.shape[0], axis=0)

    # rotation matrix is in order XYZ
    M = np.matmul(axis_d, axis_p_inv_stack)

    # get y angle from rotation matrix using inverse trigonometry.
    b = np.divide(M[:, 0, 2], M[:, 2, 2])

    angle = np.arctan(b)

    return angle


def calc_foot_offset(rtoe, ltoe, rhee, lhee, ankle_axis, flat_foot, right_sole_delta=0, left_sole_delta=0):
    """Calculate the Static foot offset angles.

    Takes in anatomically uncorrected axis or anatomically correct axis.
    Corrects the axis depending on flat-footedness.

    Calculates the offset angle between those two axes.

    It is rotated from uncorrected axis in YXZ order.

    Markers used: RTOE, LTOE, RHEE, LHEE

    Subject Measurement values used: RightSoleDelta, LeftSoleDelta

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
    flat_foot : boolean
        A boolean indicating if the feet are flat or not
    r_sole_delta : float, optional
        The right sole delta from the subject measurement file
    l_sole_delta : float, optional
        The left sole delta from the subject measurement file

    Returns
    -------
    angle : array
        The foot offset angle represented by a 2x3 array.
        The array contains the right flexion, abduction, rotation angles (1x3)
        followed by the left flexion, abduction, rotation angles (1x3).

    Notes
    -----
    The correct axis changes depending on the flat foot option.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import calc_foot_offset
    >>> rtoe = np.array([427.95, 437.1,  41.77])
    >>> ltoe = np.array([175.79, 379.5,  42.61])
    >>> rhee = np.array([406.46, 227.56,  48.76])
    >>> lhee = np.array([223.6, 173.43,  47.93])
    >>> ankle_axis = np.array([[[ 0.72,  0.69, -0.02, 393.76],
    ...                         [-0.69,  0.71, -0.12, 247.68],
    ...                         [-0.07,  0.1 ,  0.99,  87.74],
    ...                         [ 0.  ,  0.  ,  0.  ,   1.  ]],
    ...                        [[-0.28,  0.96, -0.1 ,  98.75],
    ...                         [-0.96, -0.26,  0.13, 219.47],
    ...                         [ 0.1 ,  0.13,  0.99,  80.63],
    ...                         [ 0.  ,  0.  ,  0.  ,   1.  ]]])
    >>> flat_foot = True
    >>> right_sole_delta = 0.45
    >>> left_sole_delta = 0.45
    >>> np.around(calc_foot_offset(rtoe, ltoe, rhee, lhee, ankle_axis, flat_foot, right_sole_delta, left_sole_delta), 2)
    array([[-0.08,  0.23, -0.66],
           [-0.67,  0.22, -0.3 ]])
    >>> flat_foot = False # Using the same variables and switching the flat_foot flag.
    >>> np.around(calc_foot_offset(rtoe, ltoe, rhee, lhee, ankle_axis, flat_foot, right_sole_delta, left_sole_delta), 2)
    array([[-0.08,  0.2 , -0.15],
           [-0.67,  0.19,  0.12]])
    """
    r_ankle_axis = ankle_axis[0]
    l_ankle_axis = ankle_axis[1]

    uncorrect_foot = calc_axis_uncorrect_foot(rtoe, ltoe, r_ankle_axis, l_ankle_axis)

    uncorrect_foot_right = uncorrect_foot[0]
    uncorrect_foot_left = uncorrect_foot[1]

    # Check if it is flat foot or not.
    if flat_foot == False:
        nonflat_foot = calc_axis_nonflatfoot(rtoe, ltoe, rhee, lhee, ankle_axis)
        nonflat_foot_right = nonflat_foot[0]
        nonflat_foot_left  = nonflat_foot[1]

        right_angle = calc_static_angle_ankle(uncorrect_foot_right, nonflat_foot_right)
        left_angle  = calc_static_angle_ankle(uncorrect_foot_left, nonflat_foot_left)

    # elif flat_foot == True:
    elif flat_foot == True:
        flat_foot = calc_axis_flatfoot(rtoe, ltoe, rhee, lhee, ankle_axis,
                                       right_sole_delta,
                                       left_sole_delta)
        flat_foot_right = flat_foot[0]
        flat_foot_left  = flat_foot[1]

        right_angle = calc_static_angle_ankle(uncorrect_foot_right, flat_foot_right)
        left_angle  = calc_static_angle_ankle(uncorrect_foot_left, flat_foot_left)

    angle = np.asarray([right_angle, left_angle])

    return angle


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

    # get the midpoints of the head to define the sides
    front = (lfhd + rfhd) / 2.0
    back  = (lbhd + rbhd) / 2.0
    left  = (lfhd + lbhd) / 2.0
    right = (rfhd + rbhd) / 2.0

    # Get the vectors from the sides with primary x axis facing front
    # First get the x direction
    x_axis = np.subtract(front, back)
    x_axis_norm = np.linalg.norm(x_axis, axis=1)[:, np.newaxis]
    x_axis = np.divide(x_axis, x_axis_norm)

    # get the direction of the y axis
    y_axis = np.subtract(left, right)
    y_axis_norm = np.linalg.norm(y_axis, axis=1)[:, np.newaxis]
    y_axis = np.divide(y_axis, y_axis_norm)

    # get z axis by cross-product of x axis and y axis.
    z_axis = np.cross(x_axis, y_axis)
    z_axis_norm = np.linalg.norm(z_axis, axis=1)[:, np.newaxis]
    z_axis = np.divide(z_axis, z_axis_norm)

    # make sure all x,y,z axis is orthogonal each other by cross-product
    y_axis = np.cross(z_axis, x_axis)
    y_axis_norm = np.linalg.norm(y_axis, axis=1)[:, np.newaxis]
    y_axis = np.divide(y_axis, y_axis_norm)

    x_axis = np.cross(y_axis, z_axis)
    x_axis_norm = np.linalg.norm(x_axis, axis=1)[:, np.newaxis]
    x_axis = np.divide(x_axis, x_axis_norm)

    # Create the return matrix
    num_frames = rfhd.shape[0]
    head_axis = np.column_stack([x_axis, y_axis, z_axis, front])
    head_axis = head_axis.reshape(num_frames,4,3).transpose(0,2,1)

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

    ankle_jc_r = r_ankle_axis[:, :, 3]
    ankle_jc_l = l_ankle_axis[:, :, 3]

    ankle_flexion_r = r_ankle_axis[:, :, 1] + ankle_jc_r
    ankle_flexion_l = l_ankle_axis[:, :, 1] + ankle_jc_l

    # Toe axis's origin is marker position of TOE
    r_origin = rtoe
    l_origin = ltoe

    # Right

    # z axis is from TOE marker to AJC. and normalized it.
    r_axis_z = ankle_jc_r - rtoe
    r_axis_z_div = np.linalg.norm(r_axis_z, axis=1)[:, np.newaxis]
    r_axis_z = r_axis_z / r_axis_z_div

    # bring the flexion axis of ankle axes from AnkleJointCenter function. and normalized it.
    y_flex_r = ankle_flexion_r - ankle_jc_r
    y_flex_r_div = np.linalg.norm(y_flex_r, axis=1)[:, np.newaxis]
    y_flex_r = y_flex_r / y_flex_r_div

    # x axis is calculated as a cross product of z axis and ankle flexion axis.
    r_axis_x = np.cross(y_flex_r, r_axis_z)
    r_axis_x_div = np.linalg.norm(r_axis_x, axis=1)[:, np.newaxis]
    r_axis_x = r_axis_x / r_axis_x_div

    # y axis is then perpendicularly calculated from z axis and x axis. and normalized.
    r_axis_y = np.cross(r_axis_z, r_axis_x)
    r_axis_y_div = np.linalg.norm(r_axis_y, axis=1)[:, np.newaxis]
    r_axis_y = r_axis_y / r_axis_y_div

    r_foot_axis = np.column_stack([r_axis_x, r_axis_y, r_axis_z, r_origin])

    # Left

    # z axis is from TOE marker to AJC. and normalized it.
    l_axis_z = ankle_jc_l - ltoe
    l_axis_z_div = np.linalg.norm(l_axis_z, axis=1)[:, np.newaxis]
    l_axis_z = l_axis_z / l_axis_z_div

    # bring the flexion axis of ankle axes from AnkleJointCenter function. and normalized it.
    y_flex_l = ankle_flexion_l - ankle_jc_l
    y_flex_l_div = np.linalg.norm(y_flex_l, axis=1)[:, np.newaxis]
    y_flex_l = y_flex_l / y_flex_l_div

    # x axis is calculated as a cross product of z axis and ankle flexion axis.
    l_axis_x = np.cross(y_flex_l, l_axis_z)
    l_axis_x_div = np.linalg.norm(l_axis_x, axis=1)[:, np.newaxis]
    l_axis_x = l_axis_x / l_axis_x_div

    # y axis is then perpendicularly calculated from z axis and x axis. and normalized.
    l_axis_y = np.cross(l_axis_z, l_axis_x)
    l_axis_y_div = np.linalg.norm(l_axis_y, axis=1)[:, np.newaxis]
    l_axis_y = l_axis_y / l_axis_y_div

    l_foot_axis = np.column_stack([l_axis_x, l_axis_y, l_axis_z, l_origin])

    num_frames = rtoe.shape[0]
    r_foot = r_foot_axis.reshape(num_frames,4,3).transpose(0,2,1)
    l_foot = l_foot_axis.reshape(num_frames,4,3).transpose(0,2,1)

    foot_axis = np.array([r_foot, l_foot])

    return foot_axis


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

    r_ankle_axis, l_ankle_axis = np.asarray(ankle_axis)
    ankle_jc_right = r_ankle_axis[:, :, 3]
    ankle_jc_left  = l_ankle_axis[:, :, 3]

    ankle_flexion_right = r_ankle_axis[:, :, 1] + ankle_jc_right
    ankle_flexion_left = l_ankle_axis[:, :, 1] + ankle_jc_left

    # Toe axis's origin is marker position of TOE
    right_origin = rtoe
    left_origin = ltoe

    ankle_jc_right[:, 2] += r_sole_delta
    ankle_jc_left[:, 2]  += l_sole_delta

    # Calculate the z axis
    right_axis_z = ankle_jc_right - rtoe
    right_axis_z = np.divide(right_axis_z, np.linalg.norm(right_axis_z, axis=1)[:, np.newaxis])

    # For foot flat, Z axis pointing same height of TOE marker from TOE to AJC
    heel_to_toe = rhee - rtoe
    heel_to_toe[:, 2] = 0
    heel_to_toe = np.divide(heel_to_toe, np.linalg.norm(heel_to_toe, axis=1)[:, np.newaxis])

    A = np.cross(heel_to_toe, right_axis_z)
    A /= np.linalg.norm(A, axis=1)[:, np.newaxis]
    B = np.cross(A, heel_to_toe)
    B /= np.linalg.norm(B, axis=1)[:, np.newaxis]
    C = np.cross(B, A)
    right_axis_z = C / np.linalg.norm(C, axis=1)[:, np.newaxis]

    # Bring flexion axis from ankle axis
    right_y_flex = ankle_flexion_right - ankle_jc_right
    right_y_flex = np.divide(right_y_flex, np.linalg.norm(right_y_flex, axis=1)[:, np.newaxis])

    # Calculate each x,y,z axis of foot using np.cross-product and make sure x,y,z axis is orthogonal each other.
    right_axis_x = np.cross(right_y_flex,right_axis_z)
    right_axis_x = np.divide(right_axis_x, np.linalg.norm(right_axis_x, axis=1)[:, np.newaxis])

    right_axis_y = np.cross(right_axis_z,right_axis_x)
    right_axis_y = np.divide(right_axis_y, np.linalg.norm(right_axis_y, axis=1)[:, np.newaxis])

    right_axis_z = np.cross(right_axis_x,right_axis_y)
    right_axis_z = np.divide(right_axis_z, np.linalg.norm(right_axis_z, axis=1)[:, np.newaxis])

    right_foot_axis = np.column_stack([right_axis_x, right_axis_y, right_axis_z, right_origin])

    # Left

    # Calculate the z axis of foot flat.
    left_axis_z = ankle_jc_left - ltoe
    left_axis_z = np.divide(left_axis_z, np.linalg.norm(left_axis_z, axis=1)[:, np.newaxis])

    # For foot flat, Z axis pointing same height of TOE marker from TOE to AJC
    heel_to_toe = lhee - ltoe
    heel_to_toe[:, 2] = 0
    heel_to_toe = np.divide(heel_to_toe, np.linalg.norm(heel_to_toe, axis=1)[:, np.newaxis])
    A = np.cross(heel_to_toe, left_axis_z)
    A = A / np.linalg.norm(A)
    B = np.cross(A, heel_to_toe)
    B = B / np.linalg.norm(B)
    C = np.cross(B, A)
    left_axis_z = C / np.linalg.norm(C)

    # Bring flexion axis from ankle axis
    left_y_flex = ankle_flexion_left - ankle_jc_left
    left_y_flex = np.divide(left_y_flex, np.linalg.norm(left_y_flex, axis=1)[:, np.newaxis])

    # Calculate each x,y,z axis of foot using np.cross-product and make sure (x, y, z) axes are orthogonal to each other
    left_axis_x = np.cross(left_y_flex,left_axis_z)
    left_axis_x = np.divide(left_axis_x, np.linalg.norm(left_axis_x, axis=1)[:, np.newaxis])

    left_axis_y = np.cross(left_axis_z,left_axis_x)
    left_axis_y = np.divide(left_axis_y, np.linalg.norm(left_axis_y, axis=1)[:, np.newaxis])

    left_axis_z = np.cross(left_axis_x,left_axis_y)
    left_axis_z = np.divide(left_axis_z, np.linalg.norm(left_axis_z, axis=1)[:, np.newaxis])

    left_foot_axis = np.column_stack([left_axis_x, left_axis_y, left_axis_z, left_origin])

    num_frames = rtoe.shape[0]
    r_foot = right_foot_axis.reshape(num_frames,4,3).transpose(0,2,1)
    l_foot = left_foot_axis.reshape(num_frames,4,3).transpose(0,2,1)

    axis = np.array([r_foot, l_foot])

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

    ankle_jc_right = ankle_axis[0][:, :, 3]
    ankle_jc_left = ankle_axis[1][:, :, 3]
    ankle_flexion_right = ankle_axis[0][:, :, 1]  + ankle_jc_right
    ankle_flexion_left = ankle_axis[1][:, :, 1]  + ankle_jc_left

    # Toe axis's origin is marker position of TOE
    right_origin = rtoe
    left_origin = ltoe

    # in case of non foot flat we just use the HEE marker
    right_axis_z = rhee - rtoe
    right_axis_z = np.divide(right_axis_z, np.linalg.norm(right_axis_z, axis=1)[:, np.newaxis])

    y_flex_R = ankle_flexion_right - ankle_jc_right
    y_flex_R = np.divide(y_flex_R, np.linalg.norm(y_flex_R, axis=1)[:, np.newaxis])

    right_axis_x = np.cross(y_flex_R, right_axis_z)
    right_axis_x = np.divide(right_axis_x, np.linalg.norm(right_axis_x, axis=1)[:, np.newaxis])

    right_axis_y = np.cross(right_axis_z, right_axis_x)
    right_axis_y = np.divide(right_axis_y, np.linalg.norm(right_axis_y, axis=1)[:, np.newaxis])

    r_foot_axis = np.column_stack([right_axis_x, right_axis_y, right_axis_z, right_origin])

    # Left
    left_axis_z = lhee - ltoe
    left_axis_z = np.divide(left_axis_z, np.linalg.norm(left_axis_z))

    y_flex_L = ankle_flexion_left - ankle_jc_left
    y_flex_L = np.divide(y_flex_L, np.linalg.norm(y_flex_L, axis=1)[:, np.newaxis])

    left_axis_x = np.cross(y_flex_L, left_axis_z)
    left_axis_x = np.divide(left_axis_x, np.linalg.norm(left_axis_x, axis=1)[:, np.newaxis])

    left_axis_y = np.cross(left_axis_z, left_axis_x)
    left_axis_y = np.divide(left_axis_y, np.linalg.norm(left_axis_y, axis=1)[:, np.newaxis])

    l_foot_axis = np.column_stack([left_axis_x, left_axis_y, left_axis_z, left_origin])

    num_frames = rtoe.shape[0]
    r_foot = r_foot_axis.reshape(num_frames,4,3).transpose(0,2,1)
    l_foot = l_foot_axis.reshape(num_frames,4,3).transpose(0,2,1)

    axis = np.array([r_foot, l_foot])

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

    axis_p = axis_p[:, :, :3]
    axis_d = axis_d[:, :, :3]

    axis_p_inv = np.linalg.inv(axis_p)

    # M is multiply of axis_d and axis_p_inv
    axis_d = np.transpose(axis_d, axes=(0,2,1))
    axis_p_inv = np.transpose(axis_p_inv, axes=(0,2,1))
    M = np.matmul(axis_d, axis_p_inv)

    # This is the angle calculation in YXZ Euler angle
    a = np.divide(M[:, 2, 1], np.sqrt((M[:, 2, 0] * M[:, 2, 0]) + (M[:, 2, 2] * M[:, 2, 2])))
    b = -1 * M[:, 2, 0] / M[:, 2, 2]
    g = -1 * M[:, 0, 1] / M[:, 1, 1]

    gamma =np.arctan(g)
    alpha = np.arctan(a)
    beta = np.arctan(b)

    angle = np.array([alpha, beta, gamma])
    return angle

