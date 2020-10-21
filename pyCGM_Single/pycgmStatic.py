# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:55:25 2015

@author: cadop
"""
import numpy as np
from math import *

def rotmat(x=0,y=0,z=0):
    """Rotation Matrix function
    
    This function creates and returns a rotation matrix.

    Parameters
    ----------
    x,y,z : float, optional
        Angle, which will be converted to radians, in
        each respective axis to describe the rotations. 
        The default is 0 for each unspecified angle.
    
    Returns
    -------
    Rxyz : array 
        The product of the matrix multiplication.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import rotmat
    >>> rotmat() #doctest: +NORMALIZE_WHITESPACE
    [[1.0, 0.0, 0.0], 
    [0.0, 1.0, 0.0], 
    [0.0, 0.0, 1.0]]
    >>> x = 0.5
    >>> y = 0.3
    >>> z = 0.8
    >>> np.around(rotmat(x,y,z),8)
    array([[ 0.99988882, -0.01396199,  0.00523596],
           [ 0.01400734,  0.99986381, -0.00872642],
           [-0.00511341,  0.00879879,  0.99994822]])
    >>> x = 0.5
    >>> np.around(rotmat(x),decimals=8)
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
    x = radians(x)
    y = radians(y)
    z = radians(z)
    Rx = [ [1,0,0],[0,cos(x),sin(x)*-1],[0,sin(x),cos(x)] ]
    Ry = [ [cos(y),0,sin(y)],[0,1,0],[sin(y)*-1,0,cos(y)] ]
    Rz = [ [cos(z),sin(z)*-1,0],[sin(z),cos(z),0],[0,0,1] ]
    
    Rxy = matrixmult(Rx,Ry)
    Rxyz = matrixmult(Rxy,Rz)    
    
    Ryx = matrixmult(Ry,Rx)
    Ryxz = matrixmult(Ryx,Rz)
    
    return Rxyz
    
def getDist(p0, p1):
    """Get Distance function
    
    This function calculates the distance between two 3-D positions.

    Parameters
    ----------
    p0 : array
        Position of first x,y,z coordinate.
    p1 : array
        Position of second x,y,z coordinate.
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
    >>> np.around(getDist(p0,p1),8)
    1.73205081
    >>> p0 = np.array([991.44611381, 741.95103792, 321.35500969])
    >>> p1 = np.array([117.08710839, 142.23917057, 481.95268411])
    >>> np.around(getDist(p0,p1),8)
    1072.35703347
    """
    return sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)
    
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
    calSM['MeanLegLength'] = (LeftLegLength+RightLegLength)/2
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
            iadCalc = IADcalculation(frame)
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

                Rdst = getDist(RKNE,RMKN)
                Ldst = getDist(LKNE,LMKN)
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

                Rdst = getDist(RMMA,RANK)
                Ldst = getDist(LMMA,LANK)
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
        hip_JC = hipJointCenter(frame,pelvis_origin,pelvis_axis[0],pelvis_axis[1],pelvis_axis[2],calSM)
        knee_JC = kneeJointCenter(frame,hip_JC,0,vsk=calSM)
        ankle_JC = ankleJointCenter(frame,knee_JC,0,vsk=calSM)
        angle = staticCalculation(frame,ankle_JC,knee_JC,flat_foot,calSM)
        head = headJC(frame)
        headangle = staticCalculationHead(frame,head)

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

def average(list):
    """Average Calculation function
    
    Calculates the average of the values in a given list or array.

    Parameters
    ----------
    list : list
        List or array of values.
    
    Returns
    -------
    float 
        The mean of the list.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import average
    >>> list = [1,2,3,4,5]
    >>> average(list)
    3.0
    >>> list = np.array([93.81607046, 248.95632028, 782.61762769])
    >>> np.around(average(list),8)
    375.13000614
    """
    i =0
    total = 0.0
    while(i <len(list)):
        total = total + list[i]
        i = i+1
    return total / len(list)    

def IADcalculation(frame):
    """Inter ASIS Distance (IAD) Calculation function
    
    Calculates the Inter ASIS Distance from a given frame.

    Markers used: RASI, LASI

    Parameters
    ----------
    frame : dict 
        Dictionaries of marker lists.

    Returns
    -------
    IAD : float
        The mean of the list.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import IADcalculation
    >>> frame = { 'LASI': np.array([ 183.18504333,  422.78927612, 1033.07299805]),
    ...           'RASI': np.array([ 395.36532593,  428.09790039, 1036.82763672])}
    >>> np.around(IADcalculation(frame),8)
    212.27988866
    """
    RASI = frame['RASI']
    LASI = frame['LASI']
    IAD = np.sqrt((RASI[0]-LASI[0])*(RASI[0]-LASI[0])+(RASI[1]-LASI[1])*(RASI[1]-LASI[1])+(RASI[2]-LASI[2])*(RASI[2]-LASI[2]))
    
    return IAD
    
def staticCalculationHead(frame,head):
    """Static Head Calculation function
    
    This function calculates the x,y,z axes of the head,
    and then calculates the offset of the head using the headoffCalc function.

    Parameters
    ----------
    frame : dict 
        Dictionaries of marker lists.
    head : array
        An array containing the head axis and head origin.

    Returns
    -------
    offset : float
        The head offset angle for static calibration.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import staticCalculationHead
    >>> frame = None
    >>> head = [[[100.33272997128863, 83.39303060995121, 1484.078302933558], 
    ...        [98.9655145897623, 83.57884461044797, 1483.7681493301013], 
    ...        [99.34535520789223, 82.64077714742746, 1484.7559501904173]], 
    ...        [99.58366584777832, 82.79330825805664, 1483.7968139648438]]
    >>> np.around(staticCalculationHead(frame,head),8)
    0.28546606
    """
    headAxis = head[0]
    headOrigin = head[1]
    x_axis = [headAxis[0][0]-headOrigin[0],headAxis[0][1]-headOrigin[1],headAxis[0][2]-headOrigin[2]]
    y_axis = [headAxis[1][0]-headOrigin[0],headAxis[1][1]-headOrigin[1],headAxis[1][2]-headOrigin[2]]
    z_axis = [headAxis[2][0]-headOrigin[0],headAxis[2][1]-headOrigin[1],headAxis[2][2]-headOrigin[2]]
    
    axis = [x_axis,y_axis,z_axis]
    global_axis = [[0,1,0],[-1,0,0],[0,0,1]]
    
    offset = headoffCalc(global_axis,axis)

    return offset

def headoffCalc(axisP, axisD):
    """Head Offset Calculation function
    
    Calculate head offset angle for static calibration.
    This function is only called in static trial. 
    and output will be used in dynamic later. 

    Parameters
    ----------
    axisP : list
        Shows the unit vector of axisP, the position of the proximal axis.
    axisD : list
        Shows the unit vector of axisD, the position of the distal axis.

    Returns
    -------
    angle : float
        The beta angle of the head offset.
    
    Examples
    --------
    >>> import numpy as np 
    >>> from .pycgmStatic import headoffCalc
    >>> axisP = [[0.96027586, 0.81188464, 0.8210366],
    ...         [0.24275408, 0.72003003, 0.37957337],
    ...         [0.98315477, 0.20884389, 0.68137521]]
    >>> axisD = [[0.21449582, 0.24790518, 0.94419932],
    ...         [0.79794437, 0.45267577, 0.91257356],
    ...         [0.17107064, 0.67114988, 0.85129037]]
    >>> np.around(headoffCalc(axisP,axisD),8) 
    0.94916636
    """
    axisPi = np.linalg.inv(axisP)
    
    # rotation matrix is in order XYZ
    M = matrixmult(axisD,axisPi)
    
    # get y angle from rotation matrix using inverse trigonometry.
    getB= M[0][2] / M[2][2]
    
    beta = np.arctan(getB)
    
    angle = beta
    
    return angle
    
def staticCalculation(frame,ankle_JC,knee_JC,flat_foot,vsk=None):
    """Calculate the Static angle function
    
    Takes in anatomical uncorrect axis and anatomical correct axis. 
    Correct axis depends on foot flat options.

    Calculates the offset angle between that two axis.

    It is rotated from uncorrect axis in YXZ order.
    
    Parameters
    ----------
    frame : dict 
        Dictionaries of marker lists.
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
        Returns the offset angle represented by a 2x3x3 array. 
        The array contains the right flexion, abduction, rotation angles (1x3x3)
        followed by the left flexion, abduction, rotation angles (1x3x3).

    Modifies
    --------
    The correct axis changes following to the foot flat option. 
    
    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import staticCalculation
    >>> frame = {'RTOE': np.array([427.95211792, 437.99603271,  41.77342987]),
    ...          'LTOE': np.array([175.78988647, 379.49987793,  42.61193085]),
    ...          'RHEE': np.array([406.46331787, 227.56491089,  48.75952911]),
    ...          'LHEE': np.array([223.59848022, 173.42980957,  47.92973328])}
    >>> ankle_JC = [np.array([393.76181608, 247.67829633, 87.73775041]),
    ...            np.array([98.74901939, 219.46930221, 80.6306816]),
    ...            [[np.array([394.4817575, 248.37201348, 87.715368]),
    ...            np.array([393.07114384, 248.39110006, 87.61575574]),
    ...            np.array([393.69314056, 247.78157916, 88.73002876])],
    ...            [np.array([98.47494966, 220.42553803, 80.52821783]),
    ...            np.array([97.79246671, 219.20927275, 80.76255901]),
    ...            np.array([98.84848169, 219.60345781, 81.61663775])]]]
    >>> knee_JC = [np.array([364.17774614, 292.17051722, 515.19181496]),
    ...           np.array([143.55478579, 279.90370346, 524.78408753]),
    ...           np.array([[[364.64959153, 293.06758353, 515.18513093],
    ...           [363.29019771, 292.60656648, 515.04309095],
    ...           [364.04724541, 292.24216264, 516.18067112]],
    ...           [[143.65611282, 280.88685896, 524.63197541],
    ...           [142.56434499, 280.01777943, 524.86163553],
    ...           [143.64837987, 280.04650381, 525.76940383]]])]
    >>> flat_foot = True      
    >>> vsk = { 'RightSoleDelta': 0.4532,'LeftSoleDelta': 0.4532 }
    >>> np.around(staticCalculation(frame,ankle_JC,knee_JC,flat_foot,vsk),8)
    array([[-0.08036968,  0.23192796, -0.66672181],
           [-0.67466613,  0.21812578, -0.30207993]])
    >>> flat_foot = False # Using the same variables and switching the flat_foot flag. 
    >>> np.around(staticCalculation(frame,ankle_JC,knee_JC,flat_foot,vsk),8)
    array([[-0.07971346,  0.19881323, -0.15319313],
           [-0.67470483,  0.18594096,  0.12287455]])
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

    Takes in a dictionary of x,y,z positions and marker names, as well as an index
    Calculates the pelvis joint center and axis and returns both.
    
    Markers used: RASI,LASI,RPSI,LPSI
    Other landmarks used: origin, sacrum

    Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990) and then normalized.
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

    Examples
    --------
    >>> import numpy as np 
    >>> from .pycgmStatic import pelvisJointCenter
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
        sacrum = (RPSI+LPSI)/2  
    except:
        pass #going to use sacrum marker

    if 'SACR' in frame:
        sacrum = frame['SACR']
        
        
    # REQUIRED LANDMARKS:
    # origin
    # sacrum 
    
    # Origin is Midpoint between RASI and LASI
    origin = (RASI+LASI)/2
    
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
            
def hipJointCenter(frame,pel_origin,pel_x,pel_y,pel_z,vsk=None):
    """Calculate the hip joint center function.

    Takes in a dictionary of x,y,z positions and marker names, as well as an index.
    Calculates the hip joint center and returns the hip joint center.
    
    Other landmarks used: origin, sacrum
    Subject Measurement values used: MeanLegLength, R_AsisToTrocanterMeasure, InterAsisDistance, L_AsisToTrocanterMeasure

    Hip Joint Center: Computed using Hip Joint Center Calculation (ref. Davis_1991)

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.
    pel_origin : array
        An array of pel_origin, pel_x, pel_y, pel_z each x,y,z position.
    pel_x, pel_y, pel_z : int
        Respective axes of the pelvis.
    vsk : dict, optional
        A dictionary containing subject measurements from a VSK file.

    Returns
    -------
    hip_JC : array
        Returns an array containing the left hip joint center x, y, z marker positions (1x3), 
        followed by the right hip joint center x, y, z marker positions (1x3).

    Examples
    --------
    >>> import numpy as np 
    >>> from .pycgmStatic import hipJointCenter
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
    
    mm = 7.0
    #mm = 14.0 #can this be given?
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
    
def hipAxisCenter(l_hip_jc,r_hip_jc,pelvis_axis):
    """Calculate the hip joint axis function.

    Takes in a hip joint center of x,y,z positions as well as an index.
    and takes the hip joint center and pelvis origin/axis from previous functions.
    Calculates the hip axis and returns hip joint origin and axis.

    Hip center axis: Computed by taking the mean at each x,y,z axis of the left and right hip joint center.
    Hip axis: Computed by getting the summation of the pelvis and hip center axes.

    Parameters
    ----------
    l_hip_jc, r_hip_jc: array
        Array of R_hip_jc and L_hip_jc each x,y,z position.
    pelvis_axis : array
        An array of pelvis origin and axis. The axis is also composed of 3 arrays,
        each things are x axis, y axis, z axis.
       
    Returns
    -------
    hipaxis_center, axis : array
        Returns an array that contains the hip axis center in a 1x3 array of xyz values, 
        which is then followed by a 3x2x3 array composed of the hip axis center x, y, and z 
        axis components. The xyz axis components are 2x3 arrays consisting of the axis center  
        in the first dimension and the direction of the axis in the second dimension.  

    Examples
    --------
    >>> import numpy as np 
    >>> from .pycgmStatic import hipAxisCenter
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
    hipaxis_center = [(r_hip_jc[0]+l_hip_jc[0])/2,(r_hip_jc[1]+l_hip_jc[1])/2,(r_hip_jc[2]+l_hip_jc[2])/2]
  
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

def kneeJointCenter(frame,hip_JC,delta,vsk=None):
    """Calculate the knee joint center and axis function.

    Takes in a dictionary of xyz positions and marker names, as well as an index.
    and takes the hip axis and pelvis axis.
    Calculates the knee joint axis and returns the knee origin and axis

    Markers used: RTHI, LTHI, RKNE, LKNE, hip_JC
    Subject Measurement values used: RightKneeWidth, LeftKneeWidth

    Knee joint center: Computed using Knee Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)
    
    Parameters
    ----------
    frame : dict 
        dictionaries of marker lists.
    hip_JC : array
        An array of hip_JC containing the x,y,z axes marker positions of the hip joint center. 
    delta : float
        The length from marker to joint center, retrieved from subject measurement file.
    vsk : dict, optional
        A dictionary containing subject measurements from a VSK file.

    Returns
    -------
    R, L, axis : array
        Returns an array that contains the knee axis center in a 1x3 array of xyz values, 
        which is then followed by a 2x3x3 array composed of the knee axis center x, y, and z 
        axis components. The xyz axis components are 2x3 arrays consisting of the axis center 
        in the first dimension and the direction of the axis in the second dimension.                                    
        
    Modifies
    --------
    delta is changed suitably to knee.

    Examples
    --------
    >>> import numpy as np 
    >>> from .pycgmStatic import kneeJointCenter
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
    axis_x = cross(axis_z,thi_kne_R)
    
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
    axis_x = cross(thi_kne_L,axis_z)
    
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
    """Calculate the ankle joint center and axis function.

    Takes in a dictionary of xyz positions and marker names, as well as an index.
    and takes the knee axis.
    Calculates the ankle joint axis and returns the ankle origin and axis
    
    Markers used: tib_R, tib_L, ank_R, ank_L, knee_JC
    Subject Measurement values used: RightKneeWidth, LeftKneeWidth

    Ankle Axis: Computed using Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013).
    
    Parameters
    ----------
    frame : dict 
        dictionaries of marker lists.
    knee_JC : array
        An array of knee_JC each x,y,z position.
    delta : float
        The length from marker to joint center, retrieved from subject measurement file.
    vsk : dict, optional
        A dictionary containing subject measurements from a VSK file.

    Returns
    -------
    R, L, axis : array
        Returns an array that contains the ankle axis origin in a 1x3 array of xyz values, 
        which is then followed by a 3x2x3 array composed of the ankle origin, x, y, and z 
        axis components. The xyz axis components are 2x3 arrays consisting of the origin 
        in the first dimension and the direction of the axis in the second dimension.                            

    Examples
    --------
    >>> import numpy as np 
    >>> from .pycgmStatic import ankleJointCenter
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
    
    Takes in a dictionary of xyz positions and marker names.
    and takes the ankle axis and knee axis.
    Calculate the foot joint axis by rotating incorrect foot joint axes about offset angle.
    Returns the foot axis origin and axis.
    
    In case of foot joint center, we've already make 2 kinds of axis for static offset angle. 
    and then, Call this static offset angle as an input of this function for dynamic trial. 

    Special Cases:

    (anatomical uncorrect foot axis)
    if foot flat is checked, make the reference markers instead of HEE marker which height is as same as TOE marker's height.
    elif foot flat is not checked, use the HEE marker for making Z axis.

    Markers used: RTOE,LTOE,RHEE, LHEE
    Other landmarks used: ANKLE_FLEXION_AXIS
    Subject Measurement values used: RightStaticRotOff, RightStaticPlantFlex, LeftStaticRotOff, LeftStaticPlantFlex
        
    Parameters
    ---------- 
    frame : dict 
        Dictionaries of marker lists.
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
    R, L, foot_axis : array
        Returns an array that contains the foot axis center in a 1x3 array of xyz values, 
        which is then followed by a 2x3x3 array composed of the foot axis center x, y, and z 
        axis components. The xyz axis components are 2x3 arrays consisting of the axis center 
        in the first dimension and the direction of the axis in the second dimension.   
        This function also saves the static offset angle in a global variable.     
            
    Modifies
    --------   
    Axis changes following to the static info.

    you can set the static_info by the button. and this will calculate the offset angles 
    the first setting, the foot axis show foot uncorrected anatomical reference axis(Z_axis point to the AJC from TOE)

    if press the static_info button so if static_info is not None,
    and then the static offsets angles are applied to the reference axis.
    the reference axis is Z axis point to HEE from TOE

    Examples
    --------
    >>> import numpy as np 
    >>> from .pycgmStatic import footJointCenter
    >>> frame = { 'RHEE': np.array([374.01257324, 181.57929993, 49.50960922]),
    ...           'LHEE': np.array([105.30126953, 180.2130127, 47.15660858]),
    ...           'RTOE': np.array([442.81997681, 381.62280273, 42.66047668]),
    ...           'LTOE': np.array([39.43652725, 382.44522095, 41.78911591])}
	>>> static_info = [[0.03482194, 0.14879424, 0],
	...               [0.01139704, 0.02142806, 0]]
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
    >>> [np.around(arr,8) for arr in footJointCenter(frame,static_info,ankle_JC,knee_JC,delta)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.81997681, 381.62280273,  42.66047668]), 
    array([ 39.43652725, 382.44522095,  41.78911591]), 
    array([[[442.8881541 , 381.76460597,  43.64802096],
            [441.89515447, 382.00308979,  42.66971773],
            [442.44573691, 380.70886969,  42.81754643]],
           [[ 39.50785213, 382.67891581,  42.75880631],
            [ 38.49231839, 382.14765966,  41.93027863],
            [ 39.75805858, 381.51956227,  41.98854914]]])]         
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

def headJC(frame):
    """Calculate the head joint axis function.

    Takes in a dictionary of x,y,z positions and marker names.
    Calculates the head joint center and returns the head joint center and axis.
    
    Markers used: LFHD, RFHD, LBHD, RBHD

    Parameters
    ----------
    frame : dict 
        Dictionaries of marker lists.
    
    Returns
    -------
    head_axis, origin : array
        Returns an array containing a 1x3x3 array containing the x, y, z axis 
        components of the head joint center, and a 1x3 array containing the 
        head origin x, y, z position.
    
    Examples
    --------
    >>> import numpy as np 
    >>> from .pycgmStatic import headJC
    >>> frame = {'RFHD': np.array([325.82983398, 402.55450439, 1722.49816895]),
    ...          'LFHD': np.array([184.55158997, 409.68713379, 1721.34289551]),
    ...          'RBHD': np.array([304.39898682, 242.91339111, 1694.97497559]),
    ...          'LBHD': np.array([197.8621521, 251.28889465, 1696.90197754])}
    >>> [np.around(arr,8) for arr in headJC(frame)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 255.21590218,  407.10741939, 1722.0817318 ],
           [ 254.19105385,  406.14680918, 1721.91767712],
           [ 255.18370553,  405.95974655, 1722.90744993]]), 
    array([  255.19071198,  406.12081909, 1721.92053223])]
    """
    
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

    #Add the origin back to the vector to get it in the right position
    x_axis = [x_vec[0]+origin[0],x_vec[1]+origin[1],x_vec[2]+origin[2]]
    y_axis = [y_vec[0]+origin[0],y_vec[1]+origin[1],y_vec[2]+origin[2]]
    z_axis = [z_vec[0]+origin[0],z_vec[1]+origin[1],z_vec[2]+origin[2]]
    
    head_axis =[x_axis,y_axis,z_axis]

    #Return the three axis and origin
    return [head_axis,origin]
    
def uncorrect_footaxis(frame,ankle_JC): 
    """Calculate the anatomical uncorrect foot joint center and axis function.

    Takes in a dictionary of xyz positions and marker names.
    and takes the ankle axis.
    Calculate the anatomical uncorrect foot axis.
    
    Markers used: RTOE, LTOE

    Parameters
    ----------
    frame : array
        Dictionaries of marker lists.
    ankle_JC : array
        An array of ankle_JC each x,y,z position.
           
    Returns
    -------
    R, L, foot_axis : array     
        Returns an array representing the incorrect foot joint center, the array contains a 1x3 array 
        representing the foot axis origin x, y, z positions, which is followed by a 3x2x3 array
        containing the foot axis center in the first dimension and the direction of the 
        axis in the second dimension. This will be used for calculating static offset angle 
        in static calibration.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import uncorrect_footaxis
    >>> frame = { 'RTOE': [442.81997681, 381.62280273, 42.66047668], 
    ...           'LTOE': [39.43652725, 382.44522095, 41.78911591]}
    >>> ankle_JC = [np.array([393.76181608, 247.67829633, 87.73775041]),
    ...            np.array([98.74901939, 219.46930221, 80.6306816]),
    ...            [[np.array([394.4817575, 248.37201348, 87.715368]),
    ...            np.array([393.07114384, 248.39110006, 87.61575574]),
    ...            np.array([393.69314056, 247.78157916, 88.73002876])],
    ...            [np.array([98.47494966, 220.42553803, 80.52821783]),
    ...            np.array([97.79246671, 219.20927275, 80.76255901]),
    ...            np.array([98.84848169, 219.60345781, 81.61663775])]]]
    >>> [np.around(arr,8) for arr in uncorrect_footaxis(frame,ankle_JC)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.81997681, 381.62280273,  42.66047668]), 
    array([ 39.43652725, 382.44522095,  41.78911591]), 
    array([[[442.93807347, 381.90040642,  43.61388602],
            [441.882686  , 381.97104076,  42.67518049],
            [442.49204525, 380.72744444,  42.96179781]],
           [[ 39.50071636, 382.6986218 ,  42.7543453 ],
            [ 38.49604413, 382.13712948,  41.93254235],
            [ 39.77025057, 381.52823259,  42.00765902]]])]       
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
    """Calculate the anatomical correct foot joint center and axis function which is for foot flat

    Takes in a dictionary of xyz positions and marker names.
    and takes the ankle axis.
    Calculate the anatomical correct foot axis for foot flat.

    Markers used: RTOE, LTOE, RHEE, LHEE

    Parameters
    ----------
    frame : array
        Dictionaries of marker lists.
    ankle_JC : array
        An array of ankle_JC each x,y,z position.
    vsk : dict, optional
        A dictionary containing subject measurements from a VSK file.

    Returns
    -------
    R, L, foot_axis: array
        Returns an array representing the correct foot joint center for flat feet, the array contains a 1x3 array 
        representing the foot axis origin x, y, z positions, which is followed by a 3x2x3 array
        containing the foot axis center in the first dimension and the direction of the 
        axis in the second dimension.        
         
    Modifies
    --------
    If the subject wears shoe, Soledelta is applied. then axes are changed following Soledelta.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import rotaxis_footflat
    >>> frame = { 'RHEE': [374.01257324, 181.57929993, 49.50960922],
    ...            'LHEE': [105.30126953, 180.2130127, 47.15660858],
    ...            'RTOE': [442.81997681, 381.62280273, 42.66047668], 
    ...            'LTOE': [39.43652725, 382.44522095, 41.78911591]}
    >>> ankle_JC = [np.array([393.76181608, 247.67829633, 87.73775041]),
    ...            np.array([98.74901939, 219.46930221, 80.6306816]),
    ...            [[np.array([394.4817575, 248.37201348, 87.715368]),
    ...            np.array([393.07114384, 248.39110006, 87.61575574]),
    ...           np.array([393.69314056, 247.78157916, 88.73002876])],
    ...            [np.array([98.47494966, 220.42553803, 80.52821783]),
    ...            np.array([97.79246671, 219.20927275, 80.76255901]),
    ...            np.array([98.84848169, 219.60345781, 81.61663775])]]]
    >>> vsk = { 'RightSoleDelta': 0.45, 'LeftSoleDelta': 0.45}
    >>> [np.around(arr,8) for arr in rotaxis_footflat(frame,ankle_JC,vsk)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.81997681, 381.62280273,  42.66047668]), 
    array([ 39.43652725, 382.44522095,  41.78911591]), 
    array([[[442.30666241, 381.79936348,  43.50031871],
            [442.02580128, 381.89596909,  42.1176458 ],
            [442.49471759, 380.67717784,  42.66047668]],
           [[ 39.14565179, 382.3504861 ,  42.74117514],
            [ 38.53126992, 382.15038888,  41.48320216],
            [ 39.74620554, 381.49437955,  41.78911591]]])]                 
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
    """Calculate the anatomical correct foot joint center and axis function which is for non foot flat

    Takes in a dictionary of xyz positions and marker names
    and takes the ankle axis and.
    Calculate the anatomical correct foot axis for non-foot flat.

    Markers used: RTOE, LTOE, RHEE, LHEE
        
    Parameters
    ----------
    frame : array
        Dictionaries of marker lists.
    ankle_JC : array
        An array of ankle_JC each x,y,z position.
           
    Returns
    -------
    array
        Returns an array representing the correct foot joint center for non-flat feet, the array contains a 1x3 array 
        representing the foot axis origin x, y, z positions, which is followed by a 3x2x3 array
        containing the foot axis center in the first dimension and the direction of the 
        axis in the second dimension.     

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmStatic import rotaxis_nonfootflat
    >>> frame = { 'RHEE': [374.01257324, 181.57929993, 49.50960922],
    ...            'LHEE': [105.30126953, 180.2130127, 47.15660858],
    ...            'RTOE': [442.81997681, 381.62280273, 42.66047668], 
    ...            'LTOE': [39.43652725, 382.44522095, 41.78911591]}
    >>> ankle_JC = [np.array([393.76181608, 247.67829633, 87.73775041]),
    ...            np.array([98.74901939, 219.46930221, 80.6306816]),
    ...            [[np.array([394.4817575, 248.37201348, 87.715368]),
    ...            np.array([393.07114384, 248.39110006, 87.61575574]),
    ...            np.array([393.69314056, 247.78157916, 88.73002876])],
    ...            [np.array([98.47494966, 220.42553803, 80.52821783]),
    ...            np.array([97.79246671, 219.20927275, 80.76255901]),
    ...            np.array([98.84848169, 219.60345781, 81.61663775])]]]
    >>> [np.around(arr,8) for arr in rotaxis_nonfootflat(frame,ankle_JC)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.81997681, 381.62280273,  42.66047668]), 
    array([ 39.43652725, 382.44522095,  41.78911591]), 
    array([[[442.71651135, 381.69236202,  43.65267444],
            [441.87997036, 381.94200709,  42.54007546],
            [442.49488793, 380.67767307,  42.69283623]],
           [[ 39.55544558, 382.51024763,  42.77988832],
            [ 38.49311916, 382.14149804,  41.92228333],
            [ 39.74610697, 381.4946822 ,  41.81434438]]])]                   
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
    
    This function takes in two axis and returns three angles.
    and It use inverse Euler rotation matrix in YXZ order.
    the output shows the angle in degrees.
    
    As we use arc sin we have to care about if the angle is in area between -pi/2 to pi/2
    but in case of calculate static offset angle it is in boundry under pi/2, it doesn't matter.
    
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
    >>> axisP = [[ 0.59327576, 0.10572786, 0.15773334],
    ...         [-0.13176004, -0.10067464, -0.90325703],
    ...         [0.9399765, -0.04907387, 0.75029827]]
    >>> axisD = [[0.16701015, 0.69080381, -0.37358145],
    ...         [0.1433922, -0.3923507, 0.94383974],
    ...         [-0.15507695, -0.5313784, -0.60119402]]
    >>> np.around(getankleangle(axisP,axisD),8)
    array([0.47919763, 0.99019921, 1.51695461])
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
        
def findJointC(a, b, c, delta): 
    """Calculate the Joint Center function.
    
    This function is based on physical markers, a,b,c and joint center which will be calulcated in this function are all in the same plane.

    Parameters
    ----------
    a,b,c : list 
        Three markers x,y,z position of a, b, c. 
    delta : float
        The length from marker to joint center, retrieved from subject measurement file.
    
    Returns
    -------
    mr : list
        Returns the Joint C x, y, z positions in a 1x3 array.

    Examples
    --------
    >>> import numpy as np 
    >>> from .pycgmStatic import findJointC
    >>> a = [775.40887891, 788.64742014, 514.4063264]
    >>> b = [424.5706403, 46.17046141, 305.73130661]
    >>> c = [618.98284978, 848.86492206, 133.51848843]
    >>> delta = 42.5
    >>> np.around(findJointC(a,b,c,delta),8)
    array([599.66684491, 843.26056826,  96.07876121])
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
    
    m = [(b[0]+c[0])/2,(b[1]+c[1])/2,(b[2]+c[2])/2]
    length = np.subtract(b,m)
    length = norm2d(length)
    
    # iterate_acosdomain =  1 - ( delta/norm2d(v2) - int(delta/norm2d(v2)) )
    
    # print "iterate_acosdomain:",iterate_acosdomain

    
    theta = acos(delta/norm2d(v2))

    cs = cos(theta*2)
    sn = sin(theta*2)

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
    >>> v = [50.00882192, 96.36735079, 264.84675407]
    >>> np.around(norm2d(v),8)
    286.23653105
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
    v : array
        A 3-element list.

    Returns
    -------
    array 
        The normalization of the vector returned as a float in an array.

    Examples
    --------
    >>> import numpy as np 
    >>> from .pycgmStatic import norm3d
    >>> v = [124.9784377, 368.642446, 18.42836272]
    >>> norm3d(v)
    array(389.68765369)
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
    """Matrix multiplication function
    
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
    >>> a = [12.83416835, 61.24792127, 99.59610493]
    >>> b = [14.79756689, 61.71925415, 95.44488778]
    >>> np.around(cross(a, b),8)
    array([-301.19634015,  248.82426677, -114.20491367])
    """
    c = [a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]]

    return c