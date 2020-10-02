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

    Example
    -------
    >>> import numpy as np
    >>> from math import *

    >>> rotmat() #doctest: +NORMALIZE_WHITESPACE
    [[1.0, 0.0, 0.0], 
    [0.0, 1.0, 0.0], 
    [0.0, 0.0, 1.0]]

    >>> x = 0.5
    >>> y = 0.3
    >>> z = 0.8
    >>> rotmat(x,y,z) #doctest: +NORMALIZE_WHITESPACE
    [[0.9998888175929077, -0.0139619889490318, 0.00523596383141958], 
    [0.01400733607248772, 0.9998638128275725, -0.008726415877184502], 
    [-0.005113412638268541, 0.008798787558312074, 0.9999482158335473]]

    >>> x = 0.5
    >>> rotmat(x) #doctest: +NORMALIZE_WHITESPACE
    [[1.0, 0.0, 0.0], 
    [0.0, 0.9999619230641713, -0.008726535498373935], 
    [0.0, 0.008726535498373935, 0.9999619230641713]]

    >>> x = 1
    >>> y = 1
    >>> rotmat(x,y) #doctest: +NORMALIZE_WHITESPACE
    [[0.9998476951563913, 0.0, 0.01745240643728351], 
    [0.00030458649045213493, 0.9998476951563913, -0.017449748351250485], 
    [-0.017449748351250485, 0.01745240643728351, 0.9996954135095479]]
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
            [ x, y, z ]
    p1 : array
        Position of second x,y,z coordinate.
            [ x, y, z ]
    Returns
    -------
    float 
        The distance between positions p0 and p1. 

    Example
    -------
    >>> from math import *
    >>> import numpy as np

    >>> p0 = [0,1,2]
    >>> p1 = [1,2,3]
    >>> getDist(p0,p1)
    1.7320508075688772

    >>> p0 = np.array([991.44611381, 741.95103792, 321.35500969])
    >>> p1 = np.array([117.08710839, 142.23917057, 481.95268411])
    >>> getDist(p0,p1)
    1072.3570334681392
    """
    return sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)
    
def getStatic(motionData,vsk,flat_foot=False,GCS=None):
    """ Get Static Offset function
    
    Calculate the static offset angle values and return the values in radians

    Parameters
    ----------
    motionData : dict
        Dictionary of marker lists.
            { [], [], [], ... }
    vsk : dict, optional
        Dictionary of various attributes of the skeleton.
            { [], [], [], ... }
    flat_foot : boolean, optional
        A boolean indicating of the feet are flat or not.
        The default value is False.
    GCS : array, optional
        An array containing the Global Coordinate System.
        If not provided, the default will be set to: [[1, 0, 0], [0, 1, 0], [0, 0, 1]].
    
    Returns
    -------
    calSM : dict
        Dictionary containing various marker lists of offsets.
            { [], [], [], ... }
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
    i =0
    total = 0.0
    while(i <len(list)):
        total = total + list[i]
        i = i+1
    return total / len(list)    

def IADcalculation(frame):
    RASI = frame['RASI']
    LASI = frame['LASI']
    IAD = np.sqrt((RASI[0]-LASI[0])*(RASI[0]-LASI[0])+(RASI[1]-LASI[1])*(RASI[1]-LASI[1])+(RASI[2]-LASI[2])*(RASI[2]-LASI[2]))
    
    return IAD
    
def staticCalculationHead(frame,head):
    
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
    """
    
    Calculate head offset angle for static calibration.
    This function is only called in static trial. 
    and output will be used in dynamic later. 
    
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
    """
    Calculate the Static angle function
    
    Takes in anatomical uncorrect axis and anatomical correct axis. 
    Calculates the offset angle between that two axis.

    It is rotated from uncorrect axis in YXZ order.
    
    ----------------------------------------------------------------
    INPUT: Two arrays of uncorrect axis and correct axis. 
            corect axis depends on foot flat options.
            
    OUTPUT: Returns offset angle.
            [[Right Flexion angle, Right Abduction angle, Right Rotation angle],
            [Left Flexion angle, Left Abduction angle, Left Rotation angle]]
     
    MODIFIES: the correct axis changes following to the foot flat option. 
    ----------------------------------------------------------------
    
    EXAMPLE:
            i=1
            knee_JC: [array([364.17774614, 292.17051722, 515.19181496]),
                    array([143.55478579, 279.90370346, 524.78408753]),
                    array([[[364.64959153, 293.06758353, 515.18513093],
                            [363.29019771, 292.60656648, 515.04309095],
                            [364.04724541, 292.24216264, 516.18067112]],
                           [[143.65611282, 280.88685896, 524.63197541],
                            [142.56434499, 280.01777943, 524.86163553],
                            [143.64837987, 280.04650381, 525.76940383]]])]
            ankle_JC: [array([393.76181608, 247.67829633, 87.73775041]),
                        array([98.74901939, 219.46930221, 80.6306816]),
                        [[array([394.4817575, 248.37201348, 87.715368]),
                        array([393.07114384, 248.39110006, 87.61575574]),
                        array([393.69314056, 247.78157916, 88.73002876])],
                        [array([98.47494966, 220.42553803, 80.52821783]),
                        array([97.79246671, 219.20927275, 80.76255901]),
                        array([98.84848169, 219.60345781, 81.61663775])]]]
                delta: 0 
                
            staticCalculation(frame,ankle_JC,knee_JC,flat_foot,vsk=None)
            
            >>> [[0.03482194, 0.14879424, 0],
                [0.01139704, 0.02142806, 0]]
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
    """
    Make the Pelvis Axis function

    Takes in a dictionary of x,y,z positions and marker names, as well as an index
    Calculates the pelvis joint center and axis and returns both.
    -------------------------------------------------------------------------
    
    INPUT: 
        dictionaries of marker lists.  
            { [], [], [] }
    
    OUTPUT: Returns the origin and pelvis axis also sacrum
            Pelvis = [[origin x,y,z position],
                      [[pelvis x_axis x,y,z position],
                       [pelvis y_axis x,y,z position],
                       [pelvis z_axis x,y,z position]],
                       [sacrum x,y,z position]]    
    MODIFIES: -
    
    -------------------------------------------------------------------------
    
    EXAMPLE:
            i = 3883
            frame = {...,'LASIX': 183.18504333, 'LASIY': 422.78927612, 'LASIZ': 1033.07299805,  
                            'LPSIX': 255.79994202, 'LPSIY': 241.42199707, 'LPSIZ': 1057.30065918,        
                            'RASIX': 395.36532593, 'RASIY': 428.09790039, 'RASIZ': 1036.82763672,
                            'RPSIX': 341.41815186, 'RPSIY': 246.72117615, 'RPSIZ': 1055.99145508}

            pelvisJointCenter(frame)

            >>> [array([ 289.27518463, 425.44358826, 1034.95031738]),
                array([[ 289.25243803, 426.43632163, 1034.8321521],
                    [ 288.27565385, 425.41858059, 1034.93263017],
                    [ 289.25467091, 425.56129577, 1035.94315379]]),
                array([ 298,60904694, 244.07158661, 1056.64605715])]


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

    """

    Calculate the hip joint center function.

    Takes in a dictionary of x,y,z positions and marker names, as well as an index.
    Calculates the hip joint center and returns the hip joint center.
    -------------------------------------------------------------------------

    INPUT: 
            An array of pel_origin, pel_x, pel_y, pel_z each x,y,z position.
            and pel_x,y,z is axis of pelvis.
              [(),(),()]
    
    OUTPUT: Returns the hip joint center in two array
            hip_JC = [[L_hipJC x,y,z position], [R_hipJC x,y,z position]]
    
    MODIFIES: -

    ---------------------------------------------------------------------------
    
    EXAMPLE:
            i = 1
            pel_origin = [ 251.60830688, 391.74131775, 1032.89349365]
            pel_x = [251.74063624, 392.72694721, 1032.78850073]
            pel_y = [250.61711554, 391.87232862, 1032.8741063]
            pel_z = [251.60295336, 391.84795134, 1033.88777762]
            
            hipJointCenter(frame,pel_origin,pel_x,pel_y,pel_z)

            >>> [[ 182.57097863, 339.43231855, 935.52900126],
                [308.38050472, 322.80342417, 937.98979061]]

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
    
    """

    Calculate the hip joint axis function.

    Takes in a hip joint center of x,y,z positions as well as an index.
    and takes the hip joint center and pelvis origin/axis from previous functions.
    Calculates the hip axis and returns hip joint origin and axis.
    -------------------------------------------------------------------------

    INPUT:  Array of R_hip_jc, L_hip_jc, pelvis_axis each x,y,z position.
            and pelvis_axis is array of pelvis origin and axis. the axis also composed of 3 arrays 
            each things are x axis, y axis, z axis.
            
    OUTPUT: Returns the hip Axis Center and hip Axis.
             return = [[hipaxis_center x,y,z position],
                       [array([[hipaxis_center x,y,z position],
                               [hip x_axis x,y,z position]]),
                        array([[hipaxis_center x,y,z position],
                               [hip y_axis x,y,z position]])
                        array([[hipaxis_center x,y,z position],
                               [hip z_axis x,y,z position]])]]","                   
    MODIFIES: -

    ---------------------------------------------------------------------------

    EXAMPLE:
            i = 1
            r_hip_jc = [182.57097863, 339.43231855, 935.529000126]
            l_hip_jc = [308.38050472, 322.80342417, 937.98979061]
            pelvis_axis = [array([251.60830688, 391.74131775, 1032.89349365]),
                            array([[251.74063624, 392.72694721, 1032.78850073],
                                [250.61711554, 391.87232862, 1032.8741063],
                                [251.60295336, 391.84795134, 1033.88777762]]),
                            array([231.57849121, 210.25262451, 1052.24969482])]

            hipAxisCenter(l_hip_jc,r_hip_jc,pelvis_axis)

            >>> [[245.47574168208043, 331.1178713574418, 936.75939593146768],
                [[245.60807102843359, 332.10350081526684, 936.65440301116018],
                [244.48455032769033, 331.24888223306482, 936.74000858315412],
                [245.47038814489719, 331.22450494659665, 937.75367990368613]]]

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
    
    """

    Calculate the knee joint center and axis function.

    Takes in a dictionary of xyz positions and marker names, as well as an index.
    and takes the hip axis and pelvis axis.
    Calculates the knee joint axis and returns the knee origin and axis
    -------------------------------------------------------------------
    
    INPUT:dictionaries of marker lists.  
            { [], [], [] }
           An array of hip_JC, pelvis_axis each x,y,z position.
           delta = get from subject measurement file
    
    OUTPUT:  Returns the Knee Axis Center and Knee Axis.
             return = [[kneeaxis_center x,y,z position],
                       [array([[kneeaxis_center x,y,z position],
                               [knee x_axis x,y,z position]]),
                        array([[kneeaxis_center x,y,z position],
                               [knee y_axis x,y,z position]])
                        array([[kneeaxis_center x,y,z position],
                               [knee z_axis x,y,z position]])]]                               
        
    MODIFIES: delta is changed suitably to knee

    -------------------------------------------------------------------
    EXAMPLE:
            i = 1
            frame
            = { 'RTHI': [426.50338745, 262.65310669, 673.66247559],
                'LTHI': [51.93867874, 320.01849365, 723.03186035],
                'RKNE': [416.98687744, 266.22558594, 524.04089355],
                'LKNE': [84.62355804, 286.69122314, 529.39819336],...}
                hip_JC: [[182.57097863, 339.43231855, 935.52900126],
                        [309.38050472, 32280342417, 937.98979061]]
                delta: 0

            kneeJointCenter(frame,hip_JC,delta,vsk=None)

            >>> [array([364.17774614, 292.17051722, 515.19181496]),
                array([143.55478579, 279.90370346, 524.78408753]),
                array([[[364.64959153, 293.06758353, 515.18513093],
                        [363.29019771, 292.60656648, 515.04309095],
                        [364.04724541, 292.24216264, 516.18067112]],
                       [[143.65611282, 280.88685896, 524.63197541],
                        [142.56434499, 280.01777943, 524.86163553],
                        [143.64837987, 280.04650381, 525.76940383]]])]
                   

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
    
    """

    Calculate the ankle joint center and axis function.

    Takes in a dictionary of xyz positions and marker names, as well as an index.
    and takes the knee axis.
    Calculates the ankle joint axis and returns the ankle origin and axis
    -------------------------------------------------------------------

    INPUT: dictionaries of marker lists.  
            { [], [], [] }
           An array of knee_JC each x,y,z position.
           delta = 0
    
    OUTPUT:  Returns the Ankle Axis Center and Ankle Axis.
             return = [[ankle axis_center x,y,z position],
                       [array([[ankleaxis_center x,y,z position],
                               [ankle x_axis x,y,z position]]),
                        array([[ankleaxis_center x,y,z position],
                               [ankle y_axis x,y,z position]])
                        array([[ankleaxis_center x,y,z position],
                               [ankle z_axis x,y,z position]])]]                               
    MODIFIES: -

    ---------------------------------------------------------------------

    EXAMPLE:
            i = 1
            frame
            = { 'RTIB': [433.97537231, 211.93408203, 273.3008728],
                'LTIB': [50.04016495, 235.90718079, 364.32226562],
                'RANK': [422.77005005, 217.74053955, 92.86152649],
                'LANK': [58.57380676, 208.54806519, 86.16953278],...}
            knee_JC: [array([364.17774614, 292.17051722, 515.19181496]),
                    array([143.55478579, 279.90370346, 524.78408753]),
                    array([[[364.64959153, 293.06758353, 515.18513093],
                            [363.29019771, 292.60656648, 515.04309095],
                            [364.04724541, 292.24216264, 516.18067112]],
                           [[143.65611282, 280.88685896, 524.63197541],
                            [142.56434499, 280.01777943, 524.86163553],
                            [143.64837987, 280.04650381, 525.76940383]]])]
            delta: 0
            
            ankleJointCenter(frame,knee_JC,delta,vsk=None)

            >>> [array([393.76181608, 247.67829633, 87.73775041]),
                array([98.74901939, 219.46930221, 80.6306816]),
                [[array([394.4817575, 248.37201348, 87.715368]),
                array([393.07114384, 248.39110006, 87.61575574]),
                array([393.69314056, 247.78157916, 88.73002876])],
                [array([98.47494966, 220.42553803, 80.52821783]),
                array([97.79246671, 219.20927275, 80.76255901]),
                array([98.84848169, 219.60345781, 81.61663775])]]]
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
    
    """

    Calculate the foot joint center and axis function.
    
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

    -------------------------------------------------------------------
        
    INPUT: dictionaries of marker lists.  
            { [], [], [] }
            An array of ankle_JC,knee_JC each x,y,z position.
           delta = 0
           static_info = [[R_plantar_static_angle, R_static_rotation_angle, 0], # Right Static information
                          [L_plantar_static_angle, L_static_rotation_angle, 0]] # Left Static information
        
    OUTPUT: Returns the footJointCenter and foot axis. and save the static offset angle in a global variable.
          return = [[foot axis_center x,y,z position],
                    [array([[footaxis_center x,y,z position],
                            [foot x_axis x,y,z position]]),
                    array([[footaxis_center x,y,z position],
                           [foot y_axis x,y,z position]])
                    array([[footaxis_center x,y,z position],
                            [foot z_axis x,y,z position]])]]        
         
    MODIFIES:   Axis changes following to the static info.
        
                you can set the static_info by the button. and this will calculate the offset angles 
                the first setting, the foot axis show foot uncorrected anatomical reference axis(Z_axis point to the AJC from TOE)
        
                if press the static_info button so if static_info is not None,
                and then the static offsets angles are applied to the reference axis.
                the reference axis is Z axis point to HEE from TOE

    --------------------------------------------------------------------

    EXAMPLE:
            i = 1
            frame = { 'RHEE': [374.01257324, 181.57929993, 49.50960922],
                      'LHEE': [105.30126953, 180.2130127, 47.15660858],
                      'RTOE': [442.81997681, 381.62280273, 42.66047668 
                      'LTOE': [39.43652725, 382.44522095, 41.78911591],...}
            static_info : [[0.03482194, 0.14879424, 0],
                           [0.01139704, 0.02142806, 0]]
            knee_JC: [array([364.17774614, 292.17051722, 515.19181496]),
                    array([143.55478579, 279.90370346, 524.78408753]),
                    array([[[364.64959153, 293.06758353, 515.18513093],
                            [363.29019771, 292.60656648, 515.04309095],
                            [364.04724541, 292.24216264, 516.18067112]],
                           [[143.65611282, 280.88685896, 524.63197541],
                            [142.56434499, 280.01777943, 524.86163553],
                            [143.64837987, 280.04650381, 525.76940383]]])]
            ankle_JC: [array([393.76181608, 247.67829633, 87.73775041]),
                        array([98.74901939, 219.46930221, 80.6306816]),
                        [[array([394.4817575, 248.37201348, 87.715368]),
                        array([393.07114384, 248.39110006, 87.61575574]),
                        array([393.69314056, 247.78157916, 88.73002876])],
                        [array([98.47494966, 220.42553803, 80.52821783]),
                        array([97.79246671, 219.20927275, 80.76255901]),
                        array([98.84848169, 219.60345781, 81.61663775])]]]
            delta: 0

            footJointCenter(frame,static_info,ankle_JC,knee_JC,delta,vsk=None)

            >>> [array([442.81997681, 381.62280273, 42.66047668]),
                array([39.43652725, 382.44522095, 41.78911591]),
                [[[442.88815408948221, 381.7646059422284, 43.648020966284719],
                  [441.87135392672275, 381.93856951438391, 42.680625439845173],
                  [442.51100028681969, 380.68462194642137, 42.816522573058428]],
                 [[39.507852120747259, 382.67891585204035, 42.75880629687082],
                  [38.49231838166678, 385.14765969549836, 41.930278614215709],
                  [39.758058544512153, 381.51956226668784, 41.98854919067994]]]]
                  
    """

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
    """

    Calculate the head joint axis function.

    Takes in a dictionary of x,y,z positions and marker names.
    Calculates the head joint center and returns the head joint center and axis.
    -------------------------------------------------------------------------

    INPUT:  dictionaries of marker lists.  
            { [], [], [] }
    
    OUTPUT: Returns the Head joint center and axis in three array
            head_JC = [[[head x axis x,y,z position],
                        [head y axis x,y,z position],
                        [head z axis x,y,z position]],
                        [head x,y,z position]]
    
    MODIFIES: -
    ---------------------------------------------------------------------------
    
    EXAMPLE:
            i = 1
            frame = {'RFHD': [325.82983398, 402.55450439, 1722.49816895],
                     'LFHD': [184.55158997, 409.68713379, 1721.34289551],
                     'RBHD': [304.39898682, 242.91339111, 1694.97497559],
                     'LBHD': [197.8621521, 251.28889465, 1696.90197754], ...}
            
            headJC(frame,vsk=None)

            >>> [[[255.21590217746564, 407.10741939149585, 1722.0817317995723],
                [254.19105385179665, 406.146809183757, 1721.9176771191715],
                [255.18370553356357, 405.959746549898, 1722.9074499262838]],
                [255.19071197509766, 406.12081909179687, 1721.9205322265625]]

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
    """

    Calculate the anatomical uncorrect foot joint center and axis function.

    Takes in a dictionary of xyz positions and marker names.
    and takes the ankle axis.
    Calculate the anatomical uncorrect foot axis.
    ------------------------------------------------------------------
        
    INPUT: dictionaries with each marker's list
            {[],[],[]}
           An array of ankle_JC each x,y,z position.
           delta = 0
           
    OUTPUT: Returns the incorrect footJointCenter and foot axis. This will be used for calculating static offset angle in static calibration.
          return = [[foot axis_center x,y,z position],
                    [array([[footaxis_center x,y,z position],
                            [foot x_axis x,y,z position]]),
                    array([[footaxis_center x,y,z position],
                           [foot y_axis x,y,z position]])
                    array([[footaxis_center x,y,z position],
                            [foot z_axis x,y,z position]])]]        
         
    MODIFIES:   -
    
    -----------------------------------------------------------------------

    EXAMPLE:
            i = 1
            frame = { 'RHEE': [374.01257324, 181.57929993, 49.50960922],
                      'LHEE': [105.30126953, 180.2130127, 47.15660858],
                      'RTOE': [442.81997681, 381.62280273, 42.66047668 
                      'LTOE': [39.43652725, 382.44522095, 41.78911591],...}
            knee_JC: [array([364.17774614, 292.17051722, 515.19181496]),
                    array([143.55478579, 279.90370346, 524.78408753]),
                    array([[[364.64959153, 293.06758353, 515.18513093],
                            [363.29019771, 292.60656648, 515.04309095],
                            [364.04724541, 292.24216264, 516.18067112]],
                           [[143.65611282, 280.88685896, 524.63197541],
                            [142.56434499, 280.01777943, 524.86163553],
                            [143.64837987, 280.04650381, 525.76940383]]])]
            ankle_JC: [array([393.76181608, 247.67829633, 87.73775041]),
                        array([98.74901939, 219.46930221, 80.6306816]),
                        [[array([394.4817575, 248.37201348, 87.715368]),
                        array([393.07114384, 248.39110006, 87.61575574]),
                        array([393.69314056, 247.78157916, 88.73002876])],
                        [array([98.47494966, 220.42553803, 80.52821783]),
                        array([97.79246671, 219.20927275, 80.76255901]),
                        array([98.84848169, 219.60345781, 81.61663775])]]]
            delta: 0

            uncorrect_footaxis(frame,ankle_JC,knee_JC,delta,vsk=None)

            >>> [array([ 442.82369995, 381.62716675, 42.66253662]),
                array([39.44643402, 382.45663452, 41.79634857]),
                [[[442.94168992364411, 381.90478836486795, 43.615953950635053],
                  [441.88641674118907, 381.97543028972524, 42.677120510218828],
                  [442.49570825816676, 380.73182394028481, 42.963838306964107]],
                 [[39.510624781365145, 382.71005900587886, 42.761571642881954],
                  [38.505931260238086, 382.14860120518563, 41.939771139933704],
                  [39.780101674942053, 381.5396331546591, 42.014922106201844]]]]
            
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
    """

    Calculate the anatomical correct foot joint center and axis function which is for foot flat

    Takes in a dictionary of xyz positions and marker names.
    and takes the ankle axis.
    Calculate the anatomical correct foot axis for foot flat.
    ------------------------------------------------------------------
        
    INPUT:  dictionaries of marker lists.  
            { [], [], [] }
            An array of ankle_JC each x,y,z position.
           
    OUTPUT: Returns the footJointCenter and correct foot axis for foot flat.
        return = [[foot axis_center x,y,z position],
                    [array([[footaxis_center x,y,z position],
                            [foot x_axis x,y,z position]]),
                    array([[footaxis_center x,y,z position],
                            [foot y_axis x,y,z position]])
                    array([[footaxis_center x,y,z position],
                            [foot z_axis x,y,z position]])]]        
         
    MODIFIES:   If the subject wears shoe, Soledelta is applied. then axes are changed following Soledelta.
    --------------------------------------------------------------------

    EXAMPLE:
            i = 1
            frame = { 'RHEE': [374.01257324, 181.57929993, 49.50960922],
                      'LHEE': [105.30126953, 180.2130127, 47.15660858],
                      'RTOE': [442.81997681, 381.62280273, 42.66047668 
                      'LTOE': [39.43652725, 382.44522095, 41.78911591],...}
            ankle_JC: [array([393.76181608, 247.67829633, 87.73775041]),
                        array([98.74901939, 219.46930221, 80.6306816]),
                        [[array([394.4817575, 248.37201348, 87.715368]),
                        array([393.07114384, 248.39110006, 87.61575574]),
                        array([393.69314056, 247.78157916, 88.73002876])],
                        [array([98.47494966, 220.42553803, 80.52821783]),
                        array([97.79246671, 219.20927275, 80.76255901]),
                        array([98.84848169, 219.60345781, 81.61663775])]]]

            rotaxis_footflat(frame,ankle_JC,vsk=vsk)

            >>> [array([ 442.82369995, 381.62716675, 42.66253662]),
                array([39.44643402, 382.45663452, 41.79634857]),
                [[[442.72018980280649, 381.69679347734973, 43.654724994212465],
                  [441.88371054797568, 381.94639670245357, 42.542070202169803],
                  [442.49857601931785, 380.68205069444491, 42.69494146887434]],
                 [[39.565351048740091, 382.52172872176396, 42.787116700858256],
                  [38.503083776720146, 382.15274258408243, 41.929540189406254],
                  [39.75619046909209, 381.50615441360623, 41.821617275560818]]]]
                        
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
    
    """

    Calculate the anatomical correct foot joint center and axis function which is for non foot flat

    Takes in a dictionary of xyz positions and marker names
    and takes the ankle axis and.
    Calculate the anatomical correct foot axis for non-foot flat.
    ------------------------------------------------------------------
        
    INPUT: dictionaries of marker lists.  
            { [], [], [] }
            An array of ankle_JC each x,y,z position.
           
    OUTPUT: Returns the footJointCenter and correct foot axis for non foot flat. 
          return = [[foot axis_center x,y,z position],
                    [array([[footaxis_center x,y,z position],
                            [foot x_axis x,y,z position]]),
                    array([[footaxis_center x,y,z position],
                           [foot y_axis x,y,z position]])
                    array([[footaxis_center x,y,z position],
                            [foot z_axis x,y,z position]])]]        
         
    MODIFIES:   -
    --------------------------------------------------------------------

    EXAMPLE:
            i = 1
            frame = { 'RHEE': [374.01257324, 181.57929993, 49.50960922],
                      'LHEE': [105.30126953, 180.2130127, 47.15660858],
                      'RTOE': [442.81997681, 381.62280273, 42.66047668 
                      'LTOE': [39.43652725, 382.44522095, 41.78911591],...}
            ankle_JC: [array([393.76181608, 247.67829633, 87.73775041]),
                        array([98.74901939, 219.46930221, 80.6306816]),
                        [[array([394.4817575, 248.37201348, 87.715368]),
                        array([393.07114384, 248.39110006, 87.61575574]),
                        array([393.69314056, 247.78157916, 88.73002876])],
                        [array([98.47494966, 220.42553803, 80.52821783]),
                        array([97.79246671, 219.20927275, 80.76255901]),
                        array([98.84848169, 219.60345781, 81.61663775])]]]


            rotaxis_nonfootflat(frame,ankle_JC)

            >>> [array([442.82369995, 381.62716675, 42.66253662]),
                 array([39.44643402, 382.45663452, 41.79634857]),
                 [[[442.69448539145668, 381.67161710474886, 43.653156509520521],
                   [441.88695723776266, 381.94941021535283, 42.525890228337367],
                   [442.4984051826429, 380.68155408232815, 42.66253662109375]],
                  [[39.571989586986426, 382.4975524174377, 42.787590986951152],
                   [38.503976892322832, 382.1494927199455, 41.928403387518372]
                   [39.756289407364967, 381.50585082392843, 41.796348571777344]]]]

                        
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
    """
    
    Static angle calculation function.
    
    This function takes in two axis and returns three angles.
    and It use inverse Euler rotation matrix in YXZ order.
    the output shows the angle in degrees.
    
    As we use arc sin we have to care about if the angle is in area between -pi/2 to pi/2
    but in case of calculate static offset angle it is in boundry under pi/2, it doesn't matter.
    ------------------------------------------------------
    
    INPUT: each axis show the unit vector of axis.
            axisP = [[axisP-x axis x,y,z position],
                    [axisP-y axis x,y,z position],
                    [axisP-z axis x,y,z position]]
            axisD = [[axisD-x axis x,y,z position],
                    [axisD-y axis x,y,z position],
                    [axisD-z axis x,y,z position]]
    OUTPUT: these angles are show on degree.
            angle = [gamma,beta,alpha]
    MODIFIES:
    ------------------------------------------------------
    
    EXAMPLE:
            INPUT:
            
            axisP: [[ 0.0464229   0.99648672  0.06970743]
                    [ 0.99734011 -0.04231089 -0.05935067]
                    [-0.05619277  0.07227725 -0.99580037]]
            axisD: [[-0.18067218 -0.98329158 -0.02225371]
                    [ 0.71383942 -0.1155303  -0.69071415]
                    [ 0.67660243 -0.1406784   0.7227854 ]]
                    
            OUTPUT:
            angle: [-17.197765362603256, 9.4320410147265115, 0.93765726704562813]
            
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
    """
    
    Calculate the Joint Center function.
    This function is based on physical markers, a,b,c and joint center which will be calulcated in this function are all in the same plane.
    ----------------------------------------------
    INPUT: three marker x,y,z position of a, b, c. 
            and delta which is the length from marker to joint center.
    
    OUTPUT: Joint C x,y,z position
            [joint C x position, joint C y position, joint C z position]
    
    MODIFIES: -
    
    ----------------------------------------------
    EXAMPLE: INPUT: a = [468.14532471, 325.09780884, 673.12591553]
                    b = [355.90861996, 365.38260964, 940.6974861]
                    c = [452.35180664, 329.0609436, 524.77893066]
                    delta = 59.5
             OUTPUT: c+r = [396.26807934, 347.78080454, 518.62778789]
             
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
    try:
        return sqrt((v[0]*v[0]+v[1]*v[1]+v[2]*v[2]))
    except:
        return np.nan   

def norm3d(v): 
    try:
        return np.asarray(sqrt((v[0]*v[0]+v[1]*v[1]+v[2]*v[2])))
    except:
        return np.nan
        
def normDiv(v):
    try:
        vec = sqrt((v[0]*v[0]+v[1]*v[1]+v[2]*v[2]))
        v = [v[0]/vec,v[1]/vec,v[2]/vec]
    except:
        vec = np.nan
    
    return [v[0]/vec,v[1]/vec,v[2]/vec]

def matrixmult (A, B):
    C = [[0 for row in range(len(A))] for col in range(len(B[0]))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k]*B[k][j]
    return C
 
def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]]

    return c