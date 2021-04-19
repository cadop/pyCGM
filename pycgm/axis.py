import numpy as np
import math

def hand_axis(frame, elbowJC, wristJC, vsk=None):
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
    >>> from .axis import hand_axis
    >>> frame = {'RWRA': np.array([776.51,495.68, 1108.38]),
    ...          'RWRB': np.array([830.90, 436.75, 1119.11]),
    ...          'LWRA': np.array([-249.28, 525.32, 1117.09]),
    ...          'LWRB': np.array([-311.77, 477.22, 1125.16]),
    ...          'RFIN': np.array([863.71, 524.44, 1074.54]),
    ...          'LFIN': np.array([-326.65, 558.34, 1091.04])}
    >>> elbowJC = [[np.array([633.66, 304.95, 1256.07]),
    ...            np.array([-129.16, 316.86, 1258.06])],
    ...           [[[633.81, 303.96, 1256.07],
    ...                [634.35, 305.05, 1256.79],
    ...                [632.95, 304.85, 1256.77]],
    ...                [[-129.32, 315.88, 1258.00],
    ...                [-128.45, 316.79, 1257.37],
    ...                [-128.49, 316.72, 1258.78]]],
    ...                [[793.32, 451.29, 1084.43],
    ...                [-272.45, 485.80, 1091.36]]]
    >>> wristJC = [[[793.32, 451.29, 1084.43],
    ...            [-272.45, 485.80, 1091.36]],
    ...           [[[793.77, 450.44, 1084.12],
    ...            [794.01, 451.38, 1085.15],
    ...            [792.75, 450761812234714, 1085.05]],
    ...            [[-272.92, 485.01, 1090.96],
    ...            [-271.74, 485.72, 1090.67],
    ...            [-271.94, 485.19, 1091.96]]]]
    >>> vsk = { 'RightHandThickness': 34.0, 'LeftHandThickness': 34.0}
    >>> [np.around(arr, 2) for arr in hand_axis(frame,elbowJC,wristJC,vsk)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 859.8 ,  517.27, 1051.97],
    [-324.52,  551.89, 1068.02]]), array([[[ 859.95,  517.58, 1052.91],
    [ 859.08,  517.95, 1051.86],
    [ 859.13,  516.61, 1052.3 ]],
    [[-324.61,  552.16, 1068.98],
    [-325.32,  551.29, 1068.12],
    [-323.92,  551.13, 1068.29]]])]
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

    LHND = find_joint_center(LWRI,LWJC,LFIN,L_delta)
    RHND = find_joint_center(RWRI,RWJC,RFIN,R_delta)

        # Left
    z_axis = [LWJC[0]-LHND[0],LWJC[1]-LHND[1],LWJC[2]-LHND[2]]
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    y_axis = [LWRI[0]-LWRA[0],LWRI[1]-LWRA[1],LWRI[2]-LWRA[2]]
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis,z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = np.cross(z_axis,x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    L_axis = [x_axis,y_axis,z_axis]

        # Right
    z_axis = [RWJC[0]-RHND[0],RWJC[1]-RHND[1],RWJC[2]-RHND[2]]
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    y_axis = [RWRA[0]-RWRI[0],RWRA[1]-RWRI[1],RWRA[2]-RWRI[2]]
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis,z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = np.cross(z_axis,x_axis)
    y_axis_div = np.linalg.norm(y_axis)
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

def find_joint_center(mark_a, mark_b, mark_c, delta):
    """Calculate the Joint Center.
    This function is based on physical markers mark_a, mark_b, mark_c, 
    and joint center which will be calculated in this function are all 
    in the same plane.
    Parameters
    ----------
    mark_a, mark_b, mark_c : list
        Three markers x, y, z position of a, b, c.
    delta : float
        The length from marker to joint center, retrieved from subject measurement file.
    Returns
    -------
    joint_center : array
        Returns the joint center's x, y, z positions in a 1x3 array.
    Examples
    --------
    >>> import numpy as np
    >>> from .axis import find_joint_center
    >>> mark_a = [468.14, 325.09, 673.12]
    >>> mark_b = [355.90, 365.38, 940.69]
    >>> mark_c = [452.35, 329.06, 524.77]
    >>> delta = 59.5
    >>> find_joint_center(mark_a, mark_b, mark_c, delta).round(2)
    array([396.25, 347.92, 518.63])
    """
    # make the two vector using 3 markers, which is on the same plane.
    vec_1 = (mark_a[0]-mark_c[0], mark_a[1]-mark_c[1], mark_a[2]-mark_c[2])
    vec_2 = (mark_b[0]-mark_c[0], mark_b[1]-mark_c[1], mark_b[2]-mark_c[2])

    # vec_3 is np.cross vector of vec_1, vec_2
    # and then it normalized.
    # vec_3 = np.cross(vec_1, vec_2)
    vec_3 = np.np.cross(vec_1, vec_2)
    vec_3_div = np.linalg.norm(vec_3)
    vec_3 = [vec_3[0]/vec_3_div, vec_3[1]/vec_3_div, vec_3[2]/vec_3_div]

    mid = [(mark_b[0]+mark_c[0])/2.0, (mark_b[1]+mark_c[1])/2.0, (mark_b[2]+mark_c[2])/2.0]
    length = np.subtract(mark_b, mid)
    length = np.linalg.norm(length)

    theta = math.acos(delta/np.linalg.norm(vec_2))

    cs_th = math.cos(theta*2)
    sn_th = math.sin(theta*2)

    u_x = vec_3[0]
    u_y = vec_3[1]
    u_z = vec_3[2]

    # This rotation matrix is called Rodriques' rotation formula.
    # In order to make a plane, at least 3 number of markers is required which
    # means three physical markers on the segment can make a plane.
    # then the orthogonal vector of the plane will be rotating axis.
    # joint center is determined by rotating the one vector of plane around rotating axis.

    rot = np.matrix([
        [cs_th+u_x**2.0*(1.0-cs_th),u_x*u_y*(1.0-cs_th)-u_z*sn_th,u_x*u_z*(1.0-cs_th)+u_y*sn_th],
        [u_y*u_x*(1.0-cs_th)+u_z*sn_th,cs_th+u_y**2.0*(1.0-cs_th),u_y*u_z*(1.0-cs_th)-u_x*sn_th],
        [u_z*u_x*(1.0-cs_th)-u_y*sn_th,u_z*u_y*(1.0-cs_th)+u_x*sn_th,cs_th+u_z**2.0*(1.0-cs_th)]
    ])

    r_vec = rot * (np.matrix(vec_2).transpose())
    r_vec = r_vec * length/np.linalg.norm(r_vec)

    r_vec = [r_vec[0,0], r_vec[1,0], r_vec[2,0]]
    joint_center = np.array([r_vec[0]+mid[0], r_vec[1]+mid[1], r_vec[2]+mid[2]])

    return joint_center