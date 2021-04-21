def elbowJointCenter(frame,thorax,shoulderJC,wand,vsk=None):
    """Calculate the Elbow joint axis ( Humerus) function.

    Takes in a dictionary of marker names to x, y, z positions, the thorax
    axis, and shoulder joint center.

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
    >>> frame = {'RSHO': np.array([428.88, 270.55, 1500.73]),
    ...          'LSHO': np.array([68.24, 269.01, 1510.10]),
    ...          'RELB': np.array([658.90, 326.07, 1285.28]),
    ...          'LELB': np.array([-156.32, 335.25, 1287.39]),
    ...          'RWRA': np.array([776.51,495.68, 1108.38]),
    ...          'RWRB': np.array([830.90, 436.75, 1119.11]),
    ...          'LWRA': np.array([-249.28, 525.32, 1117.09]),
    ...          'LWRB': np.array([-311.77, 477.22, 1125.16])}
    >>> thorax = [[[256.23, 365.30, 1459.66],
    ...        [257.14, 364.21, 1459.58],
    ...        [256.08, 354.32, 1458.65]],
    ...        [256.14, 364.30, 1459.65]]
    >>> shoulderJC = [np.array([429.66, 275.06, 1453.95]),
    ...            np.array([64.51, 274.93, 1463.63])]
    >>> wand = [[255.92, 364.32, 1460.62],
    ...        [256.42, 364.27, 1460.61]]
    >>> vsk = { 'RightElbowWidth': 74.0, 'LeftElbowWidth': 74.0,
    ...         'RightWristWidth': 55.0, 'LeftWristWidth': 55.0}
    >>> [np.around(arr, 2) for arr in elbowJointCenter(frame,thorax,shoulderJC,wand,vsk)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 633.66,  304.95, 1256.07],
    [-129.16,  316.86, 1258.06]]), array([[[ 633.81,  303.96, 1256.07],
    [ 634.35,  305.05, 1256.79],
    [ 632.95,  304.84, 1256.77]],
    [[-129.32,  315.88, 1258.  ],
    [-128.45,  316.79, 1257.36],
    [-128.49,  316.72, 1258.78]]]), array([[ 793.32,  451.29, 1084.43],
    [-272.46,  485.79, 1091.37]])]
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