import numpy as np

def shoulder_axis(thorax, shoulder_jc, wand):
    """Calculate the Shoulder joint axis (Clavicle) function.

    Takes in the thorax axis, wand marker and shoulder joint center.
    Calculate each shoulder joint axis and returns it.

    Parameters
    ----------
    thorax : array
        The x,y,z position of the thorax.
            thorax = [[R_thorax joint center x,y,z position],
                        [L_thorax_joint center x,y,z position],
                        [[R_thorax x axis x,y,z position],
                        [R_thorax,y axis x,y,z position],
                        [R_thorax z axis x,y,z position]]]
    shoulder_jc : array
        The x,y,z position of the shoulder joint center.
            shoulder_jc = [[R shoulder joint center x,y,z position],
                        [L shoulder joint center x,y,z position]]
    wand : array
        The x,y,z position of the wand.
            wand = [[R wand x,y,z, position],
                    [L wand x,y,z position]]

    Returns
    -------
    shoulder_jc, axis : array
        Returns the Shoulder joint center and axis in three array
            shoulder_JC = [[[[R_shoulder x axis, x,y,z position],
                        [R_shoulder y axis, x,y,z position],
                        [R_shoulder z axis, x,y,z position]],
                        [[L_shoulder x axis, x,y,z position],
                        [L_shoulder y axis, x,y,z position],
                        [L_shoulder z axis, x,y,z position]]],
                        [r_shoulder_jc_x, r_shoulder_jc_y, r_shoulder_jc_z],
                        [l_shoulder_jc_x,l_shoulder_jc_y,l_shoulder_jc_z]]

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import shoulder_axis
    >>> np.set_printoptions(suppress=True)
    >>> thorax = [[[256.23, 365.30, 1459.66],
    ...          [257.14, 364.21, 1459.58],
    ...          [256.08, 354.32, 1458.65]],
    ...          [256.14, 364.30, 1459.65]]
    >>> shoulder_jc = [np.array([429.66, 275.06, 1453.95]),
    ...              np.array([64.51, 274.93, 1463.63])]
    >>> wand = [[255.92, 364.32, 1460.62],
    ...        [256.42, 364.27, 1460.61]]
    >>> [np.around(arr, 2) for arr in shoulder_axis(thorax,shoulder_jc,wand)] #doctest: +NORMALIZE_WHITESPACE
        [array([[   0.46,    0.88,    0.09,  429.66],
            [   0.01,    0.09,   -1.  ,  275.06],
            [  -0.89,    0.46,    0.03, 1453.95],
            [   0.  ,    0.  ,    0.  ,    1.  ]]), array([[  -0.42,    0.9 ,    0.15,   64.51],
            [   0.08,   -0.13,    0.99,  274.93],
            [   0.91,    0.42,   -0.02, 1463.63],
            [   0.  ,    0.  ,    0.  ,    1.  ]])]
    """

    thorax_origin = thorax[1]

    r_shoulder_jc = shoulder_jc[0]
    l_shoulder_jc = shoulder_jc[1]

    R_wand = wand[0]
    L_wand = wand[1]
    R_wand_direc = [R_wand[0]-thorax_origin[0],R_wand[1]-thorax_origin[1],R_wand[2]-thorax_origin[2]]
    L_wand_direc = [L_wand[0]-thorax_origin[0],L_wand[1]-thorax_origin[1],L_wand[2]-thorax_origin[2]]
    R_wand_direc = R_wand_direc/np.linalg.norm(R_wand_direc)
    L_wand_direc = L_wand_direc/np.linalg.norm(L_wand_direc)

    # Right

    #Get the direction of the primary axis Z,X,Y
    z_direc = [(thorax_origin[0]-r_shoulder_jc[0]),
            (thorax_origin[1]-r_shoulder_jc[1]),
            (thorax_origin[2]-r_shoulder_jc[2])]
    z_direc = z_direc/np.linalg.norm(z_direc)
    y_direc = [R_wand_direc[0]*-1,R_wand_direc[1]*-1,R_wand_direc[2]*-1]
    x_direc = np.cross(y_direc,z_direc)
    x_direc = x_direc/np.linalg.norm(x_direc)
    y_direc = np.cross(z_direc,x_direc)
    y_direc = y_direc/np.linalg.norm(y_direc)

    r_axis = np.zeros((4, 4))
    r_axis[3, 3] = 1.0
    r_axis[0, :3] = x_direc
    r_axis[1, :3] = y_direc
    r_axis[2, :3] = z_direc
    r_axis[:3, 3] = r_shoulder_jc

    # Left

    #Get the direction of the primary axis Z,X,Y
    z_direc = [(thorax_origin[0]-l_shoulder_jc[0]),
            (thorax_origin[1]-l_shoulder_jc[1]),
            (thorax_origin[2]-l_shoulder_jc[2])]
    z_direc = z_direc/np.linalg.norm(z_direc)
    y_direc = L_wand_direc
    x_direc = np.cross(y_direc,z_direc)
    x_direc = x_direc/np.linalg.norm(x_direc)
    y_direc = np.cross(z_direc,x_direc)
    y_direc = y_direc/np.linalg.norm(y_direc)

    # backwards to account for marker size
    x_axis = [x_direc[0]+l_shoulder_jc[0],x_direc[1]+l_shoulder_jc[1],x_direc[2]+l_shoulder_jc[2]]
    y_axis = [y_direc[0]+l_shoulder_jc[0],y_direc[1]+l_shoulder_jc[1],y_direc[2]+l_shoulder_jc[2]]
    z_axis = [z_direc[0]+l_shoulder_jc[0],z_direc[1]+l_shoulder_jc[1],z_direc[2]+l_shoulder_jc[2]]

    l_axis = np.zeros((4, 4))
    l_axis[3, 3] = 1.0
    l_axis[0, :3] = x_direc
    l_axis[1, :3] = y_direc
    l_axis[2, :3] = z_direc
    l_axis[:3, 3] = l_shoulder_jc

    return([r_axis,l_axis])

    return [shoulder_jc,axis]