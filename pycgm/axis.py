import numpy as np

def wrist_axis(elbow_jc):
    """Calculate the Wrist joint axis ( Radius) function.

    Takes in the elbow axis to calculate each wrist joint axis and returns it.

    Parameters
    ----------
    elbow_jc : array
        The x,y,z position of the elbow joint center.

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
    >>> elbow_jc = [[np.array([633.66, 304.95, 1256.07]),
    ...           np.array([-129.16, 316.86, 1258.06])],
    ...           [[[633.81, 303.96, 1256.07],
    ...           [634.35, 305.05, 1256.79],
    ...           [632.95, 304.85, 1256.77]],
    ...           [[-129.32, 315.88, 1258.00],
    ...           [-128.45, 316.79, 1257.37],
    ...           [-128.49, 316.72, 1258.78]]],
    ...           [[793.32, 451.29, 1084.43],
    ...           [-272.45, 485.80, 1091.36]]]
    >>> [np.around(arr, 2) for arr in wristJointCenter(frame,shoulderJC,wand,elbow_jc)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 793.32,  451.29, 1084.43],
    [-272.45,  485.8 , 1091.36]]), array([[[ 793.76,  450.45, 1084.12],
    [ 794.01,  451.39, 1085.15],
    [ 792.75,  450.76, 1085.05]],
    [[-272.92,  485.01, 1090.96],
    [-271.73,  485.73, 1090.66],
    [-271.93,  485.19, 1091.96]]])]
    """

    # Bring Elbow joint center, axes and Wrist Joint Center for calculating Radius Axes

    REJC = elbow_jc[0][0]
    LEJC = elbow_jc[0][1]

    R_elbow_axis = elbow_jc[1][0]
    L_elbow_axis = elbow_jc[1][1]

    R_elbow_flex = [R_elbow_axis[1][0]-REJC[0],R_elbow_axis[1][1]-REJC[1],R_elbow_axis[1][2]-REJC[2]]
    L_elbow_flex = [L_elbow_axis[1][0]-LEJC[0],L_elbow_axis[1][1]-LEJC[1],L_elbow_axis[1][2]-LEJC[2]]

    RWJC = elbow_jc[2][0]
    LWJC = elbow_jc[2][1]

    # this is the axis of radius

    # right
    y_axis = R_elbow_flex
    y_axis = y_axis/ np.linalg.norm(y_axis)

    z_axis = np.subtract(REJC,RWJC)
    z_axis = z_axis/ np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis,z_axis)
    x_axis = x_axis/ np.linalg.norm(x_axis)

    z_axis = np.cross(x_axis,y_axis)
    z_axis = z_axis/ np.linalg.norm(z_axis)

    # Attach all the axes to wrist joint center.
    x_axis = [x_axis[0]+RWJC[0],x_axis[1]+RWJC[1],x_axis[2]+RWJC[2]]
    y_axis = [y_axis[0]+RWJC[0],y_axis[1]+RWJC[1],y_axis[2]+RWJC[2]]
    z_axis = [z_axis[0]+RWJC[0],z_axis[1]+RWJC[1],z_axis[2]+RWJC[2]]

    R_axis = [x_axis,y_axis,z_axis]

    # left

    y_axis = L_elbow_flex
    y_axis = y_axis/ np.linalg.norm(y_axis)

    z_axis = np.subtract(LEJC,LWJC)
    z_axis = z_axis/ np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis,z_axis)
    x_axis = x_axis/ np.linalg.norm(x_axis)

    z_axis = np.cross(x_axis,y_axis)
    z_axis = z_axis/ np.linalg.norm(z_axis)

    # Attach all the axes to wrist joint center.
    x_axis = [x_axis[0]+LWJC[0],x_axis[1]+LWJC[1],x_axis[2]+LWJC[2]]
    y_axis = [y_axis[0]+LWJC[0],y_axis[1]+LWJC[1],y_axis[2]+LWJC[2]]
    z_axis = [z_axis[0]+LWJC[0],z_axis[1]+LWJC[1],z_axis[2]+LWJC[2]]

    L_axis = [x_axis,y_axis,z_axis]

    origin = [RWJC,LWJC]

    axis = [R_axis,L_axis]

    return [origin,axis]