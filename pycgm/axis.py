import numpy as np

def wrist_axis(elbow_jc):
    """Calculate the wrist joint axis (Radius) function.

    Takes in the elbow axis to calculate each wrist joint axis and returns it.

    Parameters
    ----------
    elbow_jc : array
        A list of three elements containing a 4x4 affine matrix representing the
        right elbow, a 4x4 affine matrix representing the left elbow, and a 2x3
        matrix representing the right and left wrist joint centers.

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
    >>> from .axis import wrist_axis
    >>> np.set_printoptions(suppress=True)
    >>> elbow_jc = [ np.array([[   0.15,   -0.99,    0.  ,  633.66],
    ...        [   0.69,    0.1 ,    0.72,  304.95],
    ...        [  -0.71,   -0.1 ,    0.7 , 1256.07],
    ...        [   0.  ,    0.  ,    0.  ,    1.  ]]),
    ...        np.array([[  -0.16,   -0.98,   -0.06, -129.16],
    ...        [   0.71,   -0.07,   -0.69,  316.86],
    ...        [   0.67,   -0.14,    0.72, 1258.06],
    ...        [   0.  ,    0.  ,    0.  ,    1.  ]]),
    ...        [
    ...            [793.32, 451.29, 1084.43],
    ...            [-272.45, 485.80, 1091.36]
    ...        ]
    ...    ]
    >>> [np.around(arr, 2) for arr in wrist_axis(elbow_jc)] #doctest: +NORMALIZE_WHITESPACE
    [array([[   0.44,   -0.84,   -0.31,  793.32],
        [   0.69,    0.1 ,    0.72,  451.29],
        [  -0.57,   -0.53,    0.62, 1084.43],
        [   0.  ,    0.  ,    0.  ,    1.  ]]), array([[  -0.47,   -0.79,   -0.4 , -272.45],
        [   0.72,   -0.07,   -0.7 ,  485.8 ],
        [   0.52,   -0.61,    0.6 , 1091.36],
        [   0.  ,    0.  ,    0.  ,    1.  ]])]
    """
    # Bring Elbow joint center, axes and Wrist Joint Center for calculating Radius Axes

    rejc = [elbow_jc[0][0][3], elbow_jc[0][1][3], elbow_jc[0][2][3]]
    lejc = [elbow_jc[1][0][3], elbow_jc[1][1][3], elbow_jc[1][2][3]]

    r_elbow_flex = [elbow_jc[0][1][0], elbow_jc[0][1][1], elbow_jc[0][1][2]]
    l_elbow_flex = [elbow_jc[1][1][0], elbow_jc[1][1][1], elbow_jc[1][1][2]]

    rwjc = elbow_jc[2][0]
    lwjc = elbow_jc[2][1]

    # this is the axis of radius
    # right
    y_axis = r_elbow_flex
    y_axis = y_axis/np.linalg.norm(y_axis)

    z_axis = np.subtract(rejc,rwjc)
    z_axis = z_axis/np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis,z_axis)
    x_axis = x_axis/np.linalg.norm(x_axis)

    z_axis = np.cross(x_axis,y_axis)
    z_axis = z_axis/np.linalg.norm(z_axis)

    r_axis = np.zeros((4, 4))
    r_axis[3, 3] = 1.0
    r_axis[0, :3] = x_axis
    r_axis[1, :3] = y_axis
    r_axis[2, :3] = z_axis
    r_axis[:3, 3] = rwjc

    # left
    y_axis = l_elbow_flex
    y_axis = y_axis/np.linalg.norm(y_axis)

    z_axis = np.subtract(lejc,lwjc)
    z_axis = z_axis/np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis,z_axis)
    x_axis = x_axis/np.linalg.norm(x_axis)

    z_axis = np.cross(x_axis,y_axis)
    z_axis = z_axis/np.linalg.norm(z_axis)

    l_axis = np.zeros((4, 4))
    l_axis[3, 3] = 1.0
    l_axis[0, :3] = x_axis
    l_axis[1, :3] = y_axis
    l_axis[2, :3] = z_axis
    l_axis[:3, 3] = lwjc

    return [r_axis, l_axis]
