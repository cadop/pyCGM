import numpy as np

def wrist_axis(elbowJC):
    """Calculate the wrist joint axis (Radius) function.

    Takes in the elbow axis to calculate each wrist joint axis and returns it.

    Parameters
    ----------
    elbowJC : array
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
    >>> from .axis import wrist_axis
    >>> np.set_printoptions(suppress=True)
    >>> elbowJC = [ np.array([[   0.15,   -0.99,    0.  ,  633.66],
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
    >>> [np.around(arr, 2) for arr in wrist_axis(elbowJC)] #doctest: +NORMALIZE_WHITESPACE
    [array([[   0.44,   -0.84,   -0.31,  793.32],
        [   0.69,    0.1 ,    0.72,  451.29],
        [  -0.57,   -0.53,    0.62, 1084.43],
        [   0.  ,    0.  ,    0.  ,    1.  ]]), array([[  -0.47,   -0.79,   -0.4 , -272.45],
        [   0.72,   -0.07,   -0.7 ,  485.8 ],
        [   0.52,   -0.61,    0.6 , 1091.36],
        [   0.  ,    0.  ,    0.  ,    1.  ]])]
    """
    # Bring Elbow joint center, axes and Wrist Joint Center for calculating Radius Axes

    REJC = [elbowJC[0][0][3], elbowJC[0][1][3], elbowJC[0][2][3]]
    LEJC = [elbowJC[1][0][3], elbowJC[1][1][3], elbowJC[1][2][3]]

    R_elbow_flex = [elbowJC[0][1][0], elbowJC[0][1][1], elbowJC[0][1][2]]
    L_elbow_flex = [elbowJC[1][1][0], elbowJC[1][1][1], elbowJC[1][1][2]]

    RWJC = elbowJC[2][0]
    LWJC = elbowJC[2][1]

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

    r_axis = np.zeros((4, 4))
    r_axis[3, 3] = 1.0
    r_axis[0, :3] = x_axis
    r_axis[1, :3] = y_axis
    r_axis[2, :3] = z_axis
    r_axis[:3, 3] = RWJC

    # left
    y_axis = L_elbow_flex
    y_axis = y_axis/ np.linalg.norm(y_axis)

    z_axis = np.subtract(LEJC,LWJC)
    z_axis = z_axis/ np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis,z_axis)
    x_axis = x_axis/ np.linalg.norm(x_axis)

    z_axis = np.cross(x_axis,y_axis)
    z_axis = z_axis/ np.linalg.norm(z_axis)

    l_axis = np.zeros((4, 4))
    l_axis[3, 3] = 1.0
    l_axis[0, :3] = x_axis
    l_axis[1, :3] = y_axis
    l_axis[2, :3] = z_axis
    l_axis[:3, 3] = LWJC

    return [r_axis, l_axis]