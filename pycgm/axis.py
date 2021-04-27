import numpy as np

def wrist_axis(elbow_jc):
    r"""Calculate the wrist joint axis (Radius) function.

    Takes in the elbow axis to calculate each wrist joint axis and returns it.

    Parameters
    ----------
    elbow_jc : array
        A list of three elements containing a 4x4 affine matrix representing the
        right elbow, a 4x4 affine matrix representing the left elbow, and a list
        of two 4x4 matrices representing the left and right wrist joint centers.

    Returns
    --------
    [r_axis, l_axis] : array
        A list of two 4x4 affine matrices representing the right hand axis as
        well as the left hand axis.

    Notes
    -----
    .. math::

        \begin{matrix}
            o_{L} = \textbf{m}_{LWJC} & o_{R} = \textbf{m}_{RWJC} \\
            \hat{y}_{L} = Elbow\_Flex_{L} & \hat{y}_{R} =  Elbow\_Flex_{R} \\
            \hat{z}_{L} = \textbf{m}_{LEJC} - \textbf{m}_{LWJC} & \hat{z}_{R} = \textbf{m}_{REJC} - \textbf{m}_{RWJC} \\
            \hat{x}_{L} = \hat{y}_{L} \times \hat{z}_{L} & \hat{x}_{R} = \hat{y}_{R} \times \hat{z}_{R} \\
            \hat{z}_{L} = \hat{x}_{L} \times \hat{y}_{L} & \hat{z}_{R} = \hat{x}_{R} \times \hat{y}_{R} \\
        \end{matrix}

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import wrist_axis
    >>> np.set_printoptions(suppress=True)
    >>> elbow_jc = [ np.array([[   0.15,   -0.99,    0.  ,  633.66],
    ...        [ 0.69,  0.1,  0.72,  304.95],
    ...        [-0.71, -0.1,  0.7 , 1256.07],
    ...        [ 0.  ,  0. ,  0.  ,    1.  ]]),
    ...        np.array([[  -0.16,   -0.98,   -0.06, -129.16],
    ...        [ 0.71, -0.07, -0.69,  316.86],
    ...        [ 0.67, -0.14,  0.72, 1258.06],
    ...        [ 0.  ,  0.  ,  0.  ,    1.  ]]),
    ...        np.array([[[1, 0, 0,  793.32],
    ...             [0, 1, 0,  451.29],
    ...             [0, 0, 1, 1084.43],
    ...             [0, 0, 0,    1.  ]],
    ...            [[1, 0, 0, -272.45],
    ...             [0, 1, 0,  485.8 ],
    ...             [0, 0, 1, 1091.36],
    ...             [0, 0, 0,    1.  ]]])
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

    rejc = elbow_jc[0][:3, 3]
    lejc = elbow_jc[1][:3, 3]

    r_elbow_flex = elbow_jc[0][1, :3]
    l_elbow_flex = elbow_jc[1][1, :3]

    rwjc = elbow_jc[2][0][:3, 3]
    lwjc = elbow_jc[2][1][:3, 3]

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
