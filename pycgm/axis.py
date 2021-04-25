import math
import numpy as np


def find_joint_center(p_a, p_b, p_c, delta):
    r"""Calculate the Joint Center.

    This function is based on the physical markers p_a, p_b, p_c
    and the resulting joint center are all on the same plane.

    Parameters
    ----------
    p_a, p_b, p_c : list
        Three markers x, y, z position of a, b, c.
    delta : float
        The length from marker to joint center, retrieved from subject measurement file.

    Returns
    -------
    joint_center : array
        Returns the joint center's x, y, z positions in a 1x3 array.

    Notes
    -----
    :math:`vec_{1} = p\_a-p\_c, \ vec_{2} = (p\_b-p\_c), \ vec_{3} = vec_{1} \times vec_{2}`

    :math:`mid = \frac{(p\_b+p\_c)}{2.0}`

    :math:`length = (p\_b - mid)`

    :math:`\theta = \arccos(\frac{delta}{vec_{2}})`

    :math:`\alpha = \cos(\theta*2), \ \beta = \sin(\theta*2)`

    :math:`u_x, u_y, u_z = vec_{3}`

    .. math::

        rot =
        \begin{bmatrix}
            \alpha+u_x^2*(1-\alpha) & u_x*u_y*(1.0-\alpha)-u_z*\beta & u_x*u_z*(1.0-\alpha)+u_y*\beta \\
            u_y*u_x*(1.0-\alpha+u_z*\beta & \alpha+u_y^2.0*(1.0-\alpha) & u_y*u_z*(1.0-\alpha)-u_x*\beta \\
            u_z*u_x*(1.0-\alpha)-u_y*\beta & u_z*u_y*(1.0-\alpha)+u_x*\beta & \alpha+u_z**2.0*(1.0-\alpha) \\
        \end{bmatrix}

    :math:`r\_vec = rot * vec_2`

    :math:`r\_vec = r\_vec * \frac{length}{norm(r\_vec)}`

    :math:`joint\_center = r\_vec + mid`

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import find_joint_center
    >>> p_a = [468.14, 325.09, 673.12]
    >>> p_b = [355.90, 365.38, 940.69]
    >>> p_c = [452.35, 329.06, 524.77]
    >>> delta = 59.5
    >>> find_joint_center(p_a, p_b, p_c, delta).round(2)
    array([396.25, 347.92, 518.63])
    """

    # make the two vector using 3 markers, which is on the same plane.
    vec_1 = (p_a[0]-p_c[0], p_a[1]-p_c[1], p_a[2]-p_c[2])
    vec_2 = (p_b[0]-p_c[0], p_b[1]-p_c[1], p_b[2]-p_c[2])

    # vec_3 is cross vector of vec_1, vec_2, and then it normalized.
    vec_3 = np.cross(vec_1, vec_2)
    vec_3_div = np.linalg.norm(vec_3)
    vec_3 = [vec_3[0]/vec_3_div, vec_3[1]/vec_3_div, vec_3[2]/vec_3_div]

    mid = [(p_b[0]+p_c[0])/2.0, (p_b[1]+p_c[1])/2.0, (p_b[2]+p_c[2])/2.0]
    length = np.subtract(p_b, mid)
    length = np.linalg.norm(length)

    theta = math.acos(delta/np.linalg.norm(vec_2))

    cs_th = math.cos(theta*2)
    sn_th = math.sin(theta*2)

    u_x, u_y, u_z = vec_3

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

def hand_axis(rwra, rwrb, lwra, lwrb, rfin, lfin, wrist_jc, r_hand_thickness, l_hand_thickness):
    r"""Calculate the Hand joint axis.

    Takes in markers that correspond to (x, y, z) positions of the current
    frame as well as the wrist joint center.

    Calculates each hand joint axis and returns it.

    Markers used: RWRA, RWRB, LWRA, LWRB, RFIN, LFIN

    Subject Measurement values used: RightHandThickness, LeftHandThickness

    Parameters
    ----------
    rwra : array
        1x3 RWRA marker
    rwrb : array
        1x3 RWRB marker
    lwra : array
        1x3 LWRA marker
    lwrb : array
        1x3 LWRB marker
    rfin : array
        1x3 RFIN marker
    lfin : array
        1x3 LFIN marker
    wrist_jc : array
        The x,y,z position of the wrist joint center.
    r_hand_thickness : float
        The thickness of the right hand.
    l_hand_thickness : float
        The thickness of the left hand.

    Returns
    -------
    [r_axis, l_axis] : array
        A list of two 4x4 affine matrices representing the right hand axis as well as the
        left hand axis.
    
    Notes
    -----
    :math:`r_{delta} = (\frac{r\_hand\_thickness}{2.0} + 7.0) \hspace{1cm} l_{delta} = (\frac{l\_hand\_thickness}{2.0} + 7.0)`

    :math:`\textbf{m}_{RHND} = JC(\textbf{m}_{RWRI}, \textbf{m}_{RWJC}, \textbf{m}_{RFIN}, r_{delta})`

    :math:`\textbf{m}_{LHND} = JC(\textbf{m}_{LWRI}, \textbf{m}_{LWJC}, \textbf{m}_{LFIN}, r_{delta})`

    .. math::

        \begin{matrix}
            o_{L} = \frac{\textbf{m}_{LWRA} + \textbf{m}_{LWRB}}{2} & o_{R} = \frac{\textbf{m}_{RWRA} + \textbf{m}_{RWRB}}{2} \\
            \hat{z}_{L} = \textbf{m}_{LWJC} - \textbf{m}_{LHND} & \hat{z}_{R} = \textbf{m}_{RWJC} - \textbf{m}_{RHND} \\
            \hat{y}_{L} = \textbf{m}_{LWRI} - \textbf{m}_{LWRA} & \hat{y}_{R} = \textbf{m}_{RWRA} - \textbf{m}_{RWRI} \\
            \hat{x}_{L} = \hat{y}_{L} \times \hat{z}_{L} & \hat{x}_{R} = \hat{y}_{R} \times \hat{z}_{R} \\
            \hat{y}_{L} = \hat{z}_{L} \times \hat{x}_{L} & \hat{y}_{R} = \hat{z}_{R} \times \hat{x}_{R}
        \end{matrix}

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import hand_axis
    >>> np.set_printoptions(suppress=True)
    >>> rwra = np.array([776.51, 495.68, 1108.38])
    >>> rwrb = np.array([830.90, 436.75, 1119.11])
    >>> lwra = np.array([-249.28, 525.32, 1117.09])
    >>> lwrb = np.array([-311.77, 477.22, 1125.16])
    >>> rfin = np.array([863.71, 524.44, 1074.54])
    >>> lfin = np.array([-326.65, 558.34, 1091.04])
    >>> wrist_jc = [[
    ... [793.77, 450.44, 1084.12, 793.32],
    ... [794.01, 451.38, 1085.15, 451.29],
    ... [792.75, 450.76, 1085.05, 1084.43],
    ... [0, 0, 0, 1]
    ... ], [
    ... [-272.92, 485.01, 1090.96, -272.45],
    ... [-271.74, 485.72, 1090.67, 485.8],
    ... [-271.94, 485.19, 1091.96, 1091.36],
    ... [0, 0, 0, 1]
    ... ]]
    >>> r_hand_thickness = 34.0
    >>> l_hand_thickness = 34.0
    >>> [np.around(arr, 2) for arr in hand_axis(
    ...     rwra, rwrb, lwra, lwrb, rfin, lfin, wrist_jc, r_hand_thickness, l_hand_thickness)] #doctest: +NORMALIZE_WHITESPACE
    [array([[   0.15,    0.31,    0.94,  859.8 ],
        [  -0.73,    0.68,   -0.11,  517.27],
        [  -0.67,   -0.67,    0.33, 1051.97],
        [   0.  ,    0.  ,    0.  ,    1.  ]]), array([[  -0.09,    0.27,    0.96, -324.52],
        [  -0.8 ,   -0.59,    0.1 ,  551.89],
        [   0.6 ,   -0.76,    0.27, 1068.02],
        [   0.  ,    0.  ,    0.  ,    1.  ]])]
    """

    rwri = [(rwra[0]+rwrb[0])/2.0, (rwra[1]+rwrb[1])/2.0, (rwra[2]+rwrb[2])/2.0]
    lwri = [(lwra[0]+lwrb[0])/2.0, (lwra[1]+lwrb[1])/2.0, (lwra[2]+lwrb[2])/2.0]

    rwjc = [wrist_jc[0][0][-1], wrist_jc[0][1][-1], wrist_jc[0][2][-1]]
    lwjc = [wrist_jc[1][0][-1], wrist_jc[1][1][-1], wrist_jc[1][2][-1]]

    mm = 7.0

    r_delta =( r_hand_thickness/2.0 + mm )
    l_delta =( l_hand_thickness/2.0 + mm )

    lhnd = find_joint_center(lwri,lwjc,lfin,l_delta)
    rhnd = find_joint_center(rwri,rwjc,rfin,r_delta)

    # Left
    z_axis = [lwjc[0]-lhnd[0],lwjc[1]-lhnd[1],lwjc[2]-lhnd[2]]
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div, z_axis[1]/z_axis_div, z_axis[2]/z_axis_div]

    y_axis = [lwri[0]-lwra[0],lwri[1]-lwra[1],lwri[2]-lwra[2]]
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div, y_axis[1]/y_axis_div, y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis,z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div, x_axis[1]/x_axis_div, x_axis[2]/x_axis_div]

    y_axis = np.cross(z_axis,x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div, y_axis[1]/y_axis_div, y_axis[2]/y_axis_div]

    l_axis = np.zeros((4, 4))
    l_axis[3, 3] = 1.0
    l_axis[0, :3] = x_axis
    l_axis[1, :3] = y_axis
    l_axis[2, :3] = z_axis
    l_axis[:3, 3] = lhnd

    # Right
    z_axis = [rwjc[0]-rhnd[0],rwjc[1]-rhnd[1],rwjc[2]-rhnd[2]]
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div, z_axis[1]/z_axis_div, z_axis[2]/z_axis_div]

    y_axis = [rwra[0]-rwri[0],rwra[1]-rwri[1],rwra[2]-rwri[2]]
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div, y_axis[1]/y_axis_div, y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis,z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div, x_axis[1]/x_axis_div, x_axis[2]/x_axis_div]

    y_axis = np.cross(z_axis,x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div, y_axis[1]/y_axis_div, y_axis[2]/y_axis_div]

    r_axis = np.zeros((4, 4))
    r_axis[3, 3] = 1.0
    r_axis[0, :3] = x_axis
    r_axis[1, :3] = y_axis
    r_axis[2, :3] = z_axis
    r_axis[:3, 3] = rhnd

    return [r_axis, l_axis]
