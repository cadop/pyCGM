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
    :math:`vec_{1} = (p\_a-p\_c), \ vec_{2} = (p\_b-p\_c)`

    :math:`vec_{3} = vec_{1} \times vec_{2}`

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
