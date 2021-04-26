# -*- coding: utf-8 -*-
import numpy as np


def thorax_axis(clav, c7, strn, t10):
    r"""Make the Thorax Axis.

    Takes in CLAV, C7, STRN, T10 markers.
    Calculates the thorax axis.

    :math:`upper = (\textbf{m}_{clav} + \textbf{m}_{c7}) / 2.0`

    :math:`lower = (\textbf{m}_{strn} + \textbf{m}_{t10}) / 2.0`

    :math:`\emph{front} = (\textbf{m}_{clav} + \textbf{m}_{strn}) / 2.0`

    :math:`back = (\textbf{m}_{t10} + \textbf{m}_{c7}) / 2.0`

    :math:`\hat{z} = \frac{lower - upper}{||lower - upper||}`

    :math:`\hat{x} = \frac{\emph{front} - back}{||\emph{front} - back||}`

    :math:`\hat{y} = \frac{ \hat{z} \times \hat{x} }{||\hat{z} \times \hat{x}||}`

    :math:`\hat{z} = \frac{\hat{x} \times \hat{y} }{||\hat{x} \times \hat{y} ||}`

    Parameters
    ----------
    clav: array
        1x3 CLAV marker
    c7: array
        1x3 C7 marker
    strn: array
        1x3 STRN marker
    t10: array
        1x3 T10 marker

    Returns
    -------
    thorax : array
        4x4 affine matrix with thorax x, y, z axes and thorax origin.

    .. math::

        \begin{bmatrix}
            \hat{x}_x & \hat{x}_y & \hat{x}_z & o_x \\
            \hat{y}_x & \hat{y}_y & \hat{y}_z & o_y \\
            \hat{z}_x & \hat{z}_y & \hat{z}_z & o_z \\
            0 & 0 & 0 & 1 \\
        \end{bmatrix}

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import thorax_axis
    >>> c7 = np.array([256.78, 371.28, 1459.70])
    >>> t10 = np.array([228.64, 192.32, 1279.64])
    >>> clav = np.array([256.78, 371.28, 1459.70])
    >>> strn = np.array([251.67, 414.10, 1292.08])
    >>> thorax_axis(clav, c7, strn, t10) #doctest: +NORMALIZE_WHITESPACE
    array([[ 7.24943643e-02,  9.26336467e-01, -3.69655673e-01, 2.56272539e+02],
    [ 9.93416818e-01, -1.00026023e-01, -5.58374552e-02, 3.64795645e+02],
    [-8.86994578e-02, -3.63174261e-01, -9.27489548e-01, 1.46228759e+03],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
    """
    clav, c7, strn, t10 = map(np.asarray, [clav, c7, strn, t10])

    # Set or get a marker size as mm
    marker_size = (14.0) / 2.0

    # Get the midpoints of the upper and lower sections, as well as the front and back sections
    upper = (clav + c7)/2.0
    lower = (strn + t10)/2.0
    front = (clav + strn)/2.0
    back = (t10 + c7)/2.0

    # Get the direction of the primary axis Z (facing down)
    z_direc = lower - upper
    z = z_direc/np.linalg.norm(z_direc)

    # The secondary axis X is from back to front
    x_direc = front - back
    x = x_direc/np.linalg.norm(x_direc)

    # make sure all the axes are orthogonal to each other by cross-product
    y_direc = np.cross(z, x)
    y = y_direc/np.linalg.norm(y_direc)
    x_direc = np.cross(y, z)
    x = x_direc/np.linalg.norm(x_direc)
    z_direc = np.cross(x, y)
    z = z_direc/np.linalg.norm(z_direc)

    # move the axes about offset along the x axis.
    offset = x * marker_size

    # Add the CLAV back to the vector to get it in the right position before translating it
    o = clav - offset

    thorax = np.zeros((4, 4))
    thorax[3, 3] = 1.0
    thorax[0, :3] = x
    thorax[1, :3] = y
    thorax[2, :3] = z
    thorax[:3, 3] = o

    return thorax
