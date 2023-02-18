import math

import numpy as np 


class CalcUtils:
    @staticmethod
    def rotmat(x=0, y=0, z=0):
        r"""Rotation Matrix.

        This function creates and returns a rotation matrix.

        Parameters
        ----------
        x, y, z : float, optional
            Angle, which will be converted to radians, in
            each respective axis to describe the rotations.
            The default is 0 for each unspecified angle.

        Returns
        -------
        r_xyz : list
            The product of the matrix multiplication.

        Notes
        -----
        :math:`r_x = [ [1,0,0], [0, \cos(x), -sin(x)], [0, sin(x), cos(x)] ]`
        :math:`r_y = [ [cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)] ]`
        :math:`r_z = [ [cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1] ]`
        :math:`r_{xy} = r_x * r_y`
        :math:`r_{xyz} = r_{xy} * r_z`

        Examples
        --------
        >>> import numpy as np
        >>> from .dynamic import CalcUtils
        >>> x = 0.5
        >>> y = 0.3
        >>> z = 0.8
        >>> np.around(CalcUtils.rotmat(x, y, z), 2) #doctest: +NORMALIZE_WHITESPACE
        array([[ 1.  , -0.01,  0.01],
        [ 0.01,  1.  , -0.01],
        [-0.01,  0.01,  1.  ]])
        >>> x = 0.5
        >>> np.around(CalcUtils.rotmat(x), 2) #doctest: +NORMALIZE_WHITESPACE
        array([[ 1.  ,  0.  ,  0.  ],
        [ 0.  ,  1.  , -0.01],
        [ 0.  ,  0.01,  1.  ]])
        >>> x = 1
        >>> y = 1
        >>> np.around(CalcUtils.rotmat(x,y), 2) #doctest: +NORMALIZE_WHITESPACE
        array([[ 1.  ,  0.  ,  0.02],
        [ 0.  ,  1.  , -0.02],
        [-0.02,  0.02,  1.  ]])
        """
        x, y, z = math.radians(x), math.radians(y), math.radians(z)
        r_x = [[1, 0, 0], [0, math.cos(x), math.sin(
            x)*-1], [0, math.sin(x), math.cos(x)]]
        r_y = [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [
            math.sin(y)*-1, 0, math.cos(y)]]
        r_z = [[math.cos(z), math.sin(z)*-1, 0],
               [math.sin(z), math.cos(z), 0], [0, 0, 1]]
        r_xy = np.matmul(r_x, r_y)
        r_xyz = np.matmul(r_xy, r_z)

        return r_xyz

    @staticmethod
    def calc_joint_center(p_a, p_b, p_c, delta):
        r"""Calculate the Joint Center.

        This function is based on the physical markers p_a, p_b, p_c
        and the resulting joint center are all on the same plane.

        Parameters
        ----------
        p_a : array
            (x, y, z) position of marker a
        p_b : array 
            (x, y, z) position of marker b
        p_c : array
            (x, y, z) position of marker c
        delta : float
            The length from marker to joint center, retrieved from subject measurement file

        Returns
        -------
        joint_center : array
            (x, y, z) position of the joint center

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
        >>> from .pyCGM import calc_joint_center
        >>> p_a = np.array([468.14, 325.09, 673.12])
        >>> p_b = np.array([355.90, 365.38, 940.69])
        >>> p_c = np.array([452.35, 329.06, 524.77])
        >>> delta = 59.5
        >>> calc_joint_center(p_a, p_b, p_c, delta).round(2)
        array([396.25, 347.92, 518.63])
        """

        # make the two vector using 3 markers, which is on the same plane.

        p_a, p_b, p_c = map(np.asarray, [p_a, p_b, p_c])

        vec_1 = p_a - p_c
        vec_2 = p_b - p_c

        # vec_3 is cross vector of vec_1, vec_2, and then it normalized.
        vec_3 = np.cross(vec_1, vec_2)
        vec_3_div = np.linalg.norm(vec_3, axis=1)[:, np.newaxis]
        vec_3 = vec_3 / vec_3_div

        mid = (p_b + p_c) / 2.0
        length = np.subtract(p_b, mid)
        length = np.linalg.norm(length, axis=1)

        theta = np.arccos(delta/np.linalg.norm(vec_2, axis=1))

        alpha = np.cos(theta*2)
        beta = np.sin(theta*2)

        u_x, u_y, u_z = (vec_3[:, 0], vec_3[:, 1], vec_3[:, 2])

        # This rotation matrix is called Rodriques' rotation formula.
        # In order to make a plane, at least 3 number of markers is required which
        # means three physical markers on the segment can make a plane.
        # then the orthogonal vector of the plane will be rotating axis.
        # joint center is determined by rotating the one vector of plane around rotating axis.

        rot = np.array([ 
            [alpha+u_x**2.0*(1.0-alpha),   u_x*u_y*(1.0-alpha) - u_z*beta, u_x*u_z*(1.0-alpha)+u_y*beta],
            [u_y*u_x*(1.0-alpha)+u_z*beta, alpha+u_y**2.0 * (1.0-alpha),   u_y*u_z*(1.0-alpha)-u_x*beta],
            [u_z*u_x*(1.0-alpha)-u_y*beta, u_z*u_y*(1.0-alpha) + u_x*beta, alpha+u_z**2.0*(1.0-alpha)]
        ]).transpose(2, 0, 1)

        num_frames = vec_2.shape[0]
        vec_2 = vec_2.reshape(num_frames, 3, 1)
        r_vec = rot @ vec_2
        r_vec = r_vec * length[:, np.newaxis, np.newaxis] / np.linalg.norm(r_vec, axis=1)[:, np.newaxis]

        r_vec = np.squeeze(r_vec, axis=2)
        joint_center = r_vec + mid

        return joint_center
