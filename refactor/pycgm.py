import numpy as np


class CGM:

    def __init__(self, path_static, path_dynamic, path_measurements, path_ressults=None, path_com=None, cores=1):
        pass

    def run(self):
        """Execute the CGM calculations function

        Load in appropriate data from IO using paths.
        Perform any necessary prep on data.
        Run the static calibration trial.
        Run the dynamic trial to calculate all axes and angles.
        """
        pass

    def map(self, old, new):
        """Remap marker function

        Remaps a single marker from the expected name in CGM to a new one, using `old` and `new`.

        Parameters
        ----------
        old : str
            String containing the marker name that pycgm currently expects.
        new : str
            String containing the marker name to map `old` to.
        """
        pass

    def full_map(self, mapping):
        """Remap all markers function

        Uses the passed dictionary as the mapping for all markers.

        Parameters
        ----------
        mapping: dict
            Dictionary where each key is a string of pycgm's expected marker
            name and each value is a string of the new marker name.
        """
        pass

    @staticmethod
    def cross(a, b):
        """Cross Product function

        Given vectors a and b, calculate the cross product.

        Parameters
        ----------
        a : list
            First 3D vector.
        b : list
            Second 3D vector.

        Returns
        -------
        c : list
            The cross product of vector a and vector b.

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm import CGM
        >>> a = [6.25286248, 7.91367254, 18.63620527]
        >>> b = [3.49290439, 4.42038315, 19.23948238]
        >>> np.around(CGM.cross(a, b),8)
        array([ 6.98757956e+01, -5.52073543e+01, -1.65361000e-03])
        """
        pass

    @staticmethod
    def norm2d(v):
        """2D Vector normalization function

        This function calculates the normalization of a 3-dimensional vector.

        Parameters
        ----------
        v : list
            A 3D vector.

        Returns
        -------
        float
            The normalization of the vector as a float.

        Examples
        --------
        >>> import numpy as np 
        >>> from .pycgm import CGM
        >>> v = [105.141121037153, 101.890788777524, 326.7710280245359]
        >>> np.around(CGM.norm2d(v),8)
        358.07218955
        """
        pass

    @staticmethod
    def norm3d(v):
        """3D Vector normalization function

        This function calculates the normalization of a 3-dimensional vector.

        Parameters
        ----------
        v : list
            A 3D vector.

        Returns
        -------
        list
            The normalization of the vector returned as a float in an array.

        Examples
        --------
        >>> from .pycgm import CGM
        >>> v = [125.44928201, 143.94301493, 213.49204956]
        >>> CGM.norm3d(v)
        array(286.4192192)
        """
        pass

    @staticmethod
    def matrix_mult(a, b):
        """Matrix multiplication function

        This function returns the product of a matrix multiplication given two matrices.

        Let the dimension of the matrix A be: m by n,
        let the dimension of the matrix B be: p by q,
        multiplication will only possible if n = p,
        creating a matrix of m by q size.

        Parameters
        ----------
        a : list
            First matrix, in a 2D array format.
        b : list
            Second matrix, in a 2D array format.

        Returns
        -------
        c : list
            The product of the matrix multiplication.

        Examples
        --------
        >>> from .pycgm import CGM
        >>> A = [[11,12,13],[14,15,16]]
        >>> B = [[1,2],[3,4],[5,6]]
        >>> CGM.matrix_mult(A, B)
        [[112, 148], [139, 184]]
        """
        pass

    @staticmethod
    def rotation_matrix(x=0, y=0, z=0):
        """Rotation Matrix function

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

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm import CGM
        >>> x = 0.5
        >>> y = 0.3
        >>> z = 0.8
        >>> np.around(CGM.rotation_matrix(x, y, z), 8)
        array([[ 0.99988882, -0.01396199,  0.00523596],
               [ 0.01400734,  0.99986381, -0.00872642],
               [-0.00511341,  0.00879879,  0.99994822]])
        >>> x = 0.5
        >>> np.around(CGM.rotation_matrix(x), 8)
        array([[ 1.        ,  0.        ,  0.        ],
               [ 0.        ,  0.99996192, -0.00872654],
               [ 0.        ,  0.00872654,  0.99996192]])
        >>> x = 1
        >>> y = 1
        >>> np.around(CGM.rotation_matrix(x, y), 8)
        array([[ 9.9984770e-01,  0.0000000e+00,  1.7452410e-02],
               [ 3.0459000e-04,  9.9984770e-01, -1.7449750e-02],
               [-1.7449750e-02,  1.7452410e-02,  9.9969541e-01]])
        """
        pass

    @staticmethod
    def pelvis_axis_calc():
        pass

    @staticmethod
    def hip_axis_calc():
        pass

    @staticmethod
    def knee_axis_calc():
        pass

    @staticmethod
    def ankle_axis_calc():
        pass

    @staticmethod
    def foot_axis_calc():
        pass

    @staticmethod
    def head_axis_calc():
        pass

    @staticmethod
    def thorax_axis_calc():
        pass

    @staticmethod
    def neck_axis_calc():
        pass

    @staticmethod
    def shoulder_axis_calc():
        pass

    @staticmethod
    def elbow_axis_calc():
        pass

    @staticmethod
    def wrist_axis_calc():
        pass

    @staticmethod
    def pelvis_angle_calc():
        pass

    @staticmethod
    def hip_angle_calc():
        pass

    @staticmethod
    def knee_angle_calc():
        pass

    @staticmethod
    def ankle_angle_calc():
        pass

    @staticmethod
    def foot_angle_calc():
        pass

    @staticmethod
    def head_angle_calc():
        pass

    @staticmethod
    def thorax_angle_calc():
        pass

    @staticmethod
    def neck_angle_calc():
        pass

    @staticmethod
    def shoulder_angle_calc():
        pass

    @staticmethod
    def elbow_angle_calc():
        pass

    @staticmethod
    def wrist_angle_calc():
        pass

    @staticmethod
    def multi_calc():
        pass

    @staticmethod
    def calc():
        pass
