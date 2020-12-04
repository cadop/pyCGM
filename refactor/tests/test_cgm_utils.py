import numpy as np
import pytest

from refactor.pycgm import CGM

rounding_precision = 8


class TestCGMUtils():
    """
    This class tests the utils functions in the class CGM in pycgm.py:
    rotation_matrix
    """

    @pytest.mark.parametrize(["x", "y", "z", "expected"], [
        (0.0, 0.0, 180, [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        (0, 0, 0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        (90, 0, 0, [[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        (0, 135, 0, [[-0.70710678, 0, 0.70710678], [0, 1, 0], [-0.70710678, 0, -0.70710678]]),
        (0, 0, -60, [[0.5, 0.8660254, 0], [-0.8660254, 0.5, 0], [0, 0, 1]]),
        (90, 0, 90, [[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
        (0, 150, -30, [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]),
        (90, 180, -90, [[0, -1, 0], [0, 0, 1], [-1, 0, 0]])])
    def test_rotation_matrix(self, x, y, z, expected):
        """
        This test provides coverage of the rotation_matrix function in the class CGM in pycgm.py, defined as
        rotation_matrix(x, y, z)

        This test takes 4 parameters:
        x, y, z : float, optional
            Angle, which will be converted to radians, in each respective axis to describe the rotations.
            The default is 0 for each unspecified angle.
        expected : array
            A 3x3 ndarray which can bbe used to perform a rotation about the x, y, z axes.
        """
        result = CGM.rotation_matrix(x, y, z)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_rotation_matrix_datatypes(self):
        """
        This test provides coverage of the rotation_matrix function in the class CGM in pycgm.py, defined as
        rotation_matrix(x, y, z)

        This test checks that the resulting output from calling rotation_matrix is correct when called with ints or
        floats.
        """
        result_int = CGM.rotation_matrix(0, 150, -30)
        result_float = CGM.rotation_matrix(0.0, 150.0, -30.0)
        expected = [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]

        np.testing.assert_almost_equal(result_int, expected, rounding_precision)
        np.testing.assert_almost_equal(result_float, expected, rounding_precision)
