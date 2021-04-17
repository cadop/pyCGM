import pycgm.axis as axis
import pytest
import numpy as np

rounding_precision = 8

class TestUtils():
    """
    This class tests the utils functions in axis.py:
    findwandmarker
    cross
    norm2d
    norm3d
    norm_div
    matrixmult
    rotmat
    """
    @pytest.mark.parametrize(["vec_a", "vec_b", "expected"], [
            ([0.13232936, 0.98562946, -0.10499292], [-0.99119134, 0.13101088, -0.01938735], [-0.005353527183234709, 0.10663358915485248, 0.994283972218527]),
            ([0, 0, 0], [0, 0, 0], [0, 0, 0]),
            ([1, 1, 1], [1, 1, 1], [0, 0, 0]),
            ([0, 0, -2], [0, 4, 0], [8, 0, 0]),
            ([0, 0, 4], [-0.5, 0, 0], [0, -2, 0]),
            ([-1.5, 0, 0], [0, 4, 0], [0, 0, -6]),
            ([1, 0, 1], [0, 1, 0], [-1, 0, 1]),
            ([1, 2, 3], [3, 2, 1], [-4, 8, -4]),
            ([-2, 3, 1], [4, -1, 5], [16, 14, -10])
        ])
    def test_cross(self, vec_a, vec_b, expected):
        """
        This test provides coverage of the cross function in axis.py, defined as cross(vec_a, vec_b) where vec_a and vec_b are both 3D vectors.

        This test takes 3 parameters:
        vec_a: 3D vector
        vec_b: 3D vector
        expected: the expected result from calling cross on vec_a and vec_b. This result is the cross product of the vectors
        vec_a and vec_b.
        """
        result = axis.cross(vec_a, vec_b)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_cross_datatypes(self):
        """
        This test provides coverage of the cross function in axis.py, defined as cross(vec_a, vec_b) where vec_a and vec_b are both 3D vectors.

        This test checks that the resulting output from calling cross is correct when called with a list of ints, a numpy
        array of ints, a list of floats, and a numpy array of floats.
        """
        a_int = [-2, 3, 1]
        a_float = [-2.0, 3.0, 1.0]
        b_int = [4, -1, 5]
        b_float = [4.0, -1.0, 5.0]
        expected = [16, 14, -10]

        # Check the calling cross on a list of ints yields the expected results
        result_int_list = axis.cross(a_int, b_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling cross on a numpy array of ints yields the expected results
        result_int_nparray = axis.cross(np.array(a_int, dtype='int'), np.array(b_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling cross on a list of floats yields the expected results
        result_float_list = axis.cross(a_float, b_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling cross on a numpy array of floats yields the expected results
        result_float_nparray = axis.cross(np.array(a_float, dtype='float'), np.array(b_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["vec", "expected"], [
        ([-212.5847168, 28.09841919, -4.15808105], np.array([-4.62150006e-03,  6.10847515e-04, -9.03948887e-05])),
        ([0, 0, 0], np.array([np.nan, np.nan, np.nan])),
        ([2, 0, 0], np.array([0.5, 0, 0])),
        ([0, 0, -1], np.array([0, 0, -1])),
        ([0, 3, 4], np.array([0, 0.12, 0.16])),
        ([-3, 0, 4], np.array([-0.12, 0, 0.16])),
        ([-6, 8, 0], np.array([-0.06, 0.08, 0])),
        ([-5, 0, -12], np.array([-0.0295858, 0, -0.07100592])),
        ([1, -1, np.sqrt(2)], np.array([0.25, -0.25, 0.35355339]))])
    def test_norm_div(self, vec, expected):
        """
        This test provides coverage of the norm_div function in axis.py, defined as norm_div(vec) where vec is a 3D vector.

        This test takes 2 parameters:
        vec: 3D vector
        expected: the expected result from calling norm3d on vec. This function returns the wrong result. It is supposed
        to return the normalization division, but in the function divides the vector by the normalization twice.

        Given the vector vec, the normalization is defined by:
        normalization = :math:`\sqrt{vec_x^2 + vec_y^2 + vec_z^2}`
        where :math:`vec_x     is the x-coordinate of the vector vec

        The mathematically correct result would be defined by:
        .. math::
            \[ result = [\frac{vec_x}{norm}, \frac{vec_y}{norm}, \frac{vec_z}{norm}] \]

        But this function has an error where it divides the vector twice:
        .. math::
            \[ result = [\frac{vec_x}{norm^2}, \frac{vec_y}{norm^2}, \frac{vec_z}{norm^2}] \]
        """
        result = axis.norm_div(vec)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_norm_div_datatypes(self):
        """
        This test provides coverage of the norm_div function in axis.py, defined as norm_div(vec) where vec is a 3D vector.

        This test checks that the resulting output from calling norm_div is correct when called with a list of ints, a
        numpy array of ints, a list of floats, and a numpy array of floats.
        """
        v_int = [-6, 0, 8]
        v_float = [-6.0, 0, 8.0]
        expected = np.array([-0.06, 0, 0.08])

        # Check the calling norm_div on a list of ints yields the expected results
        result_int_list = axis.norm_div(v_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling norm_div on a numpy array of ints yields the expected results
        result_int_nparray = axis.norm_div(np.array(v_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling norm_div on a list of floats yields the expected results
        result_float_list = axis.norm_div(v_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling norm_div on a numpy array of floats yields the expected results
        result_float_nparray = axis.norm_div(np.array(v_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["matr_a", "matr_b", "expected"], [
        ([[1, 0, 0], [0, 1.0, -0.0], [0, 0.0, 1.0]], [[1.0, 0, 0.0], [0, 1, 0], [-0.0, 0, 1.0]], [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ([[1]], [[1]], [[1]]),
        # Invalid matrix dimensions
        ([[1, 2]], [[1]], [[1]]),
        ([[2], [1]], [[1, 2]], [[2, 4], [1, 2]]),
        # Invalid matrix dimensions
        ([[1, 2, 0], [0, 1, 2]], [[2, 1], [1, 4]], [[4, 9], [1, 4]]),
        ([[11, 12, 13], [14, 15, 16]], [[1, 2], [3, 4], [5, 6]], [[112, 148], [139, 184]]),
        ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]], [[58, 64], [139, 154]])])
    def test_matrixmult(self, matr_a, matr_b, expected):
        """
        This test provides coverage of the matrixmult function in axis.py, defined as matrixmult(matr_a, matr_b)
        where matr_a and matr_b are both lists that represent a matrix to be multiplied.

        This test takes 3 parameters:
        matr_a: a matrix, 2D array format
        matr_b: a matrix, 2D array format
        expected: the expected matrix from calling matrixmult on matr_a and matr_b. This is the result of multiplying the two
        matrices matr_a and matr_b. It gives the correct result for multiplying two valid matrices, but still gives a result
        in some cases when the two matrices can't be multiplied. For two matrices to be multiplied, len(matr_a[0]) need to
        be equal to len(matr_b), but this function gives an output even when this isn't true
        """
        result = axis.matrixmult(matr_a, matr_b)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_matrixmult_datatypes(self):
        """
        This test provides coverage of the matrixmult function in axis.py, defined as matrixmult(matr_a, matr_b)
        where matr_a and matr_b are both lists that represent a matrix to be multiplied.

        This test checks that the resulting output from calling matrixmult is correct when called with a list of ints,
        a numpy array of ints, a list of floats, and a numpy array of floats.
        """
        A_int = [[1, 2, 0], [0, 1, 2]]
        B_int = [[2, 1], [1, 4]]
        A_float = [[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]]
        B_float = [[2.0, 1.0], [1.0, 4.0]]
        expected = [[4, 9], [1, 4]]

        # Check the calling matrixmult on a list of ints yields the expected results
        result_int_list = axis.matrixmult(A_int, B_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling matrixmult on a numpy array of ints yields the expected results
        result_int_nparray = axis.matrixmult(np.array(A_int, dtype='int'), np.array(B_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling matrixmult on a list of floats yields the expected results
        result_float_list = axis.matrixmult(A_float, B_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling matrixmult on a numpy array of floats yields the expected results
        result_float_nparray = axis.matrixmult(np.array(A_float, dtype='float'), np.array(B_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["x", "y", "z", "expected"], [
        (0.0, 0.0, 180, [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        (0, 0, 0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        (90, 0, 0, [[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        (0, 135, 0, [[-0.70710678, 0, 0.70710678], [0, 1, 0], [-0.70710678, 0, -0.70710678]]),
        (0, 0, -60, [[0.5, 0.8660254, 0], [-0.8660254, 0.5, 0], [0, 0, 1]]),
        (90, 0, 90, [[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
        (0, 150, -30, [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]),
        (90, 180, -90, [[0, -1, 0], [0, 0, 1], [-1, 0, 0]])])
    def test_rotmat(self, x, y, z, expected):
        """
        This test provides coverage of the rotmat function in axis.py, defined as rotmat(x, y, z)
        where x, y, and z are all floats that represent the angle of rotation in a particular dimension.

        This test takes 4 parameters:
        x: angle to be rotated in the x axis
        y: angle to be rotated in the y axis
        z: angle to be rotated in the z axis
        expected: the expected rotation matrix from calling rotmat on x, y, and z. This will be a transformation
        matrix that can be used to perform a rotation in the x, y, and z directions at the values inputted.
        """
        result = axis.rotmat(x, y, z)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_rotmat_datatypes(self):
        """
        This test provides coverage of the rotmat function in axis.py, defined as rotmat(x, y, z)
        where x, y, and z are all floats that represent the angle of rotation in a particular dimension.

        This test checks that the resulting output from calling rotmat is correct when called with ints or floats.
        """
        result_int = axis.rotmat(0, 150, -30)
        result_float = axis.rotmat(0.0, 150.0, -30.0)
        expected = [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]

        np.testing.assert_almost_equal(result_int, expected, rounding_precision)
        np.testing.assert_almost_equal(result_float, expected, rounding_precision)