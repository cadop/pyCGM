import pyCGM_Single.pyCGM as pyCGM
import pytest
import numpy as np

rounding_precision = 8

class TestUtils():
    """
    This class tests the utils functions in pyCGM.py:
    findwandmarker
    cross
    norm2d
    norm3d
    normDiv
    matrixmult
    rotmat
    """
    rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]

    @pytest.mark.parametrize(["frame", "thorax", "expected"], [
        ({'RSHO': [428.88476562, 270.552948, 1500.73010254], 'LSHO': [68.24668121, 269.01049805, 1510.1072998]}, [[[256.23991128535846, 365.30496976939753, 1459.662169500559], rand_coor, rand_coor], [256.149810236564, 364.3090603933987, 1459.6553639290375]], [[255.92550222678443, 364.3226950497605, 1460.6297868417887], [256.42380097331767, 364.27770361353487, 1460.6165849382387]]),
        ({'RSHO': [0, 0, 1], 'LSHO': [0, 1, 0]}, [[[1, 0, 0], rand_coor, rand_coor], [0, 0, 0]], [[0, 1, 0], [0, 0, 1]]),
        ({'RSHO': [0, 1, 1], 'LSHO': [1, 1, 1]}, [[[1, 0, 0], rand_coor, rand_coor], [0, 0, 0]], [[0, 0.70710678, -0.70710678], [0, -0.70710678, 0.70710678]]),
        ({'RSHO': [0, 1, 1], 'LSHO': [1, 1, 1]}, [[[1, 0, 0], rand_coor, rand_coor], [-1, 0, 0]], [[-1, 0.70710678, -0.70710678], [-1, -0.70710678, 0.70710678]]),
        ({'RSHO': [1, 2, 1], 'LSHO': [2, 1, 2]}, [[[1, 0, 0], rand_coor, rand_coor], [0, 0, 0]], [[0, 0.4472136, -0.89442719], [0, -0.89442719, 0.4472136]]),
        ({'RSHO': [1, 2, 1], 'LSHO': [2, 2, 2]}, [[[1, 0, 0], rand_coor, rand_coor], [0, 0, 0]], [[0, 0.4472136, -0.89442719], [0, -0.70710678, 0.70710678]]),
        ({'RSHO': [1, 2, 2], 'LSHO': [2, 1, 2]}, [[[1, 0, 0], rand_coor, rand_coor], [0, 0, 0]], [[0, 0.70710678, -0.70710678], [0, -0.89442719, 0.4472136]]),
        ({'RSHO': [1, 1, 1], 'LSHO': [1, 1, 1]}, [[[1, 0, 1], rand_coor, rand_coor], [0, 0, 0]], [[0.70710678, 0, -0.70710678], [-0.70710678, 0, 0.70710678]]),
        ({'RSHO': [1, 1, 1], 'LSHO': [1, 1, 1]}, [[[1, 0, 1], rand_coor, rand_coor], [0, 0, 1]], [[0, 0, 0], [0, 0, 2]]),
        ({'RSHO': [0, 1, 0], 'LSHO': [0, 0, -1]}, [[[0, 3, 4], rand_coor, rand_coor], [0, 0, 0]], [[1, 0, 0], [-1, 0, 0]]),
        ({'RSHO': [1, 0, 0], 'LSHO': [0, 1, 0]}, [[[7, 0, 24], rand_coor, rand_coor], [0, 0, 0]], [[0, -1, 0], [-0.96, 0, 0.28]]),
        ({'RSHO': [1, 0, 0], 'LSHO': [0, 0, 1]}, [[[8, 0, 6], rand_coor, rand_coor], [8, 0, 0]], [[8, 1, 0], [8, -1, 0]])])
    def test_findwandmarker(self, frame, thorax, expected):
        """
        This test provides coverage of the findwandmarker function in pyCGM.py, defined as findwandmarker(frame,thorax)
        where frame is a dictionary of x, y, z positions and marker names and thorax is the thorax axis and origin.

        The function takes in the xyz position of the Right Shoulder and Left Shoulder markers, as well as the thorax
        frame, which is a list of [ xyz axis vectors, origin ]. The wand marker position is returned as a 2x3 array
        containing the right wand marker x, y, z positions (1x3) followed by the left wand marker x, y, z positions
        (1x3). The thorax axis is provided in global coordinates, which are subtracted inside the function to define
        the unit vectors.

        For the Right and Left wand markers, the function performs the same calculation, with the difference being the
        corresponding sides marker. Each wand marker is defined as the cross product between the unit vector of the
        x axis of the thorax frame, and the unit vector from the thorax frame origin to the Shoulder marker.

        Given a marker SHO, representing the right (RSHO) or left (LSHO) shoulder markers and a thorax axis TH, the
        wand marker W is defined as:

        W_R = (RSHO-TH_o) \times TH_x
        W_L = TH_x \times (LSHO-TH_o)

        where TH_o is the origin of the thorax axis, TH_x is the x unit vector of the thorax axis.

        From this calculation, it should be clear that changing the thorax y and z vectors should not have an impact on the results.

        This unit test ensure that:
        - The right and left markers do not impact the wand marker calculations for one another
        - The function requires global positions
        - The thorax y and z axis do not change the results
        """
        result = pyCGM.findwandmarker(frame, thorax)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_findwandmarker_datatypes(self):
        """
        This test provides coverage of the findwandmarker function in pyCGM.py, defined as findwandmarker(frame,thorax)
        where frame is a dictionary of x, y, z positions and marker names and thorax is the thorax axis.

        This test checks that the resulting output from calling cross is correct when called with ints or floats.
        """
        frame_int = {'RSHO': [1, 0, 0], 'LSHO': [0, 0, 1]}
        frame_float = {'RSHO': [1.0, 0.0, 0.0], 'LSHO': [0.0, 0.0, 1.0]}
        thorax_int = [[[8, 0, 6], [0, 0, 0], [0, 0, 0]], [8, 0, 0]]
        thorax_float = [[[8.0, 0.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [8.0, 0.0, 0.0]]
        expected = [[8, 1, 0], [8, -1, 0]]

        # Check that calling findwandmarker yields the expected results when frame and thorax consist of ints
        result_int_list = pyCGM.findwandmarker(frame_int, thorax_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling findwandmarker yields the expected results when frame and thorax consist of floats
        result_float_list = pyCGM.findwandmarker(frame_float, thorax_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

    @pytest.mark.parametrize(["a", "b", "expected"], [
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
    def test_cross(self, a, b, expected):
        """
        This test provides coverage of the cross function in pyCGM.py, defined as cross(a, b) where a and b are both 3D vectors.

        This test takes 3 parameters:
        a: 3D vector
        b: 3D vector
        expected: the expected result from calling cross on a and b. This result is the cross product of the vectors
        a and b.
        """
        result = pyCGM.cross(a, b)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_cross_datatypes(self):
        """
        This test provides coverage of the cross function in pyCGM.py, defined as cross(a, b) where a and b are both 3D vectors.

        This test checks that the resulting output from calling cross is correct when called with a list of ints, a numpy
        array of ints, a list of floats, and a numpy array of floats.
        """
        A_int = [-2, 3, 1]
        A_float = [-2.0, 3.0, 1.0]
        B_int = [4, -1, 5]
        B_float = [4.0, -1.0, 5.0]
        expected = [16, 14, -10]

        # Check the calling cross on a list of ints yields the expected results
        result_int_list = pyCGM.cross(A_int, B_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling cross on a numpy array of ints yields the expected results
        result_int_nparray = pyCGM.cross(np.array(A_int, dtype='int'), np.array(B_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling cross on a list of floats yields the expected results
        result_float_list = pyCGM.cross(A_float, B_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling cross on a numpy array of floats yields the expected results
        result_float_nparray = pyCGM.cross(np.array(A_float, dtype='float'), np.array(B_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["v", "expected"], [
        ([-9944.089508486479, -20189.20612828088, 150.42955108569652], 22505.812344655435),
        ([0, 0, 0], 0),
        ([2, 0, 0], 2),
        ([0, 0, -1], 1),
        ([0, 3, 4], 5),
        ([-3, 0, 4], 5),
        ([6, -8, 0], 10),
        ([-5, 0, -12], 13),
        ([1, -1, np.sqrt(2)], 2)])
    def test_norm2d(self, v, expected):
        """
        This test provides coverage of the norm2d function in pyCGM.py, defined as norm2d(v) where v is a 3D vector.

        This test takes 2 parameters:
        v: 3D vector
        expected: the expected result from calling norm2d on v. This will be the value of the normalization of vector v.
        """
        result = pyCGM.norm2d(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_norm2d_datatypes(self):
        """
        This test provides coverage of the norm2d function in pyCGM.py, defined as norm2d(v) where v is a 3D vector.

        This test checks that the resulting output from calling norm2d is correct when called with a list of ints, a
        numpy array of ints, a list of floats, and a numpy array of floats.
        """
        v_int = [6, 0, -8]
        v_float = [6.0, 0, -8.0]
        expected = 10

        # Check the calling norm2d on a list of ints yields the expected results
        result_int_list = pyCGM.norm2d(v_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling norm2d on a numpy array of ints yields the expected results
        result_int_nparray = pyCGM.norm2d(np.array(v_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling norm2d on a list of floats yields the expected results
        result_float_list = pyCGM.norm2d(v_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling norm2d on a numpy array of floats yields the expected results
        result_float_nparray = pyCGM.norm2d(np.array(v_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["v", "expected"], [
        ([-212.5847168, 28.09841919, -4.15808105], np.array(214.47394390603984)),
        ([0, 0, 0], np.array(0)),
        ([2, 0, 0], np.array(2)),
        ([0, 0, -1], np.array(1)),
        ([0, 3, 4], np.array(5)),
        ([-3, 0, 4], np.array(5)),
        ([-6, 8, 0], np.array(10)),
        ([-5, 0, -12], np.array(13)),
        ([1, -1, np.sqrt(2)], np.array(2))])
    def test_norm3d(self, v, expected):
        """
        This test provides coverage of the norm3d function in pyCGM.py, defined as norm3d(v) where v is a 3D vector.

        This test takes 2 parameters:
        v: 3D vector
        expected: the expected result from calling norm3d on v. This will be the normalization of the vector v,
        inside of a numpy array.
        """
        result = pyCGM.norm3d(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_norm3d_datatypes(self):
        """
        This test provides coverage of the norm3d function in pyCGM.py, defined as norm3d(v) where v is a 3D vector.

        This test checks that the resulting output from calling norm3d is correct when called with a list of ints, a
        numpy array of ints, a list of floats, and a numpy array of floats.
        """
        v_int = [-6, 0, 8]
        v_float = [-6.0, 0, 8.0]
        expected = np.array(10)

        # Check the calling norm3d on a list of ints yields the expected results
        result_int_list = pyCGM.norm3d(v_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling norm3d on a numpy array of ints yields the expected results
        result_int_nparray = pyCGM.norm3d(np.array(v_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling norm3d on a list of floats yields the expected results
        result_float_list = pyCGM.norm3d(v_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling norm3d on a numpy array of floats yields the expected results
        result_float_nparray = pyCGM.norm3d(np.array(v_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["v", "expected"], [
        ([-212.5847168, 28.09841919, -4.15808105], np.array([-4.62150006e-03,  6.10847515e-04, -9.03948887e-05])),
        ([0, 0, 0], np.array([np.nan, np.nan, np.nan])),
        ([2, 0, 0], np.array([0.5, 0, 0])),
        ([0, 0, -1], np.array([0, 0, -1])),
        ([0, 3, 4], np.array([0, 0.12, 0.16])),
        ([-3, 0, 4], np.array([-0.12, 0, 0.16])),
        ([-6, 8, 0], np.array([-0.06, 0.08, 0])),
        ([-5, 0, -12], np.array([-0.0295858, 0, -0.07100592])),
        ([1, -1, np.sqrt(2)], np.array([0.25, -0.25, 0.35355339]))])
    def test_normDiv(self, v, expected):
        """
        This test provides coverage of the normDiv function in pyCGM.py, defined as normDiv(v) where v is a 3D vector.

        This test takes 2 parameters:
        v: 3D vector
        expected: the expected result from calling norm3d on v. This function returns the wrong result. It is supposed
        to return the normalization division, but in the function it divides the vector by the normalization twice.
        """
        result = pyCGM.normDiv(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_normDiv_datatypes(self):
        """
        This test provides coverage of the normDiv function in pyCGM.py, defined as normDiv(v) where v is a 3D vector.

        This test checks that the resulting output from calling normDiv is correct when called with a list of ints, a
        numpy array of ints, a list of floats, and a numpy array of floats.
        """
        v_int = [-6, 0, 8]
        v_float = [-6.0, 0, 8.0]
        expected = np.array([-0.06, 0, 0.08])

        # Check the calling normDiv on a list of ints yields the expected results
        result_int_list = pyCGM.normDiv(v_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling normDiv on a numpy array of ints yields the expected results
        result_int_nparray = pyCGM.normDiv(np.array(v_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling normDiv on a list of floats yields the expected results
        result_float_list = pyCGM.normDiv(v_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling normDiv on a numpy array of floats yields the expected results
        result_float_nparray = pyCGM.normDiv(np.array(v_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["A", "B", "expected"], [
        ([[1, 0, 0], [0, 1.0, -0.0], [0, 0.0, 1.0]], [[1.0, 0, 0.0], [0, 1, 0], [-0.0, 0, 1.0]], [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ([[1]], [[1]], [[1]]),
        # Invalid matrix dimensions
        ([[1, 2]], [[1]], [[1]]),
        ([[2], [1]], [[1, 2]], [[2, 4], [1, 2]]),
        # Invalid matrix dimensions
        ([[1, 2, 0], [0, 1, 2]], [[2, 1], [1, 4]], [[4, 9], [1, 4]]),
        ([[11, 12, 13], [14, 15, 16]], [[1, 2], [3, 4], [5, 6]], [[112, 148], [139, 184]]),
        ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]], [[58, 64], [139, 154]])])
    def test_matrixmult(self, A, B, expected):
        """
        This test provides coverage of the matrixmult function in pyCGM.py, defined as matrixmult(a, b)
        where a and b are both lists that represent a matrix to be multiplied.

        This test takes 3 parameters:
        A: a matrix, 2D array format
        B: a matrix, 2D array format
        expected: the expected matrix from calling matrixmult on A and B. This is the result of multiplying the two
        matrices A and B. It gives the correct result for multiplying two valid matrices, but still gives a result
        in some cases when the two matrices can't be multiplied. For two matrices to be multiplied, len(A[0]) need to
        be equal to len(B), but this function gives an output even when this isn't true
        """
        result = pyCGM.matrixmult(A, B)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_matrixmult_datatypes(self):
        """
        This test provides coverage of the matrixmult function in pyCGM.py, defined as matrixmult(a, b)
        where a and b are both lists that represent a matrix to be multiplied.

        This test checks that the resulting output from calling matrixmult is correct when called with a list of ints,
        a numpy array of ints, a list of floats, and a numpy array of floats.
        """
        A_int = [[1, 2, 0], [0, 1, 2]]
        B_int = [[2, 1], [1, 4]]
        A_float = [[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]]
        B_float = [[2.0, 1.0], [1.0, 4.0]]
        expected = [[4, 9], [1, 4]]

        # Check the calling matrixmult on a list of ints yields the expected results
        result_int_list = pyCGM.matrixmult(A_int, B_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling matrixmult on a numpy array of ints yields the expected results
        result_int_nparray = pyCGM.matrixmult(np.array(A_int, dtype='int'), np.array(B_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling matrixmult on a list of floats yields the expected results
        result_float_list = pyCGM.matrixmult(A_float, B_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling matrixmult on a numpy array of floats yields the expected results
        result_float_nparray = pyCGM.matrixmult(np.array(A_float, dtype='float'), np.array(B_float, dtype='float'))
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
        This test provides coverage of the rotmat function in pyCGM.py, defined as rotmat(x, y, z)
        where x, y, and z are all floats that represent the angle of rotation in a particular dimension.

        This test takes 4 parameters:
        x: angle to be rotated in the x axis
        y: angle to be rotated in the y axis
        z: angle to be rotated in the z axis
        expected: the expected rotation matrix from calling rotmat on x, y, and z. This will be a transformation
        matrix that can be used to perform a rotation in the x, y, and z directions at the values inputted.
        """
        result = pyCGM.rotmat(x, y, z)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_rotmat_datatypes(self):
        """
        This test provides coverage of the rotmat function in pyCGM.py, defined as rotmat(x, y, z)
        where x, y, and z are all floats that represent the angle of rotation in a particular dimension.

        This test checks that the resulting output from calling rotmat is correct when called with ints or floats.
        """
        result_int = pyCGM.rotmat(0, 150, -30)
        result_float = pyCGM.rotmat(0.0, 150.0, -30.0)
        expected = [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]

        np.testing.assert_almost_equal(result_int, expected, rounding_precision)
        np.testing.assert_almost_equal(result_float, expected, rounding_precision)