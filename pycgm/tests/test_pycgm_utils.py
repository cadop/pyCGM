import pyCGM_Single.pyCGM as pyCGM
import pytest
import numpy as np

rounding_precision = 8

class TestUtils():
    """
    This class tests the utils functions in pyCGM.py:
    findwandmarker
    matrixmult
    rotmat
    """

    rand_coor = [
        np.random.randint(0, 10),
        np.random.randint(0, 10),
        np.random.randint(0, 10),
    ]

    @pytest.mark.parametrize(
        ["frame", "thorax_axis", "expected"],
        [
            (
                {
                    "RSHO": [428.88476562, 270.552948, 1500.73010254],
                    "LSHO": [68.24668121, 269.01049805, 1510.1072998],
                },
                np.array([[256.23991128535846, 365.30496976939753, 1459.662169500559, 256.149810236564 ],
                          [  0,                  0,                   0,              364.3090603933987],
                          [  0,                  0,                   0,             1459.6553639290375],
                          [  0,                  0,                   0,                1              ]]),
                [
                    [255.92550222678443, 364.3226950497605, 1460.6297868417887],
                    [256.42380097331767, 364.27770361353487, 1460.6165849382387],
                ],
            ),
            (
                {"RSHO": [0, 0, 1], "LSHO": [0, 1, 0]},
                np.array([[1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]]),
                [[0, 1, 0], [0, 0, 1]],
            ),
            (
                {"RSHO": [0, 1, 1], "LSHO": [1, 1, 1]},
                np.array([[1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]]),
                [[0, 0.70710678, -0.70710678], [0, -0.70710678, 0.70710678]],
            ),
            (
                {"RSHO": [0, 1, 1], "LSHO": [1, 1, 1]},
                np.array([[1, 0, 0, -1],
                          [0, 0, 0,  0],
                          [0, 0, 0,  0],
                          [0, 0, 0,  1]]),
                [[-1, 0.70710678, -0.70710678], [-1, -0.70710678, 0.70710678]],
            ),
            (
                {"RSHO": [1, 2, 1], "LSHO": [2, 1, 2]},
                np.array([[1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]]),
                [[0, 0.4472136, -0.89442719], [0, -0.89442719, 0.4472136]],
            ),
            (
                {"RSHO": [1, 2, 1], "LSHO": [2, 2, 2]},
                np.array([[1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]]),
                [[0, 0.4472136, -0.89442719], [0, -0.70710678, 0.70710678]],
            ),
            (
                {"RSHO": [1, 2, 2], "LSHO": [2, 1, 2]},
                np.array([[1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]]),
                [[0, 0.70710678, -0.70710678], [0, -0.89442719, 0.4472136]],
            ),
            (
                {"RSHO": [1, 1, 1], "LSHO": [1, 1, 1]},
                np.array([[1, 0, 1, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]]),
                [[0.70710678, 0, -0.70710678], [-0.70710678, 0, 0.70710678]],
            ),
            (
                {"RSHO": [1, 1, 1], "LSHO": [1, 1, 1]},
                np.array([[1, 0, 1, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 0, 1]]),
                [[0, 0, 0], [0, 0, 2]],
            ),
            (
                {"RSHO": [0, 1, 0], "LSHO": [0, 0, -1]},
                np.array([[0, 3, 4, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]]),
                [[1, 0, 0], [-1, 0, 0]],
            ),
            (
                {"RSHO": [1, 0, 0], "LSHO": [0, 1, 0]},
                np.array([[7, 0, 24, 0],
                          [0, 0,  0, 0],
                          [0, 0,  0, 0],
                          [0, 0,  0, 1]]),
                [[0, -1, 0], [-0.96, 0, 0.28]],
            ),
            (
                {"RSHO": [1, 0, 0], "LSHO": [0, 0, 1]},
                np.array([[8, 0, 6, 8],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]]),
                [[8, 1, 0], [8, -1, 0]],
            ),
        ],
    )
    def test_calc_marker_wand(self, frame, thorax_axis, expected):
        """
        This test provides coverage of the calc_marker_wand function in pyCGM.py, defined as 
        calc_marker_wand(rsho, lsho, thorax_axis), where rsho and lsho are (x, y, z) marker positions
        and thorax_axis is a 4x4 affine matrix representing the thorax axis and origin.

        The function takes in the (x, y, z) position of the Right Shoulder and Left Shoulder markers, as well as the thorax
        axis, which is a 4x4 affine matrix representing the thorax axis and origin.  The wand marker position is
        returned as a 2x3 array containing the right wand marker (x, y, z) positions (1x3) followed by the 
        left wand marker (x, y, z) positions (1x3).

        For the Right and Left wand markers, the function performs the same calculation, with the difference being the
        corresponding sides marker. Each wand marker is defined as the cross product between the unit vector of the
        x axis of the thorax frame, and the unit vector from the thorax frame origin to the Shoulder marker.

        Given a marker SHO, representing the right (RSHO) or left (LSHO) shoulder markers and a thorax axis TH, the
        wand marker W is defined as:

        .. math::
            W_R = (RSHO-TH_o) \times TH_x
            W_L = TH_x \times (LSHO-TH_o)

        where :math:`TH_o` is the origin of the thorax axis, :math:`TH_x` is the x unit vector of the thorax axis.

        From this calculation, it should be clear that changing the thorax y and z vectors should not have an impact
        on the results.

        This unit test ensure that:
        - The right and left markers do not impact the wand marker calculations for one another
        - The function requires global positions
        - The thorax y and z axis do not change the results
        """
        rsho = frame["RSHO"] if "RSHO" in frame else None
        lsho = frame["LSHO"] if "LSHO" in frame else None

        thorax_o = thorax_axis[:3, 3]
        thorax_axis[0, :3] -= thorax_o
        thorax_axis[1, :3] -= thorax_o
        thorax_axis[2, :3] -= thorax_o

        result = pyCGM.calc_marker_wand(rsho, lsho, thorax_axis)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_calc_marker_wand_datatypes(self):
        """
        This test provides coverage of the calc_marker_wand function in pyCGM.py, defined as 
        calc_marker_wand(rsho, lsho, thorax_axis), where rsho and lsho are (x, y, z) marker positions
        and thorax_axis is a 4x4 affine matrix representing the thorax axis and origin.

        This test checks that the resulting output from calling cross is correct when called with ints or floats.
        """
        frame_int = {'RSHO': [1, 0, 0], 'LSHO': [0, 0, 1]}
        frame_float = {'RSHO': [1.0, 0.0, 0.0], 'LSHO': [0.0, 0.0, 1.0]}
        thorax_int = np.array([[8, 0, 6, 8], 
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 1]])
        thorax_float = np.array([[8.0, 0.0, 6.0, 8.0], 
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])
        expected = [[8, 1, 0], [8, -1, 0]]


        thorax_o = thorax_int[:3, 3]
        thorax_int[0, :3] -= thorax_o
        thorax_int[1, :3] -= thorax_o
        thorax_int[2, :3] -= thorax_o

        thorax_o = thorax_float[:3, 3]
        thorax_float[0, :3] -= thorax_o
        thorax_float[1, :3] -= thorax_o
        thorax_float[2, :3] -= thorax_o

        # Check that calling calc_marker_wand yields the expected results when frame and thorax consist of ints
        rsho_int = frame_int["RSHO"] if "RSHO" in frame_int else None
        lsho_int = frame_int["LSHO"] if "LSHO" in frame_int else None
        result_int_list = pyCGM.calc_marker_wand(rsho_int, lsho_int, thorax_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling calc_marker_wand yields the expected results when frame and thorax consist of floats
        rsho_float = frame_float["RSHO"] if "RSHO" in frame_float else None
        lsho_float = frame_float["LSHO"] if "LSHO" in frame_float else None
        result_float_list = pyCGM.calc_marker_wand(rsho_float, lsho_float, thorax_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

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