from mock import patch
import pycgm.axis as axis
import pytest
import numpy as np

rounding_precision = 5


class TestLowerBodyAxis:
    """
    This class tests the lower body axis functions in pyCGM.py:
    pelvisJointCenter
    hipJointCenter
    """
    nan_3d = [np.nan, np.nan, np.nan]
    rand_int = np.random.randint(0, 10)
    rand_coor = [np.random.randint(0, 10), np.random.randint(
        0, 10), np.random.randint(0, 10)]

    @pytest.mark.parametrize(["frame", "expected"], [
        # Test from running sample data
        (
            {
                'RASI': np.array([357.90066528, 377.69210815, 1034.97253418]),
                'LASI': np.array([145.31594849, 405.79052734, 1030.81445312]),
                'RPSI': np.array([274.00466919, 205.64402771, 1051.76452637]),
                'LPSI': np.array([189.15231323, 214.86122131, 1052.73486328])
            },
            np.array([[1.32329356e-01,  9.85629458e-01, -1.04992920e-01,
                       2.51608307e+02],
                      [-9.91191344e-01,  1.31010876e-01, -1.93873483e-02,
                       3.91741318e+02],
                      [-5.35352721e-03,  1.06633589e-01,  9.94283972e-01,
                       1.03289349e+03],
                      [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                       1.00000000e+00]])
        ),
        # Test with zeros for all params
        (
            {
                'SACR': np.array([0, 0, 0]),
                'RASI': np.array([0, 0, 0]),
                'LASI': np.array([0, 0, 0]),
                'RPSI': np.array([0, 0, 0]),
                'LPSI': np.array([0, 0, 0])
            },
            np.array([[np.nan, np.nan, np.nan,  0.],
                      [np.nan, np.nan, np.nan,  0.],
                      [np.nan, np.nan, np.nan,  0.],
                      [0.,  0.,  0.,  1.]])
        ),
        # # Testing when adding values to frame['RASI'] and frame['LASI']
        (
            {
                'RASI': np.array([-6, 6, 3]),
                'LASI': np.array([-7, -9, 1]),
                'RPSI': np.array([0, 0, 0]),
                'LPSI': np.array([0, 0, 0])
            },
            np.array([[-0.94458106,  0.01927716,  0.32771179, -6.5],
                      [-0.06593805, -0.98907071, -0.13187609, -1.5],
                      [0.32158794, -0.14617634,  0.93552855,  2.],
                      [0.,  0.,  0.,  1.]])
        ),
        # Testing when adding values to frame['RPSI'] and frame['LPSI']
        (
            {
                'RASI': np.array([0, 0, 0]),
                'LASI': np.array([0, 0, 0]),
                'RPSI': np.array([1, 0, -4]),
                'LPSI': np.array([7, -2, 2])
            },
            np.array([[np.nan, np.nan, np.nan,  0.],
                      [np.nan, np.nan, np.nan,  0.],
                      [np.nan, np.nan, np.nan,  0.],
                      [0.,  0.,  0.,  1.]])
        ),
        # Testing when adding values to frame['SACR']
        (
            {
                'SACR': np.array([-4, 8, -5]),
                'RASI': np.array([0, 0, 0]),
                'LASI': np.array([0, 0, 0]),
                'RPSI': np.array([0, 0, 0]),
                'LPSI': np.array([0, 0, 0])
            },
            np.array([[np.nan, np.nan, np.nan,  0.],
                      [np.nan, np.nan, np.nan,  0.],
                      [np.nan, np.nan, np.nan,  0.],
                      [0.,  0.,  0.,  1.]])
        ),
        # Testing when adding values to frame['RASI'], frame['LASI'], frame['RPSI'] and frame['LPSI']
        (
            {
                'RASI': np.array([-6, 6, 3]),
                'LASI': np.array([-7, -9, 1]),
                'RPSI': np.array([1, 0, -4]),
                'LPSI': np.array([7, -2, 2])
            },
            np.array([[-0.95825845,  0.02592043,  0.28472598, -6.5],
                      [-0.06593805, -0.98907071, -0.13187609, -1.5],
                      [0.27819584, -0.14514566,  0.9494945,  2.],
                      [0.,  0.,  0.,  1.]])
        ),
        # Testing when adding values to frame['SACR'], frame['RASI'] and frame['LASI']
        (
            {
                'SACR': np.array([-4, 8, -5]),
                'RASI': np.array([-6, 6, 3]),
                'LASI': np.array([-7, -9, 1]),
                'RPSI': np.array([0, 0, 0]),
                'LPSI': np.array([0, 0, 0])
            },

            np.array([[-0.22928306, -0.11360872,  0.96670695, -6.5],
                      [-0.06593805, -0.98907071, -0.13187609, -1.5],
                      [0.97112381, -0.09397972,  0.21928602,  2.],
                      [0.,  0.,  0.,  1.]])
        ),
        # Testing when adding values to frame['SACR'], frame['RPSI'] and frame['LPSI']
        (
            {
                'SACR': np.array([-4, 8, -5]),
                'RASI': np.array([0, 0, 0]),
                'LASI': np.array([0, 0, 0]),
                'RPSI': np.array([1, 0, -4]),
                'LPSI': np.array([7, -2, 2])
            },
            np.array([[np.nan, np.nan, np.nan,  0.],
                      [np.nan, np.nan, np.nan,  0.],
                      [np.nan, np.nan, np.nan,  0.],
                      [0.,  0.,  0.,  1.]])
        ),
        # Testing when adding values to frame['SACR'], frame['RASI'], frame['LASI'], frame['RPSI'] and frame['LPSI']
        (
            {
                'SACR': np.array([-4, 8, -5]),
                'RASI': np.array([-6, 6, 3]),
                'LASI': np.array([-7, -9, 1]),
                'RPSI': np.array([1, 0, -4]),
                'LPSI': np.array([7, -2, 2])
            },
            np.array([[-0.22928306, -0.11360872,  0.96670695, -6.5],
                      [-0.06593805, -0.98907071, -0.13187609, -1.5],
                      [0.97112381, -0.09397972,  0.21928602,  2.],
                      [0.,  0.,  0.,  1.]])
        ),
        # Testing that when frame is composed of lists of ints
        (
            {
                'SACR': [-4, 8, -5],
                'RASI': np.array([-6, 6, 3]),
                'LASI': np.array([-7, -9, 1]),
                'RPSI': [1, 0, -4],
                'LPSI': [7, -2, 2]
            },

            np.array([[-0.22928306, -0.11360872,  0.96670695, -6.5],
                      [-0.06593805, -0.98907071, -0.13187609, -1.5],
                      [0.97112381, -0.09397972,  0.21928602,  2.],
                      [0.,  0.,  0.,  1.]])
        ),
        # Testing that when frame is composed ofe numpy arrays of ints
        (
            {
                'SACR': np.array([-4, 8, -5], dtype='int'),
                'RASI': np.array([-6, 6, 3], dtype='int'),
                'LASI': np.array([-7, -9, 1], dtype='int'),
                'RPSI': np.array([1, 0, -4], dtype='int'),
                'LPSI': np.array([7, -2, 2], dtype='int')
            },
            np.array([[-0.22928306, -0.11360872,  0.96670695, -6.5],
                      [-0.06593805, -0.98907071, -0.13187609, -1.5],
                      [0.97112381, -0.09397972,  0.21928602,  2.],
                      [0.,  0.,  0.,  1.]])
        ),
        # Testing that when frame is composed of lists of floats
        (
            {
                'SACR': [-4.0, 8.0, -5.0],
                'RASI': np.array([-6.0, 6.0, 3.0]),
                'LASI': np.array([-7.0, -9.0, 1.0]),
                'RPSI': [1.0, 0.0, -4.0],
                'LPSI': [7.0, -2.0, 2.0]
            },
            np.array([[-0.22928306, -0.11360872,  0.96670695, -6.5],
                      [-0.06593805, -0.98907071, -0.13187609, -1.5],
                      [0.97112381, -0.09397972,  0.21928602,  2.],
                      [0.,  0.,  0.,  1.]])
        ),
        # Testing that when frame is composed of numpy arrays of floats
        (
            {
                'SACR': np.array([-4.0, 8.0, -5.0], dtype='float'),
                'RASI': np.array([-6.0, 6.0, 3.0], dtype='float'),
                'LASI': np.array([-7.0, -9.0, 1.0], dtype='float'),
                'RPSI': np.array([1.0, 0.0, -4.0], dtype='float'),
                'LPSI': np.array([7.0, -2.0, 2.0], dtype='float')
            },
            np.array([[-0.22928306, -0.11360872,  0.96670695, -6.5],
                      [-0.06593805, -0.98907071, -0.13187609, -1.5],
                      [0.97112381, -0.09397972,  0.21928602,  2.],
                      [0.,  0.,  0.,  1.]])
        )
    ])
    def test_pelvis_axis(_, frame, expected):
        """
        This test provides coverage of the pelvis_axis function in axis.py, defined as pelvis_axis(frame)

        This test takes 2 parameters:
        frame: dictionary of marker lists
        expected: the expected result from calling pelvisJointCenter on frame

        This test is checking to make sure the pelvis joint center and axis are calculated correctly given the input
        parameters.

        If RPSI and LPSI are given, then the sacrum will be the midpoint of those two markers. If they are not given then the sacrum is already calculated / specified.
        The origin of the pelvis is midpoint of the RASI and LASI markers.
        The axis of the pelvis is calculated using LASI, RASI, origin, and sacrum in the Gram-Schmidt orthogonalization procedure (ref. Kadaba 1990).

        Lastly, it checks that the resulting output is correct when frame is composed of lists of ints, numpy arrays of
        ints, lists of floats, and numpy arrays of floats. frame['LASI'] and frame['RASI'] were kept as numpy arrays
        every time as list would cause an error in pyCGM.py line 111 as lists cannot be divided by floats:
        origin = (RASI+LASI)/2.0
        """
        rasi = frame["RASI"] if "RASI" in frame else None
        lasi = frame["LASI"] if "LASI" in frame else None
        rpsi = frame["RPSI"] if "RPSI" in frame else None
        lpsi = frame["LPSI"] if "LPSI" in frame else None
        sacr = frame["SACR"] if "SACR" in frame else None

        result = axis.pelvis_axis(rasi, lasi, rpsi, lpsi, sacr)
        print(np.array_repr(result, precision=8))
        np.testing.assert_almost_equal(result, expected, rounding_precision)
