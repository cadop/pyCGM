import pytest
import pyCGM_Single.pycgmStatic as pycgmStatic
import numpy as np
from mock import patch

rounding_precision = 5

class TestPycgmStaticAxis():
    """
    This class tests the axis functions in pycgmStatic.py:
        calc_static_head
        calc_axis_pelvis
        calc_joint_center_hip
        calc_axis_ankle
        calc_axis_knee
        calc_axis_head
        calc_axis_uncorrect_foot
        rotaxis_footflat
        rotaxis_nonfootflat
        calc_joint_center
    """
    nan_3d = [np.nan, np.nan, np.nan]
    rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]

    @pytest.mark.parametrize(
        ["head_axis", "expected"],
        [
            # Test from running sample data
            (
                np.array([[244.87227957886893, 326.0240255639856, 1730.4189843948805, 244.89547729492188],
                          [243.89575702706503, 325.0366593474616, 1730.1515677531293, 325.0578918457031],
                          [244.89086730509763, 324.80072493605866, 1731.1283433097797, 1730.1619873046875],
                          [0, 0, 0, 0]]),
                0.25992807335420975,
            ),
            # Test with zeros for all params
            (
                np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]]
                        ),
                np.nan
            ),
            # Testing when values are added to head x-axis
            (
                np.array([[-1, 8, 9, 0],
                          [ 0, 0, 0, 0],
                          [ 0, 0, 0, 0],
                          [ 0, 0, 0, 1]]
                        ),
                1.5707963267948966
            ),
            # Testing when values are added to head y-axis
            (
                np.array([[0, 0, 0, 0],
                          [7, 5, 7, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]]
                        ),
                np.nan
            ),
            # Testing when values are added to head z-axis
            (
                np.array([[0,  0,  0, 0],
                          [0,  0,  0, 0],
                          [0, -6, -2, 0],
                          [0,  0,  0, 1]]
                        ),
                0.0
            ),
            # Testing when values are added to head axes
            (
                np.array([[-1,  8,  9, 0],
                          [7,  5,  7, 0],
                          [0, -6, -2, 0],
                          [0,  0,  0, 1]]
                        ),
                -1.3521273809209546
            ),
            # Testing when values are added to head origin
            (
                np.array([[0, 0, 0, -4],
                          [0, 0, 0,  7],
                          [0, 0, 0,  8],
                          [0, 0, 0,  1]]
                        ),
                0.7853981633974483
            ),
            # Testing when values are added to head axes and origin
            (
                np.array([[-1,  8,  9, -4],
                          [ 7,  5,  7,  7],
                          [ 0, -6, -2,  8],
                          [ 0,  0,  0,  1]]
                        ),
                -0.09966865249116204
            ),
            # Testing that when head_axis is composed of lists of ints
            (
                [
                          [-1,  8,  9, -4],
                          [ 7,  5,  7,  7],
                          [ 3, -6, -2,  8],
                          [ 0,  0,  0,  1]
                ],
                -0.09966865249116204
            ),
            # Testing that when head_axis is composed of numpy arrays of ints
            (
                np.array([[-1,  8,  9, -4],
                          [ 7,  5,  7,  7],
                          [ 0, -6, -2,  8],
                          [ 0,  0,  0,  1]], dtype="int"),
                -0.09966865249116204,
            ),
            # Testing that when head_axis is composed of lists of floats
            (
                [
                          [-1.0,  8.0,  9.0, -4.0],
                          [ 7.0,  5.0,  7.0,  7.0],
                          [ 3.0, -6.0, -2.0,  8.0],
                          [ 0.0,  0.0,  0.0,  1.0]
                ],
                -0.09966865249116204
            ),
            # Testing that when head_axis is composed of numpy arrays of floats
            (
                np.array([[-1.0,  8.0,  9.0, -4.0],
                          [ 7.0,  5.0,  7.0,  7.0],
                          [ 3.0, -6.0, -2.0,  8.0],
                          [ 0.0,  0.0,  0.0,  1.0]], dtype="float"),
                -0.09966865249116204
            ),
        ],
    )
    def test_calc_static_head(self, head_axis, expected):
        """
        This test provides coverage of the calc_static_head function in pycgmStatic.py, defined as calc_static_head(head_axis)

        This test takes 2 parameters:
        head_axis: 4x4 affine matrix representing the head axes and origin
        expected: the expected result from calling calc_static_head on head_axis

        This function first calculates the (x, y, z) axes of the head by subtracting the given head axes by the head
        origin. It then calls calc_head_offset on this head axis and a global axis to find the head offset angles.

        This test ensures that:
        - the head axis and the head origin both have an effect on the final offset angle
        - the resulting output is correct when head_axis is composed of lists of ints, numpy arrays of ints, lists of
        floats, and numpy arrays of floats.
        """
        head_axis = np.asarray(head_axis)
        head_o = head_axis[:3, 3]
        head_axis[0, :3] -= head_o
        head_axis[1, :3] -= head_o
        head_axis[2, :3] -= head_o
        
        result = pycgmStatic.calc_static_head(head_axis)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["frame", "expected"], [
            # Test from running sample data
            (
                {
                    'RASI': np.array([357.90066528, 377.69210815, 1034.97253418]),
                    'LASI': np.array([145.31594849, 405.79052734, 1030.81445312]),
                    'RPSI': np.array([274.00466919, 205.64402771, 1051.76452637]),
                    'LPSI': np.array([189.15231323, 214.86122131, 1052.73486328])
                },
                np.array([[ 1.32329356e-01,  9.85629458e-01, -1.04992920e-01, 2.51608307e+02],
                          [-9.91191344e-01,  1.31010876e-01, -1.93873483e-02, 3.91741318e+02],
                          [-5.35352721e-03,  1.06633589e-01,  9.94283972e-01, 1.03289349e+03],
                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
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
    def test_calc_axis_pelvis(self, frame, expected):
        """
        This test provides coverage of the test_calc_axis_pelvis function in pycgmStatic.py,
        defined as test_calc_axis_pelvis(rasi, lasi, rpsi, lpsi, sacr)

        This test takes 2 parameters:
        frame: dictionary of marker lists
        expected: the expected result from calling test_calc_axis_pelvis on frame

        This test is checking to make sure the pelvis joint center and axis are calculated
        correctly given the input parameters. 

        If RPSI and LPSI are given, then the sacrum will be the midpoint of those two markers.
        If they are not given then the sacrum is already calculated / specified. 
        The origin of the pelvis is midpoint of the RASI and LASI markers.
        The axis of the pelvis is calculated using LASI, RASI, origin, and sacrum in the 
        Gram-Schmidt orthogonalization procedure (ref. Kadaba 1990). 

        Lastly, it checks that the resulting output is correct when frame is composed of lists of ints, numpy arrays of
        ints, lists of floats, and numpy arrays of floats. frame['LASI'] and frame['RASI'] were kept as numpy arrays
        every time as list as lists cannot be divided by floats e.g. o = (rasi+lasi)/2.0
        """

        rasi = frame["RASI"] if "RASI" in frame else None
        lasi = frame["LASI"] if "LASI" in frame else None
        rpsi = frame["RPSI"] if "RPSI" in frame else None
        lpsi = frame["LPSI"] if "LPSI" in frame else None
        sacr = frame["SACR"] if "SACR" in frame else None

        result = pycgmStatic.calc_axis_pelvis(rasi, lasi, rpsi, lpsi, sacr)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["pelvis_axis", "subject", "expected"], [
        # Test from running sample data
        (    
             [
                 [251.740636241119, 392.726947206848, 1032.788500732036, 251.608306884766], 
                 [250.617115540376, 391.872328624646, 1032.874106304030, 391.741317749023], 
                 [251.602953357582, 391.847951338178, 1033.887777624562, 1032.893493652344],
                 [  0,               0,                0,                   1             ]
            ],
            {
                'MeanLegLength': 940.0,
                'R_AsisToTrocanterMeasure': 72.512, 
                'L_AsisToTrocanterMeasure': 72.512, 
                'InterAsisDistance': 215.908996582031
            },
            [
                [308.38050472, 322.80342417, 937.98979061],
                [182.57097863, 339.43231855, 935.52900126]
            ]
        ),
        # Basic test with zeros for all params
        (
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            {
                'MeanLegLength': 0.0,
                'R_AsisToTrocanterMeasure': 0.0, 
                'L_AsisToTrocanterMeasure': 0.0, 
                'InterAsisDistance': 0.0
            },
            [
                [0, 0, 0], 
                [0, 0, 0]
            ]
        ),
        # Testing when values are added to pel_origin
        (
            [
                [0, 0, 0,  1],
                [0, 0, 0,  0],
                [0, 0, 0, -3],
                [0, 0, 0,  0]
            ],
            {
                'MeanLegLength': 0.0,
                'R_AsisToTrocanterMeasure': 0.0,
                'L_AsisToTrocanterMeasure': 0.0,
                'InterAsisDistance': 0.0
            },
            [
                [8.53165418, 0, -25.59496255],
                [-6.1387721, 0, 18.4163163]
            ]
        ),
        # Testing when values are added to pel_x
        (
            [
                [-5, -3, -6, 0],
                [0, 0, 0,    0],
                [0, 0, 0,    0],
                [0, 0, 0,    0]
            ],
            {
                'MeanLegLength': 0.0,
                'R_AsisToTrocanterMeasure': 0.0,
                'L_AsisToTrocanterMeasure': 0.0,
                'InterAsisDistance': 0.0
            },
            [
                [54.02442793, 32.41465676, 64.82931352],
                [54.02442793, 32.41465676, 64.82931352]
            ]
        ),
        # Testing when values are added to pel_y
        (
                [
                    [0, 0, 0,  0],
                    [4, -1, 2, 0],
                    [0, 0, 0,  0],
                    [0, 0, 0,  0]
                ],
            {
                'MeanLegLength': 0.0,
                'R_AsisToTrocanterMeasure': 0.0,
                'L_AsisToTrocanterMeasure': 0.0,
                'InterAsisDistance': 0.0
            },
            [
                [-29.34085257,  7.33521314, -14.67042628],
                [ 29.34085257, -7.33521314,  14.67042628]
            ]
        ),
        # Testing when values are added to pel_z
        (
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [3, 8, 2, 0],
                    [0, 0, 0, 0]
                ],
            {
                'MeanLegLength': 0.0,
                'R_AsisToTrocanterMeasure': 0.0,
                'L_AsisToTrocanterMeasure': 0.0,
                'InterAsisDistance': 0.0
            },
            [
                [31.82533363, 84.86755635, 21.21688909],
                [31.82533363, 84.86755635, 21.21688909]
            ]
        ),
        # Test when values are added to pel_x, pel_y, and pel_z
        (
                [
                    [-5, -3, -6, 0],
                    [ 4, -1,  2, 0],
                    [ 3,  8,  2, 0],
                    [ 0,  0,  0, 0]
                ],
            {
                'MeanLegLength': 0.0,
                'R_AsisToTrocanterMeasure': 0.0,
                'L_AsisToTrocanterMeasure': 0.0,
                'InterAsisDistance': 0.0
            },
            [
                [ 56.508909  , 124.61742625,  71.37577632],
                [115.19061413, 109.94699997, 100.71662889]
            ]
        ),
        # Test when values are added to pel_origin, pel_x, pel_y, and pel_z
        (
                [
                    [-5, -3, -6,  1],
                    [ 4, -1,  2,  0],
                    [ 3,  8,  2, -3],
                    [ 0,  0,  0,  0]
                ],
            {
                'MeanLegLength': 0.0,
                'R_AsisToTrocanterMeasure': 0.0,
                'L_AsisToTrocanterMeasure': 0.0,
                'InterAsisDistance': 0.0
            },
            [
                [ 65.04056318, 124.61742625,  45.78081377],
                [109.05184203, 109.94699997, 119.13294518]
            ]
        ),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[MeanLegLength]
        (
                [
                    [-5, -3, -6,  1],
                    [ 4, -1,  2,  0],
                    [ 3,  8,  2, -3],
                    [ 0,  0,  0,  0]
                ],
            {
                'MeanLegLength': 15.0,
                'R_AsisToTrocanterMeasure': 0.0,
                'L_AsisToTrocanterMeasure': 0.0,
                'InterAsisDistance': 0.0
            },
            [
                [ 61.83654463, 110.86920998,  41.31408931],
                [100.88576753,  97.85280235, 106.39612748]
            ]
        ),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[R_AsisToTrocanterMeasure]
        (
                [
                    [-5, -3, -6,  1],
                    [ 4, -1,  2,  0],
                    [ 3,  8,  2, -3],
                    [ 0,  0,  0,  0]
                ],
            {
                'MeanLegLength': 0.0,
                'R_AsisToTrocanterMeasure': -24.0,
                'L_AsisToTrocanterMeasure': 0.0,
                'InterAsisDistance': 0.0
            },
            [
                [-57.09307697, 115.44008189,  14.36512267],
                [109.05184203, 109.94699997, 119.13294518]
            ]
        ),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[L_AsisToTrocanterMeasure]
        (
                [
                    [-5, -3, -6,  1],
                    [ 4, -1,  2,  0],
                    [ 3,  8,  2, -3],
                    [ 0,  0,  0,  0]
                ],
            {
                'MeanLegLength': 0.0,
                'R_AsisToTrocanterMeasure': 0.0,
                'L_AsisToTrocanterMeasure': 0-7.0,
                'InterAsisDistance': 0.0
            },
            [
                [65.04056318, 124.61742625,  45.78081377],
                [73.42953032, 107.27027453, 109.97003528]
            ]
        ),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[InterAsisDistance]
        (
                [
                    [-5, -3, -6,  1],
                    [ 4, -1,  2,  0],
                    [ 3,  8,  2, -3],
                    [ 0,  0,  0,  0]
                ],
            {
                'MeanLegLength': 0.0,
                'R_AsisToTrocanterMeasure': 0.0,
                'L_AsisToTrocanterMeasure': 0.0,
                'InterAsisDistance': 11.0
            },
            [
                [ 48.54056318, 130.11742625,  18.28081377],
                [125.55184203, 104.44699997, 146.63294518]
            ]
        ),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and all values in vsk
        (
                [
                    [-5, -3, -6,  1],
                    [ 4, -1,  2,  0],
                    [ 3,  8,  2, -3],
                    [ 0,  0,  0,  0]
                ],
                {
                    'MeanLegLength': 15.0,
                    'R_AsisToTrocanterMeasure': -24.0,
                    'L_AsisToTrocanterMeasure': -7.0,
                    'InterAsisDistance': 11.0
                },
                [
                    [-76.79709552, 107.19186562, -17.60160178],
                    [ 81.76345582,  89.67607691, 124.73321758]
                ]
            ),
        # Testing that when pel_origin, pel_x, pel_y, and pel_z are lists of ints and vsk values are ints
        (
                [
                    [-5, -3, -6,  1],
                    [ 4, -1,  2,  0],
                    [ 3,  8,  2, -3],
                    [ 0,  0,  0,  0]
                ],
                {
                    'MeanLegLength': 15,
                    'R_AsisToTrocanterMeasure': -24,
                    'L_AsisToTrocanterMeasure': -7,
                    'InterAsisDistance': 11
                },
                [
                    [-76.79709552, 107.19186562, -17.60160178],
                    [ 81.76345582,  89.67607691, 124.73321758]
                ]
            ),
        # Testing that when pel_origin, pel_x, pel_y, and pel_z are numpy arrays of ints and vsk values are ints
        (
                np.array([[-5, -3, -6,  1],
                          [ 4, -1,  2,  0],
                          [ 3,  8,  2, -3],
                          [ 0,  0,  0,  0]]
                ),
                {
                    'MeanLegLength': 15,
                    'R_AsisToTrocanterMeasure': -24,
                    'L_AsisToTrocanterMeasure': -7,
                    'InterAsisDistance': 11
                },
                [
                    [-76.79709552, 107.19186562, -17.60160178],
                    [ 81.76345582,  89.67607691, 124.73321758] 
                ]
        ),
        # Testing that when pel_origin, pel_x, pel_y, and pel_z are lists of floats and vsk values are floats
        (       
                [
                    [-5.0, -3.0, -6.0,  1.0],
                    [ 4.0, -1.0,  2.0,  0.0],
                    [ 3.0,  8.0,  2.0, -3.0],
                    [ 0.0,  0.0,  0.0,  0.0]
                ],
                {
                    'MeanLegLength': 15.0,
                    'R_AsisToTrocanterMeasure': -24.0,
                    'L_AsisToTrocanterMeasure': -7.0,
                    'InterAsisDistance': 11.0
                },
                [
                    [-76.79709552, 107.19186562, -17.60160178],
                    [ 81.76345582,  89.67607691, 124.73321758] 
                ]
        ),
        # Testing that when pel_origin, pel_x, pel_y, and pel_z are numpy arrays of floats and vsk values are floats
        (
                np.array([[-5.0, -3.0, -6.0,  1.0],
                          [ 4.0, -1.0,  2.0,  0.0],
                          [ 3.0,  8.0,  2.0, -3.0],
                          [ 0.0,  0.0,  0.0,  0.0]]
                ),
                {
                    'MeanLegLength': 15.0,
                    'R_AsisToTrocanterMeasure': -24.0,
                    'L_AsisToTrocanterMeasure': -7.0,
                    'InterAsisDistance': 11
                },
                [
                    [-76.79709552, 107.19186562, -17.60160178],
                    [ 81.76345582,  89.67607691, 124.73321758] 
                ]
        )
    ])
    def test_calc_joint_center_hip(self, pelvis_axis, subject, expected):
        """
        This test provides coverage of the calc_joint_center_hip function in pycgmStatic.py,
        defined as calc_joint_center_hip(pelvis_axis, subject)
        This test takes 2 parameters:
        pelvis_axis: 4x4 affine matrix representing the pelvis axes and origin
        subject: dictionary containing subject measurements from a VSK file
        expected: the expected result from calling calc_joint_center_hip on pelvis_axis, subject
        This test is checking to make sure the hip joint center is calculated correctly given the input parameters.
        The test checks to see that the correct values in expected are updated per each input parameter added. Any
        parameter that is added should change the value of every value in expected.
        The hip joint center axis and origin are calculated using the Hip Joint Center Calculation (ref. Davis_1991).
        Lastly, it checks that the resulting output is correct when the pelvis axis is composed of lists of ints, 
        numpy arrays of ints, lists of floats, and numpy arrays of floats and vsk values are ints or floats.
        """

        pelvis_axis = np.asarray(pelvis_axis)
        pelvis_o = pelvis_axis[:3, 3]
        pelvis_axis[0, :3] -= pelvis_o
        pelvis_axis[1, :3] -= pelvis_o
        pelvis_axis[2, :3] -= pelvis_o

        result = pycgmStatic.calc_joint_center_hip(pelvis_axis, subject)
        np.testing.assert_almost_equal(result, expected, rounding_precision)



    @pytest.mark.parametrize(
        ["frame", "hip_JC", "vsk", "mock_return_val", "expected_mock_args", "expected"],
        [
            # Test from running sample data
            (
                {
                    "RTHI": np.array([426.50338745, 262.65310669, 673.66247559]),
                    "LTHI": np.array([51.93867874, 320.01849365, 723.03186035]),
                    "RKNE": np.array([416.98687744, 266.22558594, 524.04089355]),
                    "LKNE": np.array([84.62355804, 286.69122314, 529.39819336]),
                },
                [
                    [182.57097863, 339.43231855, 935.52900126],
                    [308.38050472, 322.80342417, 937.98979061]
                ],
                {"RightKneeWidth": 105.0, "LeftKneeWidth": 105.0},
                np.array([[364.17774614, 292.17051722, 515.19181496],
                         [143.55478579, 279.90370346, 524.78408753]]),
                [
                    [
                        [426.50338745, 262.65310669, 673.66247559],
                        [308.38050472, 322.80342417, 937.98979061],
                        [416.98687744, 266.22558594, 524.04089355],
                        59.5
                    ],
                    [
                        [51.93867874, 320.01849365, 723.03186035],
                        [182.57097863, 339.43231855, 935.52900126],
                        [84.62355804, 286.69122314, 529.39819336],
                        59.5
                    ],
                ],
                np.array([[[364.61959153, 293.06758353, 515.18513093, 364.17774614],
                           [363.29019771, 292.60656648, 515.04309095, 292.17051722],
                           [364.04724541, 292.24216264, 516.18067112, 515.19181496],
                           [  0,            0,            0,            1         ]],
                          [[143.65611282, 280.88685896, 524.63197541, 143.55478579],
                           [142.56434499, 280.01777943, 524.86163553, 279.90370346],
                           [143.64837987, 280.04650381, 525.76940383, 524.78408753],
                           [  0,            0,            0,            1         ]],
                ]),
            ),
            # Test with zeros for all params
            (
                {
                    "RTHI": np.array([0, 0, 0]),
                    "LTHI": np.array([0, 0, 0]),
                    "RKNE": np.array([0, 0, 0]),
                    "LKNE": np.array([0, 0, 0]),
                },
                [
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                {"RightKneeWidth": 0.0, "LeftKneeWidth": 0.0},
                np.array([[0, 0, 0],
                          [0, 0, 0]]),
                [
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        7.0
                    ], 
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        7.0
                    ]
                ],
                np.array([[[np.nan, np.nan, np.nan, 0], 
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]],
                          [[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]],
                ]),
            ),
            # Testing when values are added to frame
            (
                {
                    "RTHI": np.array([1, 2, 4]),
                    "LTHI": np.array([-1, 0, 8]),
                    "RKNE": np.array([8, -4, 5]),
                    "LKNE": np.array([8, -8, 5]),
                },
                [
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                {"RightKneeWidth": 0.0, "LeftKneeWidth": 0.0},
                np.array([[0, 0, 0],
                          [0, 0, 0]]),
                [
                    [
                        [1, 2, 4],
                        [0, 0, 0],
                        [8, -4, 5],
                        7.0
                    ], 
                    [
                        [-1, 0, 8],
                        [0, 0, 0],
                        [8, -8, 5],
                        7.0
                    ]
                ],
                np.array([[[np.nan, np.nan, np.nan,  0    ], 
                           [np.nan, np.nan, np.nan,  0    ],
                           [np.nan, np.nan, np.nan,  0    ],
                           [ 0,      0,      0,      1    ]],
                          [[np.nan, np.nan, np.nan,  0    ],
                           [np.nan, np.nan, np.nan,  0    ],
                           [np.nan, np.nan, np.nan,  0    ],
                           [ 0,      0,      0,      1    ]],
                ]),
            ),
            # # Testing when values are added to hip_JC
            (
                {
                    "RTHI": np.array([0, 0, 0]),
                    "LTHI": np.array([0, 0, 0]),
                    "RKNE": np.array([0, 0, 0]),
                    "LKNE": np.array([0, 0, 0]),
                },
                [[-8, 8, -2], [1, -9, 2]],
                {"RightKneeWidth": 0.0, "LeftKneeWidth": 0.0},
                np.array([[0, 0, 0],
                          [0, 0, 0]]),
                [
                    [
                        [0, 0, 0],
                        [1, -9, 2],
                        [0, 0, 0],
                        7.0
                    ], 
                    [
                        [0, 0, 0],
                        [-8, 8, -2],
                        [0, 0, 0],
                        7.0
                    ]
                ],
                np.array([[[ np.nan,     np.nan, np.nan,          0],
                           [ np.nan,     np.nan, np.nan,          0],
                           [ 0.10783277, -0.97049496, 0.21566555, 0],
                           [ 0,           0,          0,          1]],
                          [[np.nan,      np.nan,      np.nan,     0],
                           [np.nan,      np.nan,      np.nan,     0],
                           [-0.69631062, 0.69631062, -0.17407766, 0],
                           [ 0,          0,           0,          1]]],
                        ),
            ),
            # Testing when values are added to vsk
            (
                {
                    "RTHI": np.array([0, 0, 0]),
                    "LTHI": np.array([0, 0, 0]),
                    "RKNE": np.array([0, 0, 0]),
                    "LKNE": np.array([0, 0, 0]),
                },
                [
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                {"RightKneeWidth": 9.0, "LeftKneeWidth": -6.0},
                np.array([[0, 0, 0],
                          [0, 0, 0]]),
                [
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        11.5
                    ], 
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        4.0
                    ]
                ],
                np.array([[[np.nan, np.nan, np.nan, 0], 
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]],
                          [[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]],
                ]),
            ),
            # Testing when values are added to mock_return_val
            (
                {
                    "RTHI": np.array([0, 0, 0]),
                    "LTHI": np.array([0, 0, 0]),
                    "RKNE": np.array([0, 0, 0]),
                    "LKNE": np.array([0, 0, 0]),
                },
                [
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                {"RightKneeWidth": 0.0, "LeftKneeWidth": 0.0},
                np.array([[-5, -5, -9],
                          [3, -6, -5]]),
                [
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        7.0
                    ], 
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        7.0
                    ]
                ],
                np.array([[[ np.nan,     np.nan, np.nan,          -5],
                           [ np.nan,     np.nan, np.nan,          -5],
                           [-4.56314797, -4.56314797, -8.21366635,-9],
                           [ 0,           0,          0,           1]],
                          [[np.nan,      np.nan,      np.nan,      3],
                           [np.nan,      np.nan,      np.nan,     -6],
                           [ 2.64143142, -5.28286283, -4.4023857, -5],
                           [ 0,          0,           0,           1]]],
                        ),
            ),
            # # # Testing when values are added to frame and hip_JC
            (
                {
                    "RTHI": np.array([1, 2, 4]),
                    "LTHI": np.array([-1, 0, 8]),
                    "RKNE": np.array([8, -4, 5]),
                    "LKNE": np.array([8, -8, 5]),
                },
                [
                    [-8,  8, -2],
                    [ 1, -9,  2]
                ],
                {"RightKneeWidth": 0.0, "LeftKneeWidth": 0.0},
                np.array([[0, 0, 0],
                          [0, 0, 0]]),
                [
                    [
                        [1,  2, 4],
                        [1, -9, 2],
                        [8, -4, 5],
                        7.0
                    ], 
                    [
                        [-1, 0,  8],
                        [-8, 8, -2],
                        [8, -8,  5],
                        7.0
                    ]
                ],
                np.array([[[-0.47319376,  0.14067923,  0.86965339, 0],
                           [-0.8743339,  -0.19582873, -0.44406233, 0],
                           [ 0.10783277, -0.97049496,  0.21566555, 0],
                           [ 0,           0,           0,          1]],
                          [[-0.70710678, -0.70710678,  0.0,        0],
                           [-0.12309149,  0.12309149,  0.98473193, 0],
                           [-0.69631062,  0.69631062, -0.17407766, 0],
                           [ 0,           0,           0,          1]]],
                        ),
            ),
            # Testing when values are added to frame, hip_JC, and vsk
            (
                {
                    "RTHI": np.array([1, 2, 4]),
                    "LTHI": np.array([-1, 0, 8]),
                    "RKNE": np.array([8, -4, 5]),
                    "LKNE": np.array([8, -8, 5]),
                },
                [
                    [-8,  8, -2],
                    [ 1, -9,  2]
                ],
                {"RightKneeWidth": 9.0, "LeftKneeWidth": -6.0},
                np.array([[0, 0, 0],
                          [0, 0, 0]]),
                [
                    [
                        [1,  2, 4],
                        [1, -9, 2],
                        [8, -4, 5],
                        11.5
                    ], 
                    [
                        [-1,  0,  8],
                        [-8,  8, -2],
                        [ 8, -8,  5],
                        4.0
                    ]
                ],
                np.array([[[-0.47319376,  0.14067923,  0.86965339, 0],
                           [-0.8743339,  -0.19582873, -0.44406233, 0],
                           [ 0.10783277, -0.97049496,  0.21566555, 0],
                           [ 0,           0,           0,          1]],
                          [[-0.70710678, -0.70710678,  0,          0],
                           [-0.12309149,  0.12309149,  0.98473193, 0],
                           [-0.69631062,  0.69631062, -0.17407766, 0],
                           [ 0,           0,           0,          1]]],
                        ),
            ),
            # Testing when values are added to frame, hip_JC, vsk, and mock_return_val
            (
                {
                    "RTHI": np.array([1, 2, 4]),
                    "LTHI": np.array([-1, 0, 8]),
                    "RKNE": np.array([8, -4, 5]),
                    "LKNE": np.array([8, -8, 5]),
                },
                [
                    [-8,  8, -2],
                    [ 1, -9,  2]
                ],
                {"RightKneeWidth": 9.0, "LeftKneeWidth": -6.0},
                np.array([[-5, -5, -9],
                          [ 3, -6, -5]]),
                [
                    [
                        [1,  2, 4],
                        [1, -9, 2],
                        [8, -4, 5],
                        11.5
                    ], 
                    [
                        [-1,  0,  8],
                        [-8,  8, -2],
                        [ 8, -8,  5],
                        4.0
                    ]
                ],
                np.array([[[-5.6293369,  -4.4458078,  -8.45520089, -5],
                           [-5.62916022, -5.77484544, -8.93858368, -5],
                           [-4.54382845, -5.30411437, -8.16368549, -9],
                           [ 0,           0,           0,           1]],
                          [[2.26301154, -6.63098327, -4.75770242,  3],
                           [3.2927155,  -5.97483821, -4.04413154, -6],
                           [2.39076635, -5.22461171, -4.83384537, -5],
                           [0,           0,           0,           1]]]
                        )
            ),
            # # Testing that when hip_JC is composed of lists of ints and vsk values are ints
            (
                {
                    "RTHI": np.array([1, 2, 4]),
                    "LTHI": np.array([-1, 0, 8]),
                    "RKNE": np.array([8, -4, 5]),
                    "LKNE": np.array([8, -8, 5]),
                },
                [
                    [-8,  8, -2],
                    [ 1, -9,  2]
                ],
                {"RightKneeWidth": 9, "LeftKneeWidth": -6},
                np.array([[-5, -5, -9],
                          [3, -6, -5]]),
                [
                    [
                        [1,  2, 4],
                        [1, -9, 2],
                        [8, -4, 5],
                        11.5
                    ], 
                    [
                        [-1, 0,  8],
                        [-8, 8, -2],
                        [ 8, -8, 5],
                        4.0
                    ]
                ],
                np.array([[[-5.6293369,  -4.4458078,  -8.45520089, -5],
                           [-5.62916022, -5.77484544, -8.93858368, -5],
                           [-4.54382845, -5.30411437, -8.16368549, -9],
                           [ 0,           0,           0,           1]],
                          [[ 2.26301154, -6.63098327, -4.75770242,  3],
                           [ 3.2927155,  -5.97483821, -4.04413154, -6],
                           [ 2.39076635, -5.22461171, -4.83384537, -5],
                           [ 0,           0,           0,           1]]]
                        ),
            ),
            # # Testing that when hip_JC is composed of numpy arrays of ints and vsk values are ints
            (
                {
                    "RTHI": np.array([1, 2, 4], dtype="int"),
                    "LTHI": np.array([-1, 0, 8], dtype="int"),
                    "RKNE": np.array([8, -4, 5], dtype="int"),
                    "LKNE": np.array([8, -8, 5], dtype="int"),
                },
                np.array([[-8, 8, -2], [1, -9, 2]], dtype="int"),
                {"RightKneeWidth": 9, "LeftKneeWidth": -6},
                np.array([[-5, -5, -9],
                          [3, -6, -5]]),
                [
                    [
                        [1,  2, 4],
                        [1, -9, 2],
                        [8, -4, 5],
                        11.5
                    ], 
                    [
                        [-1,  0,  8],
                        [-8,  8, -2],
                        [ 8, -8,  5],
                        4.0
                    ]
                ],
                np.array([[[-5.6293369,  -4.4458078,  -8.45520089, -5],
                           [-5.62916022, -5.77484544, -8.93858368, -5],
                           [-4.54382845, -5.30411437, -8.16368549, -9],
                           [ 0,           0,           0,           1]],
                          [[ 2.26301154, -6.63098327, -4.75770242,  3],
                           [ 3.2927155,  -5.97483821, -4.04413154, -6],
                           [ 2.39076635, -5.22461171, -4.83384537, -5],
                           [ 0,           0,           0,           1]]]
                    ),
            ),
            # # Testing that when hip_JC is composed of lists of floats and vsk values are floats
            (
                {
                    "RTHI": np.array([1, 2, 4]),
                    "LTHI": np.array([-1, 0, 8]),
                    "RKNE": np.array([8, -4, 5]),
                    "LKNE": np.array([8, -8, 5]),
                },
                [
                    [-8.0,  8.0, -2.0],
                    [ 1.0, -9.0,  2.0]
                ],
                {"RightKneeWidth": 9.0, "LeftKneeWidth": -6.0},
                np.array([[-5, -5, -9],
                          [ 3, -6, -5]]),
                [
                    [
                        [1,  2, 4],
                        [1, -9, 2],
                        [8, -4, 5],
                        11.5
                    ], 
                    [
                        [-1, 0,  8],
                        [-8, 8, -2],
                        [ 8, -8, 5],
                        4.0
                    ]
                ],
                np.array([[[-5.6293369,  -4.4458078,  -8.45520089, -5],
                           [-5.62916022, -5.77484544, -8.93858368, -5],
                           [-4.54382845, -5.30411437, -8.16368549, -9],
                           [ 0,           0,           0,           1]],
                          [[ 2.26301154, -6.63098327, -4.75770242,  3],
                           [ 3.2927155,  -5.97483821, -4.04413154, -6],
                           [ 2.39076635, -5.22461171, -4.83384537, -5],
                           [ 0,           0,           0,           1]]]
                        )
            ),
            # # Testing that when hip_JC is composed of numpy arrays of floats and vsk values are floats
            (
                {
                    "RTHI": np.array([1.0, 2.0, 4.0], dtype="float"),
                    "LTHI": np.array([-1.0, 0.0, 8.0], dtype="float"),
                    "RKNE": np.array([8.0, -4.0, 5.0], dtype="float"),
                    "LKNE": np.array([8.0, -8.0, 5.0], dtype="float"),
                },
                np.array([[-8.0,  8.0, -2.0],
                          [ 1.0, -9.0,  2.0]], dtype="int"),
                {"RightKneeWidth": 9.0, "LeftKneeWidth": -6.0},
                np.array([[-5, -5, -9],
                          [ 3, -6, -5]]),
                [
                    [
                        [1,  2, 4],
                        [1, -9, 2],
                        [8, -4, 5],
                        11.5
                    ], 
                    [
                        [-1,  0,  8],
                        [-8,  8, -2],
                        [ 8, -8,  5],
                        4.0
                    ]
                ],
                np.array([[[-5.6293369,  -4.4458078,  -8.45520089, -5],
                           [-5.62916022, -5.77484544, -8.93858368, -5],
                           [-4.54382845, -5.30411437, -8.16368549, -9],
                           [ 0,           0,           0,           1]],
                          [[ 2.26301154, -6.63098327, -4.75770242,  3],
                           [ 3.2927155,  -5.97483821, -4.04413154, -6],
                           [ 2.39076635, -5.22461171, -4.83384537, -5],
                           [ 0,           0,           0,           1]]],
                    ),
            ),
        ])
    def test_calc_axis_knee(self, frame, hip_JC, vsk, mock_return_val, expected_mock_args, expected):
        """
        This test provides coverage of the calc_axis_knee function in pycgmStatic.py, defined as calc_axis_knee(frame, hip_JC, delta, vsk)

        This test takes 6 parameters:
        frame: dictionary of marker lists
        hip_JC: array of hip_JC containing the x, y, z positions of the right and left hip joint centers
        vsk: dictionary containing subject measurements from a VSK file
        mock_return_val: the value to be returned by the mock for calc_joint_center
        expected_mock_args: the expected arguments used to call the mocked function, calc_joint_center
        expected: the expected result from calling calc_axis_knee on frame, hip_JC, vsk, and mock_return_val

        This test is checking to make sure the knee joint center and axis are calculated correctly given the input
        parameters. This tests mocks calc_joint_center to make sure the correct parameters are being passed into it given the
        parameters passed into calc_axis_knee, and to also ensure that calc_axis_knee returns the correct value considering
        the return value of calc_joint_center, mock_return_val. 

        For each direction (L or R) D, the D knee joint center is calculated using DTHI, D hip joint center, and 
        DKNE in the Rodriques' rotation formula. The knee width for each knee is applied after the rotation in the formula as well.
        Each knee joint center and the RKNE / LKNE markers are used in the Knee Axis Calculation 
        (ref. Clinical Gait Analysis hand book, Baker2013) calculation formula.

        Lastly, it checks that the resulting output is correct when hip_JC is composed of lists of ints, numpy arrays of
        ints, lists of floats, and numpy arrays of floats and vsk values are ints and floats. The values in frame were
        kept as numpy arrays as lists would cause an error in pyCGM.py line 409 and 429 as lists cannot be subtracted
        by each other:
        thi_kne_R = rthi-rkne
        thi_kne_L = lthi-lkne
        """

        # Get marker position parameters from frame
        rthi = frame["RTHI"] if "RTHI" in frame else None
        lthi = frame["LTHI"] if "LTHI" in frame else None
        rkne = frame["RKNE"] if "RKNE" in frame else None
        lkne = frame["LKNE"] if "LKNE" in frame else None

        r_hip_jc = hip_JC[1]
        l_hip_jc = hip_JC[0]

        # Get measurement parameters from vsk
        rkne_width = vsk["RightKneeWidth"] if "RightKneeWidth" in vsk else None
        lkne_width = vsk["LeftKneeWidth"] if "LeftKneeWidth" in vsk else None

        with patch.object(pycgmStatic, 'calc_joint_center', side_effect=mock_return_val) as mock_find_joint_center:
            result = pycgmStatic.calc_axis_knee(rthi, lthi, rkne, lkne, r_hip_jc, l_hip_jc, rkne_width, lkne_width)

        right_axis = result[0]
        left_axis = result[1]

        # Add back right knee origin
        right_o = right_axis[:3, 3]
        right_axis[0, :3] += right_o
        right_axis[1, :3] += right_o
        right_axis[2, :3] += right_o

        # Add back left knee origin
        left_o = left_axis[:3, 3]
        left_axis[0, :3] += left_o
        left_axis[1, :3] += left_o
        left_axis[2, :3] += left_o

        result = np.asarray([right_axis, left_axis])

        # Asserting that there were only 2 calls to calc_joint_center
        np.testing.assert_equal(mock_find_joint_center.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to calc_joint_center
        np.testing.assert_almost_equal(expected_mock_args[0][0], mock_find_joint_center.call_args_list[0][0][0], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][1], mock_find_joint_center.call_args_list[0][0][1], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][2], mock_find_joint_center.call_args_list[0][0][2], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][3], mock_find_joint_center.call_args_list[0][0][3], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to calc_joint_center
        np.testing.assert_almost_equal(expected_mock_args[1][0], mock_find_joint_center.call_args_list[1][0][0], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][1], mock_find_joint_center.call_args_list[1][0][1], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][2], mock_find_joint_center.call_args_list[1][0][2], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][3], mock_find_joint_center.call_args_list[1][0][3], rounding_precision)

        # Asserting that calc_axis_knee returned the correct result given the return value given by mocked calc_joint_center
        np.testing.assert_almost_equal(result, expected, rounding_precision)


    @pytest.mark.parametrize(["frame", "knee_JC", "vsk", "mock_return_val", "expected_mock_args", "expected"],
        [
            # Test from running sample data
            (
                {
                    "RTIB": np.array([433.97537231, 211.93408203, 273.3008728]),
                    "LTIB": np.array([50.04016495,  235.90718079, 364.32226562]),
                    "RANK": np.array([422.77005005, 217.74053955,  92.86152649]),
                    "LANK": np.array([58.57380676,  208.54806519,  86.16953278]),
                },
                np.array([[364.17774614, 292.17051722, 515.19181496],
                          [143.55478579, 279.90370346, 524.78408753]]),
                {
                    "RightAnkleWidth": 70.0,
                    "LeftAnkleWidth": 70.0,
                    "RightTibialTorsion": 0.0,
                    "LeftTibialTorsion": 0.0,
                },
                [
                    np.array([393.76181608, 247.67829633, 87.73775041]),
                    np.array([ 98.74901939, 219.46930221, 80.6306816]),
                ],
                [
                    [
                        [433.97537231, 211.93408203, 273.3008728],
                        [364.17774614, 292.17051722, 515.19181496],
                        [422.77005005, 217.74053955,  92.86152649],
                        42.0,
                    ],
                    [
                        [ 50.04016495, 235.90718079, 364.32226562],
                        [143.55478579, 279.90370346, 524.78408753],
                        [ 58.57380676, 208.54806519,  86.16953278],
                        42.0,
                    ],
                ],
                np.array([[[394.48171575, 248.37201348, 87.715368,   393.76181608],
                           [393.07114384, 248.39110006, 87.61575574, 247.67829633],
                           [393.69314056, 247.78157916, 88.73002876,  87.73775041],
                           [  0,            0,           0,            1         ]],
                          [[ 98.47494966, 220.42553803, 80.52821783,  98.74901939],
                           [ 97.79246671, 219.20927275, 80.76255901, 219.46930221],
                           [ 98.84848169, 219.60345781, 81.61663775,  80.6306816 ],
                           [  0,            0,           0,            1         ]]]),
            ),
            # Test with zeros for all params
            (
                {
                    "RTIB": np.array([0, 0, 0]),
                    "LTIB": np.array([0, 0, 0]),
                    "RANK": np.array([0, 0, 0]),
                    "LANK": np.array([0, 0, 0]),
                },
                np.array([[0, 0, 0],
                          [0, 0, 0]]),
                {
                    "RightAnkleWidth": 0.0,
                    "LeftAnkleWidth": 0.0,
                    "RightTibialTorsion": 0.0,
                    "LeftTibialTorsion": 0.0,
                },
                [np.array([0, 0, 0]), np.array([0, 0, 0])],
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0],
                ],
                np.array([[[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]],
                          [[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]]]
                        )
            ),
            # # Testing when values are added to frame
            (
                {
                    "RTIB": np.array([-9, 6, -9]),
                    "LTIB": np.array([0, 2, -1]),
                    "RANK": np.array([1, 0, -5]),
                    "LANK": np.array([2, -4, -5]),
                },
                [
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                ],
                {
                    "RightAnkleWidth": 0.0,
                    "LeftAnkleWidth": 0.0,
                    "RightTibialTorsion": 0.0,
                    "LeftTibialTorsion": 0.0,
                },
                [np.array([0, 0, 0]), np.array([0, 0, 0])],
                [
                    [[-9, 6, -9], [0, 0, 0], [1, 0, -5], 7.0],
                    [[0, 2, -1], [0, 0, 0], [2, -4, -5], 7.0],
                ],
                np.array([[[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]],
                          [[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]]]
                        )
            ),
            # # Testing when values are added to knee_JC
            (
                {
                    "RTIB": np.array([0, 0, 0]),
                    "LTIB": np.array([0, 0, 0]),
                    "RANK": np.array([0, 0, 0]),
                    "LANK": np.array([0, 0, 0]),
                },
                [
                    np.array([-7, 1, 2]),
                    np.array([9, -8, 9]),
                ],
                {
                    "RightAnkleWidth": 0.0,
                    "LeftAnkleWidth": 0.0,
                    "RightTibialTorsion": 0.0,
                    "LeftTibialTorsion": 0.0,
                },
                [np.array([0, 0, 0]), np.array([0, 0, 0])],
                [
                    [[0, 0, 0], [-7, 1, 2], [0, 0, 0], 7.0],
                    [[0, 0, 0], [9, -8, 9], [0, 0, 0], 7.0],
                ],
                np.array([[[np.nan,     np.nan,     np.nan,      0],
                           [np.nan,     np.nan,     np.nan,      0],
                           [-0.95257934, 0.13608276, 0.27216553, 0],
                           [ 0,          0,          0,          1]],
                          [[np.nan,      np.nan,     np.nan,     0],
                           [np.nan,      np.nan,     np.nan,     0],
                           [ 0.59867109, -0.53215208, 0.59867109, 0],
                           [ 0,           0,          0,          1]]]
                        )
            ),
            # # Testing when values are added to vsk
            (
                {
                    "RTIB": np.array([0, 0, 0]),
                    "LTIB": np.array([0, 0, 0]),
                    "RANK": np.array([0, 0, 0]),
                    "LANK": np.array([0, 0, 0]),
                },
                [
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                ],
                {
                    "RightAnkleWidth": -38.0,
                    "LeftAnkleWidth": 18.0,
                    "RightTibialTorsion": 29.0,
                    "LeftTibialTorsion": -13.0,
                },
                [np.array([0, 0, 0]), np.array([0, 0, 0])],
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], -12.0],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], 16.0],
                ],
                np.array([[[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]],
                          [[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]]]
                        )
            ),
            # # Testing when values are added to mock_return_val
            (
                {
                    "RTIB": np.array([0, 0, 0]),
                    "LTIB": np.array([0, 0, 0]),
                    "RANK": np.array([0, 0, 0]),
                    "LANK": np.array([0, 0, 0]),
                },
                [
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                ],
                {
                    "RightAnkleWidth": 0.0,
                    "LeftAnkleWidth": 0.0,
                    "RightTibialTorsion": 0.0,
                    "LeftTibialTorsion": 0.0,
                },
                [np.array([2, -5, 4]), np.array([8, -3, 1])],
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0],
                ],
                np.array([[[np.nan,      np.nan,      np.nan,     2],
                           [np.nan,      np.nan,      np.nan,    -5],
                           [ 1.7018576,  -4.25464401, 3.40371521, 4],
                           [ 0,           0,          0,          1]],
                          [[np.nan,      np.nan,     np.nan,      8],
                           [np.nan,      np.nan,     np.nan,     -3],
                           [ 7.07001889, -2.65125708, 0.88375236, 1],
                           [ 0,           0,          0,          1]]]
                        )
            ),
            # # Testing when values are added to frame and knee_JC
            (
                {
                    "RTIB": np.array([-9, 6, -9]),
                    "LTIB": np.array([0, 2, -1]),
                    "RANK": np.array([1, 0, -5]),
                    "LANK": np.array([2, -4, -5]),
                },
                [
                    np.array([-7, 1, 2]),
                    np.array([9, -8, 9]),
                ],
                {
                    "RightAnkleWidth": 0.0,
                    "LeftAnkleWidth": 0.0,
                    "RightTibialTorsion": 0.0,
                    "LeftTibialTorsion": 0.0,
                },
                [np.array([0, 0, 0]), np.array([0, 0, 0])],
                [
                    [[-9, 6, -9], [-7, 1, 2], [1, 0, -5], 7.0],
                    [[0, 2, -1], [9, -8, 9], [2, -4, -5], 7.0],
                ],
                np.array([[[-0.26726124, -0.80178373, -0.53452248, 0],
                           [ 0.14547859, -0.58191437,  0.80013226, 0],
                           [-0.95257934,  0.13608276,  0.27216553, 0],
                           [ 0,           0,           0,          1]],
                          [[ 0.79317435,  0.49803971, -0.35047239, 0],
                           [-0.11165737,  0.68466825,  0.72025136, 0],
                           [ 0.59867109, -0.53215208,  0.59867109, 0],
                           [ 0,           0,           0,          1]]]
                        )
            ),
            # # Testing when values are added to frame, knee_JC, and vsk
            (
                {
                    "RTIB": np.array([-9, 6, -9]),
                    "LTIB": np.array([0, 2, -1]),
                    "RANK": np.array([1, 0, -5]),
                    "LANK": np.array([2, -4, -5]),
                },
                [
                    np.array([-7, 1, 2]),
                    np.array([9, -8, 9]),
                ],
                {
                    "RightAnkleWidth": -38.0,
                    "LeftAnkleWidth": 18.0,
                    "RightTibialTorsion": 29.0,
                    "LeftTibialTorsion": -13.0,
                },
                [np.array([0, 0, 0]), np.array([0, 0, 0])],
                [
                    [[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0],
                    [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0],
                ],
                np.array([[[-0.30428137, -0.41913816, -0.85541572, 0],
                           [-0.00233238, -0.89766624,  0.4406698,  0],
                           [-0.95257934,  0.13608276,  0.27216553, 0],
                           [ 0,           0,           0,          1]],
                          [[ 0.7477279,   0.63929183, -0.1794685,  0],
                           [-0.287221,    0.55508569,  0.7806305,  0],
                           [ 0.59867109, -0.53215208,  0.59867109, 0],
                           [ 0,           0,           0,          1]]]
                        )
                           
            ),
            # # Testing when values are added to frame, knee_JC, vsk and mock_return_val
            (
                {
                    "RTIB": np.array([-9, 6, -9]),
                    "LTIB": np.array([0, 2, -1]),
                    "RANK": np.array([1, 0, -5]),
                    "LANK": np.array([2, -4, -5]),
                },
                [
                    np.array([-7, 1, 2]),
                    np.array([9, -8, 9]),
                ],
                {
                    "RightAnkleWidth": -38.0,
                    "LeftAnkleWidth": 18.0,
                    "RightTibialTorsion": 29.0,
                    "LeftTibialTorsion": -13.0,
                },
                [np.array([2, -5, 4]), np.array([8, -3, 1])],
                [
                    [[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0],
                    [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0],
                ],
                np.array([[[1.48891678, -5.83482493, 3.7953997,   2],
                           [1.73661348, -5.07447603, 4.96181124, -5],
                           [1.18181818, -4.45454545, 3.81818182,  4],
                           [0,           0,          0,           1]],
                          [[8.87317138, -2.54514024, 1.17514093,  8],
                           [7.52412119, -2.28213872, 1.50814815, -3],
                           [8.10540926, -3.52704628, 1.84327404,  1],
                           [0,           0,          0,           1]]]
                        )
            ),
            # # Testing that when knee_JC is composed of lists of ints and vsk values are ints
            (
                {
                    "RTIB": np.array([-9, 6, -9]),
                    "LTIB": np.array([0, 2, -1]),
                    "RANK": np.array([1, 0, -5]),
                    "LANK": np.array([2, -4, -5]),
                },
                [
                    [-7, 1, 2],
                    [9, -8, 9],
                ],
                {
                    "RightAnkleWidth": -38,
                    "LeftAnkleWidth": 18,
                    "RightTibialTorsion": 29,
                    "LeftTibialTorsion": -13,
                },
                [np.array([2, -5, 4]), np.array([8, -3, 1])],
                [
                    [[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0],
                    [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0],
                ],
                np.array([[[1.48891678, -5.83482493, 3.7953997,   2],
                           [1.73661348, -5.07447603, 4.96181124, -5],
                           [1.18181818, -4.45454545, 3.81818182,  4],
                           [0,           0,          0,           1]],
                          [[8.87317138, -2.54514024, 1.17514093,  8],
                           [7.52412119, -2.28213872, 1.50814815, -3],
                           [8.10540926, -3.52704628, 1.84327404,  1],
                           [0,           0,          0,           1]]]
                        )
            ),
            # # Testing that when knee_JC is composed of numpy arrays of ints and vsk values are ints
            (
                {
                    "RTIB": np.array([-9, 6, -9]),
                    "LTIB": np.array([0, 2, -1]),
                    "RANK": np.array([1, 0, -5]),
                    "LANK": np.array([2, -4, -5]),
                },
                [
                    np.array([-7, 1, 2], dtype="int"),
                    np.array([9, -8, 9], dtype="int"),
                ],
                {
                    "RightAnkleWidth": -38,
                    "LeftAnkleWidth": 18,
                    "RightTibialTorsion": 29,
                    "LeftTibialTorsion": -13,
                },
                [np.array([2, -5, 4]), np.array([8, -3, 1])],
                [
                    [[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0],
                    [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0],
                ],
                np.array([[[1.48891678, -5.83482493, 3.7953997,   2],
                           [1.73661348, -5.07447603, 4.96181124, -5],
                           [1.18181818, -4.45454545, 3.81818182,  4],
                           [0,           0,          0,           1]],
                          [[8.87317138, -2.54514024, 1.17514093,  8],
                           [7.52412119, -2.28213872, 1.50814815, -3],
                           [8.10540926, -3.52704628, 1.84327404,  1],
                           [0,           0,          0,           1]]]
                        )
            ),
            # # Testing that when knee_JC is composed of lists of floats and vsk values are floats
            (
                {
                    "RTIB": np.array([-9.0, 6.0, -9.0]),
                    "LTIB": np.array([0.0, 2.0, -1.0]),
                    "RANK": np.array([1.0, 0.0, -5.0]),
                    "LANK": np.array([2.0, -4.0, -5.0]),
                },
                [
                    [-7.0, 1.0, 2.0],
                    [9.0, -8.0, 9.0],
                ],
                {
                    "RightAnkleWidth": -38.0,
                    "LeftAnkleWidth": 18.0,
                    "RightTibialTorsion": 29.0,
                    "LeftTibialTorsion": -13.0,
                },
                [np.array([2, -5, 4]), np.array([8, -3, 1])],
                [
                    [[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0],
                    [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0],
                ],
                np.array([[[1.48891678, -5.83482493, 3.7953997,   2],
                           [1.73661348, -5.07447603, 4.96181124, -5],
                           [1.18181818, -4.45454545, 3.81818182,  4],
                           [0,           0,          0,           1]],
                          [[8.87317138, -2.54514024, 1.17514093,  8],
                           [7.52412119, -2.28213872, 1.50814815, -3],
                           [8.10540926, -3.52704628, 1.84327404,  1],
                           [0,           0,          0,           1]]]
                        )
            ),
            # # Testing that when knee_JC is composed of numpy arrays of floats and vsk values are floats
            (
                {
                    "RTIB": np.array([-9.0, 6.0, -9.0]),
                    "LTIB": np.array([0.0, 2.0, -1.0]),
                    "RANK": np.array([1.0, 0.0, -5.0]),
                    "LANK": np.array([2.0, -4.0, -5.0]),
                },
                [
                    np.array([-7.0, 1.0, 2.0], dtype="float"),
                    np.array([9.0, -8.0, 9.0], dtype="float"),
                ],
                {
                    "RightAnkleWidth": -38.0,
                    "LeftAnkleWidth": 18.0,
                    "RightTibialTorsion": 29.0,
                    "LeftTibialTorsion": -13.0,
                },
                [np.array([2, -5, 4]), np.array([8, -3, 1])],
                [
                    [[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0],
                    [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0],
                ],
                np.array([[[1.48891678, -5.83482493, 3.7953997,   2],
                           [1.73661348, -5.07447603, 4.96181124, -5],
                           [1.18181818, -4.45454545, 3.81818182,  4],
                           [0,           0,          0,           1]],
                          [[8.87317138, -2.54514024, 1.17514093,  8],
                           [7.52412119, -2.28213872, 1.50814815, -3],
                           [8.10540926, -3.52704628, 1.84327404,  1],
                           [0,           0,          0,           1]]]
                        )
            ),
    ])
    def test_calc_axis_ankle(self, frame, knee_JC, vsk, mock_return_val, expected_mock_args, expected):
        """
        This test provides coverage of the calc_axis_ankle function in pycgmStatic.py, defined as
        calc_axis_ankle(rtib, ltib, rank, lank, r_knee_JC, l_knee_JC, rank_width, lank_width, rtib_torsion, ltib_torsion)

        This test takes 6 parameters:
        frame: dictionary of marker lists
        knee_JC: array containing the positions of the right and left knee joint centers
        vsk: dictionary containing subject measurements from a VSK file
        mock_return_val: the value to be returned by the mock for calc_joint_center
        expected_mock_args: the expected arguments used to call the mocked function, calc_joint_center
        expected: the expected result from calling calc_axis_ankle on frame, knee_JC, vsk, and mock_return_val
        This test is checking to make sure the ankle joint center and axis are calculated correctly given the input
        parameters. This tests mocks calc_joint_center to make sure the correct parameters are being passed into it given the
        parameters passed into calc_axis_ankle, and to also ensure that calc_axis_ankle returns the correct value considering
        the return value of calc_joint_center, mock_return_val.
        
        The ankle joint center left and right origin are defined by using the ANK, Tib, and KJC marker positions in the Rodriques' rotation formula.
        The ankle joint center axis is calculated using the Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013).
        Lastly, it checks that the resulting output is correct when knee_JC is composed of lists of ints, numpy arrays
        of ints, lists of floats, and numpy arrays of floats and vsk values are ints and floats. The values in frame
        were kept as numpy arrays as lists would cause an error in pycgmStatic.py line 580 and 596 as lists cannot be
        subtracted by each other:
        tib_ank_R = tib_R-ank_R
        tib_ank_L = tib_L-ank_L
        """

        # Get marker position parameters from frame
        rtib = frame["RTIB"] if "RTIB" in frame else None
        ltib = frame["LTIB"] if "LTIB" in frame else None
        rank = frame["RANK"] if "RANK" in frame else None
        lank = frame["LANK"] if "LANK" in frame else None

        r_knee_jc = knee_JC[0]
        l_knee_jc = knee_JC[1]

        # Get measurement parameters from vsk
        rank_width =   vsk["RightAnkleWidth"]    if "RightAnkleWidth"    in vsk else None
        lank_width =   vsk["LeftAnkleWidth"]     if "LeftAnkleWidth"     in vsk else None
        rtib_torsion = vsk["RightTibialTorsion"] if "RightTibialTorsion" in vsk else None
        ltib_torsion = vsk["LeftTibialTorsion"]  if "LeftTibialTorsion"  in vsk else None

        with patch.object(pycgmStatic, 'calc_joint_center', side_effect=mock_return_val) as mock_calc_joint_center:
            result = pycgmStatic.calc_axis_ankle(rtib, ltib, rank, lank, r_knee_jc, l_knee_jc, rank_width, lank_width, rtib_torsion, ltib_torsion)

        right_axis = result[0]
        left_axis = result[1]

        # Add back right knee origin
        right_o = right_axis[:3, 3]
        right_axis[0, :3] += right_o
        right_axis[1, :3] += right_o
        right_axis[2, :3] += right_o

        # Add back left knee origin
        left_o = left_axis[:3, 3]
        left_axis[0, :3] += left_o
        left_axis[1, :3] += left_o
        left_axis[2, :3] += left_o

        result = np.asarray([right_axis, left_axis])

        # Asserting that there were only 2 calls to calc_joint_center
        np.testing.assert_equal(mock_calc_joint_center.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to calc_joint_center
        np.testing.assert_almost_equal(expected_mock_args[0][0], mock_calc_joint_center.call_args_list[0][0][0], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][1], mock_calc_joint_center.call_args_list[0][0][1], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][2], mock_calc_joint_center.call_args_list[0][0][2], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][3], mock_calc_joint_center.call_args_list[0][0][3], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to calc_joint_center
        np.testing.assert_almost_equal(expected_mock_args[1][0], mock_calc_joint_center.call_args_list[1][0][0], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][1], mock_calc_joint_center.call_args_list[1][0][1], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][2], mock_calc_joint_center.call_args_list[1][0][2], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][3], mock_calc_joint_center.call_args_list[1][0][3], rounding_precision)

        # Asserting that findShoulderJC returned the correct result given the return value given by mocked calc_joint_center
        np.testing.assert_almost_equal(result, expected, rounding_precision)



    @pytest.mark.parametrize(
        ["frame", "expected"],
        [
            # Test from running sample data
            (
                {
                    "LFHD": np.array([184.55158997, 409.68713379, 1721.34289551]),
                    "RFHD": np.array([325.82983398, 402.55450439, 1722.49816895]),
                    "LBHD": np.array([197.8621521, 251.28889465, 1696.90197754]),
                    "RBHD": np.array([304.39898682, 242.91339111, 1694.97497559]),
                },
                np.array([[255.21590218, 407.10741939, 1722.0817318,   255.19071197509766],
                          [254.19105385, 406.14680918, 1721.91767712,  406.1208190917969 ],
                          [255.18370553, 405.95974655, 1722.90744993, 1721.9205322265625 ],
                          [  0,            0,             0,             1               ]]
                        )
            ),
            # Basic test with a variance of 1 in the x and y dimensions of the markers
            (
                {
                    "LFHD": np.array([1, 1, 0]),
                    "RFHD": np.array([0, 1, 0]),
                    "LBHD": np.array([1, 0, 0]),
                    "RBHD": np.array([0, 0, 0]),
                },
                np.array([[0.5, 2,  0, 0.5],
                          [1.5, 1,  0, 1  ],
                          [0.5, 1, -1, 0  ],
                          [0,   0,  0, 1  ]])

            ),
            # Setting the markers so there's no variance in the x-dimension
            (
                {
                    "LFHD": np.array([0, 1, 0]),
                    "RFHD": np.array([0, 1, 0]),
                    "LBHD": np.array([0, 0, 0]),
                    "RBHD": np.array([0, 0, 0]),
                },
                np.array([[0, 1, 0, 0],
                          [0, 1, 0, 1],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1]]
                        )
            ),
            # Setting the markers so there's no variance in the y-dimension
            (
                {
                    "LFHD": np.array([1, 0, 0]),
                    "RFHD": np.array([0, 0, 0]),
                    "LBHD": np.array([1, 0, 0]),
                    "RBHD": np.array([0, 0, 0]),
                },
                np.array([[0.5, 0, 0, 0.5],
                          [0.5, 0, 0, 0  ],
                          [0.5, 0, 0, 0  ],
                          [0,   0, 0, 1  ]]
                        )
            ),
            # Setting each marker in a different xy quadrant
            (
                {
                    "LFHD": np.array([-1, 1, 0]),
                    "RFHD": np.array([1, 1, 0]),
                    "LBHD": np.array([-1, -1, 0]),
                    "RBHD": np.array([1, -1, 0]),
                },
                np.array([[ 0, 2, 0, 0],
                          [-1, 1, 0, 1],
                          [ 0, 1, 1, 0],
                          [ 0, 0, 0, 1]]
                        )
            ),
            # Setting values of the markers so that midpoints will be on diagonals
            (
                {
                    "LFHD": np.array([-2, 1, 0]),
                    "RFHD": np.array([1, 2, 0]),
                    "LBHD": np.array([-1, -2, 0]),
                    "RBHD": np.array([2, -1, 0]),
                },
                np.array([[-0.81622777, 2.4486833,  0, -0.5],
                          [-1.4486833,  1.18377223, 0,  1.5],
                          [-0.5,        1.5,        1,  0  ],
                          [ 0,          0,          0,  1  ]]
                        )
            ),
            # Adding the value of 1 in the z dimension for all 4 markers
            (
                {
                    "LFHD": np.array([1, 1, 1]),
                    "RFHD": np.array([0, 1, 1]),
                    "LBHD": np.array([1, 0, 1]),
                    "RBHD": np.array([0, 0, 1]),
                },
                np.array([[0.5, 2, 1, 0.5],
                          [1.5, 1, 1, 1  ],
                          [0.5, 1, 0, 1  ],
                          [0,   0, 0, 1  ]])
            ),
            # Setting the z dimension value higher for LFHD and LBHD
            (
                {
                    "LFHD": np.array([1, 1, 2]),
                    "RFHD": np.array([0, 1, 1]),
                    "LBHD": np.array([1, 0, 2]),
                    "RBHD": np.array([0, 0, 1]),
                },
                np.array([[0.5,        2, 1.5,        0.5],
                          [1.20710678, 1, 2.20710678, 1  ],
                          [1.20710678, 1, 0.79289322, 1.5],
                          [0,          0, 0,          1 ]]
                        )
            ),
            # Setting the z dimension value higher for LFHD and RFHD
            (
                {
                    "LFHD": np.array([1, 1, 2]),
                    "RFHD": np.array([0, 1, 2]),
                    "LBHD": np.array([1, 0, 1]),
                    "RBHD": np.array([0, 0, 1]),
                },
                np.array([[0.5, 1.70710678, 2.70710678, 0.5],
                          [1.5, 1,          2,          1  ],
                          [0.5, 1.70710678, 1.29289322, 2  ],
                          [0,   0,          0,          1  ]]
                        )
            ),
            # Testing that when frame is composed of lists of ints
            (
                {
                    "LFHD": [1, 1, 2],
                    "RFHD": [0, 1, 2],
                    "LBHD": [1, 0, 1],
                    "RBHD": [0, 0, 1],
                },
                np.array([[0.5, 1.70710678, 2.70710678, 0.5],
                          [1.5, 1,          2,          1  ],
                          [0.5, 1.70710678, 1.29289322, 2  ],
                          [0,   0,          0,          1  ]]
                        )
            ),
            # Testing that when frame is composed of numpy arrays of ints
            (
                {
                    "LFHD": np.array([1, 1, 2], dtype="int"),
                    "RFHD": np.array([0, 1, 2], dtype="int"),
                    "LBHD": np.array([1, 0, 1], dtype="int"),
                    "RBHD": np.array([0, 0, 1], dtype="int"),
                },
                np.array([[0.5, 1.70710678, 2.70710678, 0.5],
                          [1.5, 1,          2,          1  ],
                          [0.5, 1.70710678, 1.29289322, 2  ],
                          [0,   0,          0,          1  ]]
                        )
            ),
            # Testing that when frame is composed of lists of floats
            (
                {
                    "LFHD": [1.0, 1.0, 2.0],
                    "RFHD": [0.0, 1.0, 2.0],
                    "LBHD": [1.0, 0.0, 1.0],
                    "RBHD": [0.0, 0.0, 1.0],
                },
                np.array([[0.5, 1.70710678, 2.70710678, 0.5],
                          [1.5, 1,          2,          1  ],
                          [0.5, 1.70710678, 1.29289322, 2  ],
                          [0,   0,          0,          1  ]]
                        )
            ),
            # Testing that when frame is composed of numpy arrays of floats
            (
                {
                    "LFHD": np.array([1.0, 1.0, 2.0], dtype="float"),
                    "RFHD": np.array([0.0, 1.0, 2.0], dtype="float"),
                    "LBHD": np.array([1.0, 0.0, 1.0], dtype="float"),
                    "RBHD": np.array([0.0, 0.0, 1.0], dtype="float"),
                },
                np.array([[0.5, 1.70710678, 2.70710678, 0.5],
                          [1.5, 1,          2,          1  ],
                          [0.5, 1.70710678, 1.29289322, 2  ],
                          [0,   0,          0,          1  ]]
                        )
            ),
        ],
    )
    def test_calc_axis_head(self, frame, expected):
        """
        This test provides coverage of the calc_axis_head function in pycgmStatic.py, defined as 
        calc_axis_head(lfhd, rfhd, lbhd, rbhd)

        This test takes 3 parameters:
        frame: dictionary of marker lists
        expected: the expected result from calling calc_axis_head on lfhd, rfhd, lbhd and rbhd

        This test is checking to make sure the head joint center and head joint axis are calculated correctly given
        the 4 coordinates given in frame. This includes testing when there is no variance in the coordinates,
        when the coordinates are in different quadrants, when the midpoints will be on diagonals, and when the z
        dimension is variable.

        The function uses the LFHD, RFHD, LBHD, and RBHD markers from the frame to calculate the midpoints of the 
        front, back, left, and right center positions of the head. 
        The head axis vector components are then calculated using the aforementioned midpoints.
        Finally, the axes are made orthogonal by calculating the cross product of each individual axis. 

        Lastly, it checks that the resulting output is correct when frame composed of lists of ints, numpy arrays of
        ints, lists of floats, and numpy arrays of floats.
        """
        lfhd = frame["LFHD"] if "LFHD" in frame else None
        rfhd = frame["RFHD"] if "RFHD" in frame else None
        lbhd = frame["LBHD"] if "LBHD" in frame else None
        rbhd = frame["RBHD"] if "RBHD" in frame else None

        result = pycgmStatic.calc_axis_head(lfhd, rfhd, lbhd, rbhd)

        result_o = result[:3, 3]
        result[0, :3] += result_o
        result[1, :3] += result_o
        result[2, :3] += result_o

        np.testing.assert_almost_equal(result, expected, rounding_precision)


    @pytest.mark.parametrize(
        ["frame", "ankle_axis", "expected"],
        [
            # Test from running sample data
            (
                {
                    "RTOE": np.array([433.33508301, 354.97229004, 44.27765274]),
                    "LTOE": np.array([31.77310181, 331.23657227, 42.15322876]),
                },
                np.array([[[  0,            0,           0,          397.45738291],
                           [396.73749179, 218.18875543, 87.69979179, 217.50712216],
                           [  0,            0,           0,           87.83068433],
                           [  0,            0,           0,            1         ]],
                          [[  0,            0,           0,          112.28082818],
                           [111.34886681, 175.49163538, 81.10789314, 175.83265027],
                           [  0,            0,           0,           80.98477997],
                           [  0,            0,           0,            1         ]]]
                        ),
                np.array([[[433.4256618315962,  355.25152027652007, 45.233595181827035, 433.33508301],
                           [432.36890500826763, 355.2296456773885,  44.29402798451682,  354.97229004],
                           [433.09363829389764, 354.0471962330562,  44.570749823731354,  44.27765274],
                           [   0,                 0,                 0,                   1         ]],
                          [[31.806110207058808, 331.49492345678016, 43.11871573923792,  31.77310181],
                           [30.880216288550965, 330.81014854432254, 42.29786022762896, 331.23657227],
                           [32.2221740692973,   330.36972887034574, 42.36983123198873,  42.15322876],
                           [ 0,                   0,                 0,                  1         ]]]),

            ),
            # Test with zeros for all params
            (
                {"RTOE": np.array([0, 0, 0]), "LTOE": np.array([0, 0, 0])},
                np.array([[[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]],
                          [[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]]]
                        ),
                np.array([[[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]],
                          [[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]]]
                        ),
            ),
            # Testing when values are added to frame['RTOE']
            (
                {"RTOE": np.array([-7, 3, -8]), "LTOE": np.array([0, 0, 0])},
                np.array([[[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]],
                          [[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]]]
                        ),
                np.array([[[np.nan,           np.nan,              np.nan,             -7],
                           [np.nan,           np.nan,              np.nan,              3],
                           [-6.36624977770237, 2.7283927618724446, -7.275714031659851, -8],
                           [ 0,                0,                   0,                  1]],
                          [[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]]]
                        )
            ),
            # Testing when values are added to frame['LTOE']
            (
                {"RTOE": np.array([0, 0, 0]), "LTOE": np.array([8, 0, -8])},
                np.array([[[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]],
                          [[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]]]
                        ),
                np.array([[[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]],
                          [[np.nan,            np.nan, np.nan,              8],
                           [np.nan,            np.nan, np.nan,              0],
                           [ 7.292893218813452, 0.0,   -7.292893218813452, -8],
                           [ 0,                 0,      0,                  1]]]
                        )
            ),
            # Testing when values are added to frame
            (
                {"RTOE": np.array([-7, 3, -8]), "LTOE": np.array([8, 0, -8])},
                np.array([[[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]],
                          [[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]]]
                        ),
                np.array([[[np.nan,           np.nan,              np.nan,             -7],
                           [np.nan,           np.nan,              np.nan,              3],
                           [-6.36624977770237, 2.7283927618724446, -7.275714031659851, -8],
                           [ 0,                0,                   0,                  1]],
                          [[np.nan,            np.nan, np.nan,              8],
                           [np.nan,            np.nan, np.nan,              0],
                           [ 7.292893218813452, 0.0,   -7.292893218813452, -8],
                           [ 0,                 0,      0,                  1]]]
                        )
            ),
            # Testing when values are added to right ankle origin
            (
                {"RTOE": np.array([0, 0, 0]), "LTOE": np.array([0, 0, 0])},
                np.array([[[0, 0, 0,  2],
                           [0, 0, 0, -9],
                           [0, 0, 0,  1],
                           [0, 0, 0,  1]],
                          [[0, 0, 0,  0],
                           [0, 0, 0,  0],
                           [0, 0, 0,  0],
                           [0, 0, 0,  1]]]
                        ),
                np.array([[[np.nan,               np.nan,             np.nan,               0],
                           [np.nan,               np.nan,             np.nan,               0],
                           [ 0.21566554640687682, -0.9704949588309457, 0.10783277320343841, 0],
                           [ 0,                    0,                  0,                   1]],
                          [[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]]]
                        ),
            ),
            # Testing when values are added to left ankle origin
            (
                {"RTOE": np.array([0, 0, 0]), "LTOE": np.array([0, 0, 0])},
                np.array([[[0, 0, 0,  0],
                           [0, 0, 0,  0],
                           [0, 0, 0,  0],
                           [0, 0, 0,  1]],
                          [[0, 0, 0,  3],
                           [0, 0, 0, -7],
                           [0, 0, 0,  4],
                           [0, 0, 0,  1]]]
                        ),
                np.array([[[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]],
                          [[np.nan,               np.nan,            np.nan,               0],
                           [np.nan,               np.nan,            np.nan,               0],
                           [ 0.34874291623145787, -0.813733471206735, 0.46499055497527714, 0],
                           [ 0,                    0,                 0,                   1]]]
                        ),
            ),
            # Testing when values are added to ankle y-axes
            (
                {"RTOE": np.array([0, 0, 0]), "LTOE": np.array([0, 0, 0])},
                np.array([[[ 0,  0, 0, 0],
                           [ 8, -4, 2, 0],
                           [ 0,  0, 0, 0],
                           [ 0,  0, 0, 1]],
                          [[ 0,  0, 0, 0],
                           [-9,  7, 4, 0],
                           [ 0,  0, 0, 0],
                           [ 0,  0, 0, 1]]]
                        ),
                np.array([[[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]],
                          [[np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [np.nan, np.nan, np.nan, 0],
                           [ 0,      0,      0,     1]]]
                        ),
            ),
            # Testing when values are added to ankle_axis
            (
                {"RTOE": np.array([0, 0, 0]), "LTOE": np.array([0, 0, 0])},
                np.array([[[ 0,  0, 0,  2],
                           [ 8, -4, 2, -9],
                           [ 0,  0, 0,  1],
                           [ 0,  0, 0,  1]],
                          [[ 0,  0, 0,  3],
                           [-9,  7, 4, -7],
                           [ 0,  0, 0,  4],
                           [ 0,  0, 0,  1]]]
                        ),
                np.array([[[0.21329967236760183, -0.06094276353360052, -0.9750842165376084,  0],
                           [0.9528859437838807,   0.23329276554708803,  0.1938630023560309,  0],
                           [0.21566554640687682, -0.9704949588309457,   0.10783277320343841, 0],
                           [0,                    0,                    0,                   1]],
                          [[ 0.6597830814767823,   0.5655283555515277, 0.4948373111075868,  0],
                           [-0.6656310267523443,   0.1342218942833945, 0.7341115850601987,  0],
                           [ 0.34874291623145787, -0.813733471206735,  0.46499055497527714, 0],
                           [ 0,                    0,                  0,                   1]]]),

            ),
            # Testing when values are added to frame and ankle_axis
            (
                {"RTOE": np.array([-7, 3, -8]), "LTOE": np.array([8, 0, -8])},
                np.array([[[ 0,  0, 0,  2],
                           [ 8, -4, 2, -9],
                           [ 0,  0, 0,  1],
                           [ 0,  0, 0,  1]],
                          [[ 0,  0, 0,  3],
                           [-9,  7, 4, -7],
                           [ 0,  0, 0,  4],
                           [ 0,  0, 0,  1]]]
                        ),
                np.array([[[-6.586075309097216, 2.6732173492872757, -8.849634891853084, -7],
                           [-6.249026985898898, 3.6500960420576702, -7.884178291357542,  3],
                           [-6.485504244572473, 2.3140056594299647, -7.485504244572473, -8],
                           [ 0,                 0,                   0,                  1]],
                          [[8.623180382731631,   0.5341546137699694,  -7.428751315829338,  8],
                           [7.295040915019964,   0.6999344300621451,  -7.885437867872096,  0],
                           [7.6613572692607015, -0.47409982303501746, -7.187257446225685, -8],
                           [0,                   0,                    0,                  1]]]),

            ),
            # Testing that when frame and ankle_axis are composed of lists of ints
            (
                {"RTOE": [-7, 3, -8], "LTOE": [8, 0, -8]},
                [
                    [[ 0,  0, 0,  2],
                     [ 8, -4, 2, -9],
                     [ 0,  0, 0,  1],
                     [ 0,  0, 0,  1]],
                    [[ 0,  0, 0,  3],
                     [-9,  7, 4, -7],
                     [ 0,  0, 0,  4],
                     [ 0,  0, 0,  1]]
                ],
                np.array([[[-6.586075309097216, 2.6732173492872757, -8.849634891853084, -7],
                           [-6.249026985898898, 3.6500960420576702, -7.884178291357542,  3],
                           [-6.485504244572473, 2.3140056594299647, -7.485504244572473, -8],
                           [ 0,                 0,                   0,                  1]],
                          [[8.623180382731631,   0.5341546137699694,  -7.428751315829338,  8],
                           [7.295040915019964,   0.6999344300621451,  -7.885437867872096,  0],
                           [7.6613572692607015, -0.47409982303501746, -7.187257446225685, -8],
                           [0,                   0,                    0,                  1]]]),
            ),
            # Testing that when frame and ankle_axis are composed of numpy arrays of ints
            (
                {
                    "RTOE": np.array([-7, 3, -8], dtype="int"),
                    "LTOE": np.array([8, 0, -8], dtype="int"),
                },
                np.array([[[ 0,  0, 0,  2],
                           [ 8, -4, 2, -9],
                           [ 0,  0, 0,  1],
                           [ 0,  0, 0,  1]],
                          [[ 0,  0, 0,  3],
                           [-9,  7, 4, -7],
                           [ 0,  0, 0,  4],
                           [ 0,  0, 0,  1]]], dtype="int"),
                np.array([[[-6.586075309097216, 2.6732173492872757, -8.849634891853084, -7],
                           [-6.249026985898898, 3.6500960420576702, -7.884178291357542,  3],
                           [-6.485504244572473, 2.3140056594299647, -7.485504244572473, -8],
                           [ 0,                 0,                   0,                  1]],
                          [[8.623180382731631,   0.5341546137699694,  -7.428751315829338,  8],
                           [7.295040915019964,   0.6999344300621451,  -7.885437867872096,  0],
                           [7.6613572692607015, -0.47409982303501746, -7.187257446225685, -8],
                           [0,                   0,                    0,                  1]]]),
            ),
            # Testing that when frame and ankle_axis are composed of lists of floats
            (
                {"RTOE": [-7.0, 3.0, -8.0], "LTOE": [8.0, 0.0, -8.0]},
                [
                    [[ 0.0,  0.0, 0.0,  2.0],
                     [ 8.0, -4.0, 2.0, -9.0],
                     [ 0.0,  0.0, 0.0,  1.0],
                     [ 0.0,  0.0, 0.0,  1.0]],
                    [[ 0.0,  0.0, 0.0,  3.0],
                     [-9.0,  7.0, 4.0, -7.0],
                     [ 0.0,  0.0, 0.0,  4.0],
                     [ 0.0,  0.0, 0.0,  1.0]]
                ],
                np.array([[[-6.586075309097216, 2.6732173492872757, -8.849634891853084, -7],
                           [-6.249026985898898, 3.6500960420576702, -7.884178291357542,  3],
                           [-6.485504244572473, 2.3140056594299647, -7.485504244572473, -8],
                           [ 0,                 0,                   0,                  1]],
                          [[8.623180382731631,   0.5341546137699694,  -7.428751315829338,  8],
                           [7.295040915019964,   0.6999344300621451,  -7.885437867872096,  0],
                           [7.6613572692607015, -0.47409982303501746, -7.187257446225685, -8],
                           [0,                   0,                    0,                  1]]]),
            ),
            # Testing that when frame and ankle_axis are composed of numpy arrays of floats
            (
                {
                    "RTOE": np.array([-7.0, 3.0, -8.0], dtype="float"),
                    "LTOE": np.array([8.0, 0.0, -8.0], dtype="float"),
                },
                np.array([[[ 0.0,  0.0, 0.0,  2.0],
                           [ 8.0, -4.0, 2.0, -9.0],
                           [ 0.0,  0.0, 0.0,  1.0],
                           [ 0.0,  0.0, 0.0,  1.0]],
                          [[ 0.0,  0.0, 0.0,  3.0],
                           [-9.0,  7.0, 4.0, -7.0],
                           [ 0.0,  0.0, 0.0,  4.0],
                           [ 0.0,  0.0, 0.0,  1.0]]], dtype="float"),
                np.array([[[-6.586075309097216, 2.6732173492872757, -8.849634891853084, -7],
                           [-6.249026985898898, 3.6500960420576702, -7.884178291357542,  3],
                           [-6.485504244572473, 2.3140056594299647, -7.485504244572473, -8],
                           [ 0,                 0,                   0,                  1]],
                          [[8.623180382731631,   0.5341546137699694,  -7.428751315829338,  8],
                           [7.295040915019964,   0.6999344300621451,  -7.885437867872096,  0],
                           [7.6613572692607015, -0.47409982303501746, -7.187257446225685, -8],
                           [0,                   0,                    0,                  1]]]),
            ),
        ],
    )
    def test_calc_axis_uncorrect_foot(self, frame, ankle_axis, expected):
        """
        This test provides coverage of the calc_axis_uncorrect_foot function in pycgmStatic.py,
        defined as calc_axis_uncorrect_foot(frame, ankle_axis)

        This test takes 3 parameters:
        frame: dictionaries of marker lists.
        ankle_axis: array of two 4x4 affine matrices representing the right and left ankle axes and origins.

        expected: the expected result from calling calc_axis_uncorrect_foot on frame and ankle_axis, which should be the
        anatomically incorrect foot axis

        Given a marker RTOE and the ankle JC, the right anatomically incorrect foot axis is calculated with:

        .. math::
            R = [R_x + ROrigin_x, R_y + ROrigin_y, R_z + ROrigin_z]

        where :math:`ROrigin_x` is the x coor of the foot axis's origin gotten from frame['RTOE']

        :math:`R_x` is the unit vector of :math:`Yflex_R \times R_z`

        :math:`R_y` is the unit vector of :math:`R_z \times R_x`

        :math:`R_z` is the unit vector of the axis from right toe to right ankle JC

        :math:`Yflex_R` is the unit vector of the axis from right ankle flexion to right ankle JC

        The same calculation applies for the left anatomically incorrect foot axis by replacing all the right values
        with left values

        This unit test ensures that:
        - the markers for RTOE and LTOE only effect either the right or the left axis
        - ankle_axis_R and ankle_axis_L only effect either the right or the left axis
        - the resulting output is correct when frame and ankle_axis are composed of lists of ints,
        numpy arrays of ints, lists of floats, and numpy arrays of floats.

        """
        rtoe = frame["RTOE"] if "RTOE" in frame else None
        ltoe = frame["LTOE"] if "LTOE" in frame else None

        r_ankle_axis = np.asarray(ankle_axis[0])
        right_o = r_ankle_axis[:3, 3]
        r_ankle_axis[0, :3] -= right_o
        r_ankle_axis[1, :3] -= right_o
        r_ankle_axis[2, :3] -= right_o

        l_ankle_axis = np.asarray(ankle_axis[1])
        left_o = l_ankle_axis[:3, 3]
        l_ankle_axis[0, :3] -= left_o
        l_ankle_axis[1, :3] -= left_o
        l_ankle_axis[2, :3] -= left_o

        result = pycgmStatic.calc_axis_uncorrect_foot(rtoe, ltoe, r_ankle_axis, l_ankle_axis)
        right_axis = result[0]
        left_axis = result[1]

        # Add back right ankle origin
        right_o = right_axis[:3, 3]
        right_axis[0, :3] += right_o
        right_axis[1, :3] += right_o
        right_axis[2, :3] += right_o

        # Add back left ankle origin
        left_o = left_axis[:3, 3]
        left_axis[0, :3] += left_o
        left_axis[1, :3] += left_o
        left_axis[2, :3] += left_o

        result = np.asarray([right_axis, left_axis])

        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["frame", "ankle_JC", "vsk", "expected"], [
        # Test from running sample data
        ({'RHEE': [374.01257324, 181.57929993, 49.50960922],
          'LHEE': [105.30126953, 180.2130127, 47.15660858],
          'RTOE': [442.81997681, 381.62280273, 42.66047668],
          'LTOE': [39.43652725, 382.44522095, 41.78911591]},
         [np.array([393.76181608, 247.67829633, 87.73775041]),
          np.array([98.74901939, 219.46930221, 80.6306816]),
          [[np.array(rand_coor), np.array([393.07114384, 248.39110006, 87.61575574]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([97.79246671, 219.20927275, 80.76255901]), np.array(rand_coor)]]],
         {'RightSoleDelta': 0.45, 'LeftSoleDelta': 0.45},
         [np.array([442.81997681, 381.62280273, 42.66047668]),
          np.array([ 39.43652725, 382.44522095, 41.78911591]),
          np.array([[[442.30666241, 381.79936348, 43.50031871],
                     [442.02580128, 381.89596909, 42.1176458 ],
                     [442.49471759, 380.67717784, 42.66047668]],
                    [[39.14565179, 382.3504861, 42.74117514],
                     [38.53126992, 382.15038888, 41.48320216],
                     [39.74620554, 381.49437955, 41.78911591]]])]),
        # Testing with zeros for all params
        ({'RHEE': [0, 0, 0], 'LHEE': [0, 0, 0], 'RTOE': [0, 0, 0], 'LTOE': [0, 0, 0]},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         {'RightSoleDelta': 0.0, 'LeftSoleDelta': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values for frame
        ({'RHEE': [1, -4, -9], 'LHEE': [2, -3, -1], 'RTOE': [1, 4, -6], 'LTOE': [4, 2, 2]},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         {'RightSoleDelta': 0.0, 'LeftSoleDelta': 0.0},
         [np.array([1, 4, -6]), np.array([4, 2, 2]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values for ankleJC
        ({'RHEE': [0, 0, 0], 'LHEE': [0, 0, 0], 'RTOE': [0, 0, 0], 'LTOE': [0, 0, 0]},
         [np.array([-5, -5, -1]), np.array([5, 7, 1]),
          [[np.array(rand_coor), np.array([9, 3, 7]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([-9, 2, 9]), np.array(rand_coor)]]],
         {'RightSoleDelta': 0.0, 'LeftSoleDelta': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values for vsk
        ({'RHEE': [0, 0, 0], 'LHEE': [0, 0, 0], 'RTOE': [0, 0, 0], 'LTOE': [0, 0, 0]},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values for frame and ankleJC
        ({'RHEE': [1, -4, -9], 'LHEE': [2, -3, -1], 'RTOE': [1, 4, -6], 'LTOE': [4, 2, 2]},
         [np.array([-5, -5, -1]), np.array([5, 7, 1]),
          [[np.array(rand_coor), np.array([9, 3, 7]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([-9, 2, 9]), np.array(rand_coor)]]],
         {'RightSoleDelta': 0.0, 'LeftSoleDelta': 0.0},
         [np.array([1, 4, -6]), np.array([4, 2, 2]),
          np.array([[[1.4961389383568338, 4.0, -6.868243142124459], [1.8682431421244592, 4.0, -5.503861061643166], [1.0, 3.0, -6.0]],
                    [[4.541530361073883, 1.783387855570447, 2.8122955416108235], [3.245802523504333, 2.301678990598267, 2.5832460484899826], [3.6286093236458963, 1.0715233091147407, 2.0]]])]),
        # Testing with values for frame and vsk
        ({'RHEE': [1, -4, -9], 'LHEE': [2, -3, -1], 'RTOE': [1, 4, -6], 'LTOE': [4, 2, 2]},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
         [np.array([1, 4, -6]), np.array([4, 2, 2]),
          np.array([[[0.0, 4.0, -6.0], [1.0, 4.0, -7.0], [1.0, 3.0, -6.0]],
                    [[3.071523309114741, 2.3713906763541037, 2.0], [4.0, 2.0, 1.0], [3.6286093236458963, 1.0715233091147407, 2.0]]])]),
        # Testing with values for ankleJC and vsk
        ({'RHEE': [0, 0, 0], 'LHEE': [0, 0, 0], 'RTOE': [0, 0, 0], 'LTOE': [0, 0, 0]},
         [np.array([-5, -5, -1]), np.array([5, 7, 1]),
          [[np.array(rand_coor), np.array([9, 3, 7]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([-9, 2, 9]), np.array(rand_coor)]]],
         {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values for frame, ankleJC, and vsk
        ({'RHEE': [1, -4, -9], 'LHEE': [2, -3, -1], 'RTOE': [1, 4, -6], 'LTOE': [4, 2, 2]},
         [np.array([-5, -5, -1]), np.array([5, 7, 1]),
          [[np.array(rand_coor), np.array([9, 3, 7]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([-9, 2, 9]), np.array(rand_coor)]]],
         {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
         [np.array([1, 4, -6]), np.array([4, 2, 2]),
          np.array([[[1.465329458584979, 4.0, -6.885137557090992], [1.8851375570909927, 4.0, -5.534670541415021], [1.0, 3.0, -6.0]],
                    [[4.532940727667331, 1.7868237089330676, 2.818858992574645], [3.2397085122726565, 2.304116595090937, 2.573994730184553], [3.6286093236458963, 1.0715233091147405, 2.0]]])]),
        # Testing with differing values for right: RHEE, RTOE, ankle_JC[0], and ankle_JC[2][0][1]
        ({'RHEE': [7, 3, 2], 'LHEE': [2, -3, -1], 'RTOE': [3, 7, -2], 'LTOE': [4, 2, 2]},
         [np.array([-8, 9, 9]), np.array([5, 7, 1]),
          [[np.array(rand_coor), np.array([-4, -2, -1]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([-9, 2, 9]), np.array(rand_coor)]]],
         {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
         [np.array([3, 7, -2]), np.array([4, 2, 2]),
          np.array([[[2.3588723285123567, 6.358872328512357, -1.578205479284445], [2.7017462341347014, 6.701746234134702, -2.9066914482305077], [3.7071067811865475, 6.292893218813452, -2.0]],
                    [[4.532940727667331, 1.7868237089330676, 2.818858992574645], [3.2397085122726565, 2.304116595090937, 2.573994730184553], [3.6286093236458963, 1.0715233091147405, 2.0]]])]),
        # Testing with differing values for right: LHEE, LTOE, ankle_JC[1], and ankle_JC[2][1][1]
        ({'RHEE': [1, -4, -9], 'LHEE': [5, -4, -7], 'RTOE': [1, 4, -6], 'LTOE': [5, 3, 4]},
         [np.array([-5, -5, -1]), np.array([-4, 0, -9]),
          [[np.array(rand_coor), np.array([9, 3, 7]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([6, 2, 6]), np.array(rand_coor)]]],
         {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
         [np.array([1, 4, -6]), np.array([5, 3, 4]),
          np.array([[[1.465329458584979, 4.0, -6.885137557090992], [1.8851375570909927, 4.0, -5.534670541415021], [1.0, 3.0, -6.0]],
                    [[5.828764328538102, 3.0, 3.4404022089546915], [5.5595977910453085, 3.0, 4.828764328538102], [5.0, 2.0, 4.0]]])]),
        # Testing that when thorax and ankle_JC are lists of ints and vsk values are ints
        ({'RHEE': [1, -4, -9], 'LHEE': [2, -3, -1], 'RTOE': [1, 4, -6], 'LTOE': [4, 2, 2]},
         [[-5, -5, -1], [5, 7, 1],
          [[rand_coor, [9, 3, 7], rand_coor],
           [rand_coor, [-9, 2, 9], rand_coor]]],
         {'RightSoleDelta': 1, 'LeftSoleDelta': -1},
         [np.array([1, 4, -6]), np.array([4, 2, 2]),
          np.array([[[1.4472135954999579, 4.0, -6.8944271909999157], [1.894427190999916, 4.0, -5.5527864045000417], [1.0, 3.0, -6.0]],
                    [[4.5834323811883104, 1.7666270475246759, 2.7779098415844139], [3.2777288444786272, 2.2889084622085494, 2.6283759053035944], [3.6286093236458963, 1.0715233091147407, 2.0]]])]),
        # Testing that when thorax and ankle_JC are numpy arrays of ints and vsk values are ints
        ({'RHEE': np.array([1, -4, -9], dtype='int'), 'LHEE': np.array([2, -3, -1], dtype='int'),
          'RTOE': np.array([1, 4, -6], dtype='int'), 'LTOE': np.array([4, 2, 2], dtype='int')},
         [np.array([-5, -5, -1], dtype='int'), np.array([5, 7, 1], dtype='int'),
          [np.array([rand_coor, [9, 3, 7], rand_coor], dtype='int'),
           np.array([rand_coor, [-9, 2, 9], rand_coor], dtype='int')]],
         {'RightSoleDelta': 1, 'LeftSoleDelta': -1},
         [np.array([1, 4, -6]), np.array([4, 2, 2]),
          np.array([[[1.4472135954999579, 4.0, -6.8944271909999157], [1.894427190999916, 4.0, -5.5527864045000417], [1.0, 3.0, -6.0]],
                    [[4.5834323811883104, 1.7666270475246759, 2.7779098415844139], [3.2777288444786272, 2.2889084622085494, 2.6283759053035944], [3.6286093236458963, 1.0715233091147407, 2.0]]])]),
        # Testing that when thorax and ankle_JC are lists of floats and vsk values are floats
        ({'RHEE': [1.0, -4.0, -9.0], 'LHEE': [2.0, -3.0, -1.0], 'RTOE': [1.0, 4.0, -6.0], 'LTOE': [4.0, 2.0, 2.0]},
         [[-5.0, -5.0, -1.0], [5.0, 7.0, 1.0],
          [[rand_coor, [9.0, 3.0, 7.0], rand_coor],
           [rand_coor, [-9.0, 2.0, 9.0], rand_coor]]],
         {'RightSoleDelta': 1.0, 'LeftSoleDelta': -1.0},
         [np.array([1, 4, -6]), np.array([4, 2, 2]),
          np.array([[[1.4472135954999579, 4.0, -6.8944271909999157], [1.894427190999916, 4.0, -5.5527864045000417], [1.0, 3.0, -6.0]],
                    [[4.5834323811883104, 1.7666270475246759, 2.7779098415844139], [3.2777288444786272, 2.2889084622085494, 2.6283759053035944], [3.6286093236458963, 1.0715233091147407, 2.0]]])]),
        # Testing that when thorax and ankle_JC are numpy arrays of floats and vsk values are floats
        ({'RHEE': np.array([1.0, -4.0, -9.0], dtype='float'), 'LHEE': np.array([2.0, -3.0, -1.0], dtype='float'),
          'RTOE': np.array([1.0, 4.0, -6.0], dtype='float'), 'LTOE': np.array([4.0, 2.0, 2.0], dtype='float')},
         [np.array([-5.0, -5.0, -1.0], dtype='float'), np.array([5.0, 7.0, 1.0], dtype='float'),
          [np.array([rand_coor, [9.0, 3.0, 7.0], rand_coor], dtype='float'),
           np.array([rand_coor, [-9.0, 2.0, 9.0], rand_coor], dtype='float')]],
         {'RightSoleDelta': 1.0, 'LeftSoleDelta': -1.0},
         [np.array([1, 4, -6]), np.array([4, 2, 2]),
          np.array([[[1.4472135954999579, 4.0, -6.8944271909999157], [1.894427190999916, 4.0, -5.5527864045000417], [1.0, 3.0, -6.0]],
                    [[4.5834323811883104, 1.7666270475246759, 2.7779098415844139], [3.2777288444786272, 2.2889084622085494, 2.6283759053035944], [3.6286093236458963, 1.0715233091147407, 2.0]]])])])
    def test_rotaxis_footflat(self, frame, ankle_JC, vsk, expected):
        """
        This test provides coverage of the rotaxis_footflat function in pycgmStatic.py, defined as rotaxis_footflat(frame, ankle_JC, vsk)

        This test takes 4 parameters:
        frame: dictionaries of marker lists.
        ankle_JC: array of ankle_JC each x,y,z position
        vsk: dictionary containing subject measurements from a VSK file
        expected: the expected result from calling rotaxis_footflat on frame, ankle_JC and vsk, which should be the
        anatomically correct foot axis when foot is flat.

        Given the right ankle JC and the markers :math:`TOE_R` and :math:`HEE_R`, the right anatomically correct foot
        axis is calculated with:

        .. math::
            R = [R_x + ROrigin_x, R_y + ROrigin_y, R_z + ROrigin_z]

        where :math:`ROrigin_x` is the x coor of the foot axis's origin gotten from frame['RTOE']

        :math:`R_x` is the unit vector of :math:`(AnkleFlexion_R - AnkleJC_R) \times R_z`

        :math:`R_y` is the unit vector of :math:`R_z \times R_x`

        :math:`R_z` is the unit vector of :math:`(A \times (HEE_R - TOE_R)) \times A`

        A is the unit vector of :math:`(HEE_R - TOE_R) \times (AnkleJC_R - TOE_R)`

        The same calculation applies for the left anatomically correct foot axis by replacing all the right values
        with left values

        This unit test ensures that:
        - the markers for RTOE, LTOE, RHEE, and LHEE only effect either the right or the left axis
        - ankle_JC_R and ankle_JC_L only effect either the right or the left axis
        - the resulting output is correct when frame and ankle_JC are composed of lists of ints,
        numpy arrays of ints, lists of floats, and numpy arrays of floats.
        """
        result = pycgmStatic.rotaxis_footflat(frame, ankle_JC, vsk)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

    @pytest.mark.parametrize(["frame", "ankle_JC", "expected"], [
        # Test from running sample data
        ({'RTOE': np.array([433.33508301, 354.97229004, 44.27765274]),
          'LTOE': np.array([31.77310181, 331.23657227, 42.15322876]),
          'RHEE': np.array([381.88534546, 148.47607422, 49.99120331]),
          'LHEE': np.array([122.18766785, 138.55477905, 46.29433441])},
         [np.array([397.45738291, 217.50712216, 87.83068433]), np.array([112.28082818, 175.83265027, 80.98477997]),
          [[np.array(nan_3d), np.array([396.73749179, 218.18875543, 87.69979179]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([111.34886681, 175.49163538, 81.10789314]), np.array(nan_3d)]]],
         [np.array([433.33508301, 354.97229004, 44.27765274]), np.array([31.77310181, 331.23657227, 42.15322876]),
          [[[433.2103651914497, 355.03076948530014, 45.26812011533214],
            [432.37277461595676, 355.2083164947686, 44.14254511237841],
            [433.09340548455947, 354.0023046440309, 44.30449129818456]],
           [[31.878278418984852, 331.30724434357205, 43.14516794016654],
            [30.873906948094536, 330.8173225172055, 42.27844159782351],
            [32.1978211099223, 330.33145619248916, 42.172681460633456]]]]),
        # Test with zeros for all params
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0]),
          'RHEE': np.array([0, 0, 0]), 'LHEE': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when values are added to frame
        ({'RTOE': np.array([5, -2, -2]), 'LTOE': np.array([-2, -7, -1]),
          'RHEE': np.array([3, 5, 9]), 'LHEE': np.array([-7, 6, 1])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[nan_3d, nan_3d, [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [nan_3d, nan_3d, [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing when values are added to ankle_JC
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0]),
          'RHEE': np.array([0, 0, 0]), 'LHEE': np.array([0, 0, 0])},
         [np.array([-8, 6, 2]), np.array([3, 6, -3]),
          [[np.array(nan_3d), np.array([-7, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([2, -7, -2]), np.array(nan_3d)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when values are added to frame and ankle_JC[0]
        ({'RTOE': np.array([5, -2, -2]), 'LTOE': np.array([-2, -7, -1]),
          'RHEE': np.array([3, 5, 9]), 'LHEE': np.array([-7, 6, 1])},
         [np.array([-8, 6, 2]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[[4.519177631054049, -2.7767130575280747, -1.5931503031995797], [5.8636094856901915, -2.3392751550925754, -1.6270777220883264], [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [nan_3d, nan_3d, [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing when values are added to frame and ankle_JC[1]
        ({'RTOE': np.array([5, -2, -2]), 'LTOE': np.array([-2, -7, -1]),
          'RHEE': np.array([3, 5, 9]), 'LHEE': np.array([-7, 6, 1])},
         [np.array([0, 0, 0]), np.array([3, 6, -3]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[nan_3d, nan_3d, [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [[-2.5911479210576127, -7.104320221363108, -1.7997883637838292], [-2.7240728617802468, -7.368214526980399, -0.4167877290780265], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing when values are added to frame and ankle_JC[2]
        ({'RTOE': np.array([5, -2, -2]), 'LTOE': np.array([-2, -7, -1]),
          'RHEE': np.array([3, 5, 9]), 'LHEE': np.array([-7, 6, 1])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([-7, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([2, -7, -2]), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[[5.578725405755168, -1.2684037323472404, -2.36033846018718], [4.198695813697202, -1.5720307186791873, -2.4180357583501166], [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [[-1.2572186472917923, -6.628609323645897, -1.5570860145311554], [-2.567462100766509, -7.092377551287571, -1.8182011685470592], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing when values are added to frame and ankle_JC
        ({'RTOE': np.array([5, -2, -2]), 'LTOE': np.array([-2, -7, -1]),
          'RHEE': np.array([3, 5, 9]), 'LHEE': np.array([-7, 6, 1])},
         [np.array([-8, 6, 2]), np.array([3, 6, -3]),
          [[np.array(nan_3d), np.array([-7, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([2, -7, -2]), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[[5.049326362366699, -2.8385481602338833, -1.4574100139663109], [5.987207376506346, -1.8765990779367068, -1.8990356092209417], [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [[-2.446949206712144, -7.0343807082086265, -1.8938984134242876], [-2.820959061315946, -7.381159564182403, -0.5748604861042421], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing with differing values for right: RHEE, RTOE, ankle_JC[0], and ankle_JC[2][0][1]
        ({'RTOE': np.array([-2, 9, -1]), 'LTOE': np.array([-2, -7, -1]),
          'RHEE': np.array([-1, -4, 4]), 'LHEE': np.array([-7, 6, 1])},
         [np.array([5, -1, -5]), np.array([3, 6, -3]),
          [[np.array(nan_3d), np.array([-7, -8, -5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([2, -7, -2]), np.array(nan_3d)]]],
         [np.array([-2, 9, -1]), np.array([-2, -7, -1]),
          [[[-2.1975353004951486, 9.33863194370597, -0.0800498862654514], [-2.977676633621816, 8.86339202060183, -1.1596454197108794], [-1.9283885125960567, 8.069050663748737, -0.6419425629802835]],
           [[-2.446949206712144, -7.0343807082086265, -1.8938984134242876], [-2.820959061315946, -7.381159564182403, -0.5748604861042421], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing with differing values for left: LHEE, LTOE, ankle_JC[1], and ankle_JC[2][1][1]
        ({'RTOE': np.array([5, -2, -2]), 'LTOE': np.array([5, 4, -4]),
          'RHEE': np.array([3, 5, 9]), 'LHEE': np.array([-1, 6, 9])},
         [np.array([-8, 6, 2]), np.array([0, -8, -2]),
          [[np.array(nan_3d), np.array([-7, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([-4, -9, -1]), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([5, 4, -4]),
          [[[5.049326362366699, -2.8385481602338833, -1.4574100139663109], [5.987207376506346, -1.8765990779367068, -1.8990356092209417], [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [[4.702195658033984, 4.913266648695782, -4.277950719168281], [4.140311818111685, 3.6168482384235783, -4.3378327360136195], [4.584971321680356, 4.138342892773215, -3.1007711969741028]]]]),
        # Testing that when thorax, shoulderJC, and wand are lists of ints
        ({'RTOE': [5, -2, -2], 'LTOE': [-2, -7, -1], 'RHEE': [3, 5, 9], 'LHEE': [-7, 6, 1]},
         [[-8, 6, 2], [3, 6, -3],
          [[nan_3d, [-7, 8, 5], nan_3d],
           [nan_3d, [2, -7, -2], nan_3d]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[[5.049326362366699, -2.8385481602338833, -1.4574100139663109], [5.987207376506346, -1.8765990779367068, -1.8990356092209417], [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [[-2.446949206712144, -7.0343807082086265, -1.8938984134242876], [-2.820959061315946, -7.381159564182403, -0.5748604861042421], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing that when thorax, shoulderJC and wand are numpy arrays of ints
        ({'RTOE': np.array([5, -2, -2], dtype='int'), 'LTOE': np.array([-2, -7, -1], dtype='int'),
          'RHEE': np.array([3, 5, 9], dtype='int'), 'LHEE': np.array([-7, 6, 1], dtype='int')},
         [np.array([-8, 6, 2], dtype='int'), np.array([3, 6, -3], dtype='int'),
          [[np.array(nan_3d), np.array([-7, 8, 5], dtype='int'), np.array(nan_3d)],
           [np.array(nan_3d), np.array([2, -7, -2], dtype='int'), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[[5.049326362366699, -2.8385481602338833, -1.4574100139663109], [5.987207376506346, -1.8765990779367068, -1.8990356092209417], [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [[-2.446949206712144, -7.0343807082086265, -1.8938984134242876], [-2.820959061315946, -7.381159564182403, -0.5748604861042421], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing that when thorax, shoulderJC and wand are lists of floats
        ({'RTOE': [5.0, -2.0, -2.0], 'LTOE': [-2.0, -7.0, -1.0], 'RHEE': [3.0, 5.0, 9.0], 'LHEE': [-7.0, 6.0, 1.0]},
         [[-8.0, 6.0, 2.0], [3.0, 6.0, -3.0],
          [[nan_3d, [-7.0, 8.0, 5.0], nan_3d],
           [nan_3d, [2.0, -7.0, -2.0], nan_3d]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[[5.049326362366699, -2.8385481602338833, -1.4574100139663109], [5.987207376506346, -1.8765990779367068, -1.8990356092209417], [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [[-2.446949206712144, -7.0343807082086265, -1.8938984134242876], [-2.820959061315946, -7.381159564182403, -0.5748604861042421], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing that when thorax, shoulderJC and wand are numpy arrays of floats
        ({'RTOE': np.array([5.0, -2.0, -2.0], dtype='float'), 'LTOE': np.array([-2.0, -7.0, -1.0], dtype='float'),
          'RHEE': np.array([3.0, 5.0, 9.0], dtype='float'), 'LHEE': np.array([-7.0, 6.0, 1.0], dtype='float')},
         [np.array([-8.0, 6.0, 2.0], dtype='float'), np.array([3.0, 6.0, -3.0], dtype='float'),
          [[np.array(nan_3d), np.array([-7.0, 8.0, 5.0], dtype='float'), np.array(nan_3d)],
           [np.array(nan_3d), np.array([2.0, -7.0, -2.0], dtype='float'), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[[5.049326362366699, -2.8385481602338833, -1.4574100139663109], [5.987207376506346, -1.8765990779367068, -1.8990356092209417], [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [[-2.446949206712144, -7.0343807082086265, -1.8938984134242876], [-2.820959061315946, -7.381159564182403, -0.5748604861042421], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]])])
    def test_rotaxis_nonfootflat(self, frame, ankle_JC, expected):
        """
        This test provides coverage of the rotaxis_nonfootflat function in pycgmStatic.py, defined as rotaxis_nonfootflat(frame, ankle_JC)

        This test takes 3 parameters:
        frame: dictionaries of marker lists.
        ankle_JC: array of ankle_JC each x,y,z position
        expected: the expected result from calling rotaxis_footflat on frame, ankle_JC and vsk, which should be the
        anatomically correct foot axis when foot is not flat.

        Given the right ankle JC and the markers :math:`TOE_R` and :math:`HEE_R , the right anatomically correct foot
        axis is calculated with:

        .. math::
        R is [R_x + ROrigin_x, R_y + ROrigin_y, R_z + ROrigin_z]

        where :math:`ROrigin_x` is the x coor of the foot axis's origin gotten from frame['RTOE']

        :math:`R_x` is the unit vector of :math:`YFlex_R \times R_z`

        :math:`R_y` is the unit vector of :math:`R_z \times R_x`

        :math:`R_z` is the unit vector of :math:`(HEE_R - TOE_R)`

        :math:`YFlex_R` is the unit vector of :math:`(AnkleFlexion_R - AnkleJC_R)`

        The same calculation applies for the left anatomically correct foot axis by replacing all the right values
        with left values

        This unit test ensures that:
        - the markers for RTOE, LTOE, RHEE, and LHEE only effect either the right or the left axis
        - ankle_JC_R and ankle_JC_L only effect either the right or the left axis
        - the resulting output is correct when frame and ankle_JC are composed of lists of ints,
        numpy arrays of ints, lists of floats, and numpy arrays of floats.
        """
        result = pycgmStatic.rotaxis_nonfootflat(frame, ankle_JC)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

    @pytest.mark.parametrize(["a", "b", "c", "delta", "expected"], [
        # Test from running sample data
        (
            [426.50338745, 262.65310669, 673.66247559],
            [308.38050472, 322.80342417, 937.98979061],
            [416.98687744, 266.22558594, 524.04089355],
            59.5,
            [364.17774614, 292.17051722, 515.19181496]
        ),
        # Testing with basic value in a and c
        (
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            0.0,
            [0, 0, 1]
        ),
        # Testing with value in a and basic value in c
        (
            [-7, 1, 2],
            [ 0, 0, 0],
            [ 0, 0, 1],
            0.0,
            [0, 0, 1]
        ),
        #  Testing with value in b and basic value in c
        (
            [0, 0, 0],
            [1, 4, 3],
            [0, 0, 1],
            0.0,
            [0, 0, 1]
        ),
        #  Testing with value in a and b and basic value in c
        (
            [-7, 1, 2],
            [ 1, 4, 3],
            [ 0, 0, 1],
            0.0,
            [0, 0, 1]
        ),
        #  Testing with value in a, b, and c
        (
            [-7, 1,  2],
            [ 1, 4,  3],
            [ 3, 2, -8],
            0.0,
            [ 3, 2, -8]
        ),
        # Testing with value in a, b, c and delta of 1
        (
            [-7, 1,  2],
            [ 1, 4,  3],
            [ 3, 2, -8],
            1.0,
            [3.91271, 2.361115, -7.808801]
        ),
        # Testing with value in a, b, c and delta of 20
        (
            [-7, 1,  2],
            [ 1, 4,  3],
            [ 3, 2, -8],
            10.0,
            [5.867777, 5.195449, 1.031332]
        ),
        # Testing that when a, b, and c are lists of ints and delta is an int
        (
            [-7, 1,  2],
            [ 1, 4,  3],
            [ 3, 2, -8],
            10,
            [5.867777, 5.195449, 1.031332]
        ),
        # Testing that when a, b, and c are numpy arrays of ints and delta is an int
        (
            np.array([-7, 1,  2], dtype='int'),
            np.array([ 1, 4,  3], dtype='int'),
            np.array([ 3, 2, -8], dtype='int'),
            10,
            [5.867777, 5.195449, 1.031332]
        ),
        # Testing that when a, b, and c are lists of floats and delta is a float
        (
            [-7.0, 1.0,  2.0],
            [ 1.0, 4.0,  3.0],
            [ 3.0, 2.0, -8.0],
            10.0,
            [5.867777, 5.195449, 1.031332]
        ),
        # Testing that when a, b, and c are numpy arrays of floats and delta is a float
        (
            np.array([-7.0, 1.0,  2.0], dtype='float'),
            np.array([ 1.0, 4.0,  3.0], dtype='float'),
            np.array([ 3.0, 2.0, -8.0], dtype='float'),
            10.0,
            [5.867777, 5.195449, 1.031332]
        )
    ])
    def test_calc_joint_center(self, a, b, c, delta, expected):
        """
        This test provides coverage of the calc_joint_center function in pycgmStatic.py, defined as
        calc_joint_center(p_a, p_b, p_c, delta)

        This test takes 5 parameters:
        p_a: (x, y, z) position of marker a
        p_b: (x, y, z) position of marker b
        p_c: (x, y, z) position of marker c
        delta: The length from marker to joint center, retrieved from subject measurement file
        expected: the expected result from calling calc_joint_center on p_a, p_b, p_c, and delta

        A plane will be generated using the positions of three specified markers. 
        The plane will then calculate a joint center by rotating the vector of the plane
        around the rotating axis (the orthogonal vector).

        Lastly, it checks that the resulting output is correct when a, b, and c are lists of ints,
        numpy arrays of ints, lists of floats, and numpy arrays of floats and delta is an int or a float.
        """
        result = pycgmStatic.calc_joint_center(a, b, c, delta)
        np.testing.assert_almost_equal(result, expected, rounding_precision)
