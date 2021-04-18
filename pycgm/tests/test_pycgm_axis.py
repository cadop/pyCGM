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
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["pel_origin", "pel_x", "pel_y", "pel_z", "vsk", "expected"], [
        # Test from running sample data
        ([251.608306884766, 391.741317749023, 1032.893493652344], [251.740636241119, 392.726947206848, 1032.788500732036], [250.617115540376, 391.872328624646, 1032.874106304030], [251.602953357582, 391.847951338178, 1033.887777624562],
         {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512,
             'L_AsisToTrocanterMeasure': 72.512, 'InterAsisDistance': 215.908996582031},
         [[182.57097863, 339.43231855, 935.52900126], [308.38050472, 322.80342417, 937.98979061]]),
        # Basic test with zeros for all params
        ([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0,
             'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[0, 0, 0], [0, 0, 0]]),
        # Testing when values are added to pel_origin
        ([1, 0, -3], [0, 0, 0], [0, 0, 0], [0, 0, 0],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0,
             'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[-6.1387721, 0, 18.4163163], [8.53165418, 0, -25.59496255]]),
        # Testing when values are added to pel_x
        ([0, 0, 0], [-5, -3, -6], [0, 0, 0], [0, 0, 0],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0,
             'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[54.02442793, 32.41465676, 64.82931352], [54.02442793, 32.41465676, 64.82931352]]),
        # Testing when values are added to pel_y
        ([0, 0, 0], [0, 0, 0], [4, -1, 2], [0, 0, 0],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0,
             'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[29.34085257, -7.33521314, 14.67042628], [-29.34085257,   7.33521314, -14.67042628]]),
        # Testing when values are added to pel_z
        ([0, 0, 0], [0, 0, 0], [0, 0, 0], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0,
             'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[31.82533363, 84.86755635, 21.21688909], [31.82533363, 84.86755635, 21.21688909]]),
        # Test when values are added to pel_x, pel_y, and pel_z
        ([0, 0, 0], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0,
             'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[115.19061413, 109.94699997, 100.71662889], [56.508909, 124.61742625,  71.37577632]]),
        # Test when values are added to pel_origin, pel_x, pel_y, and pel_z
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0,
             'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[109.05184203, 109.94699997, 119.13294518], [65.04056318, 124.61742625,  45.78081377]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[MeanLegLength]
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': 0.0,
             'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[100.88576753,  97.85280235, 106.39612748], [61.83654463, 110.86920998,  41.31408931]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[R_AsisToTrocanterMeasure]
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': -24.0,
             'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[109.05184203, 109.94699997, 119.13294518], [-57.09307697, 115.44008189,  14.36512267]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[L_AsisToTrocanterMeasure]
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0,
             'L_AsisToTrocanterMeasure': 0-7.0, 'InterAsisDistance': 0.0},
         [[73.42953032, 107.27027453, 109.97003528], [65.04056318, 124.61742625,  45.78081377]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[InterAsisDistance]
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0,
             'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 11.0},
         [[125.55184203, 104.44699997, 146.63294518], [48.54056318, 130.11742625,  18.28081377]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and all values in vsk
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': -24.0,
             'L_AsisToTrocanterMeasure': -7.0, 'InterAsisDistance': 11.0},
         [[81.76345582,  89.67607691, 124.73321758], [-76.79709552, 107.19186562, -17.60160178]]),
        # Testing that when pel_origin, pel_x, pel_y, and pel_z are lists of ints and vsk values are ints
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 15, 'R_AsisToTrocanterMeasure': -24,
             'L_AsisToTrocanterMeasure': -7, 'InterAsisDistance': 11},
         [[81.76345582, 89.67607691, 124.73321758], [-76.79709552, 107.19186562, -17.60160178]]),
        # Testing that when pel_origin, pel_x, pel_y, and pel_z are numpy arrays of ints and vsk values are ints
        (np.array([1, 0, -3], dtype='int'), np.array([-5, -3, -6], dtype='int'), np.array([4, -1, 2], dtype='int'),
         np.array([3, 8, 2], dtype='int'),
         {'MeanLegLength': 15, 'R_AsisToTrocanterMeasure': -24,
             'L_AsisToTrocanterMeasure': -7, 'InterAsisDistance': 11},
         [[81.76345582, 89.67607691, 124.73321758], [-76.79709552, 107.19186562, -17.60160178]]),
        # Testing that when pel_origin, pel_x, pel_y, and pel_z are lists of floats and vsk values are floats
        ([1.0, 0.0, -3.0], [-5.0, -3.0, -6.0], [4.0, -1.0, 2.0], [3.0, 8.0, 2.0],
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': -24.0,
             'L_AsisToTrocanterMeasure': -7.0, 'InterAsisDistance': 11.0},
         [[81.76345582, 89.67607691, 124.73321758], [-76.79709552, 107.19186562, -17.60160178]]),
        # Testing that when pel_origin, pel_x, pel_y, and pel_z are numpy arrays of floats and vsk values are floats
        (np.array([1.0, 0.0, -3.0], dtype='float'), np.array([-5.0, -3.0, -6.0], dtype='float'),
         np.array([4.0, -1.0, 2.0],
                  dtype='float'), np.array([3.0, 8.0, 2.0], dtype='float'),
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': -24.0,
             'L_AsisToTrocanterMeasure': -7.0, 'InterAsisDistance': 11},
         [[81.76345582, 89.67607691, 124.73321758], [-76.79709552, 107.19186562, -17.60160178]])])
    def test_hip_joint_center(self, pel_origin, pel_x, pel_y, pel_z, vsk, expected):
        """
        This test provides coverage of the hipJointCenter function in pyCGM.py, defined as hipJointCenter(frame, pel_origin, pel_x, pel_y, pel_z, vsk)

        This test takes 6 parameters:
        pel_origin: array of x,y,z position of origin of the pelvis
        pel_x: array of x,y,z position of x-axis of the pelvis
        pel_y: array of x,y,z position of y-axis of the pelvis
        pel_z: array of x,y,z position of z-axis of the pelvis
        vsk: dictionary containing subject measurements from a VSK file
        expected: the expected result from calling hipJointCenter on pel_origin, pel_x, pel_y, pel_z, and vsk

        This test is checking to make sure the hip joint center is calculated correctly given the input parameters.
        The test checks to see that the correct values in expected are updated per each input parameter added. Any
        parameter that is added should change the value of every value in expected.

        The hip joint center axis and origin are calculated using the Hip Joint Center Calculation (ref. Davis_1991).

        Lastly, it checks that the resulting output is correct when pel_origin, pel_x, pel_y, and pel_z are composed of
        lists of ints, numpy arrays of ints, lists of floats, and numpy arrays of floats and vsk values are ints or floats.
        """
        pelvis_axis = np.zeros((4, 4))
        pelvis_axis[3, 3] = 1.0
        pelvis_axis[0, :3] = np.subtract(pel_x, pel_origin)
        pelvis_axis[1, :3] = np.subtract(pel_y, pel_origin)
        pelvis_axis[2, :3] = np.subtract(pel_z, pel_origin)
        pelvis_axis[:3, 3] = pel_origin

        result = axis.hip_joint_center(pelvis_axis, vsk)
        print(pelvis_axis)
        print(result)
        print(expected)
        np.testing.assert_almost_equal(
            result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(
            result[1], expected[1], rounding_precision)

    @pytest.mark.parametrize(["l_hip_jc", "r_hip_jc", "pelvis_axis", "expected"], [
        # Test from running sample data
        (np.array([182.57097863, 339.43231855, 935.52900126]), np.array([308.38050472, 322.80342417, 937.98979061]),
         [np.array([251.60830688, 391.74131775, 1032.89349365]), np.array([[251.74063624, 392.72694721, 1032.78850073], [
             250.61711554, 391.87232862, 1032.8741063], [251.60295336, 391.84795134, 1033.88777762]]), np.array([231.57849121, 210.25262451, 1052.24969482])],
         [[245.47574167208043, 331.1178713574418, 936.7593959314677], [[245.60807102843359, 332.10350081526684, 936.6544030111602], [244.48455032769033, 331.2488822330648, 936.7400085831541], [245.47038814489719, 331.22450494659665, 937.7536799036861]]]),
        # Basic test with zeros for all params
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         [np.array([0, 0, 0]), np.array(
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[0, 0, 0], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
        # Testing when values are added to l_hip_jc
        (np.array([1, -3, 2]), np.array([0, 0, 0]),
         [np.array([0, 0, 0]), np.array(
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[0.5, -1.5, 1], [[0.5, -1.5, 1], [0.5, -1.5, 1], [0.5, -1.5, 1]]]),
        # Testing when values are added to r_hip_jc
        (np.array([0, 0, 0]), np.array([-8, 1, 4]),
         [np.array([0, 0, 0]), np.array(
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[-4, 0.5, 2], [[-4, 0.5, 2], [-4, 0.5, 2], [-4, 0.5, 2]]]),
        # Testing when values are added to l_hip_jc and r_hip_jc
        (np.array([8, -3, 7]), np.array([5, -2, -1]),
         [np.array([0, 0, 0]), np.array(
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[6.5, -2.5, 3], [[6.5, -2.5, 3], [6.5, -2.5, 3], [6.5, -2.5, 3]]]),
        # Testing when values are added to pelvis_axis[0]
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         [np.array([1, -3, 6]), np.array([[0, 0, 0], [0, 0, 0],
                                          [0, 0, 0]]), np.array(rand_coor)],
         [[0, 0, 0], [[-1, 3, -6], [-1, 3, -6], [-1, 3, -6]]]),
        # Testing when values are added to pelvis_axis[1]
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         [np.array([0, 0, 0]), np.array(
             [[1, 0, 5], [-2, -7, -3], [9, -2, 7]]), np.array(rand_coor)],
         [[0, 0, 0], [[1, 0, 5], [-2, -7, -3], [9, -2, 7]]]),
        # Testing when values are added to pelvis_axis[0] and pelvis_axis[1]
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         [np.array([-3, 0, 5]), np.array([[-4, 5, -2],
                                          [0, 0, 0], [8, 5, -1]]), np.array(rand_coor)],
         [[0, 0, 0], [[-1, 5, -7], [3, 0, -5], [11, 5, -6]]]),
        # Testing when values are added to all params
        (np.array([-5, 3, 8]), np.array([-3, -7, -1]),
         [np.array([6, 3, 9]), np.array(
             [[5, 4, -2], [0, 0, 0], [7, 2, 3]]), np.array(rand_coor)],
         [[-4, -2, 3.5], [[-5, -1, -7.5], [-10, -5, -5.5], [-3, -3, -2.5]]]),
        # Testing that when l_hip_jc, r_hip_jc, and pelvis_axis are composed of lists of ints
        (np.array([-5, 3, 8]), np.array([-3, -7, -1]),
         [[6, 3, 9], [[5, 4, -2], [0, 0, 0], [7, 2, 3]], rand_coor],
         [[-4, -2, 3.5], [[-5, -1, -7.5], [-10, -5, -5.5], [-3, -3, -2.5]]]),
        # Testing that when l_hip_jc, r_hip_jc, and pelvis_axis are composed of numpy arrays of ints
        (np.array([-5, 3, 8], dtype='int'), np.array([-3, -7, -1], dtype='int'),
         [np.array([6, 3, 9], dtype='int'), np.array(
             [[5, 4, -2], [0, 0, 0], [7, 2, 3]], dtype='int'), rand_coor],
         [[-4, -2, 3.5], [[-5, -1, -7.5], [-10, -5, -5.5], [-3, -3, -2.5]]]),
        # Testing that when l_hip_jc, r_hip_jc, and pelvis_axis are composed of lists of floats
        (np.array([-5.0, 3.0, 8.0]), np.array([-3.0, -7.0, -1.0]),
         [[6.0, 3.0, 9.0], [[5.0, 4.0, -2.0],
                            [0.0, 0.0, 0.0], [7.0, 2.0, 3.0]], rand_coor],
         [[-4, -2, 3.5], [[-5, -1, -7.5], [-10, -5, -5.5], [-3, -3, -2.5]]]),
        # Testing that when l_hip_jc, r_hip_jc, and pelvis_axis are composed of numpy arrays of floats
        (np.array([-5.0, 3.0, 8.0], dtype='float'), np.array([-3.0, -7.0, -1.0], dtype='float'),
         [np.array([6.0, 3.0, 9.0], dtype='float'), np.array(
             [[5.0, 4.0, -2.0], [0.0, 0.0, 0.0], [7.0, 2.0, 3.0]], dtype='float'), rand_coor],
         [[-4, -2, 3.5], [[-5, -1, -7.5], [-10, -5, -5.5], [-3, -3, -2.5]]])])
    def test_hip_axis(_, l_hip_jc, r_hip_jc, pelvis_axis, expected):
        """
        This test provides coverage of the hipAxisCenter function in pyCGM.py, defined as hipAxisCenter(l_hip_jc, r_hip_jc, pelvis_axis)

        This test takes 4 parameters:
        l_hip_jc: array of left hip joint center x,y,z position
        r_hip_jc: array of right hip joint center x,y,z position
        pelvis_axis: array of pelvis origin and axis
        expected: the expected result from calling hipAxisCenter on l_hip_jc, r_hip_jc, and pelvis_axis

        This test is checking to make sure the hip axis center is calculated correctly given the input parameters.

        The hip axis center is calculated using the midpoint of the right and left hip joint centers.
        Then, the given pelvis_axis variable is converted into x,y,z axis format.
        The pelvis axis is then translated to the shared hip center by calculating the sum of:
        pelvis_axis axis component + hip_axis_center axis component
        """
        origin, pelvis_axis, _ = pelvis_axis

        pelvis = np.zeros((4, 4))
        pelvis[3, 3] = 1.0
        pelvis[0, :3] = np.subtract(pelvis_axis[0], origin)
        pelvis[1, :3] = np.subtract(pelvis_axis[1], origin)
        pelvis[2, :3] = np.subtract(pelvis_axis[2], origin)
        pelvis[:3, 3] = origin

        result = axis.hip_axis(l_hip_jc, r_hip_jc, pelvis)

        expected_pelvis = np.zeros((4, 4))
        expected_pelvis[3, 3] = 1.0
        expected_pelvis[0, :3] = np.subtract(expected[1][0], expected[0])
        expected_pelvis[1, :3] = np.subtract(expected[1][1], expected[0])
        expected_pelvis[2, :3] = np.subtract(expected[1][2], expected[0])
        expected_pelvis[:3, 3] = expected[0]

        np.testing.assert_almost_equal(
            result, expected_pelvis, rounding_precision
        )
