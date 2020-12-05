#!/usr/bin/python
# -*- coding: utf-8 -*-

from mock import patch
import numpy as np
import pytest

from refactor.pycgm import CGM

rounding_precision = 8


class TestLowerBodyAxis():
    """
    This class tests the lower body axis functions in the class CGM in pycgm.py:
    pelvis_axis_calc
    hip_axis_calc
    knee_axis_calc
    ankle_axis_calc
    foot_axis_calc
    """

    nan_3d = np.array([np.nan, np.nan, np.nan])
    rand_coor = np.array([np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)])

    @pytest.mark.parametrize(["rasi", "lasi", "rpsi", "lpsi", "sacr", "expected"], [
        # Test from running sample data
        (np.array([357.90066528, 377.69210815, 1034.97253418]), np.array([145.31594849, 405.79052734, 1030.81445312]),
         np.array([274.00466919, 205.64402771, 1051.76452637]), np.array([189.15231323, 214.86122131, 1052.73486328]),
         None,
         [np.array([251.60830688, 391.74131775, 1032.89349365]), np.array([251.74063624, 392.72694721, 1032.78850073]),
          np.array([250.61711554, 391.87232862, 1032.8741063]), np.array([251.60295336, 391.84795134, 1033.88777762])]),
        # Test with zeros for all params
        (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
         [np.array([0, 0, 0]), np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)]),
        # Testing when adding values to rasi and lasi
        (np.array([-6, 6, 3]), np.array([-7, -9, 1]), np.array([0, 0, 0]), np.array([0, 0, 0]), None,
         [np.array([-6.5, -1.5, 2.0]), np.array([-7.44458106, -1.48072284, 2.32771179]),
          np.array([-6.56593805, -2.48907071, 1.86812391]), np.array([-6.17841206, -1.64617634, 2.93552855])]),
        # Testing when adding values to rpsi and lpsi
        (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([1, 0, -4]), np.array([7, -2, 2]), None,
         [np.array([0, 0, 0]), np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)]),
        # Testing when adding values to sacr
        (np.array([0, 0, 0]), np.array([0, 0, 0]), None, None, np.array([-4, 8, -5]),
         [np.array([0, 0, 0]), np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)]),
        # Testing when adding values to rasi, lasi, rpsi, lpsi
        (np.array([-6, 6, 3]), np.array([-7, -9, 1]), np.array([1, 0, -4]), np.array([7, -2, 2]), None,
         [np.array([-6.5, -1.5, 2.0]), np.array([-7.45825845, -1.47407957, 2.28472598]),
          np.array([-6.56593805, -2.48907071, 1.86812391]), np.array([-6.22180416, -1.64514566, 2.9494945])]),
        # Testing when adding values to rasi, lasi, and sacr
        (np.array([-6, 6, 3]), np.array([-7, -9, 1]), None, None, np.array([-4, 8, -5]),
         [np.array([-6.5, -1.5, 2.0]), np.array([-6.72928306, -1.61360872, 2.96670695]),
          np.array([-6.56593805, -2.48907071, 1.86812391]), np.array([-5.52887619, -1.59397972, 2.21928602])]),
        # Testing when adding values to rpsi, lpsi, and sacr
        (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([1, 0, -4]), np.array([7, -2, 2]), np.array([-4, 8, -5]),
         [np.array([0, 0, 0]), np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)]),
        # Testing when adding values to rasi, lasi, rpsi, lpsi, and sacr
        (np.array([-6, 6, 3]), np.array([-7, -9, 1]), np.array([1, 0, -4]), np.array([7, -2, 2]), np.array([-4, 8, -5]),
         [np.array([-6.5, -1.5, 2.0]), np.array([-6.72928306, -1.61360872, 2.96670695]),
          np.array([-6.56593805, -2.48907071, 1.86812391]), np.array([-5.52887619, -1.59397972, 2.21928602])]),
        # Testing that when rasi, lasi, rpsi, lpsi, and sacr are composed of numpy arrays of ints
        (np.array([-6, 6, 3], dtype='int'), np.array([-7, -9, 1], dtype='int'), np.array([1, 0, -4], dtype='int'),
         np.array([7, -2, 2], dtype='int'), np.array([-4, 8, -5], dtype='int'),
         [np.array([-6.5, -1.5, 2.0]), np.array([-6.72928306, -1.61360872, 2.96670695]),
          np.array([-6.56593805, -2.48907071, 1.86812391]), np.array([-5.52887619, -1.59397972, 2.21928602])]),
        # Testing that when rasi, lasi, rpsi, lpsi, and sacr are composed of numpy arrays of floats
        (np.array([-6.0, 6.0, 3.0], dtype='float'), np.array([-7.0, -9.0, 1.0], dtype='float'),
         np.array([1.0, 0.0, -4.0], dtype='float'), np.array([7.0, -2.0, 2.0], dtype='float'),
         np.array([-4.0, 8.0, -5.0], dtype='float'),
         [np.array([-6.5, -1.5, 2.0]), np.array([-6.72928306, -1.61360872, 2.96670695]),
          np.array([-6.56593805, -2.48907071, 1.86812391]), np.array([-5.52887619, -1.59397972, 2.21928602])])])
    def test_pelvis_axis_calc(self, rasi, lasi, rpsi, lpsi, sacr, expected):
        """
        This test provides coverage of the pelvis_axis_calc function in the class CGM in pycgm.py, defined as
        pelvis_axis_calc(rasi, lasi, rpsi, lpsi, sacr)

        This test takes 6 parameters:
        rasi, lasi : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        rpsi, lpsi, sacr : array, optional
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        expected : array
            A 4x3 ndarray that contains the pelvis origin and the pelvis x, y, and z axis components.

        This test is checking to make sure the pelvis joint center and axis are calculated correctly given the input
        parameters.

        If sacr marker is not present, the mean of rpsi and lpsi markers will be used instead.
        The pelvis origin is the midpoint of the rasi and lasi markers.
        x axis is computed with a Gram-Schmidt orthogonalization procedure (ref. Kadaba 1990).
        y axis is computed by subtracting rasi from lasi.
        z axis is cross product of x axis and y axis.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - rpsi and lpsi are only used if sacr isn't given.
        - the resulting output is correct when rasi, lasi, rpsi, lpsi, and sacr are composed of numpy arrays of ints
        and numpy arrays of floats. Lists were not tested as lists would cause errors on the following lines in
        pycgm.py as lists cannot be divided by floats:
        origin = (rasi + lasi) / 2.0
        sacrum = (rpsi + lpsi) / 2.0
        """
        result = CGM.pelvis_axis_calc(rasi, lasi, rpsi, lpsi, sacr)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["pelvis_axis", "measurements", "expected"], [
        # Test from running sample data
        (np.array([[251.60830688, 391.74131775, 1032.89349365], [251.74063624, 392.72694721, 1032.78850073],
                   [250.61711554, 391.87232862, 1032.8741063], [251.60295336, 391.84795134, 1033.88777762]]),
         {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512, 'L_AsisToTrocanterMeasure': 72.512,
          'InterAsisDistance': 215.908996582031},
         np.array([[308.38050352, 322.80342433, 937.98979092], [182.57097799, 339.43231799, 935.52900136],
                   [245.47574075, 331.11787116, 936.75939614], [245.60807011, 332.10350062, 936.65440322],
                   [244.48454941, 331.24888203, 936.74000879], [245.47038723, 331.22450475, 937.75368011]])),
        # Basic test with zeros for all params
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])),
        # Testing when values are added to pel_o
        (np.array([[1, 0, -3], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[8.53165418, 0., -25.59496255], [-6.1387721, 0., 18.4163163], [1.19644104, 0., -3.58932313],
                   [0.19644104, 0, -0.58932313], [0.19644104, 0, -0.58932313], [0.19644104, 0, -0.58932313]])),
        # Testing when values are added to pel_x
        (np.array([[0, 0, 0], [-5, -3, -6], [0, 0, 0], [0, 0, 0]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[54.02442793, 32.41465676, 64.82931352], [54.02442793, 32.41465676, 64.82931352],
                   [54.02442793, 32.41465676, 64.82931352], [49.02442793, 29.41465676, 58.82931352],
                   [54.02442793, 32.41465676, 64.82931352], [54.02442793, 32.41465676, 64.82931352]])),
        # Testing when values are added to pel_y
        (np.array([[0, 0, 0], [0, 0, 0], [4, -1, 2], [0, 0, 0]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array(
             [[-29.34085257, 7.33521314, -14.67042628], [29.34085257, -7.33521314, 14.67042628], [0, 0, 0], [0, 0, 0],
              [4, -1, 2], [0, 0, 0]])),
        # Testing when values are added to pel_z
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[31.82533363, 84.86755635, 21.21688909], [31.82533363, 84.86755635, 21.21688909],
                   [31.82533363, 84.86755635, 21.21688909], [31.82533363, 84.86755635, 21.21688909],
                   [31.82533363, 84.86755635, 21.21688909], [34.82533363, 92.86755635, 23.21688909]])),
        # Test when values are added to pel_x, pel_y, and pel_z
        (np.array([[0, 0, 0], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[56.508909, 124.61742625, 71.37577632], [115.19061413, 109.94699997, 100.71662889],
                   [85.84976156, 117.28221311, 86.04620261], [80.84976156, 114.28221311, 80.04620261],
                   [89.84976156, 116.28221311, 88.04620261], [88.84976156, 125.28221311, 88.04620261]])),
        # Test when values are added to pel_o, pel_x, pel_y, and pel_z
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[65.04056318, 124.61742625, 45.78081377], [109.05184203, 109.94699997, 119.13294518],
                   [87.04620261, 117.28221311, 82.45687948], [81.04620261, 114.28221311, 79.45687948],
                   [90.04620261, 116.28221311, 87.45687948], [89.04620261, 125.28221311, 87.45687948]])),
        # Test when values are added to pel_o, pel_x, pel_y, pel_z, and measurements[MeanLegLength]
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[61.83654463, 110.86920998, 41.31408931], [100.88576753, 97.85280235, 106.39612748],
                   [81.36115608, 104.36100616, 73.8551084], [75.36115608, 101.36100616, 70.8551084],
                   [84.36115608, 103.36100616, 78.8551084], [83.36115608, 112.36100616, 78.8551084]])),
        # Test when values are added to pel_o, pel_x, pel_y, pel_z, and measurements[R_AsisToTrocanterMeasure]
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[-57.09307697, 115.44008189, 14.36512267], [109.05184203, 109.94699997, 119.13294518],
                   [25.97938253, 112.69354093, 66.74903393], [19.97938253, 109.69354093, 63.74903393],
                   [28.97938253, 111.69354093, 71.74903393], [27.97938253, 120.69354093, 71.74903393]])),
        # Test when values are added to pel_o, pel_x, pel_y, pel_z, and measurements[R_AsisToTrocanterMeasure]
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[-57.09307697, 115.44008189, 14.36512267], [109.05184203, 109.94699997, 119.13294518],
                   [25.97938253, 112.69354093, 66.74903393], [19.97938253, 109.69354093, 63.74903393],
                   [28.97938253, 111.69354093, 71.74903393], [27.97938253, 120.69354093, 71.74903393]])),
        # Test when values are added to pel_o, pel_x, pel_y, pel_z, and measurements[L_AsisToTrocanterMeasure]
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0 - 7.0,
          'InterAsisDistance': 0.0},
         np.array([[65.04056318, 124.61742625, 45.78081377], [73.42953032, 107.27027453, 109.97003528],
                   [69.23504675, 115.94385039, 77.87542453], [63.23504675, 112.94385039, 74.87542453],
                   [72.23504675, 114.94385039, 82.87542453], [71.23504675, 123.94385039, 82.87542453]])),
        # Test when values are added to pel_o, pel_x, pel_y, pel_z, and measurements[InterAsisDistance]
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 11.0},
         np.array([[48.54056318, 130.11742625, 18.28081377], [125.55184203, 104.44699997, 146.63294518],
                   [87.04620261, 117.28221311, 82.45687948], [81.04620261, 114.28221311, 79.45687948],
                   [90.04620261, 116.28221311, 87.45687948], [89.04620261, 125.28221311, 87.45687948]])),
        # Test when values are added to pel_o, pel_x, pel_y, pel_z, and all values in measurements
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': -7.0,
          'InterAsisDistance': 11.0},
         np.array([[-76.79709552, 107.19186562, -17.60160178], [81.76345582, 89.67607691, 124.73321758],
                   [2.48318015, 98.43397127, 53.5658079], [-3.51681985, 95.43397127, 50.5658079],
                   [5.48318015, 97.43397127, 58.5658079], [4.48318015, 106.43397127, 58.5658079]])),
        # Testing that when pel_o, pel_x, pel_y, and pel_z are numpy arrays of ints and measurements values are ints
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]], dtype='int'),
         {'MeanLegLength': 15, 'R_AsisToTrocanterMeasure': -24, 'L_AsisToTrocanterMeasure': -7,
          'InterAsisDistance': 11},
         np.array([[-76.79709552, 107.19186562, -17.60160178], [81.76345582, 89.67607691, 124.73321758],
                   [2.48318015, 98.43397127, 53.5658079], [-3.51681985, 95.43397127, 50.5658079],
                   [5.48318015, 97.43397127, 58.5658079], [4.48318015, 106.43397127, 58.5658079]])),
        # Testing that when pel_o, pel_x, pel_y, and pel_z are numpy arrays of floats and measurements values are floats
        (np.array([[1.0, 0.0, -3.0], [-5.0, -3.0, -6.0], [4.0, -1.0, 2.0], [3.0, 8.0, 2.0]], dtype='float'),
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': -7.0,
          'InterAsisDistance': 11},
         np.array([[-76.79709552, 107.19186562, -17.60160178], [81.76345582, 89.67607691, 124.73321758],
                   [2.48318015, 98.43397127, 53.5658079], [-3.51681985, 95.43397127, 50.5658079],
                   [5.48318015, 97.43397127, 58.5658079], [4.48318015, 106.43397127, 58.5658079]]))])
    def test_hip_axis_calc(self, pelvis_axis, measurements, expected):
        """
        This test provides coverage of the hip_axis_calc function in the class CGM in pycgm.py, defined as
        hip_axis_calc(pelvis_axis, measurements)

        This test takes 3 parameters:
        pelvis_axis : array
            A 4x3 ndarray that contains the pelvis origin and the pelvis x, y, and z axis components.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        expected : array
            A 4x3 ndarray that contains the hip origin and the hip x, y, and z axis components.

        This test is checking to make sure the hip joint center and axis are calculated correctly given the input
        parameters.

        The hip origin is calculated using the Hip Joint Center Calculation (ref. Davis_1991).
        The hip center axis is calculated by taking the mean at each x, y, z axis of the left and right hip joint
        center.
        The hip axis is calculated by getting the summation of the pelvis and hip center axis.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when pelvis_axis is composed of numpy arrays of ints and numpy arrays of
        floats. Lists were not tested as lists would cause errors on the following lines in pycgm.py as lists cannot
        be subtracted by each other:
        pelvis_xaxis = pel_x - pel_o
        pelvis_yaxis = pel_y - pel_o
        pelvis_zaxis = pel_z - pel_o
        """
        result = CGM.hip_axis_calc(pelvis_axis, measurements)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(
        ["rthi", "lthi", "rkne", "lkne", "hip_origin", "measurements", "mock_return_val", "expected_mock_args",
         "expected"], [
            # Test from running sample data
            (np.array([426.50338745, 262.65310669, 673.66247559]), np.array([51.93867874, 320.01849365, 723.03186035]),
             np.array([416.98687744, 266.22558594, 524.04089355]), np.array([84.62355804, 286.69122314, 529.39819336]),
             [[308.38050472, 322.80342417, 937.98979061], [182.57097863, 339.43231855, 935.52900126]],
             {'RightKneeWidth': 105.0, 'LeftKneeWidth': 105.0},
             [np.array([364.17774614, 292.17051722, 515.19181496]),
              np.array([143.55478579, 279.90370346, 524.78408753])],
             [[[426.50338745, 262.65310669, 673.66247559], [308.38050472, 322.80342417, 937.98979061],
               [416.98687744, 266.22558594, 524.04089355], 59.5],
              [[51.93867874, 320.01849365, 723.03186035], [182.57097863, 339.43231855, 935.52900126],
               [84.62355804, 286.69122314, 529.39819336], 59.5]],
             np.array([[364.17774614, 292.17051722, 515.19181496], [364.61959153, 293.06758353, 515.18513093],
                       [363.29019771, 292.60656648, 515.04309095], [364.04724541, 292.24216264, 516.18067112],
                       [143.55478579, 279.90370346, 524.78408753], [143.65611282, 280.88685896, 524.63197541],
                       [142.56434499, 280.01777943, 524.86163553], [143.64837987, 280.04650381, 525.76940383]])),
            # Test with zeros for all params
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             [[0, 0, 0], [0, 0, 0]],
             {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to rthi
            (np.array([1, 2, 4]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             [[0, 0, 0], [0, 0, 0]],
             {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[1, 2, 4], [0, 0, 0], [0, 0, 0], 7.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to lthi
            (np.array([0, 0, 0]), np.array([-1, 0, 8]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             [[0, 0, 0], [0, 0, 0]],
             {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0], [[-1, 0, 8], [0, 0, 0], [0, 0, 0], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to rkne
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([8, -4, 5]), np.array([0, 0, 0]),
             [[0, 0, 0], [0, 0, 0]],
             {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[0, 0, 0], [0, 0, 0], [8, -4, 5], 7.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to lkne
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([8, -8, 5]),
             [[0, 0, 0], [0, 0, 0]],
             {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0], [[0, 0, 0], [0, 0, 0], [8, -8, 5], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to hip_origin
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             [[1, -9, 2], [-8, 8, -2]],
             {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[0, 0, 0], [1, -9, 2], [0, 0, 0], 7.0], [[0, 0, 0], [-8, 8, -2], [0, 0, 0], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, [0.10783277, -0.97049496, 0.21566555], [0, 0, 0], nan_3d, nan_3d,
                       [-0.69631062, 0.69631062, -0.17407766]])),
            # Testing when values are added to measurements
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             [[0, 0, 0], [0, 0, 0]],
             {'RightKneeWidth': 9.0, 'LeftKneeWidth': -6.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 11.5], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 4.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to mock_return_val
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             [[0, 0, 0], [0, 0, 0]],
             {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
             [np.array([-5, -5, -9]), np.array([3, -6, -5])],
             [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
             np.array(
                 [[-5, -5, -9], nan_3d, nan_3d, [-4.56314797, -4.56314797, -8.21366635], [3, -6, -5], nan_3d, nan_3d,
                  [2.64143142, -5.28286283, -4.4023857]])),
            # Testing when values are added to rthi, lthi, rkne, lkne, and hip_origin
            (np.array([1, 2, 4]), np.array([-1, 0, 8]), np.array([8, -4, 5]), np.array([8, -8, 5]),
             [[1, -9, 2], [-8, 8, -2]],
             {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 7.0], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 7.0]],
             np.array([[0, 0, 0], [-0.47319376, 0.14067923, 0.86965339], [-0.8743339, -0.19582873, -0.44406233],
                       [0.10783277, -0.97049496, 0.21566555], [0, 0, 0], [-0.70710678, -0.70710678, 0.0],
                       [-0.12309149, 0.12309149, 0.98473193], [-0.69631062, 0.69631062, -0.17407766]])),
            # Testing when values are added to rthi, lthi, rkne, lkne, hip_origin, and measurements
            (np.array([1, 2, 4]), np.array([-1, 0, 8]), np.array([8, -4, 5]), np.array([8, -8, 5]),
             [[1, -9, 2], [-8, 8, -2]],
             {'RightKneeWidth': 9.0, 'LeftKneeWidth': -6.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 11.5], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 4.0]],
             np.array([[0, 0, 0], [-0.47319376, 0.14067923, 0.86965339], [-0.8743339, -0.19582873, -0.44406233],
                       [0.10783277, -0.97049496, 0.21566555], [0, 0, 0], [-0.70710678, -0.70710678, 0.0],
                       [-0.12309149, 0.12309149, 0.98473193], [-0.69631062, 0.69631062, -0.17407766]])),
            # Testing when values are added to rthi, lthi, rkne, lkne, hip_origin, measurements, and mock_return_val
            (np.array([1, 2, 4]), np.array([-1, 0, 8]), np.array([8, -4, 5]), np.array([8, -8, 5]),
             [[1, -9, 2], [-8, 8, -2]],
             {'RightKneeWidth': 9.0, 'LeftKneeWidth': -6.0},
             [np.array([-5, -5, -9]), np.array([3, -6, -5])],
             [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 11.5], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 4.0]],
             np.array([[-5, -5, -9], [-5.6293369, -4.4458078, -8.45520089], [-5.62916022, -5.77484544, -8.93858368],
                       [-4.54382845, -5.30411437, -8.16368549], [3, -6, -5], [2.26301154, -6.63098327, -4.75770242],
                       [3.2927155, -5.97483821, -4.04413154], [2.39076635, -5.22461171, -4.83384537]])),
            # Testing that when rthi, lthi, and hip_origin are composed of lists of ints and measurements values are
            # ints
            ([1, 2, 4], [-1, 0, 8], np.array([8, -4, 5]), np.array([8, -8, 5]),
             [[1, -9, 2], [-8, 8, -2]],
             {'RightKneeWidth': 9, 'LeftKneeWidth': -6},
             [np.array([-5, -5, -9]), np.array([3, -6, -5])],
             [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 11.5], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 4.0]],
             np.array([[-5, -5, -9], [-5.6293369, -4.4458078, -8.45520089], [-5.62916022, -5.77484544, -8.93858368],
                       [-4.54382845, -5.30411437, -8.16368549], [3, -6, -5], [2.26301154, -6.63098327, -4.75770242],
                       [3.2927155, -5.97483821, -4.04413154], [2.39076635, -5.22461171, -4.83384537]])),
            # Testing that when rthi, lthi, rkne, lkne, and hip_origin are composed of numpy arrays of ints and
            # measurements values are ints
            (np.array([1, 2, 4], dtype='int'), np.array([-1, 0, 8], dtype='int'), np.array([8, -4, 5], dtype='int'),
             np.array([8, -8, 5], dtype='int'), np.array([[1, -9, 2], [-8, 8, -2]], dtype='int'),
             {'RightKneeWidth': 9, 'LeftKneeWidth': -6},
             [np.array([-5, -5, -9]), np.array([3, -6, -5])],
             [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 11.5], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 4.0]],
             np.array([[-5, -5, -9], [-5.6293369, -4.4458078, -8.45520089], [-5.62916022, -5.77484544, -8.93858368],
                       [-4.54382845, -5.30411437, -8.16368549], [3, -6, -5], [2.26301154, -6.63098327, -4.75770242],
                       [3.2927155, -5.97483821, -4.04413154], [2.39076635, -5.22461171, -4.83384537]])),
            # Testing that when rthi, lthi, and hip_origin are composed of lists of floats and measurements values
            # are floats
            ([1.0, 2.0, 4.0], [-1.0, 0.0, 8.0], np.array([8.0, -4.0, 5.0]), np.array([8.0, -8.0, 5.0]),
             [[1.0, -9.0, 2.0], [-8.0, 8.0, -2.0]],
             {'RightKneeWidth': 9.0, 'LeftKneeWidth': -6.0},
             [np.array([-5, -5, -9]), np.array([3, -6, -5])],
             [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 11.5], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 4.0]],
             np.array([[-5, -5, -9], [-5.6293369, -4.4458078, -8.45520089], [-5.62916022, -5.77484544, -8.93858368],
                       [-4.54382845, -5.30411437, -8.16368549], [3, -6, -5], [2.26301154, -6.63098327, -4.75770242],
                       [3.2927155, -5.97483821, -4.04413154], [2.39076635, -5.22461171, -4.83384537]])),
            # Testing that when rthi, lthi, rkne, lkne, and hip_origin are composed of numpy arrays of floats and
            # measurements values are floats
            (np.array([1.0, 2.0, 4.0], dtype='float'), np.array([-1.0, 0.0, 8.0], dtype='float'),
             np.array([8.0, -4.0, 5.0], dtype='float'), np.array([8.0, -8.0, 5.0], dtype='float'),
             np.array([[1.0, -9.0, 2.0], [-8.0, 8.0, -2.0]], dtype='float'),
             {'RightKneeWidth': 9.0, 'LeftKneeWidth': -6.0},
             [np.array([-5, -5, -9]), np.array([3, -6, -5])],
             [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 11.5], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 4.0]],
             np.array([[-5, -5, -9], [-5.6293369, -4.4458078, -8.45520089], [-5.62916022, -5.77484544, -8.93858368],
                       [-4.54382845, -5.30411437, -8.16368549], [3, -6, -5], [2.26301154, -6.63098327, -4.75770242],
                       [3.2927155, -5.97483821, -4.04413154], [2.39076635, -5.22461171, -4.83384537]]))])
    def test_knee_axis_calc(self, rthi, lthi, rkne, lkne, hip_origin, measurements, mock_return_val, expected_mock_args,
                            expected):
        """
        This test provides coverage of the knee_axis_calc function in the class CGM in pycgm.py, defined as
        knee_axis_calc(rthi, lthi, rkne, lkne, hip_origin, measurement)

        This test takes 9 parameters:
        rthi, lthi, rkne, lkne : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        hip_origin : array
            A 2x3 ndarray of the right and left hip origin vectors (joint centers).
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        mock_return_val : list
            The value to be returned by the mock for find_joint_center
        expected_mock_args : list
            The expected arguments used to call the mocked function, find_joint_center
        expected : array
            An 8x3 ndarray that contains the right knee origin, right knee x, y, and z axis components,
            left knee origin, and left knee x, y, and z axis components.

        This test is checking to make sure the knee joint center and axis are calculated correctly given the input
        parameters. This tests mocks find_joint_center to make sure the correct parameters are being passed into it
        given the parameters passed into knee_axis_calc, expected_mock_args, and to also ensure that knee_axis_calc
        returns the correct value considering the return value of find_joint_center, mock_return_val.

        Calculated using Knee Axis Calculation (ref. Clinical Gait Analysis hand book, Baker2013)

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when rthi, lthi, and hip_origin are composed of lists of ints, numpy arrays
        of ints, lists of floats, and numpy arrays of floats and measurements values are ints and floats. The values
        of rkne and lkne were kept as numpy arrays as lists would cause errors on the following lines in pycgm.py as
        lists cannot be subtracted by each other:
        axis_x = np.cross(axis_z, rkne - r_hip_jc)
        axis_x = np.cross(lkne - l_hip_jc, axis_z)
        """
        with patch.object(CGM, 'find_joint_center', side_effect=mock_return_val) as mock_find_joint_center:
            result = CGM.knee_axis_calc(rthi, lthi, rkne, lkne, hip_origin, measurements)

        # Asserting that there were only 2 calls to find_joint_center
        np.testing.assert_equal(mock_find_joint_center.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to find_joint_center
        np.testing.assert_almost_equal(expected_mock_args[0][0], mock_find_joint_center.call_args_list[0][0][0],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][1], mock_find_joint_center.call_args_list[0][0][1],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][2], mock_find_joint_center.call_args_list[0][0][2],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][3], mock_find_joint_center.call_args_list[0][0][3],
                                       rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to find_joint_center
        np.testing.assert_almost_equal(expected_mock_args[1][0], mock_find_joint_center.call_args_list[1][0][0],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][1], mock_find_joint_center.call_args_list[1][0][1],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][2], mock_find_joint_center.call_args_list[1][0][2],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][3], mock_find_joint_center.call_args_list[1][0][3],
                                       rounding_precision)

        # Asserting that knee_axis_calc returned the correct result given the return value given by mocked
        # find_joint_center
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(
        ["rtib", "ltib", "rank", "lank", "knee_origin", "measurements", "mock_return_val", "expected_mock_args",
         "expected"], [
            # Test from running sample data
            (np.array([433.97537231, 211.93408203, 273.3008728]), np.array([50.04016495, 235.90718079, 364.32226562]),
             np.array([422.77005005, 217.74053955, 92.86152649]), np.array([58.57380676, 208.54806519, 86.16953278]),
             np.array([[364.17774614, 292.17051722, 515.19181496], [143.55478579, 279.90370346, 524.78408753]]),
             {'RightAnkleWidth': 70.0, 'LeftAnkleWidth': 70.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
             [np.array([393.76181608, 247.67829633, 87.73775041]), np.array([98.74901939, 219.46930221, 80.6306816])],
             [[[433.97537231, 211.93408203, 273.3008728], [364.17774614, 292.17051722, 515.19181496],
               [422.77005005, 217.74053955, 92.86152649], 42.0],
              [[50.04016495, 235.90718079, 364.32226562], [143.55478579, 279.90370346, 524.78408753],
               [58.57380676, 208.54806519, 86.16953278], 42.0]],
             np.array([[393.76181608, 247.67829633, 87.73775041], [394.48171575, 248.37201348, 87.715368],
                       [393.07114384, 248.39110006, 87.61575574], [393.69314056, 247.78157916, 88.73002876],
                       [98.74901939, 219.46930221, 80.6306816], [98.47494966, 220.42553803, 80.52821783],
                       [97.79246671, 219.20927275, 80.76255901], [98.84848169, 219.60345781, 81.61663775]])),
            # Test with zeros for all params
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to rtib
            (np.array([-9, 6, -9]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[-9, 6, -9], [0, 0, 0], [0, 0, 0], 7.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to ltib
            (np.array([0, 0, 0]), np.array([0, 2, -1]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0], [[0, 2, -1], [0, 0, 0], [0, 0, 0], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to rank
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([1, 0, -5]), np.array([0, 0, 0]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[0, 0, 0], [0, 0, 0], [1, 0, -5], 7.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to lank
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([2, -4, -5]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0], [[0, 0, 0], [0, 0, 0], [2, -4, -5], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to rtib, ltib, rank, and lank
            (np.array([-9, 6, -9]), np.array([0, 2, -1]), np.array([1, 0, -5]), np.array([2, -4, -5]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[-9, 6, -9], [0, 0, 0], [1, 0, -5], 7.0], [[0, 2, -1], [0, 0, 0], [2, -4, -5], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to knee_origin
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([[-7, 1, 2], [9, -8, 9]]),
             {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[0, 0, 0], [-7, 1, 2], [0, 0, 0], 7.0], [[0, 0, 0], [9, -8, 9], [0, 0, 0], 7.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, [-0.95257934, 0.13608276, 0.27216553], [0, 0, 0], nan_3d, nan_3d,
                       [0.59867109, -0.53215208, 0.59867109]])),
            # Testing when values are added to measurements
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightAnkleWidth': -38.0, 'LeftAnkleWidth': 18.0, 'RightTibialTorsion': 29.0, 'LeftTibialTorsion': -13.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[0, 0, 0], [0, 0, 0], [0, 0, 0], -12.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 16.0]],
             np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to mock_return_val
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
             [np.array([2, -5, 4]), np.array([8, -3, 1])],
             [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
             np.array([[2, -5, 4], nan_3d, nan_3d, [1.7018576, -4.25464401, 3.40371521], [8, -3, 1], nan_3d, nan_3d,
                       [7.07001889, -2.65125708, 0.88375236]])),
            # Testing when values are added to rtib, ltib, rank, lank, and knee_origin
            (np.array([-9, 6, -9]), np.array([0, 2, -1]), np.array([1, 0, -5]), np.array([2, -4, -5]),
             np.array([[-7, 1, 2], [9, -8, 9]]),
             {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], 7.0], [[0, 2, -1], [9, -8, 9], [2, -4, -5], 7.0]],
             np.array([[0, 0, 0], [-0.26726124, -0.80178373, -0.53452248], [0.14547859, -0.58191437, 0.80013226],
                       [-0.95257934, 0.13608276, 0.27216553], [0, 0, 0], [0.79317435, 0.49803971, -0.35047239],
                       [-0.11165737, 0.68466825, 0.72025136], [0.59867109, -0.53215208, 0.59867109]])),
            # Testing when values are added to rtib, ltib, rank, lank, knee_origin, and measurements
            (np.array([-9, 6, -9]), np.array([0, 2, -1]), np.array([1, 0, -5]), np.array([2, -4, -5]),
             np.array([[-7, 1, 2], [9, -8, 9]]),
             {'RightAnkleWidth': -38.0, 'LeftAnkleWidth': 18.0, 'RightTibialTorsion': 29.0, 'LeftTibialTorsion': -13.0},
             [np.array([0, 0, 0]), np.array([0, 0, 0])],
             [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0], [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0]],
             np.array([[0, 0, 0], [-0.30428137, -0.41913816, -0.85541572], [-0.00233238, -0.89766624, 0.4406698],
                       [-0.95257934, 0.13608276, 0.27216553], [0, 0, 0], [0.7477279, 0.63929183, -0.1794685],
                       [-0.287221, 0.55508569, 0.7806305], [0.59867109, -0.53215208, 0.59867109]])),
            # Testing when values are added to rtib, ltib, rank, lank, knee_origin, measurements, and mock_return_val
            (np.array([-9, 6, -9]), np.array([0, 2, -1]), np.array([1, 0, -5]), np.array([2, -4, -5]),
             np.array([[-7, 1, 2], [9, -8, 9]]),
             {'RightAnkleWidth': -38.0, 'LeftAnkleWidth': 18.0, 'RightTibialTorsion': 29.0, 'LeftTibialTorsion': -13.0},
             [np.array([2, -5, 4]), np.array([8, -3, 1])],
             [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0], [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0]],
             np.array([[2, -5, 4], [1.48891678, -5.83482493, 3.7953997], [1.73661348, -5.07447603, 4.96181124],
                       [1.18181818, -4.45454545, 3.81818182], [8, -3, 1], [8.87317138, -2.54514024, 1.17514093],
                       [7.52412119, -2.28213872, 1.50814815], [8.10540926, -3.52704628, 1.84327404]])),
            # Testing that when rank, lank, and knee_origin are composed of lists of ints and measurements values are
            #  ints
            (np.array([-9, 6, -9]), np.array([0, 2, -1]), [1, 0, -5], [2, -4, -5],
             [[-7, 1, 2], [9, -8, 9]],
             {'RightAnkleWidth': -38, 'LeftAnkleWidth': 18, 'RightTibialTorsion': 29, 'LeftTibialTorsion': -13},
             [np.array([2, -5, 4]), np.array([8, -3, 1])],
             [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0], [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0]],
             np.array([[2, -5, 4], [1.48891678, -5.83482493, 3.7953997], [1.73661348, -5.07447603, 4.96181124],
                       [1.18181818, -4.45454545, 3.81818182], [8, -3, 1], [8.87317138, -2.54514024, 1.17514093],
                       [7.52412119, -2.28213872, 1.50814815], [8.10540926, -3.52704628, 1.84327404]])),
            # Testing that when rank, lank, and knee_origin are composed of numpy arrays of ints and measurements
            # values are ints
            (np.array([-9, 6, -9], dtype='int'), np.array([0, 2, -1], dtype='int'), np.array([1, 0, -5], dtype='int'),
             np.array([2, -4, -5], dtype='int'),
             np.array([[-7, 1, 2], [9, -8, 9]], dtype='int'),
             {'RightAnkleWidth': -38, 'LeftAnkleWidth': 18, 'RightTibialTorsion': 29, 'LeftTibialTorsion': -13},
             [np.array([2, -5, 4]), np.array([8, -3, 1])],
             [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0], [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0]],
             np.array([[2, -5, 4], [1.48891678, -5.83482493, 3.7953997], [1.73661348, -5.07447603, 4.96181124],
                       [1.18181818, -4.45454545, 3.81818182], [8, -3, 1], [8.87317138, -2.54514024, 1.17514093],
                       [7.52412119, -2.28213872, 1.50814815], [8.10540926, -3.52704628, 1.84327404]])),
            # Testing that when rank, lank, and knee_origin are composed of lists of floats and measurements values
            # are floats
            (np.array([-9.0, 6.0, -9.0]), np.array([0.0, 2.0, -1.0]), [1.0, 0.0, -5.0], [2.0, -4.0, -5.0],
             [[-7.0, 1.0, 2.0], [9.0, -8.0, 9.0]],
             {'RightAnkleWidth': -38.0, 'LeftAnkleWidth': 18.0, 'RightTibialTorsion': 29.0, 'LeftTibialTorsion': -13.0},
             [np.array([2, -5, 4]), np.array([8, -3, 1])],
             [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0], [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0]],
             np.array([[2, -5, 4], [1.48891678, -5.83482493, 3.7953997], [1.73661348, -5.07447603, 4.96181124],
                       [1.18181818, -4.45454545, 3.81818182], [8, -3, 1], [8.87317138, -2.54514024, 1.17514093],
                       [7.52412119, -2.28213872, 1.50814815], [8.10540926, -3.52704628, 1.84327404]])),
            # Testing that when rank, lank, and knee_origin are composed of numpy arrays of floats and measurements
            # values are floats
            (np.array([-9, 6, -9], dtype='float'), np.array([0, 2, -1], dtype='float'),
             np.array([1, 0, -5], dtype='float'), np.array([2, -4, -5], dtype='float'),
             np.array([[-7, 1, 2], [9, -8, 9]], dtype='float'),
             {'RightAnkleWidth': -38.0, 'LeftAnkleWidth': 18.0, 'RightTibialTorsion': 29.0, 'LeftTibialTorsion': -13.0},
             [np.array([2, -5, 4]), np.array([8, -3, 1])],
             [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0], [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0]],
             np.array([[2, -5, 4], [1.48891678, -5.83482493, 3.7953997], [1.73661348, -5.07447603, 4.96181124],
                       [1.18181818, -4.45454545, 3.81818182], [8, -3, 1], [8.87317138, -2.54514024, 1.17514093],
                       [7.52412119, -2.28213872, 1.50814815], [8.10540926, -3.52704628, 1.84327404]]))])
    def test_ankle_axis_calc(self, rtib, ltib, rank, lank, knee_origin, measurements, mock_return_val,
                             expected_mock_args, expected):
        """
        This test provides coverage of the ankle_axis_calc function in the class CGM in pycgm.py, defined as
        ankle_axis_calc(rtib, ltib, rank, lank, knee_origin, measurements)

        This test takes 9 parameters:
        rtib, ltib, rank, lank : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        knee_origin : array
            A 2x3 ndarray of the right and left knee origin vectors (joint centers).
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        mock_return_val : list
            The value to be returned by the mock for find_joint_center
        expected_mock_args : list
            The expected arguments used to call the mocked function, find_joint_center
        expected : array
            An 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z axis components, left ankle
            origin, and left ankle x, y, and z axis components.

        This test is checking to make sure the ankle joint center and axis are calculated correctly given the input
        parameters. This tests mocks find_joint_center to make sure the correct parameters are being passed into it
        given the parameters passed into ankle_axis_calc, expected_mock_args, and to also ensure that ankle_axis_calc
        returns the correct value considering the return value of find_joint_center, mock_return_val.

        Calculated using Ankle Axis Calculation (ref. Clinical Gait Analysis hand book, Baker2013).

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when rank, lank, and knee_origin are composed of lists of ints,
        numpy arrays of ints, lists of floats, and numpy arrays of floats and measurements values are ints and
        floats. The values of rtib and ltib were kept as numpy arrays as lists would cause errors on the following
        lines in pycgm.py as lists cannot be subtracted by each other:
        tib_ank_r = rtib - rank
        tib_ank_l = ltib - lank
        """
        with patch.object(CGM, 'find_joint_center', side_effect=mock_return_val) as mock_find_joint_center:
            result = CGM.ankle_axis_calc(rtib, ltib, rank, lank, knee_origin, measurements)

        # Asserting that there were only 2 calls to find_joint_center
        np.testing.assert_equal(mock_find_joint_center.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to find_joint_center
        np.testing.assert_almost_equal(expected_mock_args[0][0], mock_find_joint_center.call_args_list[0][0][0],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][1], mock_find_joint_center.call_args_list[0][0][1],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][2], mock_find_joint_center.call_args_list[0][0][2],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][3], mock_find_joint_center.call_args_list[0][0][3],
                                       rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to find_joint_center
        np.testing.assert_almost_equal(expected_mock_args[1][0], mock_find_joint_center.call_args_list[1][0][0],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][1], mock_find_joint_center.call_args_list[1][0][1],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][2], mock_find_joint_center.call_args_list[1][0][2],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][3], mock_find_joint_center.call_args_list[1][0][3],
                                       rounding_precision)

        # Asserting that ankle_axis_calc returned the correct result given the return value given by mocked
        # find_joint_center
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["rtoe", "ltoe", "ankle_axis", "measurements", "expected"], [
        # Test from running sample data
        (np.array([442.81997681, 381.62280273, 42.66047668]), np.array([39.43652725, 382.44522095, 41.78911591]),
         np.array([[393.76181608, 247.67829633, 87.73775041], [394.48171575, 248.37201348, 87.715368],
                   [393.07114384, 248.39110006, 87.61575574], [393.69314056, 247.78157916, 88.73002876],
                   [98.74901939, 219.46930221, 80.6306816], [98.47494966, 220.42553803, 80.52821783],
                   [97.79246671, 219.20927275, 80.76255901], [98.84848169, 219.60345781, 81.61663775]]),
         {'RightStaticRotOff': 0.015683497632642047, 'RightStaticPlantFlex': 0.2702417907002757,
          'LeftStaticRotOff': 0.009402910292403022, 'LeftStaticPlantFlex': 0.20251085737834015},
         np.array([[442.81997681, 381.62280273, 42.66047668], [442.8462412676692, 381.6513024007671, 43.65972537588915],
                   [441.8773505621594, 381.95630350196393, 42.67574106247485],
                   [442.48716163075153, 380.68048378251575, 42.69610043598381],
                   [39.43652725, 382.44522095, 41.78911591],
                   [39.566526257915626, 382.50901000467115, 42.778575967950964],
                   [38.493133283871245, 382.1460684058263, 41.932348504971834],
                   [39.74166341694723, 381.493150197213, 41.81040458481808]])),
        # Test with zeros for all params
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         np.array([[0, 0, 0], rand_coor, [0, 0, 0], rand_coor, [0, 0, 0], rand_coor, [0, 0, 0], rand_coor]),
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
        # Testing when adding values in rtoe
        (np.array([4, 0, -3]), np.array([0, 0, 0]),
         np.array([[0, 0, 0], rand_coor, [0, 0, 0], rand_coor, [0, 0, 0], rand_coor, [0, 0, 0], rand_coor]),
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         np.array([[4, 0, -3], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
        # Testing when adding values in  ltoe
        (np.array([0, 0, 0]), np.array([-1, 7, 2]),
         np.array([[0, 0, 0], rand_coor, [0, 0, 0], rand_coor, [0, 0, 0], rand_coor, [0, 0, 0], rand_coor]),
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [-1, 7, 2], nan_3d, nan_3d, nan_3d])),
        # Testing when adding values in rtoe and ltoe
        (np.array([4, 0, -3]), np.array([-1, 7, 2]),
         np.array([[0, 0, 0], rand_coor, [0, 0, 0], rand_coor, [0, 0, 0], rand_coor, [0, 0, 0], rand_coor]),
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         np.array([[4, 0, -3], nan_3d, nan_3d, nan_3d, [-1, 7, 2], nan_3d, nan_3d, nan_3d])),
        # Testing when adding values in measurements
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         np.array([[0, 0, 0], rand_coor, [0, 0, 0], rand_coor, [0, 0, 0], rand_coor, [0, 0, 0], rand_coor]),
         {'RightStaticRotOff': -12.0, 'RightStaticPlantFlex': 20.0, 'LeftStaticRotOff': 34.0,
          'LeftStaticPlantFlex': -70.0},
         np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
        # Testing when adding values to ankle_jc_r and ankle_flexion_r
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         np.array([[-3, 5, 2], rand_coor, [0, 0, 0], rand_coor, [2, 3, 9], rand_coor, [0, 0, 0], rand_coor]),
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
        # Testing when adding values to ankle_jc_l and ankle_flexion_l
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         np.array([[0, 0, 0], rand_coor, [-1, 0, 2], rand_coor, [0, 0, 0], rand_coor, [9, 3, -4], rand_coor]),
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         np.array([[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
        # Testing when adding values to ankle_axis
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         np.array([[-3, 5, 2], rand_coor, [-1, 0, 2], rand_coor, [2, 3, 9], rand_coor, [9, 3, -4], rand_coor]),
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         np.array([[0, 0, 0], [-0.84215192, -0.33686077, -0.42107596], [-0.23224564, -0.47815279, 0.8470135],
                   [-0.48666426, 0.81110711, 0.32444284], [0, 0, 0], [0.39230172, -0.89525264, 0.21123939],
                   [0.89640737, 0.32059014, -0.30606502], [0.20628425, 0.30942637, 0.92827912]])),
        # Testing when adding values in rtoe, ltoe, and measurements
        (np.array([4, 0, -3]), np.array([-1, 7, 2]),
         np.array([[0, 0, 0], rand_coor, [0, 0, 0], rand_coor, [0, 0, 0], rand_coor, [0, 0, 0], rand_coor]),
         {'RightStaticRotOff': -12.0, 'RightStaticPlantFlex': 20.0, 'LeftStaticRotOff': 34.0,
          'LeftStaticPlantFlex': -70.0},
         np.array([[4, 0, -3], nan_3d, nan_3d, nan_3d, [-1, 7, 2], nan_3d, nan_3d, nan_3d])),
        # Testing when adding values in rtoe, ltoe, and ankle_axis
        (np.array([4, 0, -3]), np.array([-1, 7, 2]),
         np.array([[-3, 5, 2], rand_coor, [-1, 0, 2], rand_coor, [2, 3, 9], rand_coor, [9, 3, -4], rand_coor]),
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         np.array([[4, 0, -3], [3.31958618, -0.27216553, -3.68041382], [3.79484752, -0.82060994, -2.46660354],
                   [3.29647353, 0.50251891, -2.49748109], [-1, 7, 2], [-1.49065338, 6.16966351, 1.73580203],
                   [-0.20147784, 6.69287609, 1.48227684], [-0.65125708, 6.53500945, 2.81373347]])),
        # Testing when adding values in rtoe, ltoe, ankle_axis, and measurements
        (np.array([4, 0, -3]), np.array([-1, 7, 2]),
         np.array([[-3, 5, 2], rand_coor, [-1, 0, 2], rand_coor, [2, 3, 9], rand_coor, [9, 3, -4], rand_coor]),
         {'RightStaticRotOff': -12.0, 'RightStaticPlantFlex': 20.0, 'LeftStaticRotOff': 34.0,
          'LeftStaticPlantFlex': -70.0},
         np.array([[4, 0, -3], [3.08005417, 0.34770638, -2.81889243], [4.00614173, -0.44911697, -2.10654814],
                   [4.3919974, 0.82303962, -2.58897224], [-1, 7, 2], [-1.58062909, 6.83398388, 1.20293758],
                   [-1.59355918, 7.75640754, 2.27483654], [-0.44272327, 7.63268181, 1.46226738]])),
        # Testing that when rtoe, ltoe, and ankle_axis are composed of lists of ints and measurements values are ints
        (np.array([4, 0, -3]), np.array([-1, 7, 2]),
         np.array([[-3, 5, 2], rand_coor, [-1, 0, 2], rand_coor, [2, 3, 9], rand_coor, [9, 3, -4], rand_coor]),
         {'RightStaticRotOff': -12, 'RightStaticPlantFlex': 20, 'LeftStaticRotOff': 34, 'LeftStaticPlantFlex': -70},
         np.array([[4, 0, -3], [3.08005417, 0.34770638, -2.81889243], [4.00614173, -0.44911697, -2.10654814],
                   [4.3919974, 0.82303962, -2.58897224], [-1, 7, 2], [-1.58062909, 6.83398388, 1.20293758],
                   [-1.59355918, 7.75640754, 2.27483654], [-0.44272327, 7.63268181, 1.46226738]])),
        # Testing that when rtoe, ltoe, and ankle_axis are composed of numpy arrays of ints and measurements values
        # are ints
        (np.array([4, 0, -3], dtype='int'), np.array([-1, 7, 2], dtype='int'),
         np.array([[-3, 5, 2], rand_coor, [-1, 0, 2], rand_coor, [2, 3, 9], rand_coor, [9, 3, -4], rand_coor],
                  dtype='int'),
         {'RightStaticRotOff': -12, 'RightStaticPlantFlex': 20, 'LeftStaticRotOff': 34, 'LeftStaticPlantFlex': -70},
         np.array([[4, 0, -3], [3.08005417, 0.34770638, -2.81889243], [4.00614173, -0.44911697, -2.10654814],
                   [4.3919974, 0.82303962, -2.58897224], [-1, 7, 2], [-1.58062909, 6.83398388, 1.20293758],
                   [-1.59355918, 7.75640754, 2.27483654], [-0.44272327, 7.63268181, 1.46226738]])),
        # Testing that when rtoe, ltoe, and ankle_axis are composed of lists of floats and measurements values are
        # floats
        (np.array([4.0, 0.0, -3.0]), np.array([-1.0, 7.0, 2.0]),
         np.array(
             [[-3.0, 5.0, 2.0], rand_coor, [-1.0, 0.0, 2.0], rand_coor, [2.0, 3.0, 9.0], rand_coor, [9.0, 3.0, -4.0],
              rand_coor]),
         {'RightStaticRotOff': -12.0, 'RightStaticPlantFlex': 20.0, 'LeftStaticRotOff': 34.0,
          'LeftStaticPlantFlex': -70.0},
         np.array([[4, 0, -3], [3.08005417, 0.34770638, -2.81889243], [4.00614173, -0.44911697, -2.10654814],
                   [4.3919974, 0.82303962, -2.58897224], [-1, 7, 2], [-1.58062909, 6.83398388, 1.20293758],
                   [-1.59355918, 7.75640754, 2.27483654], [-0.44272327, 7.63268181, 1.46226738]])),
        # Testing that when rtoe, ltoe, and ankle_axis are composed of numpy arrays of floats and measurements values
        #  are floats
        (np.array([4.0, 0.0, -3.0], dtype='float'), np.array([-1.0, 7.0, 2.0], dtype='float'),
         np.array([[-3, 5, 2], rand_coor, [-1, 0, 2], rand_coor, [2, 3, 9], rand_coor, [9, 3, -4], rand_coor],
                  dtype='float'),
         {'RightStaticRotOff': -12.0, 'RightStaticPlantFlex': 20.0, 'LeftStaticRotOff': 34.0,
          'LeftStaticPlantFlex': -70.0},
         np.array([[4, 0, -3], [3.08005417, 0.34770638, -2.81889243], [4.00614173, -0.44911697, -2.10654814],
                   [4.3919974, 0.82303962, -2.58897224], [-1, 7, 2], [-1.58062909, 6.83398388, 1.20293758],
                   [-1.59355918, 7.75640754, 2.27483654], [-0.44272327, 7.63268181, 1.46226738]]))])
    def test_foot_axis_calc(self, rtoe, ltoe, ankle_axis, measurements, expected):
        """
        This test provides coverage of the foot_axis_calc function in the class CGM in pycgm.py, defined as
        foot_axis_calc(rtoe, ltoe, ankle_axis, measurements)

        This test takes 5 parameters:
        rtoe, ltoe : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_axis : array
            An 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        expected : array
            An 8x3 ndarray that contains the right foot origin, right foot x, y, and z axis components,
            left foot origin, and left foot x, y, and z axis components.

        This test is checking to make sure the foot joint center and axis are calculated correctly given the input
        parameters. It calculates the right and left foot joint axis by rotating uncorrect foot joint axes about
        offset angle.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when rtoe, ltoe, and ankle_axis are composed of lists of ints, numpy arrays
        of ints, lists of floats, and numpy arrays of floats and measurements values are ints and floats.
        """
        result = CGM.foot_axis_calc(rtoe, ltoe, ankle_axis, measurements)
        np.testing.assert_almost_equal(result, expected, rounding_precision)


class TestUpperBodyAxis():
    """
    This class tests the upper body axis functions in the class CGM in pycgm.py:
    head_axis_calc
    thorax_axis_calc
    shoulder_axis_calc
    elbow_axis_calc
    """

    nan_3d = np.array([np.nan, np.nan, np.nan])
    rand_coor = np.array([np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)])

    @pytest.mark.parametrize(["rfhd", "lfhd", "rbhd", "lbhd", "measurements", "expected"], [
        # Test from running sample data
        (np.array([325.82983398, 402.55450439, 1722.49816895]), np.array([184.55158997, 409.68713379, 1721.34289551]),
         np.array([304.39898682, 242.91339111, 1694.97497559]), np.array([197.8621521, 251.28889465, 1696.90197754]),
         {'HeadOffset': 0.2571990469310653},
         [[255.19071197509766, 406.1208190917969, 1721.9205322265625],
          [255.21685582510975, 407.11593887758056, 1721.8253843887082],
          [254.19105385179665, 406.146809183757, 1721.9176771191715],
          [255.19034370229795, 406.2160090443217, 1722.9159912851449]]),
        # Basic test with a variance of 1 in the x and y dimensions of the markers
        (np.array([0, 1, 0]), np.array([1, 1, 0]), np.array([0, 0, 0]), np.array([1, 0, 0]),
         {'HeadOffset': 0.0},
         [[0.5, 1, 0], [0.5, 2, 0], [1.5, 1, 0], [0.5, 1, -1]]),
        # Setting the markers so there's no variance in the x-dimension
        (np.array([0, 1, 0]), np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
         {'HeadOffset': 0.0},
         [[0, 1, 0], nan_3d, nan_3d, nan_3d]),
        # Setting the markers so there's no variance in the y-dimension
        (np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([1, 0, 0]),
         {'HeadOffset': 0.0},
         [[0.5, 0, 0], nan_3d, nan_3d, nan_3d]),
        # Setting each marker in a different xy quadrant
        (np.array([1, 1, 0]), np.array([-1, 1, 0]), np.array([1, -1, 0]), np.array([-1, -1, 0]),
         {'HeadOffset': 0.0},
         [[0, 1, 0], [0, 2, 0], [-1, 1, 0], [0, 1, 1]]),
        # Setting values of the markers so that midpoints will be on diagonals
        (np.array([1, 2, 0]), np.array([-2, 1, 0]), np.array([2, -1, 0]), np.array([-1, -2, 0]),
         {'HeadOffset': 0.0},
         [[-0.5, 1.5, 0], [-0.81622777, 2.4486833, 0], [-1.4486833, 1.18377223, 0], [-0.5, 1.5, 1]]),
        # Adding the value of 1 in the z dimension for all 4 markers
        (np.array([0, 1, 1]), np.array([1, 1, 1]), np.array([0, 0, 1]), np.array([1, 0, 1]),
         {'HeadOffset': 0.0},
         [[0.5, 1, 1], [0.5, 2, 1], [1.5, 1, 1], [0.5, 1, 0]]),
        # Setting the z dimension value higher for lfhd and lbhd
        (np.array([0, 1, 1]), np.array([1, 1, 2]), np.array([0, 0, 1]), np.array([1, 0, 2]),
         {'HeadOffset': 0.0},
         [[0.5, 1, 1.5], [0.5, 2, 1.5], [1.20710678, 1, 2.20710678], [1.20710678, 1, 0.79289322]]),
        # Setting the z dimension value higher for lfhd and rfhd
        (np.array([0, 1, 2]), np.array([1, 1, 2]), np.array([0, 0, 1]), np.array([1, 0, 1]),
         {'HeadOffset': 0.0},
         [[0.5, 1, 2], [0.5, 1.70710678, 2.70710678], [1.5, 1, 2], [0.5, 1.70710678, 1.29289322]]),
        # Adding a value for HeadOffset
        (np.array([0, 1, 0]), np.array([1, 1, 0]), np.array([0, 0, 0]), np.array([1, 0, 0]),
         {'HeadOffset': 0.5},
         [[0.5, 1, 0], [0.5, 1.87758256, 0.47942554], [1.5, 1, 0], [0.5, 1.47942554, -0.87758256]]),
        # Testing that when rfhd, lfhd, rbhd, and lbhd are numpy arrays of ints and headOffset is an int
        (np.array([0, 1, 0], dtype='int'), np.array([1, 1, 0], dtype='int'), np.array([0, 0, 0], dtype='int'),
         np.array([1, 0, 0], dtype='int'),
         {'HeadOffset': 1},
         [[0.5, 1, 0], [0.5, 1.5403023058681398, 0.8414709848078965], [1.5, 1, 0],
          [0.5, 1.8414709848078965, -0.5403023058681398]]),
        # Testing that when rfhd, lfhd, rbhd, and lbhd are numpy arrays of floats and headOffset is a float
        (np.array([0.0, 1.0, 0.0], dtype='float'), np.array([1.0, 1.0, 0.0], dtype='float'),
         np.array([0.0, 0.0, 0.0], dtype='float'), np.array([1.0, 0.0, 0.0], dtype='float'),
         {'HeadOffset': 1.0},
         [[0.5, 1, 0], [0.5, 1.5403023058681398, 0.8414709848078965], [1.5, 1, 0],
          [0.5, 1.8414709848078965, -0.5403023058681398]])])
    def test_head_axis_calc(self, rfhd, lfhd, rbhd, lbhd, measurements, expected):
        """
        This test provides coverage of the head_axis_calc function in the class CGM in pycgm.py, defined as
        head_axis_calc(rfhd, lfhd, rbhd, lbhd, measurements)

        This test takes 6 parameters:
        rfhd, lfhd, rbhd, lbhd : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        expected : array
            A 4x3 ndarray that contains the head origin and the head x, y, and z axis components.

        This test is checking to make sure the head joint center and axis are calculated correctly given the input
        parameters.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when rfhd, lfhd, rbhd, and lbhd are composed of numpy arrays of ints and
        numpy arrays of floats and measurements values are ints and floats. The values of rfhd, lfhd, rbhd,
        and lbhd were kept as numpy arrays as lists would cause errors from lines like the following in pycgm.py as
        lists cannot be added together:
        front = (rfhd + lfhd) / 2.0
        """
        result = CGM.head_axis_calc(rfhd, lfhd, rbhd, lbhd, measurements)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["clav", "c7", "strn", "t10", "expected"], [
        # Test from running sample data
        (np.array([256.78051758, 371.28042603, 1459.70300293]), np.array([251.22619629, 229.75683594, 1533.77624512]),
         np.array([251.67492676, 414.10391235, 1292.08508301]), np.array([228.64323425, 192.32041931, 1279.6418457]),
         [[256.149810236564, 364.3090603933987, 1459.6553639290375],
          [256.23991128535846, 365.30496976939753, 1459.662169500559],
          [257.1435863244796, 364.21960599061947, 1459.588978712983],
          [256.0843053658035, 364.32180498523223, 1458.6575930699294]]),
        # Basic test with a variance of 1 in the x and y dimensions of the markers
        (np.array([1, 0, 0]), np.array([1, 1, 0]), np.array([0, 0, 0]), np.array([0, 1, 0]),
         [[1, 7, 0], [1, 6, 0], [1, 7, 1], [0, 7, 0]]),
        # Setting the markers so there's no variance in the x-dimension
        (np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([0, 1, 0]),
         [nan_3d, nan_3d, nan_3d, nan_3d]),
        # Setting the markers so there's no variance in the y-dimension
        (np.array([1, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
         [nan_3d, nan_3d, nan_3d, nan_3d]),
        # Setting each marker in a different xy quadrant
        (np.array([-1, -1, 0]), np.array([-1, 1, 0]), np.array([1, -1, 0]), np.array([1, 1, 0]),
         [[-1, 6, 0], [-1, 5, 0], [-1, 6, -1], [0, 6, 0]]),
        # Setting values of the markers so that midpoints will be on diagonals
        (np.array([-1, -2, 0]), np.array([-2, 1, 0]), np.array([2, -1, 0]), np.array([1, 2, 0]),
         [[-3.21359436, 4.64078309, 0], [-2.8973666, 3.69209979, 0], [-3.21359436, 4.64078309, -1],
          [-2.26491106, 4.95701085, 0]]),
        # Adding the value of 1 in the z dimension for all 4 markers
        (np.array([1, 0, 1]), np.array([1, 1, 1]), np.array([0, 0, 1]), np.array([0, 1, 1]),
         [[1, 7, 1], [1, 6, 1], [1, 7, 2], [0, 7, 1]]),
        # Setting the z dimension value higher for c7 and clav
        (np.array([1, 0, 2]), np.array([1, 1, 2]), np.array([0, 0, 1]), np.array([0, 1, 1]),
         [[1, 7, 2], [1, 6, 2], [0.29289322, 7, 2.70710678], [0.29289322, 7, 1.29289322]]),
        # Setting the z dimension value higher for c7 and t10
        (np.array([1, 0, 1]), np.array([1, 1, 2]), np.array([0, 0, 1]), np.array([0, 1, 2]),
         [[1, 4.94974747, 5.94974747], [1, 4.24264069, 5.24264069], [1, 4.24264069, 6.65685425],
          [0, 4.94974747, 5.94974747]]),
        # Testing that when clav, c7, strn, and t10 are numpy arrays of ints
        (np.array([1, 0, 1], dtype='int'), np.array([1, 1, 2], dtype='int'), np.array([0, 0, 1], dtype='int'),
         np.array([0, 1, 2], dtype='int'),
         [[1, 4.94974747, 5.94974747], [1, 4.24264069, 5.24264069], [1, 4.24264069, 6.65685425],
          [0, 4.94974747, 5.94974747]]),
        # Testing that when clav, c7, strn, and t10 are numpy arrays of floats
        (np.array([1.0, 0.0, 1.0], dtype='float'), np.array([1.0, 1.0, 2.0], dtype='float'),
         np.array([0.0, 0.0, 1.0], dtype='float'), np.array([0.0, 1.0, 2.0], dtype='float'),
         [[1, 4.94974747, 5.94974747], [1, 4.24264069, 5.24264069], [1, 4.24264069, 6.65685425],
          [0, 4.94974747, 5.94974747]])])
    def test_thorax_axis_calc(self, clav, c7, strn, t10, expected):
        """
        This test provides coverage of the thorax_axis_calc function in the class CGM in pycgm.py, defined as
        thorax_axis_calc(clav, c7, strn, t10)

        This test takes 6 parameters:
        clav, c7, strn, t10 : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        expected : array
            A 4x3 ndarray that contains the thorax origin and the thorax x, y, and z axis components.

        This test is checking to make sure the thorax joint center and axis are calculated correctly given the input
        parameters.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when clav, c7, strn, and t10 are composed of numpy arrays of ints and numpy
        arrays of floats . The values of clav, c7, strn, and t10 were kept as numpy arrays as lists would cause
        errors from lines like the following in pycgm.py as lists cannot be added together:
        upper = (clav + c7) / 2.0
        """
        result = CGM.thorax_axis_calc(clav, c7, strn, t10)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(
        ["rsho", "lsho", "thorax_origin", "wand", "measurements", "mock_return_val", "expected_mock_args", "expected"],
        [
            # Test from running sample data
            (np.array([428.88496562, 270.552948, 1500.73010254]),
             np.array([68.24668121, 269.01049805, 1510.1072998]),
             np.array([256.14981023656401, 364.30906039339868, 1459.6553639290375]),
             np.array([[255.92550222678443, 364.32269504976051, 1460.6297868417887],
                       [256.42380097331767, 364.27770361353487, 1460.6165849382387]]),
             {'RightShoulderOffset': 40.0, 'LeftShoulderOffset': 40.0},
             [[429.66971693, 275.06718208, 1453.95397769], [64.51952733, 274.93442161, 1463.63133339]],
             [[[255.92550223, 364.32269505, 1460.62978684], [256.14981024, 364.30906039, 1459.65536393],
               [428.88496562, 270.552948, 1500.73010254], 47.0],
              [[256.42380097, 364.27770361, 1460.61658494], [256.14981024, 364.30906039, 1459.65536393],
               [68.24668121, 269.01049805, 1510.1072998], 47.0]],
             [[429.66971693, 275.06718208, 1453.95397769], [430.1275099, 275.95136234, 1454.04698775],
              [429.68641377, 275.16322961, 1452.95874099], [428.7808149, 275.52434742, 1453.98318456],
              [64.51952733, 274.93442161, 1463.63133339], [64.10400325, 275.83192827, 1463.77905454],
              [64.59882848, 274.80838069, 1464.62018374], [65.42564601, 275.35702721, 1463.61253313]]),
            # Basic test with zeros for all params
            (np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([0, 0, 0]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightShoulderOffset': 0.0, 'LeftShoulderOffset': 0.0},
             [[0, 0, 0], [0, 0, 0]],
             [[[0, 0, 0], [0, 0, 0], np.array([0, 0, 0]), 7.0], [[0, 0, 0], [0, 0, 0], np.array([0, 0, 0]), 7.0]],
             [[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d]),
            # Testing when values are added to rsho and lsho
            (np.array([2, -1, 3]), np.array([-3, 1, 2]),
             np.array([0, 0, 0]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightShoulderOffset': 0.0, 'LeftShoulderOffset': 0.0},
             [[0, 0, 0], [0, 0, 0]],
             [[[0, 0, 0], [0, 0, 0], np.array([2, -1, 3]), 7.0], [[0, 0, 0], [0, 0, 0], np.array([-3, 1, 2]), 7.0]],
             [[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d]),
            # Testing when a value is added to thorax_origin
            (np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([5, -2, 7]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightShoulderOffset': 0.0, 'LeftShoulderOffset': 0.0},
             [[0, 0, 0], [0, 0, 0]],
             [[[0, 0, 0], [5, -2, 7], np.array([0, 0, 0]), 7.0], [[0, 0, 0], [5, -2, 7], np.array([0, 0, 0]), 7.0]],
             [[0, 0, 0], nan_3d, nan_3d, [0.56613852, -0.22645541, 0.79259392], [0, 0, 0], nan_3d, nan_3d,
              [0.56613852, -0.22645541, 0.79259392]]),
            # Testing when a value is added to wand
            (np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([0, 0, 0]),
             np.array([[2, 6, -4], [-3, 5, 2]]),
             {'RightShoulderOffset': 0.0, 'LeftShoulderOffset': 0.0},
             [[0, 0, 0], [0, 0, 0]],
             [[[2, 6, -4], [0, 0, 0], np.array([0, 0, 0]), 7.0], [[-3, 5, 2], [0, 0, 0], np.array([0, 0, 0]), 7.0]],
             [[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d]),
            # Testing when values are added to RightShoulderOffset and LeftShoulderOffset
            (np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([0, 0, 0]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightShoulderOffset': 20.0, 'LeftShoulderOffset': -20.0},
             [[0, 0, 0], [0, 0, 0]],
             [[[0, 0, 0], [0, 0, 0], np.array([0, 0, 0]), 27.0], [[0, 0, 0], [0, 0, 0], np.array([0, 0, 0]), -13.0]],
             [[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d]),
            # Testing when values are added to mock_return_val
            (np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([0, 0, 0]),
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightShoulderOffset': 0.0, 'LeftShoulderOffset': 0.0},
             [[8, -1, -4], [-7, -9, -1]],
             [[[0, 0, 0], [0, 0, 0], np.array([0, 0, 0]), 7.0], [[0, 0, 0], [0, 0, 0], np.array([0, 0, 0]), 7.0]],
             [[8, -1, -4], nan_3d, nan_3d, [7.11111111, -0.88888889, -3.55555556], [-7, -9, -1], nan_3d, nan_3d,
              [-6.38840716, -8.21366635, -0.91262959]]),
            # Testing when values are added to all params
            (np.array([3, -5, 2]), np.array([-7, 3, 9]),
             np.array([-1, -9, -5]),
             np.array([[-7, -1, 5], [5, -9, 2]]),
             {'RightShoulderOffset': -6.0, 'LeftShoulderOffset': 42.0},
             [[8, -1, -4], [-7, -9, -1]],
             [[[-7, -1, 5], [-1, -9, -5], np.array([3, -5, 2]), 1.0],
              [[5, -9, 2], [-1, -9, -5], np.array([-7, 3, 9]), 49.0]],
             [[8, -1, -4], [7.57573593, -0.43431458, -4.70710678], [8.51498105, -1.49157282, -4.70224688],
              [7.2551547, -1.66208471, -4.08276059], [-7, -9, -1], [-7, -8, -1], [-6.4452998, -9, -0.16794971],
              [-6.16794971, -9, -1.5547002]]),
            # Testing that when rsho, lsho, and wand are lists of ints and measurements values are ints
            ([3, -5, 2], [-7, 3, 9],
             np.array([-1, -9, -5]),
             [[-7, -1, 5], [5, -9, 2]],
             {'RightShoulderOffset': -6, 'LeftShoulderOffset': 42},
             [[8, -1, -4], [-7, -9, -1]],
             [[[-7, -1, 5], [-1, -9, -5], np.array([3, -5, 2]), 1.0],
              [[5, -9, 2], [-1, -9, -5], np.array([-7, 3, 9]), 49.0]],
             [[8, -1, -4], [7.57573593, -0.43431458, -4.70710678], [8.51498105, -1.49157282, -4.70224688],
              [7.2551547, -1.66208471, -4.08276059], [-7, -9, -1], [-7, -8, -1], [-6.4452998, -9, -0.16794971],
              [-6.16794971, -9, -1.5547002]]),
            # Testing that when rsho, lsho, and wand are numpy arrays of ints and measurements values are ints
            (np.array([3, -5, 2], dtype='int'), np.array([-7, 3, 9], dtype='int'),
             np.array([-1, -9, -5], dtype='int'),
             np.array([[-7, -1, 5], [5, -9, 2]], dtype='int'),
             {'RightShoulderOffset': -6, 'LeftShoulderOffset': 42},
             [[8, -1, -4], [-7, -9, -1]],
             [[[-7, -1, 5], [-1, -9, -5], np.array([3, -5, 2]), 1.0],
              [[5, -9, 2], [-1, -9, -5], np.array([-7, 3, 9]), 49.0]],
             [[8, -1, -4], [7.57573593, -0.43431458, -4.70710678], [8.51498105, -1.49157282, -4.70224688],
              [7.2551547, -1.66208471, -4.08276059], [-7, -9, -1], [-7, -8, -1], [-6.4452998, -9, -0.16794971],
              [-6.16794971, -9, -1.5547002]]),
            # Testing that when rsho, lsho, and wand are lists of floats and measurements values are floats
            ([3.0, -5.0, 2.0], [-7.0, 3.0, 9.0],
             np.array([-1.0, -9.0, -5.0]),
             [[-7.0, -1.0, 5.0], [5.0, -9.0, 2.0]],
             {'RightShoulderOffset': -6.0, 'LeftShoulderOffset': 42.0},
             [[8, -1, -4], [-7, -9, -1]],
             [[[-7, -1, 5], [-1, -9, -5], np.array([3, -5, 2]), 1.0],
              [[5, -9, 2], [-1, -9, -5], np.array([-7, 3, 9]), 49.0]],
             [[8, -1, -4], [7.57573593, -0.43431458, -4.70710678], [8.51498105, -1.49157282, -4.70224688],
              [7.2551547, -1.66208471, -4.08276059], [-7, -9, -1], [-7, -8, -1], [-6.4452998, -9, -0.16794971],
              [-6.16794971, -9, -1.5547002]]),
            # Testing that when rsho, lsho, and wand are numpy arrays of floats and measurements values are floats
            (np.array([3.0, -5.0, 2.0], dtype='float'), np.array([-7.0, 3.0, 9.0], dtype='float'),
             np.array([-1.0, -9.0, -5.0], dtype='float'),
             np.array([[-7.0, -1.0, 5.0], [5.0, -9.0, 2.0]], dtype='float'),
             {'RightShoulderOffset': -6.0, 'LeftShoulderOffset': 42.0},
             [[8, -1, -4], [-7, -9, -1]],
             [[[-7, -1, 5], [-1, -9, -5], np.array([3, -5, 2]), 1.0],
              [[5, -9, 2], [-1, -9, -5], np.array([-7, 3, 9]), 49.0]],
             [[8, -1, -4], [7.57573593, -0.43431458, -4.70710678], [8.51498105, -1.49157282, -4.70224688],
              [7.2551547, -1.66208471, -4.08276059], [-7, -9, -1], [-7, -8, -1], [-6.4452998, -9, -0.16794971],
              [-6.16794971, -9, -1.5547002]])])
    def test_shoulder_axis_calc(self, rsho, lsho, thorax_origin, wand, measurements, mock_return_val,
                                expected_mock_args, expected):
        """
        This test provides coverage of the shoulder_axis_calc function in the class CGM in pycgm.py, defined as
        shoulder_axis_calc(rsho, lsho, thorax_origin, wand, measurements)

        This test takes 8 parameters:
        rsho, lsho : dict
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        thorax_origin : ndarray
            A 1x3 ndarray of the thorax origin vector (joint center).
        wand : ndarray
            A 2x3 ndarray containing the right wand marker x, y, and z positions and the
            left wand marker x, y, and z positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        mock_return_val : list
            The value to be returned by the mock for find_joint_center
        expected_mock_args : list
            The expected arguments used to call the mocked function, find_joint_center
        expected : array
            An 8x3 ndarray that contains the right shoulder origin, right shoulder x, y, and z axis components,
            left shoulder origin, and left shoulder x, y, and z axis components.

        This test is checking to make sure the shoulder joint center and axis are calculated correctly given the
        input parameters. This tests mocks find_joint_center to make sure the correct parameters are being passed
        into it given the parameters passed into shoulder_axis_calc, expected_mock_args, and to also ensure that
        shoulder_axis_calc returns the correct value considering the return value of find_joint_center, mock_return_val.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when rsho, lsho, and wand are composed of lists of ints, numpy arrays of
        ints, lists of floats, and numpy arrays of floats and measurements values are ints and floats. thorax_origin
        was kept as numpy arrays as lists would cause errors in lines like the following in pycgm.py as lists cannot
        be subtracted by each other:
        r_wand_direc = r_wand - thorax_origin
        """
        with patch.object(CGM, 'find_joint_center', side_effect=mock_return_val) as mock_find_joint_center:
            result = CGM.shoulder_axis_calc(rsho, lsho, thorax_origin, wand, measurements)

        # Asserting that there were only 2 calls to find_joint_center
        np.testing.assert_equal(mock_find_joint_center.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to find_joint_center
        np.testing.assert_almost_equal(expected_mock_args[0][0], mock_find_joint_center.call_args_list[0][0][0],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][1], mock_find_joint_center.call_args_list[0][0][1],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][2], mock_find_joint_center.call_args_list[0][0][2],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][3], mock_find_joint_center.call_args_list[0][0][3],
                                       rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to find_joint_center
        np.testing.assert_almost_equal(expected_mock_args[1][0], mock_find_joint_center.call_args_list[1][0][0],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][1], mock_find_joint_center.call_args_list[1][0][1],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][2], mock_find_joint_center.call_args_list[1][0][2],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][3], mock_find_joint_center.call_args_list[1][0][3],
                                       rounding_precision)

        # Asserting that shoulder_axis_calc returned the correct result given the return value given by mocked
        # find_joint_center
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(
        ["rsho", "lsho", "relb", "lelb", "rwra", "rwrb", "lwra", "lwrb", "thorax_axis", "shoulder_origin",
         "measurements", "mock_return_val", "expected_mock_args", "expected"], [
            # Test from running sample data
            (np.array([428.88476562, 270.552948, 1500.73010254]), np.array([68.24668121, 269.01049805, 1510.1072998]),
             np.array([658.90338135, 326.07580566, 1285.28515625]),
             np.array([-156.32162476, 335.2583313, 1287.39916992]),
             np.array([776.51898193, 495.68103027, 1108.38464355]),
             np.array([830.9072876, 436.75341797, 1119.11901855]),
             np.array([-249.28146362, 525.32977295, 1117.09057617]),
             np.array([-311.77532959, 477.22512817, 1125.1619873]),
             [[256.149810236564, 364.3090603933987, 1459.6553639290375], rand_coor,
              [257.1435863244796, 364.21960599061947, 1459.588978712983], rand_coor],
             np.array([[429.66951995, 275.06718615, 1453.95397813], [64.51952734, 274.93442161, 1463.6313334]]),
             {'RightElbowWidth': 74.0, 'LeftElbowWidth': 74.0, 'RightWristWidth': 55.0, 'LeftWristWidth': 55.0},
             [[633.66707588, 304.95542115, 1256.07799541], [-129.16952219, 316.8671644, 1258.06440717]],
             [[[429.78392325, 96.82482443, 904.56444296], [429.66951995, 275.06718615, 1453.95397813],
               [658.90338135, 326.07580566, 1285.28515625], -44.0],
              [[-409.6146956, 530.62802087, 1671.68201453], [64.51952734, 274.93442161, 1463.6313334],
               [-156.32162476, 335.2583313, 1287.39916992], 44.0]],
             [[633.66707587, 304.95542115, 1256.07799541], [633.8107013869995, 303.96579004975194, 1256.07658506845],
              [634.3524799178464, 305.0538658933253, 1256.799473014224],
              [632.9532180390149, 304.85083190737765, 1256.770431750491], [-129.16952218, 316.8671644, 1258.06440717],
              [-129.32391792749496, 315.8807291324946, 1258.008662931836],
              [-128.45117135279028, 316.79382333592827, 1257.3726028780698],
              [-128.49119037560908, 316.72030884193634, 1258.7843373067021]]),
            # Test with zeros for all params
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             [[0, 0, 0], rand_coor, [0, 0, 0], rand_coor],
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
             [[0, 0, 0], [0, 0, 0]],
             [[nan_3d, [0, 0, 0], [0, 0, 0], -7.0], [nan_3d, [0, 0, 0], [0, 0, 0], 7.0]],
             [[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d]),
            # Testing when values are added to rsho, lsho, relb, lelb, rwra, rwrb, lwra, and lwrb
            ((np.array([9, -7, -6]), np.array([3, -8, 5]), np.array([-9, 1, -4]), np.array([-4, 1, -6]),
              np.array([2, -3, 9]), np.array([-4, -2, -7]), np.array([-9, 1, -1]), np.array([-3, -4, -9]),
              [[0, 0, 0], rand_coor, [0, 0, 0], rand_coor],
              np.array([[0, 0, 0], [0, 0, 0]]),
              {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
              [[0, 0, 0], [0, 0, 0]],
              [[[149.87576359540907, -228.48721408225754, -418.8422716102348], [0, 0, 0], [-9, 1, -4], -7.0],
               [[282.73117218166414, -326.69276820761615, -251.76957615571214], [0, 0, 0], [-4, 1, -6], 7.0]],
              [[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d])),
            # Testing when values are added to thorax_axis
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             [[-5, -2, -3], rand_coor, [-9, 5, -5], rand_coor],
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
             [[0, 0, 0], [0, 0, 0]],
             [[nan_3d, [0, 0, 0], [0, 0, 0], -7.0], [nan_3d, [0, 0, 0], [0, 0, 0], 7.0]],
             [[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d]),
            # Testing when values are added to shoulder_origin
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             [[0, 0, 0], rand_coor, [0, 0, 0], rand_coor],
             np.array([[-2, -8, -3], [5, -3, 2]]),
             {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
             [[0, 0, 0], [0, 0, 0]],
             [[nan_3d, [-2, -8, -3], [0, 0, 0], -7.0], [nan_3d, [5, -3, 2], [0, 0, 0], 7.0]],
             [[0, 0, 0], nan_3d, nan_3d, [-0.2279211529192759, -0.9116846116771036, -0.3418817293789138], [0, 0, 0],
              nan_3d, nan_3d, [0.8111071056538127, -0.48666426339228763, 0.3244428422615251]]),
            # Testing when values are added to measurements
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             [[0, 0, 0], rand_coor, [0, 0, 0], rand_coor],
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightElbowWidth': -38.0, 'LeftElbowWidth': 6.0, 'RightWristWidth': 47.0, 'LeftWristWidth': -7.0},
             [[0, 0, 0], [0, 0, 0]],
             [[nan_3d, [0, 0, 0], [0, 0, 0], 12.0], [nan_3d, [0, 0, 0], [0, 0, 0], 10.0]],
             [[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d]),
            # Testing when values are added to mock_return_val
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
             [[0, 0, 0], rand_coor, [0, 0, 0], rand_coor],
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
             [[5, 4, -4], [6, 3, 5]],
             [[nan_3d, [0, 0, 0], [0, 0, 0], -7.0], [nan_3d, [0, 0, 0], [0, 0, 0], 7.0]],
             [[5, 4, -4], nan_3d, nan_3d, [4.337733821467478, 3.4701870571739826, -3.4701870571739826], [6, 3, 5],
              nan_3d, nan_3d, [5.2828628343993635, 2.6414314171996818, 4.4023856953328036]]),
            # Testing when values are added to rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb, and thorax_axis
            (np.array([9, -7, -6]), np.array([3, -8, 5]), np.array([-9, 1, -4]), np.array([-4, 1, -6]),
             np.array([2, -3, 9]), np.array([-4, -2, -7]), np.array([-9, 1, -1]), np.array([-3, -4, -9]),
             [[-5, -2, -3], rand_coor, [-9, 5, -5], rand_coor],
             np.array([[0, 0, 0], [0, 0, 0]]),
             {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
             [[0, 0, 0], [0, 0, 0]],
             [[[149.87576359540907, -228.48721408225754, -418.8422716102348], [0, 0, 0], [-9, 1, -4], -7.0],
              [[282.73117218166414, -326.69276820761615, -251.76957615571214], [0, 0, 0], [-4, 1, -6], 7.0]],
             [[0, 0, 0], nan_3d, nan_3d, nan_3d, [0, 0, 0], nan_3d, nan_3d, nan_3d]),
            # Testing when values are added to rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb, thorax_axis,
            # and shoulder_origin
            (np.array([9, -7, -6]), np.array([3, -8, 5]), np.array([-9, 1, -4]), np.array([-4, 1, -6]),
             np.array([2, -3, 9]), np.array([-4, -2, -7]), np.array([-9, 1, -1]), np.array([-3, -4, -9]),
             [[-5, -2, -3], rand_coor, [-9, 5, -5], rand_coor],
             np.array([[-2, -8, -3], [5, -3, 2]]),
             {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
             [[0, 0, 0], [0, 0, 0]],
             [[[-311.42865408643604, -195.76081109238007, 342.15327877363165], [-2, -8, -3], [-9, 1, -4], -7.0],
              [[183.9753004933977, -292.7114070209339, -364.32791656553934], [5, -3, 2], [-4, 1, -6], 7.0]],
             [[0, 0, 0], [-0.9661174276011973, 0.2554279765068226, -0.03706298561739535],
              [0.12111591199009825, 0.3218504585188577, -0.9390118307103527],
              [-0.2279211529192759, -0.9116846116771036, -0.3418817293789138], [0, 0, 0],
              [-0.40160401780320154, -0.06011448807273248, 0.9138383123989052],
              [-0.4252287337918506, -0.8715182976051595, -0.24420561192811296],
              [0.8111071056538127, -0.48666426339228763, 0.3244428422615251]]),
            # Testing when values are added to rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb, thorax_axis,
            # shoulder_origin, and measurements
            (np.array([9, -7, -6]), np.array([3, -8, 5]), np.array([-9, 1, -4]), np.array([-4, 1, -6]),
             np.array([2, -3, 9]), np.array([-4, -2, -7]), np.array([-9, 1, -1]), np.array([-3, -4, -9]),
             [[-5, -2, -3], rand_coor, [-9, 5, -5], rand_coor],
             np.array([[-2, -8, -3], [5, -3, 2]]),
             {'RightElbowWidth': -38.0, 'LeftElbowWidth': 6.0, 'RightWristWidth': 47.0, 'LeftWristWidth': -7.0},
             [[0, 0, 0], [0, 0, 0]],
             [[[-311.42865408643604, -195.76081109238007, 342.15327877363165], [-2, -8, -3], [-9, 1, -4], 12.0],
              [[183.9753004933977, -292.7114070209339, -364.32791656553934], [5, -3, 2], [-4, 1, -6], 10.0]],
             [[0, 0, 0], [-0.9685895544902782, 0.17643783885299713, 0.17522546605219314],
              [-0.09942948749879241, 0.3710806621909213, -0.9232621075099284],
              [-0.2279211529192759, -0.9116846116771036, -0.3418817293789138], [0, 0, 0],
              [-0.10295276972565287, 0.4272479327059481, 0.8982538233730544],
              [-0.5757655689286321, -0.7619823480470763, 0.29644040025096585],
              [0.8111071056538127, -0.48666426339228763, 0.3244428422615251]]),
            # Testing when values are added to rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb, thorax_axis,
            # shoulder_origin, measurements and mock_return_val
            (np.array([9, -7, -6]), np.array([3, -8, 5]), np.array([-9, 1, -4]), np.array([-4, 1, -6]),
             np.array([2, -3, 9]), np.array([-4, -2, -7]), np.array([-9, 1, -1]), np.array([-3, -4, -9]),
             [[-5, -2, -3], rand_coor, [-9, 5, -5], rand_coor],
             np.array([[-2, -8, -3], [5, -3, 2]]),
             {'RightElbowWidth': -38.0, 'LeftElbowWidth': 6.0, 'RightWristWidth': 47.0, 'LeftWristWidth': -7.0},
             [[5, 4, -4], [6, 3, 5]],
             [[[-311.42865408643604, -195.76081109238007, 342.15327877363165], [-2, -8, -3], [-9, 1, -4], 12.0],
              [[183.9753004933977, -292.7114070209339, -364.32791656553934], [5, -3, 2], [-4, 1, -6], 10.0]],
             [[5, 4, -4], [4.156741342815987, 4.506819397152288, -3.8209778344606247],
              [4.809375978699987, 4.029428853750657, -4.981221904092206],
              [4.4974292889675835, 3.138450209658714, -3.928204184138226], [6, 3, 5],
              [6.726856988207308, 2.5997910101837682, 5.558132316896694],
              [5.329224487433077, 2.760784472038086, 5.702022893446135],
              [5.852558043845103, 2.1153482630706173, 4.557674131535308]]),
            # Testing that when rsho, lsho, relb, lelb, rwra, lwra, thorax_axis, and shoulder_origin are lists of
            # ints and measurements values are ints
            ([9, -7, -6], [3, -8, 5], [-9, 1, -4], [-4, 1, -6], np.array([2, -3, 9]), [-4, -2, -7],
             np.array([-9, 1, -1]), [-3, -4, -9],
             [[-5, -2, -3], rand_coor, [-9, 5, -5], rand_coor],
             [[-2, -8, -3], [5, -3, 2]],
             {'RightElbowWidth': -38, 'LeftElbowWidth': 6, 'RightWristWidth': 47, 'LeftWristWidth': -7},
             [[0, 0, 0], [0, 0, 0]],
             [[[-311.42865408643604, -195.76081109238007, 342.15327877363165], [-2, -8, -3], [-9, 1, -4], 12.0],
              [[183.9753004933977, -292.7114070209339, -364.32791656553934], [5, -3, 2], [-4, 1, -6], 10.0]],
             [[0, 0, 0], [-0.9685895544902782, 0.17643783885299713, 0.17522546605219314],
              [-0.09942948749879241, 0.3710806621909213, -0.9232621075099284],
              [-0.2279211529192759, -0.9116846116771036, -0.3418817293789138], [0, 0, 0],
              [-0.10295276972565287, 0.4272479327059481, 0.8982538233730544],
              [-0.5757655689286321, -0.7619823480470763, 0.29644040025096585],
              [0.8111071056538127, -0.48666426339228763, 0.3244428422615251]]),
            # Testing that when rsho, lsho, relb, lelb, rwra, lwra, thorax_axis, and shoulder_origin are numpy arrays
            #  of ints and measurements values are ints
            (np.array([9, -7, -6], dtype='int'), np.array([3, -8, 5], dtype='int'), np.array([-9, 1, -4], dtype='int'),
             np.array([-4, 1, -6], dtype='int'), np.array([2, -3, 9], dtype='int'), np.array([-4, -2, -7], dtype='int'),
             np.array([-9, 1, -1], dtype='int'), np.array([-3, -4, -9], dtype='int'),
             np.array([[-5, -2, -3], rand_coor, [-9, 5, -5], rand_coor], dtype='int'),
             np.array([[-2, -8, -3], [5, -3, 2]], dtype='int'),
             {'RightElbowWidth': -38, 'LeftElbowWidth': 6, 'RightWristWidth': 47, 'LeftWristWidth': -7},
             [[5, 4, -4], [6, 3, 5]],
             [[[-311.42865408643604, -195.76081109238007, 342.15327877363165], [-2, -8, -3], [-9, 1, -4], 12.0],
              [[183.9753004933977, -292.7114070209339, -364.32791656553934], [5, -3, 2], [-4, 1, -6], 10.0]],
             [[5, 4, -4], [4.156741342815987, 4.506819397152288, -3.8209778344606247],
              [4.809375978699987, 4.029428853750657, -4.981221904092206],
              [4.4974292889675835, 3.138450209658714, -3.928204184138226], [6, 3, 5],
              [6.726856988207308, 2.5997910101837682, 5.558132316896694],
              [5.329224487433077, 2.760784472038086, 5.702022893446135],
              [5.852558043845103, 2.1153482630706173, 4.557674131535308]]),
            # Testing that when rsho, lsho, relb, lelb, rwra, lwra, thorax_axis, and shoulder_origin are lists of
            # floats and measurements values are floats
            ([9.0, -7.0, -6.0], [3.0, -8.0, 5.0], [-9.0, 1.0, -4.0], [-4.0, 1.0, -6.0], np.array([2.0, -3.0, 9.0]),
             [-4.0, -2.0, -7.0], np.array([-9.0, 1.0, -1.0]), [-3.0, -4.0, -9.0],
             [[-5.0, -2.0, -3.0], rand_coor, [-9.0, 5.0, -5.0], rand_coor],
             [[-2.0, -8.0, -3.0], [5.0, -3.0, 2.0]],
             {'RightElbowWidth': -38.0, 'LeftElbowWidth': 6.0, 'RightWristWidth': 47.0, 'LeftWristWidth': -7.0},
             [[0, 0, 0], [0, 0, 0]],
             [[[-311.42865408643604, -195.76081109238007, 342.15327877363165], [-2, -8, -3], [-9, 1, -4], 12.0],
              [[183.9753004933977, -292.7114070209339, -364.32791656553934], [5, -3, 2], [-4, 1, -6], 10.0]],
             [[0, 0, 0], [-0.9685895544902782, 0.17643783885299713, 0.17522546605219314],
              [-0.09942948749879241, 0.3710806621909213, -0.9232621075099284],
              [-0.2279211529192759, -0.9116846116771036, -0.3418817293789138], [0, 0, 0],
              [-0.10295276972565287, 0.4272479327059481, 0.8982538233730544],
              [-0.5757655689286321, -0.7619823480470763, 0.29644040025096585],
              [0.8111071056538127, -0.48666426339228763, 0.3244428422615251]]),
            # Testing that when rsho, lsho, relb, lelb, rwra, lwra, thorax_axis, and shoulder_origin are numpy arrays
            #  of floats and measurements values are floats
            (np.array([9.0, -7.0, -6.0], dtype='float'), np.array([3.0, -8.0, 5.0], dtype='float'),
             np.array([-9.0, 1.0, -4.0], dtype='float'), np.array([-4.0, 1.0, -6.0], dtype='float'),
             np.array([2.0, -3.0, 9.0], dtype='float'), np.array([-4.0, -2.0, -7.0], dtype='float'),
             np.array([-9.0, 1.0, -1.0], dtype='float'), np.array([-3.0, -4.0, -9.0], dtype='float'),
             np.array([[-5.0, -2.0, -3.0], rand_coor, [-9.0, 5.0, -5.0], rand_coor], dtype='float'),
             np.array([[-2.0, -8.0, -3.0], [5.0, -3.0, 2.0]], dtype='float'),
             {'RightElbowWidth': -38.0, 'LeftElbowWidth': 6.0, 'RightWristWidth': 47.0, 'LeftWristWidth': -7.0},
             [[5, 4, -4], [6, 3, 5]],
             [[[-311.42865408643604, -195.76081109238007, 342.15327877363165], [-2, -8, -3], [-9, 1, -4], 12.0],
              [[183.9753004933977, -292.7114070209339, -364.32791656553934], [5, -3, 2], [-4, 1, -6], 10.0]],
             [[5, 4, -4], [4.156741342815987, 4.506819397152288, -3.8209778344606247],
              [4.809375978699987, 4.029428853750657, -4.981221904092206],
              [4.4974292889675835, 3.138450209658714, -3.928204184138226], [6, 3, 5],
              [6.726856988207308, 2.5997910101837682, 5.558132316896694],
              [5.329224487433077, 2.760784472038086, 5.702022893446135],
              [5.852558043845103, 2.1153482630706173, 4.557674131535308]])])
    def test_elbow_axis_calc(self, rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb, thorax_axis, shoulder_origin,
                             measurements, mock_return_val, expected_mock_args, expected):
        """
        This test provides coverage of the elbow_axis_calc function in the class CGM in pycgm.py, defined as
        elbow_axis_calc(rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb, thorax_axis, shoulder_origin, measurements)

        This test takes 14 parameters:
        rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        thorax_axis : ndarray
            A 4x3 ndarray that contains the thorax origin and the thorax x, y, and z axis components.
        shoulder_origin : ndarray
            A 2x3 ndarray of the right and left shoulder origin vectors (joint centers).
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        mock_return_val : list
            The value to be returned by the mock for find_joint_center
        expected_mock_args : list
            The expected arguments used to call the mocked function, find_joint_center
        expected : array
            An 8x3 ndarray that contains the right elbow origin, right elbow x, y, and z axis components, left elbow
            origin, and left elbow x, y, and z axis components.

        This test is checking to make sure the elbow joint center and axis are calculated correctly given the input
        parameters. This tests mocks find_joint_center to make sure the correct parameters are being passed into it
        given the parameters passed into elbow_axis_calc, expected_mock_args, and to also ensure that elbow_axis_calc
        returns the correct value considering the return value of find_joint_center, mock_return_val.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when rsho, lsho, relb, lelb, rwra, lwra, thorax_axis, and shoulder_origin
        are composed of lists of ints, numpy arrays of ints, lists of floats, and numpy arrays of floats and
        measurements values are ints and floats. rwrb and lwrb were kept as numpy arrays as lists would cause errors
        in lines like the following in pycgm.py as lists cannot be subtracted by each other:
        rwri = (rwra + rwrb) / 2.0
        lwri = (lwra + lwrb) / 2.0
        """
        with patch.object(CGM, 'find_joint_center', side_effect=mock_return_val) as mock_find_joint_center:
            result = CGM.elbow_axis_calc(rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb, thorax_axis, shoulder_origin,
                                         measurements)

        # Asserting that there were only 2 calls to find_joint_center
        np.testing.assert_equal(mock_find_joint_center.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to find_joint_center
        np.testing.assert_almost_equal(expected_mock_args[0][0], mock_find_joint_center.call_args_list[0][0][0],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][1], mock_find_joint_center.call_args_list[0][0][1],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][2], mock_find_joint_center.call_args_list[0][0][2],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][3], mock_find_joint_center.call_args_list[0][0][3],
                                       rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to find_joint_center
        np.testing.assert_almost_equal(expected_mock_args[1][0], mock_find_joint_center.call_args_list[1][0][0],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][1], mock_find_joint_center.call_args_list[1][0][1],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][2], mock_find_joint_center.call_args_list[1][0][2],
                                       rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][3], mock_find_joint_center.call_args_list[1][0][3],
                                       rounding_precision)

        # Asserting that elbow_axis_calc returned the correct result given the return value given by mocked
        # find_joint_center
        np.testing.assert_almost_equal(result, expected, rounding_precision)


class TestAxisUtils():
    """
    This class tests the axis util functions in the class CGM in pycgm.py:
    find_joint_center
    """

    @pytest.mark.parametrize(["a", "b", "c", "delta", "expected"], [
        # Test from running sample data
        (np.array([426.50338745, 262.65310669, 673.66247559]), np.array([308.38050472, 322.80342417, 937.98979061]),
         np.array([416.98687744, 266.22558594, 524.04089355]), 59.5, [364.17774614, 292.17051722, 515.19181496]),
        # Testing with basic value in a and c
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0.0, [0, 0, 1]),
        # Testing with value in a and basic value in c
        (np.array([-7.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0.0, [0, 0, 1]),
        #  Testing with value in b and basic value in c
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 4.0, 3.0]), np.array([0.0, 0.0, 1.0]), 0.0, [0, 0, 1]),
        #  Testing with value in a and b and basic value in c
        (np.array([-7.0, 1.0, 2.0]), np.array([1.0, 4.0, 3.0]), np.array([0.0, 0.0, 1.0]), 0.0, [0, 0, 1]),
        #  Testing with value in a, b, and c
        (np.array([-7.0, 1.0, 2.0]), np.array([1.0, 4.0, 3.0]), np.array([3.0, 2.0, -8.0]), 0.0, [3, 2, -8]),
        # Testing with value in a, b, c and delta of 1
        (np.array([-7.0, 1.0, 2.0]), np.array([1.0, 4.0, 3.0]), np.array([3.0, 2.0, -8.0]), 1.0,
         [3.91270955, 2.36111526, -7.80880104]),
        # Testing with value in a, b, c and delta of 20
        (np.array([-7.0, 1.0, 2.0]), np.array([1.0, 4.0, 3.0]), np.array([3.0, 2.0, -8.0]), 10.0,
         [5.86777669, 5.19544877, 1.03133235])])
    def test_find_joint_center(self, a, b, c, delta, expected):
        """
        This test provides coverage of the find_joint_center function in the class CGM in pycgm.py, defined as
        find_joint_center(a, b, c, delta)

        This test takes 5 parameters:
        a, b, c : array
            A 1x3 ndarray representing x, y, and z coordinates of the marker.
        delta : float
            The length from marker to joint center, retrieved from subject measurement file.
        expected : array
            A 1x3 ndarray for the x, y, z positions in Joint Center

        Calculated using Rodrigues' rotation formula
        """
        result = CGM.find_joint_center(a, b, c, delta)
        np.testing.assert_almost_equal(result, expected, rounding_precision)
