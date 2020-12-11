#!/usr/bin/python
# -*- coding: utf-8 -*-

from mock import patch
import pytest
import numpy as np
from pycgm.pycgm import StaticCGM, CGM 


rounding_precision = 8
class TestStaticCGMAxis:
    """
    This class tests the axis functions in pycgm.StaticCGM:
        static_calculation_head
        pelvis_axis_calc
        hip_axis_calc
        knee_axis_calc
        ankle_axis_calc
        foot_axis_calc
        flat_foot_axis_calc
        non_flat_foot_axis_calc
    """
    nan_3d = [np.nan, np.nan, np.nan]
    rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]

    @pytest.mark.parametrize(["head_axis", "expected"], [
        # Test from running sample data
        ([[244.89547729492188, 325.0578918457031, 1730.1619873046875],
          [244.87227957886893, 326.0240255639856, 1730.4189843948805],
          [243.89575702706503, 325.0366593474616, 1730.1515677531293],
          [244.89086730509763, 324.80072493605866, 1731.1283433097797]],
          0.25992807335420975),
        # Test with zeros for all params
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         np.nan),
        # Testing when values are added to head[0][0]
        ([[0, 0, 0], [-1, 8, 9], [0, 0, 0], [0, 0, 0]],
         1.5707963267948966),
        # Testing when values are added to head[0][1]
        ([[0, 0, 0], [0, 0, 0], [7, 5, 7], [0, 0, 0]],
         np.nan),
        # Testing when values are added to head[0][2]
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [3, -6, -2]],
         0.0),
        # Testing when values are added to head[0]
        ([[0, 0, 0], [-1, 8, 9], [7, 5, 7], [3, -6, -2]],
         -1.3521273809209546),
        # Testing when values are added to head[1]
        ([[-4, 7, 8], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         0.7853981633974483),
        # Testing when values are added to head
        ([[-4, 7, 8], [-1, 8, 9], [7, 5, 7], [3, -6, -2]],
         -0.09966865249116204),
        # Testing that when head is composed of lists of ints
        ([[-4, 7, 8], [-1, 8, 9], [7, 5, 7], [3, -6, -2]],
         -0.09966865249116204),
        # Testing that when head is composed of numpy arrays of ints
        (np.array([[-4, 7, 8], [-1, 8, 9], [7, 5, 7], [3, -6, -2]], dtype='int'),
         -0.09966865249116204),
        # Testing that when head is composed of lists of floats
        ([[-4.0, 7.0, 8.0], [-1.0, 8.0, 9.0], [7.0, 5.0, 7.0], [3.0, -6.0, -2.0]],
         -0.09966865249116204),
        # Testing that when head is composed of numpy arrays of floats
        (np.array([[-4.0, 7.0, 8.0], [-1.0, 8.0, 9.0], [7.0, 5.0, 7.0], [3.0, -6.0, -2.0]], dtype='float'),
         -0.09966865249116204)])
    def test_static_calculation_head(self, head_axis, expected):
        """
        This test provides coverage of the static_calculation_head function in 
        pycgm.StaticCGM, defined as static_calculation_head(head_axis)

        This function first calculates the x, y, z axes of the head by subtracting the given head origin from the head
        axes. It then calculates the offset angle using the global axis as the proximal axis, and the 
        head axis as the distal axis using inverse tangent.

        This test ensures that:
        - the head axis and the head origin both have an effect on the final offset angle
        - the resulting output is correct when head is composed of lists of ints, numpy arrays of ints, lists of
        floats, and numpy arrays of floats.
        """
        result = StaticCGM.static_calculation_head(head_axis)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

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
        This test provides coverage of the pelvis_axis_calc function in the class StaticCGM in pycgm.py, defined as
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
        result = StaticCGM.pelvis_axis_calc(rasi, lasi, rpsi, lpsi, sacr)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["pelvis_axis", "measurements", "expected"], [
        # Test from running sample data
        (np.array([[251.608306884766, 391.741317749023, 1032.893493652344],
                   [251.740636241119, 392.726947206848, 1032.788500732036],
                   [250.617115540376, 391.872328624646, 1032.874106304030],
                   [251.602953357582, 391.847951338178, 1033.887777624562]]),
         {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512, 'L_AsisToTrocanterMeasure': 72.512,
          'InterAsisDistance': 215.908996582031},
         np.array([[308.38050472, 322.80342417, 937.98979061], [182.57097863, 339.43231855, 935.52900126],
                   [245.47574167, 331.11787136, 936.75939593], [245.60807103, 332.10350082, 936.65440301],
                   [244.48455033, 331.24888223, 936.74000858], [245.47038814, 331.22450495, 937.7536799]])),
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
        This test provides coverage of the hip_axis_calc function in the class StaticCGM in pycgm.py, defined as
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
        result = StaticCGM.hip_axis_calc(pelvis_axis, measurements)
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
    def test_knee_axis_calc(self, rthi, lthi, rkne, lkne, hip_origin, measurements, mock_return_val, expected_mock_args, expected):
        """
        This test provides coverage of the knee_axis_calc function in the class StaticCGM in pycgm.py, defined as
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
            result = StaticCGM.knee_axis_calc(rthi, lthi, rkne, lkne, hip_origin, measurements)

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
        This test provides coverage of the ankle_axis_calc function in the class StaticCGM in pycgm.py, defined as
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
            result = StaticCGM.ankle_axis_calc(rtib, ltib, rank, lank, knee_origin, measurements)

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

    @pytest.mark.parametrize(["rtoe", "ltoe", "ankle_axis", "expected"],
        # Test from running sample data
        ([ ([ np.array([433.33508301, 354.97229004, 44.27765274]), 
            np.array([31.77310181, 331.23657227, 42.15322876]), 
            np.array([np.array([397.45738291, 217.50712216, 87.83068433]),
            np.array(rand_coor), np.array([396.73749179, 218.18875543, 87.69979179]), np.array(rand_coor),
            np.array([112.28082818, 175.83265027, 80.98477997]),
            np.array(rand_coor), np.array([111.34886681, 175.49163538, 81.10789314]), np.array(rand_coor)]), 
            np.array([[433.33508301, 354.97229004, 44.27765274],
            [433.4256618315962, 355.25152027652007, 45.233595181827035],
            [432.36890500826763, 355.2296456773885, 44.29402798451682],
            [433.09363829389764, 354.0471962330562, 44.570749823731354],
            [31.77310181, 331.23657227, 42.15322876],
            [31.806110207058808, 331.49492345678016, 43.11871573923792],
            [30.880216288550965, 330.81014854432254, 42.29786022762896],
            [32.2221740692973, 330.36972887034574, 42.36983123198873]])
        ]),
        # Test with zeros for all params
        ([ np.array([0, 0, 0]), 
            np.array([0, 0, 0]), 
            np.array([ [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor)]), 
            np.array([ [0, 0, 0], nan_3d, 
            nan_3d, nan_3d, 
            [0, 0, 0], nan_3d, 
            nan_3d, nan_3d]) ]),
       # Testing when values are added to rtoe
        ([ np.array([-7, 3, -8]), 
            np.array([0, 0, 0]), 
            np.array([ [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor)]), 
            np.array([ [-7, 3, -8], nan_3d, 
            nan_3d, [-6.36624977770237, 2.7283927618724446, -7.275714031659851], 
            [0, 0, 0], nan_3d, 
            nan_3d, nan_3d]) 
        ]),
        # Testing when values are added to ltoe
        ([ np.array([0, 0, 0]), 
            np.array([8, 0, -8]), 
            np.array([ [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor)]), 
            np.array([ [0, 0, 0], nan_3d, 
            nan_3d, nan_3d, 
            [8, 0, -8], nan_3d, 
            nan_3d, [7.292893218813452, 0.0, -7.292893218813452]]) ]),
        # Testing when values are added to both rtoe and ltoe
        ([ np.array([-7, 3, -8]), 

            np.array([8, 0, -8]), 

            np.array([ [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor)]), 

            np.array([ [-7, 3, -8], nan_3d, 
            nan_3d, [-6.36624977770237, 2.7283927618724446, -7.275714031659851], 
            [8, 0, -8], nan_3d, 
            nan_3d, [7.292893218813452, 0.0, -7.292893218813452]]) ]),
        # Testing when values are added to ankle_axis[1]
        ([ np.array([0, 0, 0]), 
            np.array([0, 0, 0]), 
            np.array([ [2, -9, 1], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor)]), 
            np.array([ [0, 0, 0], nan_3d, 
            nan_3d, [0.21566554640687682, -0.9704949588309457, 0.10783277320343841], 
            [0, 0, 0], nan_3d, 
            nan_3d, nan_3d]) ]),
        # Testing when values are added to ankle_axis
        ([ np.array([0, 0, 0]), 
            np.array([0, 0, 0]), 
            np.array([ [ 2, -9, 1], np.array(rand_coor), 
            [8, -4, 2], np.array(rand_coor), 
            [3, -7, 4], np.array(rand_coor), 
            [-9, 7, 4], np.array(rand_coor)]), 
            np.array([ np.array([0, 0, 0]), [0.21329967236760183, -0.06094276353360052, -0.9750842165376084], 
            [0.9528859437838807, 0.23329276554708803, 0.1938630023560309], [0.21566554640687682, -0.9704949588309457, 0.10783277320343841], 
            np.array([0, 0, 0]), [0.6597830814767823, 0.5655283555515277, 0.4948373111075868], 
            [-0.6656310267523443, 0.1342218942833945, 0.7341115850601987], [0.34874291623145787, -0.813733471206735, 0.46499055497527714]]) ]),
        # Testing when values are added to rtoe, ltoe and ankle_axis
        ([ np.array([-7, 3, -8]), 

            np.array([8, 0, -8]), 

            np.array([ [2, -9, 1], np.array(rand_coor), 
            [8, -4, 2], np.array(rand_coor), 
            [3, -7, 4], np.array(rand_coor), 
            [-9, 7, 4], np.array(rand_coor)]), 

            np.array([ [-7, 3, -8], [-6.586075309097216, 2.6732173492872757, -8.849634891853084], 
            [-6.249026985898898, 3.6500960420576702, -7.884178291357542], [-6.485504244572473, 2.3140056594299647, -7.485504244572473], 
            [8, 0, -8], [8.623180382731631, 0.5341546137699694, -7.428751315829338], 
            [7.295040915019964, 0.6999344300621451, -7.885437867872096], [7.6613572692607015, -0.47409982303501746, -7.187257446225685]]) ]),
        # Testing that when rtoe, ltoe and ankle are composed of numpy arrays of floats
        ([ np.array([-7.0, 3.0, -8.0], dtype='float'), 
            np.array([8.0, 0.0, -8.0], dtype='float'), 
            np.array([ np.array([2.0, -9.0, 1.0], dtype='float'), np.array(rand_coor, dtype='float'), 
            np.array([8.0, -4.0, 2.0], dtype='float'), np.array(rand_coor, dtype='float'), 
            np.array([3.0, -7.0, 4.0], dtype='float'), np.array(rand_coor, dtype='float'), 
            np.array([-9.0, 7.0, 4.0], dtype='float'), np.array(rand_coor, dtype='float')]), 
            np.array([ np.array([-7, 3, -8]), [-6.586075309097216, 2.6732173492872757, -8.849634891853084], 
            [-6.249026985898898, 3.6500960420576702, -7.884178291357542], [-6.485504244572473, 2.3140056594299647, -7.485504244572473], 
            np.array([8, 0, -8]), [8.623180382731631, 0.5341546137699694, -7.428751315829338], 
            [7.295040915019964, 0.6999344300621451, -7.885437867872096], [7.6613572692607015, -0.47409982303501746, -7.187257446225685]]) ])
        ]))
    def test_foot_axis_calc(self, rtoe, ltoe, ankle_axis, expected):
        """
        This test provides coverage of the uncorrect_footaxis function in pycgmStatic.py, defined as foot_axis_calc(rtoe, ltoe, ankle_axis)

        This test takes 3 parameters:
        rtoe, ltoe : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_axis: 
            A 8x3 ndarray that contains the right then left ankle origin and the ankle x, y, and z axis components.
        expected:
            A 8x3 ndarray that contains the right then left foot origin and the foot x, y, and z axis components.

        expected: the expected result from calling uncorrect_footaxis with the correct parameters, which should be the
        anatomically incorrect foot axis

        Given a marker RTOE and the ankle JC, the right anatomically incorrect foot axis is calculated with:

        .. math::
            R = [R_x + ROrigin_x, R_y + ROrigin_y, R_z + ROrigin_z]

        where :math:`ROrigin_x` is the x coor of the foot axis's origin gotten from rtoe

        :math:`R_x` is the unit vector of :math:`Yflex_R \times R_z`

        :math:`R_y` is the unit vector of :math:`R_z \times R_x`

        :math:`R_z` is the unit vector of the axis from right toe to right ankle JC

        :math:`Yflex_R` is the unit vector of the axis from right ankle flexion to right ankle JC

        The same calculation applies for the left anatomically incorrect foot axis by replacing all the right values
        with left values

        This unit test ensures that:
        - the markers for RTOE and LTOE only effect either the right or the left axis
        - ankle_JC_R and ankle_JC_L only effect either the right or the left axis
        - the resulting output is correct when markers and ankle_JC are composed of lists of ints,
        numpy arrays of ints, lists of floats, and numpy arrays of floats.

        """
        result = StaticCGM.foot_axis_calc(rtoe, ltoe, ankle_axis)
        np.testing.assert_almost_equal(result, expected, rounding_precision)
    
    @pytest.mark.parametrize(["rtoe", "ltoe", "rhee", "lhee","ankle_axis", "measurements", "expected"],
        ([
        # Test from running sample data
        ([ np.array([442.81997681, 381.62280273, 42.66047668]), 
            np.array([39.43652725, 382.44522095, 41.78911591]),
            np.array([374.01257324, 181.57929993, 49.50960922]), 
            np.array([105.30126953, 180.2130127, 47.15660858]),  
            np.array([ [393.76181608, 247.67829633, 87.73775041], np.array(rand_coor), 
            np.array([393.07114384, 248.39110006, 87.61575574]), np.array(rand_coor), 
            [98.74901939, 219.46930221, 80.6306816], np.array(rand_coor), 
            np.array([97.79246671, 219.20927275, 80.76255901]), np.array(rand_coor)]), 
            {'RightSoleDelta': 0.45, 'LeftSoleDelta': 0.45},
            np.array([[442.81997681, 381.62280273, 42.66047668], 
                     [442.30666241, 381.79936348, 43.50031871],
                     [442.02580128, 381.89596909, 42.1176458 ],
                     [442.49471759, 380.67717784, 42.66047668], 
                     [ 39.43652725, 382.44522095, 41.78911591], 
                     [39.14565179, 382.3504861, 42.74117514],
                     [38.53126992, 382.15038888, 41.48320216],
                     [39.74620554, 381.49437955, 41.78911591]]) ]),
        # Testing with zeros for all params
        ([ np.array([0, 0, 0]), 
            np.array([0, 0, 0]),
            np.array([0, 0, 0]), 
            np.array([0, 0, 0]),  
            np.array([ [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor)]), 
            {'RightSoleDelta': 0.0, 'LeftSoleDelta': 0.0},
            np.array([ [0, 0, 0], nan_3d, 
            nan_3d, nan_3d, 
            [0, 0, 0], nan_3d, 
            nan_3d, nan_3d]) ]),
        # Test with marker values only
        ([ np.array([1, 4, -6]), 
            np.array([4, 2, 2]),
            np.array([1, -4, -9]), 
            np.array([2, -3, -1]),  
            np.array([ [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor), 
            [0, 0, 0], np.array(rand_coor)]), 
            {'RightSoleDelta': 0.0, 'LeftSoleDelta': 0.0},
            np.array([ [1, 4, -6], nan_3d, 
            nan_3d, nan_3d, 
            [4, 2, 2], nan_3d, 
            nan_3d, nan_3d]) ]),
        # Test with ankle_axis values only
        ([ np.array([0, 0, 0]), 
            np.array([0, 0, 0]),
            np.array([0, 0, 0]), 
            np.array([0, 0, 0]),  
            np.array([ np.array([-5, -5, -1]), np.array(rand_coor), 
            np.array([9, 3, 7]), np.array(rand_coor), 
            np.array([-5, -5, -1]), np.array(rand_coor), 
            np.array([-9, 2, 9]), np.array(rand_coor)]), 
            {'RightSoleDelta': 0.0, 'LeftSoleDelta': 0.0},
            np.array([ np.array([0, 0, 0]), nan_3d, 
            nan_3d, nan_3d, 
            np.array([0, 0, 0]), nan_3d, 
            nan_3d, nan_3d]) ]),
        # Test with marker values and ankle_axis values
        ([ np.array([1, 4, -6]), 
            np.array([4, 2, 2]),
            np.array([1, -4, -9]), 
            np.array([2, -3, -1]),  
            np.array([ np.array([-5, -5, -1]), np.array(rand_coor), 
            np.array([9, 3, 7]), np.array(rand_coor), 
            np.array([5, 7, 1]), np.array(rand_coor), 
            np.array([-9, 2, 9]), np.array(rand_coor)]), 
            {'RightSoleDelta': 0.0, 'LeftSoleDelta': 0.0},
            np.array([ np.array([1, 4, -6]), [1.4961389383568338, 4.0, -6.868243142124459], 
            [1.8682431421244592, 4.0, -5.503861061643166], [1.0, 3.0, -6.0], 
            np.array([4, 2, 2]), [4.541530361073883, 1.783387855570447, 2.8122955416108235], 
            [3.245802523504333, 2.301678990598267, 2.5832460484899826], [3.6286093236458963, 1.0715233091147407, 2.0]]) ]),
        # Test with marker and measurement values
        ([ np.array([1, 4, -6]), 
            np.array([4, 2, 2]),
            np.array([1, -4, -9]), 
            np.array([2, -3, -1]),  
            np.array([ np.array([0, 0, 0]), np.array(rand_coor), 
            np.array([0, 0, 0]), np.array(rand_coor), 
            np.array([0, 0, 0]), np.array(rand_coor), 
            np.array([0, 0, 0]), np.array(rand_coor)]), 
            {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
            np.array([ np.array([1, 4, -6]), [0.0, 4.0, -6.0], [1.0, 4.0, -7.0], [1.0, 3.0, -6.0], 
            np.array([4, 2, 2]), [3.071523309114741, 2.3713906763541037, 2.0], [4.0, 2.0, 1.0], [3.6286093236458963, 1.0715233091147407, 2.0]]) ]),
        # Test with ankle_axis and measurement values
        ([ np.array([0, 0, 0]), 
            np.array([0, 0, 0]),
            np.array([0, 0, 0]), 
            np.array([0, 0, 0]),  
            np.array([ np.array([-5, -5, -1]), np.array(rand_coor), 
            np.array([9, 3, 7]), np.array(rand_coor), 
            np.array([5, 7, 1]), np.array(rand_coor), 
            np.array([-9, 2, 9]), np.array(rand_coor)]), 
            {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
            np.array([ np.array([0, 0, 0]), nan_3d, 
            nan_3d, nan_3d, 
            np.array([0, 0, 0]), nan_3d, 
            nan_3d, nan_3d]) ]),
        # Test with markers, ankle_axis, and measurement values
        ([ np.array([1, 4, -6]), 
            np.array([4, 2, 2]),
            np.array([1, -4, -9]), 
            np.array([2, -3, -1]),  
            np.array([ np.array([-5, -5, -1]), np.array(rand_coor), 
            np.array([9, 3, 7]), np.array(rand_coor), 
            np.array([5, 7, 1]), np.array(rand_coor), 
            np.array([-9, 2, 9]), np.array(rand_coor)]), 
            {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
            np.array([ np.array([1, 4, -6]), [1.465329458584979, 4.0, -6.885137557090992], 
            [1.8851375570909927, 4.0, -5.534670541415021], [1.0, 3.0, -6.0], np.array([4, 2, 2]), 
            [4.532940727667331, 1.7868237089330676, 2.818858992574645], [3.2397085122726565, 2.304116595090937, 2.573994730184553], 
            [3.6286093236458963, 1.0715233091147405, 2.0]]) ]),
        # Test with markers, ankle_axis, and measurement values as numpy arrays of floats and vsk values are floats
        ([ np.array([1.0, 4.0, -6.0], dtype='float'), 
            np.array([4.0, 2.0, 2.0], dtype='float'),
            np.array([1.0, -4.0, -9.0], dtype='float'), 
            np.array([2.0, -3.0, -1.0], dtype='float'),  
            np.array([ np.array([-5.0, -5.0, -1.0], dtype='float'), np.array(rand_coor, dtype='float'), 
            np.array([9.0, 3.0, 7.0], dtype='float'), np.array(rand_coor, dtype='float'), 
            np.array([5.0, 7.0, 1.0], dtype='float'), np.array(rand_coor, dtype='float'), 
            np.array([-9.0, 2.0, 9.0], dtype='float'), np.array(rand_coor, dtype='float')]), 
            {'RightSoleDelta': 1.0, 'LeftSoleDelta': -1.0},
            np.array([ np.array([1.0, 4.0, -6.0]), [1.4472135954999579, 4.0, -6.8944271909999157], 
            [1.894427190999916, 4.0, -5.5527864045000417], [1.0, 3.0, -6.0], 
            np.array([4.0, 2.0, 2.0]), [4.5834323811883104, 1.7666270475246759, 2.7779098415844139], 
            [3.2777288444786272, 2.2889084622085494, 2.6283759053035944], [3.6286093236458963, 1.0715233091147407, 2.0]]) ])]))
    def test_flat_foot_axis_calc(self, rtoe, ltoe, rhee, lhee, ankle_axis, measurements, expected):
        """
        This test provides coverage of the flat_foot_axis_calc method in StaticCGM in the file pycgm.py, defined as flat_foot_axis_calc(rtoe, ltoe, rhee, lhee, ankle_axis, measurements)

        This test takes 7 parameters:
        rtoe, ltoe, rhee, lhee : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_axis : array
            An 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.
        measurements : 
            A dictionary containing the subject measurements given from the file input.
        expected : array
            The expected result from calling flat_foot_axis_calc with the correct parameters, which should be the
            anatomically correct foot axis when foot is flat.

        Given the right ankle JC and the markers :math:`rtoe` and :math:`rhee`, the right anatomically correct foot
        axis is calculated with:

        .. math::
            R = [R_x + ROrigin_x, R_y + ROrigin_y, R_z + ROrigin_z]

        where :math:`ROrigin_x` is the x coor of the foot axis's origin gotten from rtoe

        :math:`R_x` is the unit vector of :math:`(AnkleFlexion_R - AnkleJC_R) \times R_z`

        :math:`R_y` is the unit vector of :math:`R_z \times R_x`

        :math:`R_z` is the unit vector of :math:`(A \times (rhee - rtoe)) \times A`

        A is the unit vector of :math:`(rhee - rtoe) \times (AnkleJC_R - rtoe)`

        The same calculation applies for the left anatomically correct foot axis by replacing all the right values
        with left values

        This unit test ensures that:
        - the markers for RTOE, LTOE, RHEE, and LHEE only effect either the right or the left axis
        - ankle_JC_R and ankle_JC_L only effect either the right or the left axis
        - the resulting output is correct when markers and ankle_JC are composed of lists of ints,
        numpy arrays of ints, lists of floats, and numpy arrays of floats.
        """
        result = StaticCGM.flat_foot_axis_calc(rtoe, ltoe, rhee, lhee, ankle_axis, measurements)
        np.testing.assert_almost_equal(result, expected, rounding_precision)
    
    @pytest.mark.parametrize(["rtoe", "ltoe", "rhee", "lhee","ankle_axis", "expected"], [
            ([ #Testing from running sample data 
            np.array([433.33508301, 354.97229004, 44.27765274]), 
            np.array([31.77310181, 331.23657227, 42.15322876]),
            np.array([381.88534546, 148.47607422, 49.99120331]),
            np.array([122.18766785, 138.55477905, 46.29433441]), 
            np.array([ np.array([397.45738291, 217.50712216, 87.83068433]), np.array(rand_coor), 
            np.array([396.73749179, 218.18875543, 87.69979179]), np.array(rand_coor), 
            np.array([112.28082818, 175.83265027, 80.98477997]), np.array(rand_coor), 
            np.array([111.34886681, 175.49163538, 81.10789314]), np.array(rand_coor)]),
            np.array([ [433.33508301, 354.97229004, 44.27765274], [433.2103651914497, 355.03076948530014, 45.26812011533214],
            [432.37277461595676, 355.2083164947686, 44.14254511237841],[433.09340548455947, 354.0023046440309, 44.30449129818456], 
            [31.77310181, 331.23657227, 42.15322876], [31.878278418984852, 331.30724434357205, 43.14516794016654],
            [30.873906948094536, 330.8173225172055, 42.27844159782351],
            [32.1978211099223, 330.33145619248916, 42.172681460633456]]) ]),
            ([ #Testing all values with zeroes 
            np.array([0, 0, 0]), 
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]), 
            np.array([ [0, 0, 0], np.array(rand_coor), 
            np.array([0, 0, 0]), np.array(rand_coor), 
            np.array([0, 0, 0]), np.array(rand_coor), 
            np.array([0, 0, 0]), np.array(rand_coor)]), 
            np.array([ np.array([0, 0, 0]), nan_3d, 
            nan_3d, nan_3d, 
            np.array([0, 0, 0]), nan_3d, 
            nan_3d, nan_3d]) ]),
            ([ #Testing with marker values only
            np.array([5, -2, -2]), 
            np.array([-2, -7, -1]),
            np.array([3, 5, 9]),
            np.array([-7, 6, 1]), 
            np.array([ [0, 0, 0], np.array(rand_coor), 
            np.array([0, 0, 0]), np.array(rand_coor), 
            np.array([0, 0, 0]), np.array(rand_coor), 
            np.array([0, 0, 0]), np.array(rand_coor)]), 
            np.array([ np.array([5, -2, -2]), nan_3d, 
            nan_3d, [4.848380391284219, -1.4693313694947676, -1.1660921520632064], 
            np.array([-2, -7, -1]), nan_3d, 
            nan_3d, [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]) ]),
            ([ #Testing ankle_axis values only
            np.array([0, 0, 0]), 
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]), 
            np.array([ [-8, 6, 2], np.array(rand_coor), 
            np.array([3, 6, -3]), np.array(rand_coor), 
            np.array([-7, 8, 5]), np.array(rand_coor), 
            np.array([2, -7, -2]), np.array(rand_coor)]), 
            np.array([ np.array([0, 0, 0]), nan_3d, 
            nan_3d, nan_3d, 
            np.array([0, 0, 0]), nan_3d, 
            nan_3d, nan_3d]) ]),
            ([ #Testing marker and ankle_axis values
            np.array([5, -2, -2]), 
            np.array([-2, -7, -1]),
            np.array([3, 5, 9]),
            np.array([-7, 6, 1]), 
            np.array([ [-8, 6, 2], np.array(rand_coor), 
            np.array([-7, 8, 5]), np.array(rand_coor), 
            np.array([3, 6, -3]), np.array(rand_coor), 
            np.array([2, -7, -2]), np.array(rand_coor)]), 
            np.array([ np.array([5, -2, -2]), [5.049326362366699, -2.8385481602338833, -1.4574100139663109], 
            [5.987207376506346, -1.8765990779367068, -1.8990356092209417], [4.848380391284219, -1.4693313694947676, -1.1660921520632064], 
            np.array([-2, -7, -1]), [-2.446949206712144, -7.0343807082086265, -1.8938984134242876], 
            [-2.820959061315946, -7.381159564182403, -0.5748604861042421], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]) ]),
            ([ #Testing with different right markers and right ankle ankle_axis values
            np.array([-2, 9, -1]), 
            np.array([-2, -7, -1]),
            np.array([-1, -4, 4]),
            np.array([-7, 6, 1]), 
            np.array([ [5, -1, -5], np.array(rand_coor), 
            np.array([-7, -8, -5]), np.array(rand_coor), 
            np.array([3, 6, -3]), np.array(rand_coor), 
            np.array([2, -7, -2]), np.array(rand_coor)]),
            np.array([ np.array([-2, 9, -1]), [-2.1975353004951486, 9.33863194370597, -0.0800498862654514], 
            [-2.977676633621816, 8.86339202060183, -1.1596454197108794], [-1.9283885125960567, 8.069050663748737, -0.6419425629802835], 
            np.array([-2, -7, -1]), [-2.446949206712144, -7.0343807082086265, -1.8938984134242876], 
            [-2.820959061315946, -7.381159564182403, -0.5748604861042421], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]) ]),
            ([ #Testing different left markers and left ankle ankle_axis values
            np.array([5, -2, -2]), 
            np.array([5, 4, -4]),
            np.array([3, 5, 9]),
            np.array([-1, 6, 9]), 
            np.array([ [-8, 6, 2], np.array(rand_coor), 
            np.array([-7, 8, 5]), np.array(rand_coor), 
            np.array([0, -8, -2]), np.array(rand_coor), 
            np.array([-4, -9, -1]), np.array(rand_coor)]), 
            np.array([ np.array([5, -2, -2]), [5.049326362366699, -2.8385481602338833, -1.4574100139663109], 
            [5.987207376506346, -1.8765990779367068, -1.8990356092209417], [4.848380391284219, -1.4693313694947676, -1.1660921520632064], 
            np.array([5, 4, -4]), [4.702195658033984, 4.913266648695782, -4.277950719168281], 
            [4.140311818111685, 3.6168482384235783, -4.3378327360136195], [4.584971321680356, 4.138342892773215, -3.1007711969741028]]) ]),
            ([ #Testing when all values are numpy arrays of floats   
            np.array([5.0, -2.0, -2.0], dtype='float'), 
            np.array([-2.0, -7.0, -1.0], dtype='float'),
            np.array([3.0, 5.0, 9.0], dtype='float'),
            np.array([-7.0, 6.0, 1.0], dtype='float'), 
            np.array([ np.array([-8.0, 6.0, 2.0],dtype='float'), np.array(rand_coor, dtype='float'), 
            np.array([-7.0, 8.0, 5.0], dtype='float'), np.array(rand_coor, dtype='float'), 
            np.array([3.0, 6.0, -3.0], dtype='float'), np.array(rand_coor, dtype='float'), 
            np.array([2.0, -7.0, -2.0], dtype='float'), np.array(rand_coor, dtype='float')]), 
            np.array([ np.array([5.0, -2.0, -2.0]), [5.049326362366699, -2.8385481602338833, -1.4574100139663109], 
            [5.987207376506346, -1.8765990779367068, -1.8990356092209417], [4.848380391284219, -1.4693313694947676, -1.1660921520632064], 
            np.array([-2.0, -7.0, -1.0]), [-2.446949206712144, -7.0343807082086265, -1.8938984134242876], 
            [-2.820959061315946, -7.381159564182403, -0.5748604861042421], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]) ])
            ])
    def test_non_flat_foot_axis_calc(self, rtoe, ltoe, rhee, lhee, ankle_axis, expected):
        """
        This test provides coverage of the non_flat_foot_axis_calc function in StaticCGM in the filepycgm.py, 
        defined as non_flat_foot_axis_calc(rtoe, ltoe, rhee, lhee, ankle_axis)

        This test takes 6 parameters:
        rtoe, ltoe, rhee, lhee : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_axis : ndarray
            An 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.
        expected: ndarray
            The expected result, which will be an array containing the right and left foot origin and XYZ axes.

        Given the right ankle JC and the markers :math:`TOE_R` and :math:`HEE_R , the right anatomically correct foot
        axis is calculated with:

        .. math::
        R is [R_x + ROrigin_x, R_y + ROrigin_y, R_z + ROrigin_z]

        where :math:`ROrigin_x` is the x coor of the foot axis's origin gotten from rtoe

        :math:`R_x` is the unit vector of :math:`YFlex_R \times R_z`

        :math:`R_y` is the unit vector of :math:`R_z \times R_x`

        :math:`R_z` is the unit vector of :math:`(HEE_R - TOE_R)`

        :math:`YFlex_R` is the unit vector of :math:`(AnkleFlexion_R - AnkleJC_R)`

        The same calculation applies for the left anatomically correct foot axis by replacing all the right values
        with left values

        This unit test ensures that:
        - the markers for RTOE, LTOE, RHEE, and LHEE only effect either the right or the left axis
        - ankle_JC_R and ankle_JC_L only effect either the right or the left axis
        - the resulting output is correct when markers and ankle_JC are composed of lists of ints,
        numpy arrays of ints, lists of floats, and numpy arrays of floats.
        """
        result = StaticCGM.non_flat_foot_axis_calc(rtoe, ltoe, rhee, lhee, ankle_axis)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

