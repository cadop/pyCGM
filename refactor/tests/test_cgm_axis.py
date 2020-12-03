from refactor.pycgm import CGM
import pytest
import numpy as np

rounding_precision = 8

class TestLowerBodyAxis():
    """
    This class tests the lower body axis functions in the class CGM in pycgm.py:
    pelvis_axis_calc
    hip_axis_calc
    """

    nan_3d = [np.nan, np.nan, np.nan]

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

        This unit test ensure that:
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
        (np.array([[251.608306884766, 391.741317749023, 1032.893493652344],
                   [251.740636241119, 392.726947206848, 1032.788500732036],
                   [250.617115540376, 391.872328624646, 1032.874106304030],
                   [251.602953357582, 391.847951338178, 1033.887777624562]]),
         {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512, 'L_AsisToTrocanterMeasure': 72.512,
          'InterAsisDistance': 215.908996582031},
         np.array([[245.47574167, 331.11787136, 936.75939593], [245.60807103, 332.10350082, 936.65440301],
                   [244.48455033, 331.24888223, 936.74000858], [245.47038814, 331.22450495, 937.7536799]])),
        # Basic test with zeros for all params
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])),
        # Testing when values are added to pel_o
        (np.array([[1, 0, -3], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[1.19644104, 0., -3.58932313], [0.19644104, 0, -0.58932313], [0.19644104, 0, -0.58932313],
                   [0.19644104, 0, -0.58932313]])),
        # Testing when values are added to pel_x
        (np.array([[0, 0, 0], [-5, -3, -6], [0, 0, 0], [0, 0, 0]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[54.02442793, 32.41465676, 64.82931352], [49.02442793, 29.41465676, 58.82931352],
                   [54.02442793, 32.41465676, 64.82931352], [54.02442793, 32.41465676, 64.82931352]])),
        # Testing when values are added to pel_y
        (np.array([[0, 0, 0], [0, 0, 0], [4, -1, 2], [0, 0, 0]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[0, 0, 0], [0, 0, 0], [4, -1, 2], [0, 0, 0]])),
        # Testing when values are added to pel_z
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[31.82533363, 84.86755635, 21.21688909], [31.82533363, 84.86755635, 21.21688909],
                   [31.82533363, 84.86755635, 21.21688909], [34.82533363, 92.86755635, 23.21688909]])),
        # Test when values are added to pel_x, pel_y, and pel_z
        (np.array([[0, 0, 0], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[85.84976156, 117.28221311, 86.04620261], [80.84976156, 114.28221311, 80.04620261],
                   [89.84976156, 116.28221311, 88.04620261], [88.84976156, 125.28221311, 88.04620261]])),
        # Test when values are added to pel_o, pel_x, pel_y, and pel_z
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[87.04620261, 117.28221311, 82.45687948], [81.04620261, 114.28221311, 79.45687948],
                   [90.04620261, 116.28221311, 87.45687948], [89.04620261, 125.28221311, 87.45687948]])),
        # Test when values are added to pel_o, pel_x, pel_y, pel_z, and measurements[MeanLegLength]
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[81.36115608, 104.36100616, 73.8551084], [75.36115608, 101.36100616, 70.8551084],
                   [84.36115608, 103.36100616, 78.8551084], [83.36115608, 112.36100616, 78.8551084]])),
        # Test when values are added to pel_o, pel_x, pel_y, pel_z, and measurements[R_AsisToTrocanterMeasure]
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[25.97938253, 112.69354093, 66.74903393], [19.97938253, 109.69354093, 63.74903393],
                   [28.97938253, 111.69354093, 71.74903393], [27.97938253, 120.69354093, 71.74903393]])),
        # Test when values are added to pel_o, pel_x, pel_y, pel_z, and measurements[R_AsisToTrocanterMeasure]
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 0.0},
         np.array([[25.97938253, 112.69354093, 66.74903393], [19.97938253, 109.69354093, 63.74903393],
                   [28.97938253, 111.69354093, 71.74903393], [27.97938253, 120.69354093, 71.74903393]])),
        # Test when values are added to pel_o, pel_x, pel_y, pel_z, and measurements[L_AsisToTrocanterMeasure]
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0 - 7.0,
          'InterAsisDistance': 0.0},
         np.array([[69.23504675, 115.94385039, 77.87542453], [63.23504675, 112.94385039, 74.87542453],
                   [72.23504675, 114.94385039, 82.87542453], [71.23504675, 123.94385039, 82.87542453]])),
        # Test when values are added to pel_o, pel_x, pel_y, pel_z, and measurements[InterAsisDistance]
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0,
          'InterAsisDistance': 11.0},
         np.array([[87.04620261, 117.28221311, 82.45687948], [81.04620261, 114.28221311, 79.45687948],
                   [90.04620261, 116.28221311, 87.45687948], [89.04620261, 125.28221311, 87.45687948]])),
        # Test when values are added to pel_o, pel_x, pel_y, pel_z, and all values in measurements
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]]),
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': -7.0,
          'InterAsisDistance': 11.0},
         np.array([[2.48318015, 98.43397127, 53.5658079], [-3.51681985, 95.43397127, 50.5658079],
                   [5.48318015, 97.43397127, 58.5658079], [4.48318015, 106.43397127, 58.5658079]])),
        # Testing that when pel_o, pel_x, pel_y, and pel_z are numpy arrays of ints and measurements values are ints
        (np.array([[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2]], dtype='int'),
         {'MeanLegLength': 15, 'R_AsisToTrocanterMeasure': -24, 'L_AsisToTrocanterMeasure': -7,
          'InterAsisDistance': 11},
         np.array([[2.48318015, 98.43397127, 53.5658079], [-3.51681985, 95.43397127, 50.5658079],
                   [5.48318015, 97.43397127, 58.5658079], [4.48318015, 106.43397127, 58.5658079]])),
        # Testing that when pel_o, pel_x, pel_y, and pel_z are numpy arrays of floats and measurements values are floats
        (np.array([[1.0, 0.0, -3.0], [-5.0, -3.0, -6.0], [4.0, -1.0, 2.0], [3.0, 8.0, 2.0]], dtype='float'),
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': -7.0,
          'InterAsisDistance': 11},
         np.array([[2.48318015, 98.43397127, 53.5658079], [-3.51681985, 95.43397127, 50.5658079],
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

        This unit test ensure that:
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