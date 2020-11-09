import pytest
import pyCGM_Single.pycgmStatic as pycgmStatic
import numpy as np
from unittest.mock import patch

rounding_precision = 8

class TestPycgmStaticAxis():
    """
    This class tests the axis functions in pycgmStatic.py:
    staticCalculationHead
    pelvisJointCenter
    hipJointCenter
    hipAxisCenter
    kneeJointCenter
    ankleJointCenter
    footJointCenter
    headJC
    uncorrect_footaxis
    rotaxis_footflat
    rotaxis_nonfootflat
    findJointC
    """
    nan_3d = [np.nan, np.nan, np.nan]
    rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]

    @pytest.mark.parametrize(["head", "expected"], [
        # Test from running sample data
        ([[[244.87227957886893, 326.0240255639856, 1730.4189843948805],
           [243.89575702706503, 325.0366593474616, 1730.1515677531293],
           [244.89086730509763, 324.80072493605866, 1731.1283433097797]],
          [244.89547729492188, 325.0578918457031, 1730.1619873046875]],
         0.25992807335420975),
        # Test with zeros for all params
        ([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]],
         np.nan),
        # Testing when values are added to head[0][0]
        ([[[-1, 8, 9], [0, 0, 0], [0, 0, 0]], [0, 0, 0]],
         1.5707963267948966),
        # Testing when values are added to head[0][1]
        ([[[0, 0, 0], [7, 5, 7], [0, 0, 0]], [0, 0, 0]],
         np.nan),
        # Testing when values are added to head[0][2]
        ([[[0, 0, 0], [0, 0, 0], [3, -6, -2]], [0, 0, 0]],
         0.0),
        # Testing when values are added to head[0]
        ([[[-1, 8, 9], [7, 5, 7], [3, -6, -2]], [0, 0, 0]],
         -1.3521273809209546),
        # Testing when values are added to head[1]
        ([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [-4, 7, 8]],
         0.7853981633974483),
        # Testing when values are added to head
        ([[[-1, 8, 9], [7, 5, 7], [3, -6, -2]], [-4, 7, 8]],
         -0.09966865249116204)])
    def testStaticCalculationHead(self, head, expected):
        """
       This test provides coverage of the staticCalculationHead function in pycgmStatic.py, defined as staticCalculationHead(frame, head)

       This test takes 2 parameters:
       head: array containing the head axis and head origin
       expected: the expected result from calling staticCalculationHead on head
       """
        result = pycgmStatic.staticCalculationHead(None, head)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["frame", "expected"], [
        # Test from running sample data
        ({'RASI': np.array([357.90066528, 377.69210815, 1034.97253418]),
          'LASI': np.array([145.31594849, 405.79052734, 1030.81445312]),
          'RPSI': np.array([274.00466919, 205.64402771, 1051.76452637]),
          'LPSI': np.array([189.15231323, 214.86122131, 1052.73486328])},
         [np.array([251.60830688, 391.74131775, 1032.89349365]),
          np.array([[251.74063624, 392.72694721, 1032.78850073], [250.61711554, 391.87232862, 1032.8741063], [251.60295336, 391.84795134, 1033.88777762]]),
          np.array([231.57849121, 210.25262451, 1052.24969482])]),
        # Test with zeros for all params
        ({'SACR': np.array([0, 0, 0]), 'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0]),
          'RPSI': np.array([0, 0, 0]), 'LPSI': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([nan_3d, nan_3d, nan_3d]), np.array([0, 0, 0])]),
        # Testing when adding values to frame['RASI'] and frame['LASI']
        ({'RASI': np.array([-6, 6, 3]), 'LASI': np.array([-7, -9, 1]), 'RPSI': np.array([0, 0, 0]),
          'LPSI': np.array([0, 0, 0])},
         [np.array([-6.5, -1.5, 2.0]),
          np.array([[-7.44458106, -1.48072284, 2.32771179], [-6.56593805, -2.48907071, 1.86812391], [-6.17841206, -1.64617634, 2.93552855]]),
          np.array([0, 0, 0])]),
        # Testing when adding values to frame['RPSI'] and frame['LPSI']
        ({'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0]), 'RPSI': np.array([1, 0, -4]),
          'LPSI': np.array([7, -2, 2])},
         [np.array([0, 0, 0]), np.array([nan_3d, nan_3d, nan_3d]), np.array([4., -1.0, -1.0])]),
        # Testing when adding values to frame['SACR']
        ({'SACR': np.array([-4, 8, -5]), 'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0]),
          'RPSI': np.array([0, 0, 0]), 'LPSI': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([nan_3d, nan_3d, nan_3d]), np.array([-4, 8, -5, ])]),
        # Testing when adding values to frame['RASI'], frame['LASI'], frame['RPSI'] and frame['LPSI']
        ({'RASI': np.array([-6, 6, 3]), 'LASI': np.array([-7, -9, 1]), 'RPSI': np.array([1, 0, -4]),
          'LPSI': np.array([7, -2, 2])},
         [np.array([-6.5, -1.5, 2.0]),
          np.array([[-7.45825845, -1.47407957, 2.28472598], [-6.56593805, -2.48907071, 1.86812391], [-6.22180416, -1.64514566, 2.9494945]]),
          np.array([4.0, -1.0, -1.0])]),
        # Testing when adding values to frame['SACR'], frame['RASI'] and frame['LASI']
        ({'SACR': np.array([-4, 8, -5]), 'RASI': np.array([-6, 6, 3]), 'LASI': np.array([-7, -9, 1]),
          'RPSI': np.array([0, 0, 0]), 'LPSI': np.array([0, 0, 0])},
         [np.array([-6.5, -1.5, 2.0]),
          np.array([[-6.72928306, -1.61360872, 2.96670695], [-6.56593805, -2.48907071, 1.86812391], [-5.52887619, -1.59397972, 2.21928602]]),
          np.array([-4, 8, -5])]),
        # Testing when adding values to frame['SACR'], frame['RPSI'] and frame['LPSI']
        ({'SACR': np.array([-4, 8, -5]), 'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0]),
          'RPSI': np.array([1, 0, -4]), 'LPSI': np.array([7, -2, 2])},
         [np.array([0, 0, 0]), np.array([nan_3d, nan_3d, nan_3d]), np.array([-4, 8, -5])]),
        # Testing when adding values to frame['SACR'], frame['RASI'], frame['LASI'], frame['RPSI'] and frame['LPSI']
        ({'SACR': np.array([-4, 8, -5]), 'RASI': np.array([-6, 6, 3]), 'LASI': np.array([-7, -9, 1]),
          'RPSI': np.array([1, 0, -4]), 'LPSI': np.array([7, -2, 2])},
         [np.array([-6.5, -1.5, 2.0]),
          np.array([[-6.72928306, -1.61360872, 2.96670695], [-6.56593805, -2.48907071, 1.86812391], [-5.52887619, -1.59397972, 2.21928602]]),
          np.array([-4, 8, -5])])])
    def testPelvisJointCenter(self, frame, expected):
        """
        This test provides coverage of the pelvisJointCenter function in pycgmStatic.py, defined as pelvisJointCenter(frame)
        This test takes 2 parameters:
        frame: dictionary of marker lists
        expected: the expected result from calling pelvisJointCenter on frame
        """
        result = pycgmStatic.pelvisJointCenter(frame)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

    @pytest.mark.parametrize(["pel_origin", "pel_x", "pel_y", "pel_z", "vsk", "expected"], [
        # Test from running sample data
        ([251.608306884766, 391.741317749023, 1032.893493652344], [251.740636241119, 392.726947206848, 1032.788500732036], [250.617115540376, 391.872328624646, 1032.874106304030], [251.602953357582, 391.847951338178, 1033.887777624562],
         {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512, 'L_AsisToTrocanterMeasure': 72.512, 'InterAsisDistance': 215.908996582031},
         [[182.57097863, 339.43231855, 935.52900126], [308.38050472, 322.80342417, 937.98979061]]),
        # Basic test with zeros for all params
        ([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[0, 0, 0], [0, 0, 0]]),
        # Testing when values are added to pel_origin
        ([1, 0, -3], [0, 0, 0], [0, 0, 0], [0, 0, 0],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[-6.1387721, 0, 18.4163163], [8.53165418, 0, -25.59496255]]),
        # Testing when values are added to pel_x
        ([0, 0, 0], [-5, -3, -6], [0, 0, 0], [0, 0, 0],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[54.02442793, 32.41465676, 64.82931352], [54.02442793, 32.41465676, 64.82931352]]),
        # Testing when values are added to pel_y
        ([0, 0, 0], [0, 0, 0], [4, -1, 2], [0, 0, 0],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[29.34085257, -7.33521314, 14.67042628], [-29.34085257,   7.33521314, -14.67042628]]),
        # Testing when values are added to pel_z
        ([0, 0, 0], [0, 0, 0], [0, 0, 0], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[31.82533363, 84.86755635, 21.21688909], [31.82533363, 84.86755635, 21.21688909]]),
        # Test when values are added to pel_x, pel_y, and pel_z
        ([0, 0, 0], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[115.19061413, 109.94699997, 100.71662889], [56.508909  , 124.61742625,  71.37577632]]),
        # Test when values are added to pel_origin, pel_x, pel_y, and pel_z
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[109.05184203, 109.94699997, 119.13294518], [65.04056318, 124.61742625,  45.78081377]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[MeanLegLength]
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[100.88576753,  97.85280235, 106.39612748], [61.83654463, 110.86920998,  41.31408931]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[R_AsisToTrocanterMeasure]
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[109.05184203, 109.94699997, 119.13294518], [-57.09307697, 115.44008189,  14.36512267]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[L_AsisToTrocanterMeasure]
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0-7.0, 'InterAsisDistance': 0.0},
         [[73.42953032, 107.27027453, 109.97003528], [65.04056318, 124.61742625,  45.78081377]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[InterAsisDistance]
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 11.0},
         [[125.55184203, 104.44699997, 146.63294518], [48.54056318, 130.11742625,  18.28081377]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and all values in vsk
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': -7.0, 'InterAsisDistance': 11.0},
         [[81.76345582,  89.67607691, 124.73321758], [-76.79709552, 107.19186562, -17.60160178]])])
    def testHipJointCenter(self, pel_origin, pel_x, pel_y, pel_z, vsk, expected):
        """
        This test provides coverage of the hipJointCenter function in pycgmStatic.py, defined as hipJointCenter(frame, pel_origin, pel_x, pel_y, pel_z, vsk)
        This test takes 6 parameters:
        pel_origin: array of x,y,z position of origin of the pelvis
        pel_x: array of x,y,z position of x-axis of the pelvis
        pel_y: array of x,y,z position of y-axis of the pelvis
        pel_z: array of x,y,z position of z-axis of the pelvis
        vsk: dictionary containing subject measurements from a VSK file
        expected: the expected result from calling hipJointCenter on pel_origin, pel_x, pel_y, pel_z, and vsk
        """
        result = pycgmStatic.hipJointCenter(None, pel_origin, pel_x, pel_y, pel_z, vsk)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)

    @pytest.mark.parametrize(["l_hip_jc", "r_hip_jc", "pelvis_axis", "expected"], [
        # Test from running sample data
        ([182.57097863, 339.43231855, 935.52900126], [308.38050472, 322.80342417, 937.98979061],
         [np.array([251.60830688, 391.74131775, 1032.89349365]), np.array([[251.74063624, 392.72694721, 1032.78850073], [250.61711554, 391.87232862, 1032.8741063], [251.60295336, 391.84795134, 1033.88777762]]), np.array([231.57849121, 210.25262451, 1052.24969482])],
         [[245.47574167208043, 331.1178713574418, 936.7593959314677], [[245.60807102843359, 332.10350081526684, 936.6544030111602], [244.48455032769033, 331.2488822330648, 936.7400085831541], [245.47038814489719, 331.22450494659665, 937.7536799036861]]]),
        # Basic test with zeros for all params
        ([0, 0, 0], [0, 0, 0],
         [np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[0, 0, 0], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
        # Testing when values are added to l_hip_jc
        ([1, -3, 2], [0, 0, 0],
         [np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[0.5, -1.5, 1], [[0.5, -1.5, 1], [0.5, -1.5, 1], [0.5, -1.5, 1]]]),
        # Testing when values are added to r_hip_jc
        ([0, 0, 0], [-8, 1, 4],
         [np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[-4, 0.5, 2], [[-4, 0.5, 2], [-4, 0.5, 2], [-4, 0.5, 2]]]),
        # Testing when values are added to l_hip_jc and r_hip_jc
        ([8, -3, 7], [5, -2, -1],
         [np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[6.5, -2.5, 3], [[6.5, -2.5, 3], [6.5, -2.5, 3], [6.5, -2.5, 3]]]),
        # Testing when values are added to pelvis_axis[0]
        ([0, 0, 0], [0, 0, 0],
         [np.array([1, -3, 6]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[0, 0, 0], [[-1, 3, -6], [-1, 3, -6], [-1, 3, -6]]]),
        # Testing when values are added to pelvis_axis[1]
        ([0, 0, 0], [0, 0, 0],
         [np.array([0, 0, 0]), np.array([[1, 0, 5], [-2, -7, -3], [9, -2, 7]]), np.array(rand_coor)],
         [[0, 0, 0], [[1, 0, 5], [-2, -7, -3], [9, -2, 7]]]),
        # Testing when values are added to pelvis_axis[0] and pelvis_axis[1]
        ([0, 0, 0], [0, 0, 0],
         [np.array([-3, 0, 5]), np.array([[-4, 5, -2], [0, 0, 0], [8, 5, -1]]), np.array(rand_coor)],
         [[0, 0, 0], [[-1, 5, -7], [3, 0, -5], [11, 5, -6]]]),
        # Testing when values are added to all params
        ([-5, 3, 8], [-3, -7, -1],
         [np.array([6, 3, 9]), np.array([[5, 4, -2], [0, 0, 0], [7, 2, 3]]), np.array(rand_coor)],
         [[-4, -2, 3.5], [[-5, -1, -7.5], [-10, -5, -5.5], [-3, -3, -2.5]]])])
    def testHipAxisCenter(self, l_hip_jc, r_hip_jc, pelvis_axis, expected):
        """
        This test provides coverage of the hipAxisCenter function in pycgmStatic.py, defined as hipAxisCenter(l_hip_jc, r_hip_jc, pelvis_axis)
        This test takes 4 parameters:
        l_hip_jc: array of left hip joint center x,y,z position
        r_hip_jc: array of right hip joint center x,y,z position
        pelvis_axis: array of pelvis origin and axis
        expected: the expected result from calling hipAxisCenter on l_hip_jc, r_hip_jc, and pelvis_axis
        """
        result = pycgmStatic.hipAxisCenter(l_hip_jc, r_hip_jc, pelvis_axis)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)

    @pytest.mark.parametrize(["frame", "hip_JC", "vsk", "mockReturnVal", "expectedMockArgs", "expected"], [
        # Test from running sample data
        ({'RTHI': np.array([426.50338745, 262.65310669, 673.66247559]),
          'LTHI': np.array([51.93867874, 320.01849365, 723.03186035]),
          'RKNE': np.array([416.98687744, 266.22558594, 524.04089355]),
          'LKNE': np.array([84.62355804, 286.69122314, 529.39819336])},
         [[182.57097863, 339.43231855, 935.52900126], [308.38050472, 322.80342417, 937.98979061]],
         {'RightKneeWidth': 105.0, 'LeftKneeWidth': 105.0},
         [np.array([364.17774614, 292.17051722, 515.19181496]), np.array([143.55478579, 279.90370346, 524.78408753])],
         [[[426.50338745, 262.65310669, 673.66247559], [308.38050472, 322.80342417, 937.98979061], [416.98687744, 266.22558594, 524.04089355], 59.5],
          [[51.93867874, 320.01849365, 723.03186035], [182.57097863, 339.43231855, 935.52900126], [84.62355804, 286.69122314, 529.39819336], 59.5]],
         [np.array([364.17774614, 292.17051722, 515.19181496]),
          np.array([143.55478579, 279.90370346, 524.78408753]),
          np.array([[[364.61959153, 293.06758353, 515.18513093], [363.29019771, 292.60656648, 515.04309095], [364.04724541, 292.24216264, 516.18067112]],
                    [[143.65611282, 280.88685896, 524.63197541], [142.56434499, 280.01777943, 524.86163553], [143.64837987, 280.04650381, 525.76940383]]])]),
        # Test with zeros for all params
        ({'RTHI': np.array([0, 0, 0]), 'LTHI': np.array([0, 0, 0]), 'RKNE': np.array([0, 0, 0]), 'LKNE': np.array([0, 0, 0])},
         [[0, 0, 0], [0, 0, 0]],
         {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing when values are added to frame
        ({'RTHI': np.array([1, 2, 4]), 'LTHI': np.array([-1, 0, 8]), 'RKNE': np.array([8, -4, 5]), 'LKNE': np.array([8, -8, 5])},
         [[0, 0, 0], [0, 0, 0]],
         {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[1, 2, 4], [0, 0, 0], [8, -4, 5], 7.0], [[-1, 0, 8], [0, 0, 0], [8, -8, 5], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing when values are added to hip_JC
        ({'RTHI': np.array([0, 0, 0]), 'LTHI': np.array([0, 0, 0]), 'RKNE': np.array([0, 0, 0]), 'LKNE': np.array([0, 0, 0])},
         [[-8, 8, -2], [1, -9, 2]],
         {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[0, 0, 0], [1, -9, 2], [0, 0, 0], 7.0], [[0, 0, 0], [-8, 8, -2], [0, 0, 0], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, [0.10783277, -0.97049496, 0.21566555]],
                    [nan_3d, nan_3d, [-0.69631062, 0.69631062, -0.17407766]]])]),
        # Testing when values are added to vsk
        ({'RTHI': np.array([0, 0, 0]), 'LTHI': np.array([0, 0, 0]), 'RKNE': np.array([0, 0, 0]), 'LKNE': np.array([0, 0, 0])},
         [[0, 0, 0], [0, 0, 0]],
         {'RightKneeWidth': 9.0, 'LeftKneeWidth': -6.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 11.5], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 4.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing when values are added to mockReturnVal
        ({'RTHI': np.array([0, 0, 0]), 'LTHI': np.array([0, 0, 0]), 'RKNE': np.array([0, 0, 0]), 'LKNE': np.array([0, 0, 0])},
         [[0, 0, 0], [0, 0, 0]],
         {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
         [np.array([-5, -5, -9]), np.array([3, -6, -5])],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
         [np.array([-5, -5, -9]), np.array([3, -6, -5]),
          np.array([[nan_3d, nan_3d, [-4.56314797, -4.56314797, -8.21366635]],
                    [nan_3d, nan_3d, [2.64143142, -5.28286283, -4.4023857]]])]),
        # Testing when values are added to frame and hip_JC
        ({'RTHI': np.array([1, 2, 4]), 'LTHI': np.array([-1, 0, 8]), 'RKNE': np.array([8, -4, 5]), 'LKNE': np.array([8, -8, 5])},
         [[-8, 8, -2], [1, -9, 2]],
         {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 7.0], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[[-0.0512465, -0.22206816, -0.97368348], [0.99284736, 0.09394289, -0.07368069], [0.10783277, -0.97049496, 0.21566555]],
                    [[-0.68318699, -0.71734633, -0.1366374 ], [-0.22001604, 0.02378552, 0.97520623], [-0.69631062, 0.69631062, -0.17407766]]])]),
        # Testing when values are added to frame, hip_JC, and vsk
        ({'RTHI': np.array([1, 2, 4]), 'LTHI': np.array([-1, 0, 8]), 'RKNE': np.array([8, -4, 5]),
          'LKNE': np.array([8, -8, 5])},
         [[-8, 8, -2], [1, -9, 2]],
         {'RightKneeWidth': 9.0, 'LeftKneeWidth': -6.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 11.5], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 4.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[[-0.0512465, -0.22206816, -0.97368348], [0.99284736 ,0.09394289, -0.07368069], [0.10783277, -0.97049496, 0.21566555]],
                    [[-0.68318699, -0.71734633, -0.1366374 ], [-0.22001604, 0.02378552, 0.97520623], [-0.69631062, 0.69631062, -0.17407766]]])]),
        # Testing when values are added to frame, hip_JC, vsk, and mockReturnVal
        ({'RTHI': np.array([1, 2, 4]), 'LTHI': np.array([-1, 0, 8]), 'RKNE': np.array([8, -4, 5]), 'LKNE': np.array([8, -8, 5])},
         [[-8, 8, -2], [1, -9, 2]],
         {'RightKneeWidth': 9.0, 'LeftKneeWidth': -6.0},
         [np.array([-5, -5, -9]), np.array([3, -6, -5])],
         [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 11.5], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 4.0]],
         [np.array([-5, -5, -9]), np.array([3, -6, -5]),
          np.array([[[-5.65539698, -5.75053525, -8.91543265], [-4.39803462, -5.58669523, -9.54168847], [-4.54382845, -5.30411437, -8.16368549]],
                    [[2.57620655, -6.14126448, -5.89467506], [2.32975119, -6.6154814, -4.58533245], [2.39076635, -5.22461171, -4.83384537]]])])])
    def testKneeJointCenter(self, frame, hip_JC, vsk, mockReturnVal, expectedMockArgs, expected):
        """
        This test provides coverage of the kneeJointCenter function in pycgmStatic.py, defined as kneeJointCenter(frame, hip_JC, delta, vsk)
        This test takes 6 parameters:
        frame: dictionary of marker lists
        hip_JC: array of hip_JC containing the x,y,z axes marker positions of the hip joint center
        vsk: dictionary containing subject measurements from a VSK file
        mockReturnVal: the value to be returned by the mock for findJointC
        expectedMockArgs: the expected arguments used to call the mocked function, findJointC
        expected: the expected result from calling kneeJointCenter on frame, hip_JC, vsk, and mockReturnVal
        """
        with patch.object(pycgmStatic, 'findJointC', side_effect=mockReturnVal) as mock_findJointC:
            result = pycgmStatic.kneeJointCenter(frame, hip_JC, None, vsk)

        # Asserting that there were only 2 calls to findJointC
        np.testing.assert_equal(mock_findJointC.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to findJointC
        np.testing.assert_almost_equal(expectedMockArgs[0][0], mock_findJointC.call_args_list[0][0][0], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][1], mock_findJointC.call_args_list[0][0][1], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][2], mock_findJointC.call_args_list[0][0][2], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][3], mock_findJointC.call_args_list[0][0][3], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to findJointC
        np.testing.assert_almost_equal(expectedMockArgs[1][0], mock_findJointC.call_args_list[1][0][0], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][1], mock_findJointC.call_args_list[1][0][1], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][2], mock_findJointC.call_args_list[1][0][2], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][3], mock_findJointC.call_args_list[1][0][3], rounding_precision)

        # Asserting that findShoulderJC returned the correct result given the return value given by mocked findJointC
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

    @pytest.mark.parametrize(["frame", "knee_JC", "vsk", "mockReturnVal", "expectedMockArgs", "expected"], [
        # Test from running sample data
        ({'RTIB': np.array([433.97537231, 211.93408203, 273.3008728 ]), 'LTIB': np.array([50.04016495, 235.90718079, 364.32226562]),
          'RANK': np.array([422.77005005, 217.74053955, 92.86152649]), 'LANK': np.array([58.57380676, 208.54806519, 86.16953278])},
         [np.array([364.17774614, 292.17051722, 515.19181496]), np.array([143.55478579, 279.90370346, 524.78408753]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': 70.0, 'LeftAnkleWidth': 70.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
         [np.array([393.76181608, 247.67829633,  87.73775041]), np.array([98.74901939, 219.46930221,  80.6306816])],
         [[[433.97537231, 211.93408203, 273.3008728 ], [364.17774614, 292.17051722, 515.19181496], [422.77005005, 217.74053955, 92.86152649], 42.0],
          [[50.04016495, 235.90718079, 364.32226562], [143.55478579, 279.90370346, 524.78408753], [58.57380676, 208.54806519, 86.16953278], 42.0]],
         [np.array([393.76181608, 247.67829633, 87.73775041]), np.array([98.74901939, 219.46930221, 80.6306816]),
          [[np.array([394.48171575, 248.37201348, 87.715368]),
            np.array([393.07114384, 248.39110006, 87.61575574]),
            np.array([393.69314056, 247.78157916, 88.73002876])],
           [np.array([98.47494966, 220.42553803, 80.52821783]),
            np.array([97.79246671, 219.20927275, 80.76255901]),
            np.array([98.84848169, 219.60345781, 81.61663775])]]]),
        # Test with zeros for all params
        ({'RTIB': np.array([0, 0, 0]), 'LTIB': np.array([0, 0, 0]), 'RANK': np.array([0, 0, 0]), 'LANK': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0],
          [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)],
           [np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)]]]),
        # Testing when values are added to frame
        ({'RTIB': np.array([-9, 6, -9]), 'LTIB': np.array([0, 2, -1]), 'RANK': np.array([1, 0, -5]),
          'LANK': np.array([2, -4, -5])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[-9, 6, -9], [0, 0, 0], [1, 0, -5], 7.0],
          [[0, 2, -1], [0, 0, 0], [2, -4, -5], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)],
           [np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)]]]),
        # Testing when values are added to knee_JC
        ({'RTIB': np.array([0, 0, 0]), 'LTIB': np.array([0, 0, 0]), 'RANK': np.array([0, 0, 0]), 'LANK': np.array([0, 0, 0])},
         [np.array([-7, 1, 2]), np.array([9, -8, 9]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[0, 0, 0], [-7, 1, 2], [0, 0, 0], 7.0],
          [[0, 0, 0], [9, -8, 9], [0, 0, 0], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array(nan_3d), np.array([-0.95257934, 0.13608276, 0.27216553])],
           [np.array(nan_3d), np.array(nan_3d), np.array([0.59867109, -0.53215208, 0.59867109])]]]),
        # Testing when values are added to vsk
        ({'RTIB': np.array([0, 0, 0]), 'LTIB': np.array([0, 0, 0]), 'RANK': np.array([0, 0, 0]), 'LANK': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': -38.0, 'LeftAnkleWidth': 18.0, 'RightTibialTorsion': 29.0, 'LeftTibialTorsion': -13.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0], -12.0],
          [[0, 0, 0], [0, 0, 0], [0, 0, 0], 16.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)],
           [np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)]]]),
        # Testing when values are added to mockReturnVal
        ({'RTIB': np.array([0, 0, 0]), 'LTIB': np.array([0, 0, 0]), 'RANK': np.array([0, 0, 0]), 'LANK': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
         [np.array([2, -5, 4]), np.array([8, -3, 1])],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0],
          [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
         [np.array([2, -5, 4]), np.array([8, -3, 1]),
          [[np.array(nan_3d), np.array(nan_3d), np.array([1.7018576 , -4.25464401, 3.40371521])],
           [np.array(nan_3d), np.array(nan_3d), np.array([7.07001889, -2.65125708, 0.88375236])]]]),
        # Testing when values are added to frame and knee_JC
        ({'RTIB': np.array([-9, 6, -9]), 'LTIB': np.array([0, 2, -1]), 'RANK': np.array([1, 0, -5]), 'LANK': np.array([2, -4, -5])},
         [np.array([-7, 1, 2]), np.array([9, -8, 9]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], 7.0],
          [[0, 2, -1], [9, -8, 9], [2, -4, -5], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array([-0.26726124, -0.80178373, -0.53452248]), np.array([0.14547859, -0.58191437, 0.80013226]), np.array([-0.95257934, 0.13608276, 0.27216553])],
           [np.array([0.79317435, 0.49803971, -0.35047239]), np.array([-0.11165737, 0.68466825, 0.72025136]), np.array([0.59867109, -0.53215208, 0.59867109])]]]),
        # Testing when values are added to frame, knee_JC, and vsk
        ({'RTIB': np.array([-9, 6, -9]), 'LTIB': np.array([0, 2, -1]), 'RANK': np.array([1, 0, -5]), 'LANK': np.array([2, -4, -5])},
         [np.array([-7, 1, 2]), np.array([9, -8, 9]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': -38.0, 'LeftAnkleWidth': 18.0, 'RightTibialTorsion': 29.0, 'LeftTibialTorsion': -13.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0],
          [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array([-0.30428137, -0.41913816, -0.85541572]), np.array([-0.00233238, -0.89766624, 0.4406698]), np.array([-0.95257934, 0.13608276, 0.27216553])],
           [np.array([0.7477279, 0.63929183, -0.1794685]), np.array([-0.287221, 0.55508569, 0.7806305]), np.array([0.59867109, -0.53215208, 0.59867109])]]]),
        # Testing when values are added to frame, knee_JC, vsk and mockReturnVal
        ({'RTIB': np.array([-9, 6, -9]), 'LTIB': np.array([0, 2, -1]), 'RANK': np.array([1, 0, -5]), 'LANK': np.array([2, -4, -5])},
         [np.array([-7, 1, 2]), np.array([9, -8, 9]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': -38.0, 'LeftAnkleWidth': 18.0, 'RightTibialTorsion': 29.0, 'LeftTibialTorsion': -13.0},
         [np.array([2, -5, 4]), np.array([8, -3, 1])],
         [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0],
          [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0]],
         [np.array([2, -5, 4]), np.array([8, -3, 1]),
          [[np.array([1.48891678, -5.83482493, 3.7953997 ]), np.array([1.73661348, -5.07447603, 4.96181124]), np.array([1.18181818, -4.45454545, 3.81818182])],
           [np.array([8.87317138, -2.54514024, 1.17514093]), np.array([7.52412119, -2.28213872, 1.50814815]), np.array([8.10540926, -3.52704628, 1.84327404])]]])])
    def testAnkleJointCenter(self, frame, knee_JC, vsk, mockReturnVal, expectedMockArgs, expected):
        """
        This test provides coverage of the ankleJointCenter function in pycgmStatic.py, defined as ankleJointCenter(frame, knee_JC, delta, vsk)
        This test takes 6 parameters:
        frame: dictionary of marker lists
        knee_JC: array of knee_JC each x,y,z position.
        vsk: dictionary containing subject measurements from a VSK file
        mockReturnVal: the value to be returned by the mock for findJointC
        expectedMockArgs: the expected arguments used to call the mocked function, findJointC
        expected: the expected result from calling ankleJointCenter on frame, knee_JC, vsk, and mockReturnVal
        """
        with patch.object(pycgmStatic, 'findJointC', side_effect=mockReturnVal) as mock_findJointC:
            result = pycgmStatic.ankleJointCenter(frame, knee_JC, None, vsk)

        # Asserting that there were only 2 calls to findJointC
        np.testing.assert_equal(mock_findJointC.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to findJointC
        np.testing.assert_almost_equal(expectedMockArgs[0][0], mock_findJointC.call_args_list[0][0][0], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][1], mock_findJointC.call_args_list[0][0][1], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][2], mock_findJointC.call_args_list[0][0][2], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][3], mock_findJointC.call_args_list[0][0][3], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to findJointC
        np.testing.assert_almost_equal(expectedMockArgs[1][0], mock_findJointC.call_args_list[1][0][0], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][1], mock_findJointC.call_args_list[1][0][1], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][2], mock_findJointC.call_args_list[1][0][2], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][3], mock_findJointC.call_args_list[1][0][3], rounding_precision)

        # Asserting that findShoulderJC returned the correct result given the return value given by mocked findJointC
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

    @pytest.mark.parametrize(["frame", "static_info", "ankle_JC", "expected"], [
        # Test from running sample data
        ({'RTOE': np.array([442.81997681, 381.62280273, 42.66047668]), 'LTOE': np.array([39.43652725, 382.44522095, 41.78911591])},
         [[0.03482194, 0.14879424, np.random.randint(0, 10)], [0.01139704, 0.02142806, np.random.randint(0, 10)]],
         [np.array([393.76181608, 247.67829633, 87.73775041]),
          np.array([98.74901939, 219.46930221, 80.6306816]),
          [[np.array(nan_3d), np.array([393.07114384, 248.39110006, 87.61575574]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([97.79246671, 219.20927275, 80.76255901]), np.array(nan_3d)]]],
         [np.array([442.81997681, 381.62280273, 42.66047668]),
          np.array([39.43652725, 382.44522095, 41.78911591]),
          np.array([[[442.8881541, 381.76460597, 43.64802096],
                     [441.89515447, 382.00308979, 42.66971773],
                     [442.44573691, 380.70886969, 42.81754643]],
                    [[39.50785213, 382.67891581, 42.75880631],
                     [38.49231839, 382.14765966, 41.93027863],
                     [39.75805858, 381.51956227, 41.98854914]]])]),
        # Test with zeros for all params
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         [[0, 0, np.random.randint(0, 10)], [0, 0, np.random.randint(0, 10)]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values added to frame
        ({'RTOE': np.array([-1, -1, -5]), 'LTOE': np.array([-5, -6, 1])},
         [[0, 0, np.random.randint(0, 10)], [0, 0, np.random.randint(0, 10)]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([-1, -1, -5]), np.array([-5, -6, 1]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values added to static_info
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         [[-6, 7, np.random.randint(0, 10)], [2, -9, np.random.randint(0, 10)]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values added to ankle_JC
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         [[0, 0, np.random.randint(0, 10)], [0, 0, np.random.randint(0, 10)]],
         [np.array([6, 0, 3]), np.array([1, 4, -3]),
          [[np.array(nan_3d), np.array([-2, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([1, -6, 8]), np.array(nan_3d)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[[0.3713906763541037, 0.5570860145311556, -0.7427813527082074], [-0.24913643956121992, 0.8304547985373997, 0.49827287912243984], [0.8944271909999159, 0.0, 0.4472135954999579]],
                   [[-0.6855829496241487, 0.538672317561831, 0.4897021068743917], [0.701080937355391, 0.3073231506215415, 0.6434578466138523], [0.19611613513818404, 0.7844645405527362, -0.5883484054145521]]])]),
          # Testing with values added to frame and static_info
        ({'RTOE': np.array([-1, -1, -5]), 'LTOE': np.array([-5, -6, 1])},
         [[-6, 7, np.random.randint(0, 10)], [2, -9, np.random.randint(0, 10)]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([-1, -1, -5]), np.array([-5, -6, 1]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values added to frame and ankle_JC
        ({'RTOE': np.array([-1, -1, -5]), 'LTOE': np.array([-5, -6, 1])},
         [[0, 0, np.random.randint(0, 10)], [0, 0, np.random.randint(0, 10)]],
         [np.array([6, 0, 3]), np.array([1, 4, -3]),
          [[np.array(nan_3d), np.array([-2, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([1, -6, 8]), np.array(nan_3d)]]],
         [np.array([-1, -1, -5]), np.array([-5, -6, 1]),
          np.array([[[-0.4764529245456802, -0.34134400184779123, -5.540435690791556], [-1.544126730072802, -0.25340750990010874, -4.617213172448785], [-0.3443899318928142, -0.9063414188418306, -4.250731350734645]],
                    [[-5.617369411832039, -5.417908840272649, 1.5291737815703186], [-4.3819280753253675, -6.057228881914318, 1.7840356822261547], [-4.513335736607712, -5.188892894346187, 0.6755571577384749]]])]),
        # Testing with values added to static_info and ankle_JC
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         [[-6, 7, np.random.randint(0, 10)], [2, -9, np.random.randint(0, 10)]],
         [np.array([6, 0, 3]), np.array([1, 4, -3]),
          [[np.array(nan_3d), np.array([-2, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([1, -6, 8]), np.array(nan_3d)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[[0.8676189717605698, 0.41998838044559317, -0.2661711481957037], [-0.35944921047092726, 0.8996435491853136, 0.2478663944569317], [0.3435601620283683, -0.11937857722363693, 0.9315123028533232]],
                    [[0.5438323231671144, -0.8140929502604927, -0.20371321168453085], [0.12764145145799288, 0.32016712879535714, -0.9387228928222822], [0.829429963377473, 0.48450560159311296, 0.27802923924749284]]])]),
        # Testing with values added to frame, static_info and ankle_JC
        ({'RTOE': np.array([-1, -1, -5]), 'LTOE': np.array([-5, -6, 1])},
         [[-6, 7, np.random.randint(0, 10)], [2, -9, np.random.randint(0, 10)]],
         [np.array([6, 0, 3]), np.array([1, 4, -3]),
          [[np.array(nan_3d), np.array([-2, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([1, -6, 8]), np.array(nan_3d)]]],
         [np.array([-1, -1, -5]), np.array([-5, -6, 1]),
          np.array([[[-0.17456964188738444, -0.44190534702217665, -4.915176169482615], [-1.564451151846412, -0.1819624820720035, -4.889503319319258], [-1.0077214691178664, -1.139086223544123, -4.009749828914483]],
                    [[-4.638059331793927, -6.864633064377841, 0.6515626072260268], [-4.6226610672854616, -5.522323332954951, 0.2066272429566376], [-4.147583269429562, -5.844325128086398, 1.4991503297587707]]])])])
    def testFootJointCenter(self, frame, static_info, ankle_JC, expected):
        """
        This test provides coverage of the footJointCenter function in pycgmStatic.py, defined as footJointCenter(frame, static_info, ankle_JC, knee_JC, delta)

        This test takes 4 parameters:
        frame: dictionaries of marker lists
        static_info: array containing offset angles
        ankle_JC: array of ankle_JC each x,y,z position
        expected: the expected result from calling footJointCenter on frame, static_info, and ankle_JC
        """
        result = pycgmStatic.footJointCenter(frame, static_info, ankle_JC, None, None)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

    @pytest.mark.parametrize(["frame", "expected"], [
        # Test from running sample data
        ({'LFHD': np.array([184.55158997, 409.68713379, 1721.34289551]), 'RFHD': np.array([325.82983398, 402.55450439, 1722.49816895]), 'LBHD': np.array([197.8621521 , 251.28889465, 1696.90197754]), 'RBHD': np.array([304.39898682, 242.91339111, 1694.97497559])},
         [[[255.21590218, 407.10741939, 1722.0817318], [254.19105385, 406.14680918, 1721.91767712], [255.18370553, 405.95974655, 1722.90744993]], [255.19071197509766, 406.1208190917969, 1721.9205322265625]]),
        # Basic test with a variance of 1 in the x and y dimensions of the markers
        ({'LFHD': np.array([1, 1, 0]), 'RFHD': np.array([0, 1, 0]), 'LBHD': np.array([1, 0, 0]), 'RBHD': np.array([0, 0, 0])},
         [[[0.5, 2, 0], [1.5, 1, 0], [0.5, 1, -1]], [0.5, 1, 0]]),
        # Setting the markers so there's no variance in the x-dimension
        ({'LFHD': np.array([0, 1, 0]), 'RFHD': np.array([0, 1, 0]), 'LBHD': np.array([0, 0, 0]), 'RBHD': np.array([0, 0, 0])},
         [[nan_3d, nan_3d, nan_3d], [0, 1, 0]]),
        # Setting the markers so there's no variance in the y-dimension
        ({'LFHD': np.array([1, 0, 0]), 'RFHD': np.array([0, 0, 0]), 'LBHD': np.array([1, 0, 0]), 'RBHD': np.array([0, 0, 0])},
         [[nan_3d, nan_3d, nan_3d], [0.5, 0, 0]]),
        # Setting each marker in a different xy quadrant
        ({'LFHD': np.array([-1, 1, 0]), 'RFHD': np.array([1, 1, 0]), 'LBHD': np.array([-1, -1, 0]), 'RBHD': np.array([1, -1, 0])},
         [[[0, 2, 0], [-1, 1, 0], [0, 1, 1]], [0, 1, 0]]),
        # Setting values of the markers so that midpoints will be on diagonals
        ({'LFHD': np.array([-2, 1, 0]), 'RFHD': np.array([1, 2, 0]), 'LBHD': np.array([-1, -2, 0]), 'RBHD': np.array([2, -1, 0])},
         [[[-0.81622777, 2.4486833 ,  0], [-1.4486833, 1.18377223, 0], [-0.5, 1.5,  1]], [-0.5, 1.5, 0]]),
        # Adding the value of 1 in the z dimension for all 4 markers
        ({'LFHD': np.array([1, 1, 1]), 'RFHD': np.array([0, 1, 1]), 'LBHD': np.array([1, 0, 1]), 'RBHD': np.array([0, 0, 1])},
         [[[0.5, 2, 1], [1.5, 1, 1], [0.5, 1, 0]], [0.5, 1, 1]]),
        # Setting the z dimension value higher for LFHD and LBHD
        ({'LFHD': np.array([1, 1, 2]), 'RFHD': np.array([0, 1, 1]), 'LBHD': np.array([1, 0, 2]), 'RBHD': np.array([0, 0, 1])},
         [[[0.5, 2, 1.5], [1.20710678, 1, 2.20710678], [1.20710678, 1, 0.79289322]], [0.5, 1, 1.5]]),
        # Setting the z dimension value higher for LFHD and RFHD
        ({'LFHD': np.array([1, 1, 2]), 'RFHD': np.array([0, 1, 2]), 'LBHD': np.array([1, 0, 1]), 'RBHD': np.array([0, 0, 1])},
         [[[0.5, 1.70710678, 2.70710678], [1.5, 1, 2], [0.5, 1.70710678, 1.29289322]], [0.5, 1, 2]])])
    def testHeadJC(self, frame, expected):
        """
        This test provides coverage of the headJC function in pycgmStatic.py, defined as headJC(frame)
        This test takes 3 parameters:
        frame: dictionary of marker lists
        expected: the expected result from calling headJC on frame
        """
        result = pycgmStatic.headJC(frame)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)

    @pytest.mark.parametrize(["frame", "ankle_JC", "expected"], [
        # Test from running sample data
        ({'RTOE': np.array([433.33508301, 354.97229004, 44.27765274]),
          'LTOE': np.array([31.77310181, 331.23657227, 42.15322876])},
         [np.array([397.45738291, 217.50712216, 87.83068433]), np.array([112.28082818, 175.83265027, 80.98477997]),
          [[np.array(rand_coor), np.array([396.73749179, 218.18875543, 87.69979179]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([111.34886681, 175.49163538, 81.10789314]), np.array(rand_coor)]]],
         [np.array([433.33508301, 354.97229004, 44.27765274]), np.array([31.77310181, 331.23657227, 42.15322876]),
          [[[433.4256618315962, 355.25152027652007, 45.233595181827035],
            [432.36890500826763, 355.2296456773885, 44.29402798451682],
            [433.09363829389764, 354.0471962330562, 44.570749823731354]],
           [[31.806110207058808, 331.49492345678016, 43.11871573923792],
            [30.880216288550965, 330.81014854432254, 42.29786022762896],
            [32.2221740692973, 330.36972887034574, 42.36983123198873]]]]),
        # Test with zeros for all params
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when values are added to frame
        ({'RTOE': np.array([-7, 3, -8]), 'LTOE': np.array([8, 0, -8])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         [np.array([-7, 3, -8]), np.array([8, 0, -8]),
          [[nan_3d, nan_3d, [-6.36624977770237, 2.7283927618724446, -7.275714031659851]],
           [nan_3d, nan_3d, [7.292893218813452, 0.0, -7.292893218813452]]]]),
        # Testing when values are added to ankle_JC[0]
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         [np.array([2, -9, 1]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[nan_3d, nan_3d, [0.21566554640687682, -0.9704949588309457, 0.10783277320343841]],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when values are added to ankle_JC[1]
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([3, -7, 4]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, [0.34874291623145787, -0.813733471206735, 0.46499055497527714]]]]),
        # Testing when values are added to ankle_JC[2]
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([8, -4, 2]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([-9, 7, 4]), np.array(rand_coor)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when values are added to ankle_JC
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         [np.array([2, -9, 1]), np.array([3, -7, 4]),
          [[np.array(rand_coor), np.array([8, -4, 2]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([-9, 7, 4]), np.array(rand_coor)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[[0.21329967236760183, -0.06094276353360052, -0.9750842165376084], [0.9528859437838807, 0.23329276554708803, 0.1938630023560309], [0.21566554640687682, -0.9704949588309457, 0.10783277320343841]],
           [[0.6597830814767823, 0.5655283555515277, 0.4948373111075868], [-0.6656310267523443, 0.1342218942833945, 0.7341115850601987], [0.34874291623145787, -0.813733471206735, 0.46499055497527714]]]]),
        # Testing when values are added to frame and ankle_JC
        ({'RTOE': np.array([-7, 3, -8]), 'LTOE': np.array([8, 0, -8])},
         [np.array([2, -9, 1]), np.array([3, -7, 4]),
          [[np.array(rand_coor), np.array([8, -4, 2]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([-9, 7, 4]), np.array(rand_coor)]]],
         [np.array([-7, 3, -8]), np.array([8, 0, -8]),
          [[[-6.586075309097216, 2.6732173492872757, -8.849634891853084], [-6.249026985898898, 3.6500960420576702, -7.884178291357542], [-6.485504244572473, 2.3140056594299647, -7.485504244572473]],
           [[8.623180382731631, 0.5341546137699694, -7.428751315829338], [7.295040915019964, 0.6999344300621451, -7.885437867872096], [7.6613572692607015, -0.47409982303501746, -7.187257446225685]]]])])
    def testUncorrect_footaxis(self, frame, ankle_JC, expected):
        """
        This test provides coverage of the uncorrect_footaxis function in pycgmStatic.py, defined as uncorrect_footaxis(frame, ankle_JC)

        This test takes 3 parameters:
        frame: dictionaries of marker lists.
        ankle_JC: array of ankle_JC each x,y,z position
        expected: the expected result from calling uncorrect_footaxis on frame and ankle_JC
        """
        result = pycgmStatic.uncorrect_footaxis(frame, ankle_JC)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

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
                    [[4.532940727667331, 1.7868237089330676, 2.818858992574645], [3.2397085122726565, 2.304116595090937, 2.573994730184553], [3.6286093236458963, 1.0715233091147405, 2.0]]])])])
    def testRotaxis_footflat(self, frame, ankle_JC, vsk, expected):
        """
        This test provides coverage of the rotaxis_footflat function in pycgmStatic.py, defined as rotaxis_footflat(frame, ankle_JC, vsk)

        This test takes 4 parameters:
        frame: dictionaries of marker lists.
        ankle_JC: array of ankle_JC each x,y,z position
        vsk: dictionary containing subject measurements from a VSK file
        expected: the expected result from calling rotaxis_footflat on frame, ankle_JC and vsk
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
        ({'RTOE': np.array([0, 0, 0]),
          'LTOE': np.array([0, 0, 0]),
          'RHEE': np.array([0, 0, 0]),
          'LHEE': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when values are added to frame
        ({'RTOE': np.array([5, -2, -2]),
          'LTOE': np.array([-2, -7, -1]),
          'RHEE': np.array([3, 5, 9]),
          'LHEE': np.array([-7, 6, 1])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[nan_3d, nan_3d, [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [nan_3d, nan_3d, [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing when values are added to ankle_JC
        ({'RTOE': np.array([0, 0, 0]),
          'LTOE': np.array([0, 0, 0]),
          'RHEE': np.array([0, 0, 0]),
          'LHEE': np.array([0, 0, 0])},
         [np.array([-8, 6, 2]), np.array([3, 6, -3]),
          [[np.array(nan_3d), np.array([-7, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([2, -7, -2]), np.array(nan_3d)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when values are added to frame and ankle_JC[0]
        ({'RTOE': np.array([5, -2, -2]),
          'LTOE': np.array([-2, -7, -1]),
          'RHEE': np.array([3, 5, 9]),
          'LHEE': np.array([-7, 6, 1])},
         [np.array([-8, 6, 2]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[[4.519177631054049, -2.7767130575280747, -1.5931503031995797], [5.8636094856901915, -2.3392751550925754, -1.6270777220883264], [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [nan_3d, nan_3d, [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing when values are added to frame and ankle_JC[1]
        ({'RTOE': np.array([5, -2, -2]),
          'LTOE': np.array([-2, -7, -1]),
          'RHEE': np.array([3, 5, 9]),
          'LHEE': np.array([-7, 6, 1])},
         [np.array([0, 0, 0]), np.array([3, 6, -3]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[nan_3d, nan_3d, [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [[-2.5911479210576127, -7.104320221363108, -1.7997883637838292], [-2.7240728617802468, -7.368214526980399, -0.4167877290780265], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing when values are added to frame and ankle_JC[2]
        ({'RTOE': np.array([5, -2, -2]),
          'LTOE': np.array([-2, -7, -1]),
          'RHEE': np.array([3, 5, 9]),
          'LHEE': np.array([-7, 6, 1])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([-7, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([2, -7, -2]), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[[5.578725405755168, -1.2684037323472404, -2.36033846018718], [4.198695813697202, -1.5720307186791873, -2.4180357583501166], [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [[-1.2572186472917923, -6.628609323645897, -1.5570860145311554], [-2.567462100766509, -7.092377551287571, -1.8182011685470592], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]]),
        # Testing when values are added to frame and ankle_JC
        ({'RTOE': np.array([5, -2, -2]),
          'LTOE': np.array([-2, -7, -1]),
          'RHEE': np.array([3, 5, 9]),
          'LHEE': np.array([-7, 6, 1])},
         [np.array([-8, 6, 2]), np.array([3, 6, -3]),
          [[np.array(nan_3d), np.array([-7, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([2, -7, -2]), np.array(nan_3d)]]],
         [np.array([5, -2, -2]), np.array([-2, -7, -1]),
          [[[5.049326362366699, -2.8385481602338833, -1.4574100139663109], [5.987207376506346, -1.8765990779367068, -1.8990356092209417], [4.848380391284219, -1.4693313694947676, -1.1660921520632064]],
           [[-2.446949206712144, -7.0343807082086265, -1.8938984134242876], [-2.820959061315946, -7.381159564182403, -0.5748604861042421], [-2.355334527259351, -6.076130229125688, -0.8578661890962597]]]])])
    def testRotaxis_nonfootflat(self, frame, ankle_JC, expected):
        """
        This test provides coverage of the rotaxis_nonfootflat function in pycgmStatic.py, defined as rotaxis_nonfootflat(frame, ankle_JC)

        This test takes 3 parameters:
        frame: dictionaries of marker lists.
        ankle_JC: array of ankle_JC each x,y,z position
        expected: the expected result from calling rotaxis_nonfootflat on frame and ankle_JC
        """
        result = pycgmStatic.rotaxis_nonfootflat(frame, ankle_JC)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

    @pytest.mark.parametrize(["a", "b", "c", "delta", "expected"], [
        # Test from running sample data
        ([426.50338745, 262.65310669, 673.66247559],
         [308.38050472, 322.80342417, 937.98979061],
         [416.98687744, 266.22558594, 524.04089355],
         59.5,
         [364.17774614, 292.17051722, 515.19181496]),
        # Testing with basic value in a and c
        ([1, 0, 0], [0, 0, 0], [0, 0, 1], 0.0, [0, 0, 1]),
        # Testing with value in a and basic value in c
        ([-7, 1, 2], [0, 0, 0], [0, 0, 1], 0.0, [0, 0, 1]),
        #  Testing with value in b and basic value in c
        ([0, 0, 0], [1, 4, 3], [0, 0, 1], 0.0, [0, 0, 1]),
        #  Testing with value in a and b and basic value in c
        ([-7, 1, 2], [1, 4, 3], [0, 0, 1], 0.0, [0, 0, 1]),
        #  Testing with value in a, b, and c
        ([-7, 1, 2], [1, 4, 3], [3, 2, -8], 0.0, [3, 2, -8]),
        # Testing with value in a, b, c and delta of 1
        ([-7, 1, 2], [1, 4, 3], [3, 2, -8], 1.0, [3.91270955, 2.36111526, -7.80880104]),
        # Testing with value in a, b, c and delta of 20
        ([-7, 1, 2], [1, 4, 3], [3, 2, -8], 10.0, [5.86777669, 5.19544877, 1.031332352])])
    def testfindJointC(self, a, b, c, delta, expected):
        """
        This test provides coverage of the findJointC function in pycgmStatic.py, defined as findJointC(a, b, c, delta)
        This test takes 5 parameters:
        a: list markers of x,y,z position
        b: list markers of x,y,z position
        c: list markers of x,y,z position
        delta: length from marker to joint center, retrieved from subject measurement file
        expected: the expected result from calling findJointC on a, b, c, and delta
        """
        result = pycgmStatic.findJointC(a, b, c, delta)
        np.testing.assert_almost_equal(result, expected, rounding_precision)
