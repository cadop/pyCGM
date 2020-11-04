import unittest
import pyCGM_Single.pycgmStatic as pycgmStatic
import numpy as np

rounding_precision = 8

class TestPycgmStaticAxis(unittest.TestCase):

    def testPelvisJointCenter(self):
        nan_3d = [np.nan, np.nan, np.nan]
        testcases = [
            # Test from running sample data
            [{'RASI': np.array([357.90066528, 377.69210815, 1034.97253418]), 'LASI': np.array([145.31594849, 405.79052734, 1030.81445312]),
              'RPSI': np.array([274.00466919, 205.64402771, 1051.76452637]), 'LPSI': np.array([189.15231323, 214.86122131, 1052.73486328])},
             [np.array([251.60830688, 391.74131775, 1032.89349365]),
              np.array([[251.74063624, 392.72694721, 1032.78850073], [250.61711554, 391.87232862, 1032.8741063], [251.60295336, 391.84795134, 1033.88777762]]),
              np.array([231.57849121, 210.25262451, 1052.24969482])]],
            # Test with zeros for all params
            [{'SACR': np.array([0, 0, 0]), 'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0]),
              'RPSI': np.array([0, 0, 0]), 'LPSI': np.array([0, 0, 0])},
             [np.array([0, 0, 0]), np.array([nan_3d, nan_3d, nan_3d]), np.array([0, 0, 0])]],
            # Testing when adding values to frame['RASI'] and frame['LASI']
            [{'RASI': np.array([-6, 6, 3]), 'LASI': np.array([-7, -9, 1]), 'RPSI': np.array([0, 0, 0]), 'LPSI': np.array([0, 0, 0])},
             [np.array([-6.5, -1.5,  2.0]),
              np.array([[-7.44458106, -1.48072284, 2.32771179], [-6.56593805, -2.48907071, 1.86812391], [-6.17841206, -1.64617634, 2.93552855]]),
              np.array([0, 0, 0])]],
            # Testing when adding values to frame['RPSI'] and frame['LPSI']
            [{'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0]), 'RPSI': np.array([1, 0, -4]), 'LPSI': np.array([7, -2, 2])},
             [np.array([0, 0, 0]), np.array([nan_3d, nan_3d, nan_3d]), np.array([4., -1.0, -1.0])]],
            # Testing when adding values to frame['SACR']
            [{'SACR': np.array([-4, 8, -5]), 'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0]),
              'RPSI': np.array([0, 0, 0]), 'LPSI': np.array([0, 0, 0])},
             [np.array([0, 0, 0]), np.array([nan_3d, nan_3d, nan_3d]), np.array([-4,  8, -5,])]],
            # Testing when adding values to frame['RASI'], frame['LASI'], frame['RPSI'] and frame['LPSI']
            [{'RASI': np.array([-6, 6, 3]), 'LASI': np.array([-7, -9, 1]), 'RPSI': np.array([1, 0, -4]), 'LPSI': np.array([7, -2, 2])},
             [np.array([-6.5, -1.5,  2.0]),
              np.array([[-7.45825845, -1.47407957, 2.28472598], [-6.56593805, -2.48907071, 1.86812391], [-6.22180416, -1.64514566, 2.9494945]]),
              np.array([4.0, -1.0, -1.0])]],
            # Testing when adding values to frame['SACR'], frame['RASI'] and frame['LASI']
            [{'SACR': np.array([-4, 8, -5]), 'RASI': np.array([-6, 6, 3]), 'LASI': np.array([-7, -9, 1]),
              'RPSI': np.array([0, 0, 0]), 'LPSI': np.array([0, 0, 0])},
             [np.array([-6.5, -1.5,  2.0]),
              np.array([[-6.72928306, -1.61360872, 2.96670695], [-6.56593805, -2.48907071, 1.86812391], [-5.52887619, -1.59397972, 2.21928602]]),
              np.array([-4, 8, -5])]],
            # Testing when adding values to frame['SACR'], frame['RPSI'] and frame['LPSI']
            [{'SACR': np.array([-4, 8, -5]), 'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0]),
              'RPSI': np.array([1, 0, -4]), 'LPSI': np.array([7, -2, 2])},
             [np.array([0, 0, 0]), np.array([nan_3d, nan_3d, nan_3d]), np.array([-4,  8, -5])]],
            # Testing when adding values to frame['SACR'], frame['RASI'], frame['LASI'], frame['RPSI'] and frame['LPSI']
            [{'SACR': np.array([-4, 8, -5]), 'RASI': np.array([-6, 6, 3]), 'LASI': np.array([-7, -9, 1]),
              'RPSI': np.array([1, 0, -4]), 'LPSI': np.array([7, -2, 2])},
             [np.array([-6.5, -1.5,  2.0]),
              np.array([[-6.72928306, -1.61360872, 2.96670695], [-6.56593805, -2.48907071, 1.86812391], [-5.52887619, -1.59397972,  2.21928602]]),
              np.array([-4,  8, -5])]]]
        for testcase in testcases:
            result = pycgmStatic.pelvisJointCenter(testcase[0])
            np.testing.assert_almost_equal(result[0], testcase[1][0], rounding_precision)
            np.testing.assert_almost_equal(result[1], testcase[1][1], rounding_precision)
            np.testing.assert_almost_equal(result[2], testcase[1][2], rounding_precision)

    def testHipJointCenter(self):
        testcases = [
            # Test from running sample data
            [[251.608306884766, 391.741317749023, 1032.893493652344], [251.740636241119, 392.726947206848, 1032.788500732036], [250.617115540376, 391.872328624646, 1032.874106304030], [251.602953357582, 391.847951338178, 1033.887777624562],
             {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512, 'L_AsisToTrocanterMeasure': 72.512, 'InterAsisDistance': 215.908996582031},
             [[182.57097863, 339.43231855, 935.52900126], [308.38050472, 322.80342417, 937.98979061]]],
            # Basic test with zeros for all params
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
             [[0, 0, 0], [0, 0, 0]]],
            # Testing when values are added to pel_origin
            [[1, 0, -3], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
             [[-6.1387721, 0, 18.4163163], [8.53165418, 0, -25.59496255]]],
            # Testing when values are added to pel_x
            [[0, 0, 0], [-5, -3, -6], [0, 0, 0], [0, 0, 0],
             {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
             [[54.02442793, 32.41465676, 64.82931352], [54.02442793, 32.41465676, 64.82931352]]],
            # Testing when values are added to pel_y
            [[0, 0, 0], [0, 0, 0], [4, -1, 2], [0, 0, 0],
             {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
             [[29.34085257, -7.33521314, 14.67042628], [-29.34085257,   7.33521314, -14.67042628]]],
            # Testing when values are added to pel_z
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [3, 8, 2],
             {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
             [[31.82533363, 84.86755635, 21.21688909], [31.82533363, 84.86755635, 21.21688909]]],
            # Test when values are added to pel_x, pel_y, and pel_z
            [[0, 0, 0], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
             {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
             [[115.19061413, 109.94699997, 100.71662889], [56.508909  , 124.61742625,  71.37577632]]],
            # Test when values are added to pel_origin, pel_x, pel_y, and pel_z
            [[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
             {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
             [[109.05184203, 109.94699997, 119.13294518], [65.04056318, 124.61742625,  45.78081377]]],
            # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[MeanLegLength]
            [[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
             {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
             [[100.88576753,  97.85280235, 106.39612748], [61.83654463, 110.86920998,  41.31408931]]],
            # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[R_AsisToTrocanterMeasure]
            [[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
             {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
             [[109.05184203, 109.94699997, 119.13294518], [-57.09307697, 115.44008189,  14.36512267]]],
            # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[L_AsisToTrocanterMeasure]
            [[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
             {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0-7.0, 'InterAsisDistance': 0.0},
             [[73.42953032, 107.27027453, 109.97003528], [65.04056318, 124.61742625,  45.78081377]]],
            # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[InterAsisDistance]
            [[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
             {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 11.0},
             [[125.55184203, 104.44699997, 146.63294518], [48.54056318, 130.11742625,  18.28081377]]],
            # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and all values in vsk
            [[1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
             {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': -7.0, 'InterAsisDistance': 11.0},
             [[81.76345582,  89.67607691, 124.73321758], [-76.79709552, 107.19186562, -17.60160178]]]]
        for testcase in testcases:
            result = pycgmStatic.hipJointCenter(None, testcase[0], testcase[1], testcase[2], testcase[3], testcase[4])
            np.testing.assert_almost_equal(result[0], testcase[5][0], rounding_precision)
            np.testing.assert_almost_equal(result[1], testcase[5][1], rounding_precision)

    def testHipAxisCenter(self):
        rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]
        testcases = [
            # Test from running sample data
            [[182.57097863, 339.43231855, 935.52900126], [308.38050472, 322.80342417, 937.98979061],
             [np.array([251.60830688, 391.74131775, 1032.89349365]), np.array(
                 [[251.74063624, 392.72694721, 1032.78850073], [250.61711554, 391.87232862, 1032.8741063],
                  [251.60295336, 391.84795134, 1033.88777762]]), np.array([231.57849121, 210.25262451, 1052.24969482])],
             [[245.47574167208043, 331.1178713574418, 936.7593959314677],
              [[245.60807102843359, 332.10350081526684, 936.6544030111602],
               [244.48455032769033, 331.2488822330648, 936.7400085831541],
               [245.47038814489719, 331.22450494659665, 937.7536799036861]]]],
            # Basic test with zeros for all params
            [[0, 0, 0], [0, 0, 0],
             [np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
             [[0, 0, 0], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
            # Testing when values are added to l_hip_jc
            [[1, -3, 2], [0, 0, 0],
             [np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
             [[0.5, -1.5, 1], [[0.5, -1.5, 1], [0.5, -1.5, 1], [0.5, -1.5, 1]]]],
            # Testing when values are added to r_hip_jc
            [[0, 0, 0], [-8, 1, 4],
             [np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
             [[-4, 0.5, 2], [[-4, 0.5, 2], [-4, 0.5, 2], [-4, 0.5, 2]]]],
            # Testing when values are added to l_hip_jc and r_hip_jc
            [[8, -3, 7], [5, -2, -1],
             [np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
             [[6.5, -2.5, 3], [[6.5, -2.5, 3], [6.5, -2.5, 3], [6.5, -2.5, 3]]]],
            # Testing when values are added to pelvis_axis[0]
            [[0, 0, 0], [0, 0, 0],
             [np.array([1, -3, 6]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
             [[0, 0, 0], [[-1, 3, -6], [-1, 3, -6], [-1, 3, -6]]]],
            # Testing when values are added to pelvis_axis[1]
            [[0, 0, 0], [0, 0, 0],
             [np.array([0, 0, 0]), np.array([[1, 0, 5], [-2, -7, -3], [9, -2, 7]]), np.array(rand_coor)],
             [[0, 0, 0], [[1, 0, 5], [-2, -7, -3], [9, -2, 7]]]],
            # Testing when values are added to pelvis_axis[0] and pelvis_axis[1]
            [[0, 0, 0], [0, 0, 0],
             [np.array([-3, 0, 5]), np.array([[-4, 5, -2], [0, 0, 0], [8, 5, -1]]), np.array(rand_coor)],
             [[0, 0, 0], [[-1, 5, -7], [3, 0, -5], [11, 5, -6]]]],
            # Testing when values are added to all params
            [[-5, 3, 8], [-3, -7, -1],
             [np.array([6, 3, 9]), np.array([[5, 4, -2], [0, 0, 0], [7, 2, 3]]), np.array(rand_coor)],
             [[-4, -2, 3.5], [[-5, -1, -7.5], [-10, -5, -5.5], [-3, -3, -2.5]]]]]
        for testcase in testcases:
            result = pycgmStatic.hipAxisCenter(testcase[0], testcase[1], testcase[2])
            np.testing.assert_almost_equal(result[0], testcase[3][0], rounding_precision)
            np.testing.assert_almost_equal(result[1], testcase[3][1], rounding_precision)

    def testKneeJointCenter(self):
        testcases = [
            # Test from running sample data
            [{'RTHI': np.array([426.50338745, 262.65310669, 673.66247559]),
              'LTHI': np.array([51.93867874, 320.01849365, 723.03186035]),
              'RKNE': np.array([416.98687744, 266.22558594, 524.04089355]),
              'LKNE': np.array([84.62355804, 286.69122314, 529.39819336])},
             [[182.57097863, 339.43231855, 935.52900126], [308.38050472, 322.80342417, 937.98979061]],
             0,
             {'RightKneeWidth': 105.0, 'LeftKneeWidth': 105.0},
             [np.array([364.17774614, 292.17051722, 515.19181496]),
              np.array([143.55478579, 279.90370346, 524.78408753]),
              np.array([[[364.61959153, 293.06758353, 515.18513093], [363.29019771, 292.60656648, 515.04309095],
                         [364.04724541, 292.24216264, 516.18067112]],
                        [[143.65611282, 280.88685896, 524.63197541], [142.56434499, 280.01777943, 524.86163553],
                         [143.64837987, 280.04650381, 525.76940383]]])]],

            # Testing when
            # [{'RTHI': np.array([0, 0, 0]), 'LTHI': np.array([0, 0, 0]), 'RKNE': np.array([0, 0, 0]), 'LKNE': np.array([0, 0, 0])},
            # [[0, 0, 0], [0, 0, 0]],
            # 0,
            # {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
            # [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])]],
        ]
        for testcase in testcases:
            result = pycgmStatic.kneeJointCenter(testcase[0], testcase[1], testcase[2], testcase[3])
            np.testing.assert_almost_equal(result[0], testcase[4][0], rounding_precision)
            np.testing.assert_almost_equal(result[1], testcase[4][1], rounding_precision)
            np.testing.assert_almost_equal(result[2], testcase[4][2], rounding_precision)

    def testAnkleJointCenter(self):
        testcases = [
            # Test from running sample data
            [{'RTIB': np.array([433.97537231, 211.93408203, 273.3008728 ]),
              'LTIB': np.array([50.04016495, 235.90718079, 364.32226562]),
              'RANK': np.array([422.77005005, 217.74053955, 92.86152649]),
              'LANK': np.array([58.57380676, 208.54806519, 86.16953278])},
             [np.array([364.17774614, 292.17051722, 515.19181496]),
              np.array([143.55478579, 279.90370346, 524.78408753]),
              np.array([[[364.61959153, 293.06758353, 515.18513093],
                         [363.29019771, 292.60656648, 515.04309095],
                         [364.04724541, 292.24216264, 516.18067112]],
                        [[143.65611282, 280.88685896, 524.63197541], [142.56434499, 280.01777943, 524.86163553], [143.64837987, 280.04650381, 525.76940383]]])],
             0,
             {'RightAnkleWidth': 70.0, 'LeftAnkleWidth': 70.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
             [np.array([393.76181608, 247.67829633, 87.73775041]),
              np.array([98.74901939, 219.46930221, 80.6306816]),
              [[np.array([394.48171575, 248.37201348, 87.715368]),
                np.array([393.07114384, 248.39110006, 87.61575574]),
                np.array([393.69314056, 247.78157916, 88.73002876])],
               [np.array([98.47494966, 220.42553803, 80.52821783]),
                np.array([97.79246671, 219.20927275, 80.76255901]),
                np.array([98.84848169, 219.60345781, 81.61663775])]]]],

            # Testing when
            #[{'RTIB': np.array([0, 0, 0]), 'LTIB': np.array([0, 0, 0]), 'RANK': np.array([0, 0, 0]), 'LANK': np.array([0, 0, 0])},
             #[np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])],
             #0,
             #{'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
             #[np.array([0, 0, 0]), np.array([0, 0, 0]), [[np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])],
              # [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])]]]],
        ]
        for testcase in testcases:
            result = pycgmStatic.ankleJointCenter(testcase[0], testcase[1], testcase[2], testcase[3])
            np.testing.assert_almost_equal(result[0], testcase[4][0], rounding_precision)
            np.testing.assert_almost_equal(result[1], testcase[4][1], rounding_precision)
            np.testing.assert_almost_equal(result[2], testcase[4][2], rounding_precision)

    def testHeadJC(self):
        nan_3d = [np.nan, np.nan, np.nan]
        testcases = [
            # Test from running sample data
            [{'LFHD': np.array([184.55158997, 409.68713379, 1721.34289551]),
              'RFHD': np.array([325.82983398, 402.55450439, 1722.49816895]),
              'LBHD': np.array([197.8621521, 251.28889465, 1696.90197754]),
              'RBHD': np.array([304.39898682, 242.91339111, 1694.97497559])},
             [[[255.21590218, 407.10741939, 1722.0817318],
               [254.19105385, 406.14680918, 1721.91767712],
               [255.18370553, 405.95974655, 1722.90744993]],
              [255.19071197509766, 406.1208190917969, 1721.9205322265625]]],
            # Basic test with a variance of 1 in the x and y dimensions of the markers
            [{'LFHD': np.array([1, 1, 0]), 'RFHD': np.array([0, 1, 0]), 'LBHD': np.array([1, 0, 0]),
              'RBHD': np.array([0, 0, 0])},
             [[[0.5, 2, 0], [1.5, 1, 0], [0.5, 1, -1]], [0.5, 1, 0]]],
            # Setting the markers so there's no variance in the x-dimension
            [{'LFHD': np.array([0, 1, 0]), 'RFHD': np.array([0, 1, 0]), 'LBHD': np.array([0, 0, 0]),
              'RBHD': np.array([0, 0, 0])},
             [[nan_3d, nan_3d, nan_3d], [0, 1, 0]]],
            # Setting the markers so there's no variance in the y-dimension
            [{'LFHD': np.array([1, 0, 0]), 'RFHD': np.array([0, 0, 0]), 'LBHD': np.array([1, 0, 0]),
              'RBHD': np.array([0, 0, 0])},
             [[nan_3d, nan_3d, nan_3d], [0.5, 0, 0]]],
            # Setting each marker in a different xy quadrant
            [{'LFHD': np.array([-1, 1, 0]), 'RFHD': np.array([1, 1, 0]), 'LBHD': np.array([-1, -1, 0]),
              'RBHD': np.array([1, -1, 0])},
             [[[0, 2, 0], [-1, 1, 0], [0, 1, 1]], [0, 1, 0]]],
            # Setting values of the markers so that midpoints will be on diagonals
            [{'LFHD': np.array([-2, 1, 0]), 'RFHD': np.array([1, 2, 0]), 'LBHD': np.array([-1, -2, 0]),
              'RBHD': np.array([2, -1, 0])},
             [[[-0.81622777, 2.4486833, 0], [-1.4486833, 1.18377223, 0], [-0.5, 1.5, 1]], [-0.5, 1.5, 0]]],
            # Adding the value of 1 in the z dimension for all 4 markers
            [{'LFHD': np.array([1, 1, 1]), 'RFHD': np.array([0, 1, 1]), 'LBHD': np.array([1, 0, 1]),
              'RBHD': np.array([0, 0, 1])},
             [[[0.5, 2, 1], [1.5, 1, 1], [0.5, 1, 0]], [0.5, 1, 1]]],
            # Setting the z dimension value higher for LFHD and LBHD
            [{'LFHD': np.array([1, 1, 2]), 'RFHD': np.array([0, 1, 1]), 'LBHD': np.array([1, 0, 2]),
              'RBHD': np.array([0, 0, 1])},
             [[[0.5, 2, 1.5], [1.20710678, 1, 2.20710678], [1.20710678, 1, 0.79289322]], [0.5, 1, 1.5]]],
            # Setting the z dimension value higher for LFHD and RFHD
            [{'LFHD': np.array([1, 1, 2]), 'RFHD': np.array([0, 1, 2]), 'LBHD': np.array([1, 0, 1]),
              'RBHD': np.array([0, 0, 1])},
             [[[0.5, 1.70710678, 2.70710678], [1.5, 1, 2], [0.5, 1.70710678, 1.29289322]], [0.5, 1, 2]]]]
        for testcase in testcases:
            result = pycgmStatic.headJC(testcase[0])
            np.testing.assert_almost_equal(result[0], testcase[1][0], rounding_precision)
            np.testing.assert_almost_equal(result[1], testcase[1][1], rounding_precision)

    def testfindJointC(self):
        testcases = [
            # Test from running sample data
            [[426.50338745, 262.65310669, 673.66247559],
             [308.38050472, 322.80342417, 937.98979061],
             [416.98687744, 266.22558594, 524.04089355],
             59.5,
             [364.17774614, 292.17051722, 515.19181496]],
        ]
        for testcase in testcases:
            result = pycgmStatic.findJointC(testcase[0], testcase[1], testcase[2], testcase[3])
            np.testing.assert_almost_equal(result, testcase[4], rounding_precision)