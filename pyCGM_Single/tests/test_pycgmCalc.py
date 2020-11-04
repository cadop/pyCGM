import sys
import unittest
import pyCGM_Single.pycgmCalc as pycgmCalc
import pyCGM_Single.pycgmIO as pycgmIO
import pyCGM_Single.pycgmStatic as pycgmStatic
import pyCGM_Single.pyCGM_Helpers as pyCGM_Helpers
import numpy as np

rounding_precision = 8


class TestPycgmCalc(unittest.TestCase):

    file_names = pyCGM_Helpers.getfilenames(x=2)
    c3d_file, vsk_file = file_names[1:3]
    c3d_data = pycgmIO.loadC3D(c3d_file)[0]
    vsk_data = pycgmIO.loadVSK(vsk_file, False)
    vsk = pycgmStatic.getStatic(c3d_data, vsk_data, flat_foot=False)

    

    # calcKinetics is equivalent to pycgmKinetics.getKinetics() - would be duplicate test

    def test_calcAngles(self):
        result = pycgmCalc.calcAngles(self.c3d_data, vsk=self.vsk)
        if type(result) == tuple:
            joint_angle_vals, joint_center_locs = result
        else:
            joint_angle_vals = result

        # Tests pelx
        pelx_expected_result =    np.array([[ -0.45646046,  -5.76277607,   4.80620732],
                                            [  3.03831198,  -7.02211832, -17.40722232],
                                            [ -3.00351427,  -4.5429072 ,  -1.7375533 ],
                                            [  3.74245063,   1.83607381, -21.13452485],
                                            [ -0.37416136,  -0.26949328, -23.95627279],
                                            [  3.73277638,  -8.72559625,  29.38805467],
                                            [  4.1585744 ,   1.11320439,  -4.87112244],
                                            [-83.62619447,  -5.22096754,  75.92556425],
                                            [ 87.39254842, 169.33246201, -64.83149434],
                                            [  0.59355258,  -0.16900286, -88.78337074],
                                            [ -3.25087941,   4.24956567, 267.59917792],
                                            [  4.17050116,  -3.7462158 ,  -3.90225314],
                                            [ -9.86327857,   4.10321636,  -2.80874737],
                                            [ 11.79826334,  47.0848481 ,  15.79677853],
                                            [  4.01711439,  40.08073157,   7.55558543],
                                            [ 37.89341078,  -0.        ,   0.        ],
                                            [ 36.32373414,   0.        ,  -0.        ],
                                            [  9.85050913,  15.46090162, 126.03268251],
                                            [  6.67871519,  17.79096147, 123.74619493]])

        pelx = joint_angle_vals[0]
        np.testing.assert_almost_equal(pelx.round(rounding_precision)[0], pelx_expected_result, rounding_precision)

    def test_Calc(self):
        """
        This test provides coverage of the Calc function in pycgmCalc.py,
        defined as Calc(start, end, data, vsk), where start and end are integer
        indices, data is either a list of dicts or a list, and vsk is
        either a dict or a list.
        
        The function is a wrapper for calcFrames, with the only modification
        being the addition of a start and end point for the data list.
        """
        pass

    def test_calcFrames(self):
        """
        This test provides coverage of the calcFrames function in pycgmCalc.py,
        defined as calcFrames(data, vsk), where data is either a list of dicts
        or a list, and vsk is either a dict or a list.
        """

        accuracy_tests = [
            0
            ]

        """xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"""
        
        # Testing case of data being a list of dictionaries and vsk being a dict
        angles, joints = pycgmCalc.calcFrames(self.c3d_data, self.vsk)

        # Testing angles
        # There are over 250 pieces of data per frame, making it a bit excessive when testing
        # accuracy of multiple frames in a row
        # Hence, only testing the first 20 elements of the first 5 frames here
        # This covers pelx through lankley in pyCGM.JointAngleCalc
        expected_angles = np.array([[-0.45646046,   -5.76277607,    4.80620732,    3.03831198,
                                     -7.02211832,  -17.40722232,   -3.00351427,   -4.5429072 ,
                                     -1.7375533 ,    3.74245063,    1.83607381,  -21.13452485,
                                     -0.37416136,   -0.26949328,  -23.95627279,    3.73277638,
                                     -8.72559625,   29.38805467,    4.1585744 ,    1.11320439],
                                    [-0.45789927,   -5.75510865,    4.80248808,    3.02959298,
                                     -7.01771319,  -17.40062371,   -3.00790103,   -4.54741702,
                                     -1.7593788 ,    3.74217923,    1.83141621,  -21.13601231,
                                     -0.37011802,   -0.26310096,  -23.95115687,    3.73051125,
                                     -8.7234315 ,   29.38219349,    4.15715309,    1.10940092],
                                    [-0.45608902,   -5.75700103,    4.78977193,    3.03176127,
                                     -7.02096677,  -17.42333757,   -3.01080904,   -4.54412767,
                                     -1.72618007,    3.76002588,    1.83372416,  -21.06900299,
                                     -0.37404409,   -0.2695666 ,  -23.94131247,    3.75661308,
                                     -8.69521377,   29.30572748,    4.16097315,    1.11289797],
                                    [-0.45851004,   -5.75217475,    4.78699316,    3.02112921,
                                     -7.01761788,  -17.38526492,   -3.0127356 ,   -4.54803339,
                                     -1.76091744,    3.73957047,    1.82914314,  -21.14547928,
                                     -0.37367971,   -0.26559781,  -23.95681519,    3.72862659,
                                     -8.71443697,   29.35783269,    4.15269901,    1.10437245],
                                    [-0.45979952,   -5.75626564,    4.78014806,    3.02853276,
                                     -7.02092575,  -17.39239597,   -3.0037155 ,   -4.54858795,
                                     -1.76358003,    3.75179103,    1.83587525,  -21.14139803,
                                     -0.36570291,   -0.26727198,  -23.94861123,    3.7383873 ,
                                     -8.71386289,   29.35627745,    4.16058229,    1.10477669]
                                    ])

        result_angles = np.array([np.around(frame[:20], rounding_precision) for frame in angles[0:5]])
        np.testing.assert_almost_equal(result_angles, expected_angles, rounding_precision)

        # Testing joints
        accuracy_tests = [
            (0, 'Pelvis'),
            (0, 'RKnee'),
            (1, 'Pelvis'),
            (1, 'RKnee'),
            ]
        accuracy_results = [
            np.array([246.152565,  353.26243591, 1031.71362305]),
            np.array([363.02405738, 263.18822023, 515.0972345]),
            np.array([246.16200256,  353.27105713, 1031.71856689]),
            np.array([363.01220104, 263.21150444, 515.08412259])
            ]
        
        for i in range(len(accuracy_tests)):
            result = np.around(joints[accuracy_tests[i][0]][accuracy_tests[i][1]], rounding_precision)
            expected = accuracy_results[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)
