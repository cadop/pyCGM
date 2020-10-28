import sys
import unittest
import pyCGM_Single.pycgmCalc as pycgmCalc
import pyCGM_Single.pycgmIO as pycgmIO
import pyCGM_Single.pycgmStatic as pycgmStatic
import pyCGM_Single.pyCGM_Helpers as pyCGM_Helpers
import numpy as np

rounding_precision = 8
#np.set_printoptions(precision=rounding_precision, threshold=sys.maxsize, suppress=True)


class TestPycgmCalc(unittest.TestCase):

    file_names = pyCGM_Helpers.getfilenames(x=2)
    c3d_file, vsk_file = file_names[1:3]
    c3d_data = pycgmIO.loadC3D(c3d_file)[0]
    vsk_data = pycgmIO.loadVSK(vsk_file, False)
    vsk = pycgmStatic.getStatic(c3d_data, vsk_data, flat_foot=False)

    # Function is equivalent to pycgmKinetics.getKinetics() - would be duplicate test
    # def test_calcKinetics(self):
    #     pass

    def test_calcAngles(self):
        result = pycgmCalc.calcAngles(self.c3d_data, vsk=self.vsk)
        if type(result) == tuple:
            joint_angle_vals, joint_center_locs = result
        else:
            joint_angle_vals = result

        # Joint angles example
        doc_expected_result =   np.array([[ -0.45646046,  -5.76277607,   4.80620732],
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
                                          [ 37.89341078,   0         ,   0         ],
                                          [ 36.32373414,   0         ,   0         ],
                                          [  9.85050913,  15.46090162, 126.03268251],
                                          [  6.67871519, 17.79096147 , 123.74619493]])

        log_output = False
        if log_output:
            file_path = "pyCGM_Single\\tests\\test_calcAngles_log.txt"
            with open(file_path, "w") as f:
                f.write(str(joint_angle_vals[0].round(rounding_precision)))

        frame1 = joint_angle_vals[0]
        np.testing.assert_almost_equal(frame1.round(rounding_precision)[0], doc_expected_result, rounding_precision)

    def test_Calc(self):
        start, end = 0, 3
        angles, joints = pycgmCalc.Calc(start, end, self.c3d_data, self.vsk)

        # Angle examples
        expected_angles = [-0.45646046, -0.45789927, -0.45608902]
        result_angles = [np.around(angles[x][0], rounding_precision) for x in range(0, 3)]
        np.testing.assert_almost_equal(result_angles, expected_angles, rounding_precision)

        # Joint examples
        expected_joints = [np.array([246.152565  , 353.26243591, 1031.71362305]),
                           np.array([246.16200256, 353.27105713, 1031.71856689])]
        result_joints = [np.around(joints[x]['Pelvis'], rounding_precision) for x in range(0, 2)]
        np.testing.assert_almost_equal(result_joints, expected_joints, rounding_precision)

    def test_calcFrames(self):
        angles, joints = pycgmCalc.calcFrames(self.c3d_data, self.vsk)

        # First example
        expected_angle = -0.45646046  # TODO find a more appropriate name for this and second example
        result_angle = np.around(angles[0][0], rounding_precision)
        np.testing.assert_almost_equal(result_angle, expected_angle, rounding_precision)

        # Second example
        expected_joint_pelvis = np.array([246.152565, 353.26243591, 1031.71362305])
        result_joint_pelvis = np.around(joints[0]['Pelvis'], rounding_precision)
        np.testing.assert_almost_equal(result_joint_pelvis, expected_joint_pelvis, rounding_precision)


if __name__ == "__main__":
    TestPycgmCalc().test_calcAngles()