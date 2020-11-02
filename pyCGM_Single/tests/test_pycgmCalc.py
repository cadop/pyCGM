import sys
import unittest
import pyCGM_Single.pycgmCalc as pycgmCalc
import pyCGM_Single.pycgmIO as pycgmIO
import pyCGM_Single.pycgmStatic as pycgmStatic
import pyCGM_Single.pyCGM_Helpers as pyCGM_Helpers
import numpy as np

rounding_precision = 8
log_outputs = False
num_frames_to_log = 5


class TestPycgmCalc(unittest.TestCase):

    file_names = pyCGM_Helpers.getfilenames(x=2)
    c3d_file, vsk_file = file_names[1:3]
    c3d_data = pycgmIO.loadC3D(c3d_file)[0]
    vsk_data = pycgmIO.loadVSK(vsk_file, False)
    vsk = pycgmStatic.getStatic(c3d_data, vsk_data, flat_foot=False)

    angle_first_truncated_frames =    np.array([[-0.45646046,   -5.76277607,    4.80620732,    3.03831198,
                                                 -7.02211832,  -17.40722232,   -3.00351427,   -4.5429072 ,
                                                 -1.7375533 ,    3.74245063,    1.83607381,  -21.13452485,
                                                 -0.37416136,   -0.26949328,  -23.95627279,    3.73277638,
                                                 -8.72559625,   29.38805467,    4.1585744 ,    1.11320439,
                                                 -4.87112244,  -83.62619447,   -5.22096754,   75.92556425,
                                                 87.39254842,  169.33246201,  -64.83149434,    0.59355258,
                                                 -0.16900286,  -88.78337074,   -3.25087941,    4.24956567,
                                                267.59917792,    4.17050116,   -3.7462158 ,   -3.90225314,
                                                 -9.86327857,    4.10321636,   -2.80874737,   11.79826334],
                                                [-0.45789927,   -5.75510865,    4.80248808,    3.02959298,
                                                 -7.01771319,  -17.40062371,   -3.00790103,   -4.54741702,
                                                 -1.7593788 ,    3.74217923,    1.83141621,  -21.13601231,
                                                 -0.37011802,   -0.26310096,  -23.95115687,    3.73051125,
                                                 -8.7234315 ,   29.38219349,    4.15715309,    1.10940092,
                                                 -4.85273336,  -83.6085162 ,   -5.24024702,   75.92783589,
                                                 87.3853236 ,  169.33009136,  -64.82876599,    0.58946631,
                                                 -0.16436273,  -88.78116681,   -3.25044085,    4.25004022,
                                                267.60769109,    4.17564044,   -3.74143379,   -3.89562355,
                                                 -9.85663741,    4.10250599,   -2.81298875,   11.79771854],
                                                [-0.45608902,   -5.75700103,    4.78977193,    3.03176127,
                                                 -7.02096677,  -17.42333757,   -3.01080904,   -4.54412767,
                                                 -1.72618007,    3.76002588,    1.83372416,  -21.06900299,
                                                 -0.37404409,   -0.2695666 ,  -23.94131247,    3.75661308,
                                                 -8.69521377,   29.30572748,    4.16097315,    1.11289797,
                                                 -4.86964096,  -83.66601966,   -5.21747026,   75.91079546,
                                                 87.38890645,  169.32774842,  -64.84307092,    0.59093957,
                                                 -0.15963595,  -88.76912434,   -3.25222856,    4.25258176,
                                                267.62415115,    4.18378324,   -3.74363921,   -3.89160576,
                                                 -9.86223772,    4.09983468,   -2.81668274,   11.77871629],
                                                [-0.45851004,   -5.75217475,    4.78699316,    3.02112921,
                                                 -7.01761788,  -17.38526492,   -3.0127356 ,   -4.54803339,
                                                 -1.76091744,    3.73957047,    1.82914314,  -21.14547928,
                                                 -0.37367971,   -0.26559781,  -23.95681519,    3.72862659,
                                                 -8.71443697,   29.35783269,    4.15269901,    1.10437245,
                                                 -4.82842005,  -83.5965817 ,   -5.26317964,   75.92470593,
                                                 87.37772568,  169.32456408,  -64.83035467,    0.58605962,
                                                 -0.13992447,  -88.75816241,   -3.25293414,    4.25709514,
                                                267.64121165,    4.20861357,   -3.73813173,   -3.88544801,
                                                 -9.8630116 ,    4.10003968,   -2.83074759,   11.7617372],
                                                [-0.45979952,   -5.75626564,    4.78014806,    3.02853276,
                                                 -7.02092575,  -17.39239597,   -3.0037155 ,   -4.54858795,
                                                 -1.76358003,    3.75179103,    1.83587525,  -21.14139803,
                                                 -0.36570291,   -0.26727198,  -23.94861123,    3.7383873 ,
                                                 -8.71386289,   29.35627745,    4.16058229,    1.10477669,
                                                 -4.83037467,  -83.60502241,   -5.25556537,   75.92648129,
                                                 87.39021682,  169.33415412,  -64.82791648,    0.57469817,
                                                 -0.1344207 ,  -88.7578598 ,   -3.24539137,    4.26333532,
                                                267.65859551,    4.21988315,   -3.71897901,   -3.86732727,
                                                 -9.87478703,    4.09123313,   -2.84079787,   11.74402988]
                                                ])

    # Function is equivalent to pycgmKinetics.getKinetics() - would be duplicate test
    # def test_calcKinetics(self):
    #     pass

    def test_calcAngles(self):
        result = pycgmCalc.calcAngles(self.c3d_data, vsk=self.vsk)
        if type(result) == tuple:
            joint_angle_vals, joint_center_locs = result
        else:
            joint_angle_vals = result

        if log_outputs:
            with np.printoptions(precision=rounding_precision, threshold=sys.maxsize, suppress=True):
                file_path = "pyCGM_Single\\tests\\test_calcAngles_log"
                with open(file_path + ".txt", "w") as f:
                    # Generally the result of these functions is a list of each frame, so [0]
                    # is used to access only the first frame (note last few lines of pycgmCalc.calcFrames())
                    f.write(str(joint_angle_vals[0:num_frames_to_log]))

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
        start, end = 0, 5
        angles, joints = pycgmCalc.Calc(start, end, self.c3d_data, self.vsk)

        if log_outputs:
            with np.printoptions(precision=rounding_precision, threshold=sys.maxsize, suppress=True):
                file_path = "pyCGM_Single\\tests\\test_Calc_log"
                with open(file_path + "1.txt", "w") as f:
                    f.write(str(angles[0:num_frames_to_log]))
                with open(file_path + "2.txt", "w") as f:
                    f.write(str(joints[0:num_frames_to_log]))

        # Angle test
        # Same comment from test_calcFrames applies
        expected_angles = self.angle_first_truncated_frames

        result_angles = np.array([np.around(frame[:40], rounding_precision) for frame in angles[0:5]])
        np.testing.assert_almost_equal(result_angles, expected_angles, rounding_precision)

        # Joint test
        expected_joints = [np.array([246.152565  , 353.26243591, 1031.71362305]),
                           np.array([246.16200256, 353.27105713, 1031.71856689])]
        result_joints = [np.around(joints[x]['Pelvis'], rounding_precision) for x in range(0, 2)]
        np.testing.assert_almost_equal(result_joints, expected_joints, rounding_precision)

    def test_calcFrames(self):
        angles, joints = pycgmCalc.calcFrames(self.c3d_data, self.vsk)

        if log_outputs:
            with np.printoptions(precision=rounding_precision, threshold=sys.maxsize, suppress=True):
                file_path = "pyCGM_Single\\tests\\test_calcFrames_log"
                with open(file_path + "1.txt", "w") as f:
                    f.write(str(angles[0:num_frames_to_log]))
                with open(file_path + "2.txt", "w") as f:
                    f.write(str(joints[0:num_frames_to_log]))

        # Angle test
        # The raw data from a single frame is excessively large for the purpose of testing more than one frame
        # Thus, only using the first 40 elements per frame on first 5 frames
        expected_angles = self.angle_first_truncated_frames

        result_angles = np.array([np.around(frame[:40], rounding_precision) for frame in angles[0:5]])
        np.testing.assert_almost_equal(result_angles, expected_angles, rounding_precision)

        # Second example
        expected_joint_pelvis = np.array([246.152565, 353.26243591, 1031.71362305])
        result_joint_pelvis = np.around(joints[0]['Pelvis'], rounding_precision)
        np.testing.assert_almost_equal(result_joint_pelvis, expected_joint_pelvis, rounding_precision)

        print(np.around(angles[0], rounding_precision))


if __name__ == "__main__":
    TestPycgmCalc().test_calcAngles()