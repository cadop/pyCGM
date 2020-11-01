from unittest import TestCase
import pyCGM_Single.pyCGM as pyCGM
import numpy as np

rounding_precision = 8

class TestUtils(TestCase):

    def test_findwandmarker(self):
        rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]
        testcases = [
            [{'RSHO': [428.88476562, 270.552948, 1500.73010254], 'LSHO': [68.24668121, 269.01049805, 1510.1072998]}, [[[256.23991128535846, 365.30496976939753, 1459.662169500559], rand_coor, rand_coor], [256.149810236564, 364.3090603933987, 1459.6553639290375]], [[255.92550222678443, 364.3226950497605, 1460.6297868417887], [256.42380097331767, 364.27770361353487, 1460.6165849382387]]],
            [{'RSHO': [0, 0, 1], 'LSHO': [0, 1, 0]}, [[[1, 0, 0], rand_coor, rand_coor], [0, 0, 0]], [[0, 1, 0], [0, 0, 1]]],
            [{'RSHO': [0, 1, 1], 'LSHO': [1, 1, 1]}, [[[1, 0, 0], rand_coor, rand_coor], [0, 0, 0]], [[0, 0.70710678, -0.70710678], [0, -0.70710678, 0.70710678]]],
            [{'RSHO': [0, 1, 1], 'LSHO': [1, 1, 1]}, [[[1, 0, 0], rand_coor, rand_coor], [-1, 0, 0]], [[-1, 0.70710678, -0.70710678], [-1, -0.70710678, 0.70710678]]],
            [{'RSHO': [1, 2, 1], 'LSHO': [2, 1, 2]}, [[[1, 0, 0], rand_coor, rand_coor], [0, 0, 0]], [[0, 0.4472136, -0.89442719], [0, -0.89442719, 0.4472136]]],
            [{'RSHO': [1, 1, 1], 'LSHO': [1, 1, 1]}, [[[1, 0, 1], rand_coor, rand_coor], [0, 0, 0]], [[0.70710678, 0, -0.70710678], [-0.70710678, 0, 0.70710678]]],
            [{'RSHO': [1, 1, 1], 'LSHO': [1, 1, 1]}, [[[1, 0, 1], rand_coor, rand_coor], [0, 0, 1]], [[0, 0, 0], [0, 0, 2]]],
            [{'RSHO': [0, 1, 0], 'LSHO': [0, 0, -1]}, [[[0, 3, 4], rand_coor, rand_coor], [0, 0, 0]], [[1, 0, 0], [-1, 0, 0]]],
            [{'RSHO': [1, 0, 0], 'LSHO': [0, 1, 0]}, [[[7, 0, 24], rand_coor, rand_coor], [0, 0, 0]], [[0, -1, 0], [-0.96, 0, 0.28]]],
            [{'RSHO': [1, 0, 0], 'LSHO': [0, 0, 1]}, [[[8, 0, 6], rand_coor, rand_coor], [8, 0, 0]], [[8, 1, 0], [8, -1, 0]]]]
        for testcase in testcases:
            result = pyCGM.findwandmarker(testcase[0], testcase[1])
            np.testing.assert_almost_equal(result, testcase[2], rounding_precision)

    def test_findwandmarker_datatypes(self):
        frame_int = {'RSHO': [1, 0, 0], 'LSHO': [0, 0, 1]}
        frame_float = {'RSHO': [1.0, 0.0, 0.0], 'LSHO': [0.0, 0.0, 1.0]}
        thorax_int = [[[8, 0, 6], [0, 0, 0], [0, 0, 0]], [8, 0, 0]]
        thorax_float = [[[8.0, 0.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [8.0, 0.0, 0.0]]
        expected = [[8, 1, 0], [8, -1, 0]]

        result_int_list = pyCGM.findwandmarker(frame_int, thorax_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        result_float_list = pyCGM.findwandmarker(frame_float, thorax_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

    def test_cross(self):
        testcases = [[[0.13232936, 0.98562946, -0.10499292], [-0.99119134, 0.13101088, -0.01938735], [-0.005353527183234709, 0.10663358915485248, 0.994283972218527]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                     [[0, 0, -2], [0, 4, 0], [8, 0, 0]],
                     [[0, 0, 4], [-0.5, 0, 0], [0, -2, 0]],
                     [[-1.5, 0, 0], [0, 4, 0], [0, 0, -6]],
                     [[1, 0, 1], [0, 1, 0], [-1, 0, 1]],
                     [[1, 2, 3], [3, 2, 1], [-4, 8, -4]],
                     [[-2, 3, 1], [4, -1, 5], [16, 14, -10]]]
        for testcase in testcases:
            result = pyCGM.cross(testcase[0], testcase[1])
            np.testing.assert_almost_equal(result, testcase[2], rounding_precision)

    def test_cross_datatypes(self):
        A_int = [-2, 3, 1]
        A_float = [-2.0, 3.0, 1.0]
        B_int = [4, -1, 5]
        B_float = [4.0, -1.0, 5.0]
        expected = [16, 14, -10]

        result_int_list = pyCGM.cross(A_int, B_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        result_int_nparray = pyCGM.cross(np.array(A_int, dtype='int'), np.array(B_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        result_float_list = pyCGM.cross(A_float, B_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        result_float_nparray = pyCGM.cross(np.array(A_float, dtype='float'), np.array(B_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    def test_norm2d(self):
        testcases = [[[-9944.089508486479, -20189.20612828088, 150.42955108569652], 22505.812344655435],
                     [[0, 0, 0], 0],
                     [[2, 0, 0], 2],
                     [[0, 0, -1], 1],
                     [[0, 3, 4], 5],
                     [[-3, 0, 4], 5],
                     [[6, -8, 0], 10],
                     [[-5, 0, -12], 13],
                     [[1, -1, np.sqrt(2)], 2]]
        for testcase in testcases:
            result = pyCGM.norm2d(testcase[0])
            np.testing.assert_almost_equal(result, testcase[1], rounding_precision)

    def test_norm3d_datatypes(self):
        v_int = [6, 0, -8]
        v_float = [6.0, 0, -8.0]
        expected = 10

        result_int_list = pyCGM.norm3d(v_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        result_int_nparray = pyCGM.norm3d(np.array(v_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        result_float_list = pyCGM.norm3d(v_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        result_float_nparray = pyCGM.norm3d(np.array(v_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    def test_norm3d(self):
        testcases = [[[-212.5847168, 28.09841919, -4.15808105], np.array(214.47394390603984)],
                     [[0, 0, 0], np.array(0)],
                     [[2, 0, 0], np.array(2)],
                     [[0, 0, -1], np.array(1)],
                     [[0, 3, 4], np.array(5)],
                     [[-3, 0, 4], np.array(5)],
                     [[-6, 8, 0], np.array(10)],
                     [[-5, 0, -12], np.array(13)],
                     [[1, -1, np.sqrt(2)], np.array(2)]]
        for testcase in testcases:
            result = pyCGM.norm3d(testcase[0])
            np.testing.assert_almost_equal(result, testcase[1], rounding_precision)

    def test_norm3d_datatypes(self):
        v_int = [-6, 0, 8]
        v_float = [-6.0, 0, 8.0]
        expected = np.array(10)

        result_int_list = pyCGM.norm3d(v_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        result_int_nparray = pyCGM.norm3d(np.array(v_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        result_float_list = pyCGM.norm3d(v_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        result_float_nparray = pyCGM.norm3d(np.array(v_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)


    def test_matrixmult(self):
        testcases = [[[[1, 0, 0], [0, 1.0, -0.0], [0, 0.0, 1.0]], [[1.0, 0, 0.0], [0, 1, 0], [-0.0, 0, 1.0]], [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
                     [[[1]], [[1]], [[1]]],
                     [[[2], [1]], [[1, 2]], [[2, 4], [1, 2]]],
                     [[[1, 2, 0], [0, 1, 2]], [[2, 1], [1, 4]], [[4, 9], [1, 4]]],
                     [[[11,12,13],[14,15,16]], [[1, 2], [3, 4], [5, 6]], [[112, 148], [139, 184]]],
                     [[[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]], [[58, 64], [139, 154]]]]
        for testcase in testcases:
            result = pyCGM.matrixmult(testcase[0], testcase[1])
            np.testing.assert_almost_equal(result, testcase[2], rounding_precision)

    def test_matrixmult_datatypes(self):
        A_int = [[1, 2, 0], [0, 1, 2]]
        B_int = [[2, 1], [1, 4]]
        A_float = [[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]]
        B_float = [[2.0, 1.0], [1.0, 4.0]]
        expected = [[4, 9], [1, 4]]

        result_int_list = pyCGM.matrixmult(A_int, B_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        result_int_nparray = pyCGM.matrixmult(np.array(A_int, dtype='int'), np.array(B_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        result_float_list = pyCGM.matrixmult(A_float, B_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        result_float_nparray = pyCGM.matrixmult(np.array(A_float, dtype='float'), np.array(B_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    def test_rotmat(self):
        testcases = [[0.0, 0.0, 180, [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]],
                     [0, 0, 0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                     [90, 0, 0, [[1, 0, 0], [0, 0, -1], [0, 1, 0]]],
                     [0, 135, 0, [[-0.70710678, 0, 0.70710678], [0, 1, 0], [-0.70710678, 0, -0.70710678]]],
                     [0, 0, -60, [[0.5, 0.8660254, 0], [-0.8660254, 0.5, 0], [0, 0, 1]]],
                     [90, 0, 90, [[0, -1, 0], [0, 0, -1], [1, 0, 0]]],
                     [0, 150, -30, [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]],
                     [90, 180, -90, [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]]]
        for testcase in testcases:
            result = pyCGM.rotmat(testcase[0], testcase[1], testcase[2])
            np.testing.assert_almost_equal(result, testcase[3], rounding_precision)

    def test_rotmat_datatypes(self):
        result_int = pyCGM.rotmat(0, 150, -30)
        result_float = pyCGM.rotmat(0.0, 150.0, -30.0)
        expected = [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]

        np.testing.assert_almost_equal(result_int, expected, rounding_precision)
        np.testing.assert_almost_equal(result_float, expected, rounding_precision)