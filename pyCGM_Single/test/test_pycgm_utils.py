import unittest
import pyCGM_Single.pyCGM as pyCGM
import numpy as np
from parameterized import parameterized

rounding_precision = 8

class TestUpperBodyAxis(unittest.TestCase):

    @parameterized.expand([[{'RSHO': [428.88476562, 270.552948, 1500.73010254],
                             'LSHO': [68.24668121, 269.01049805, 1510.1072998]},
                            [[[256.23991128535846, 365.30496976939753, 1459.662169500559],
                              [257.1435863244796, 364.21960599061947, 1459.588978712983],
                              [256.0843053658035, 364.32180498523223, 1458.6575930699294]],
                             [256.149810236564, 364.3090603933987, 1459.6553639290375]],
                            [[255.92550222678443, 364.3226950497605, 1460.6297868417887],
                             [256.42380097331767, 364.27770361353487, 1460.6165849382387]]],
                           ])
    def testFindWandMarker(self, frame, thorax, expected):
        result = pyCGM.findwandmarker(frame, thorax)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @parameterized.expand([[[0.13232936, 0.98562946, -0.10499292],
                            [-0.99119134, 0.13101088, -0.01938735],
                            [-0.005353527183234709, 0.10663358915485248, 0.994283972218527]],
                           [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],
                           [[1, 1, 1],
                            [1, 1, 1],
                            [0, 0, 0]],
                           [[0, 0, -2],
                            [0, 4, 0],
                            [8, 0, 0]],
                           [[0, 0, 4],
                            [-0.5, 0, 0],
                            [0, -2, 0]],
                           [[-1.5, 0, 0],
                            [0, 4, 0],
                            [0, 0, -6]],
                           [[1, 0, 1],
                            [0, 1, 0],
                            [-1, 0, 1]],
                           [[1, 2, 3],
                            [3, 2, 1],
                            [-4, 8, -4]],
                           [[-2, 3, 1],
                            [4, -1, 5],
                            [16, 14, -10]]])
    def testCross(self, a, b, expected):
        result = pyCGM.cross(a, b)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @parameterized.expand([[[-9944.089508486479, -20189.20612828088, 150.42955108569652],
                                22505.812344655435],
                           [[0, 0, 0],
                            0],
                           [[2, 0, 0],
                            2],
                           [[0, 3, 4],
                            5],
                           [[0, -1, 0],
                            1],
                           [[1, -1, np.sqrt(2)],
                            2],
                           [[-5, 0, -12],
                            13]])
    def testNorm2d(self, v, expected):
        result = pyCGM.norm2d(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @parameterized.expand([[[-212.5847168, 28.09841919, -4.15808105],
                            np.array(214.47394390603984)],
                           [[0, 0, 0],
                            np.array(0)],
                           [[2, 0, 0],
                            np.array(2)],
                           [[0, 3, 4],
                            np.array(5)],
                           [[0, -3, -4],
                            np.array(5)],
                           [[-1, 0, 0],
                            np.array(1)],
                           [[1, -1, np.sqrt(2)],
                            np.array(2)],
                           [[-5, 0, -12],
                            np.array(13)]])
    def testNorm3d(self, v, expected):
        result = pyCGM.norm3d(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    '''
    normDiv needs fixed, issue #31 created
    @parameterized.expand([[[1.44928201, 1.94301493, 2.49204956],
                            [0.11991375545853512, 0.16076527243190078, 0.20619245907039865]],
                           [[0, 0, 0],
                            [np.nan, np.nan, np.nan]],
                           [[1, 0, 0],
                            [1, 0, 0]],
                           [[0, 2, 0],
                            [0, 0.5, 0]],
                           [[0, 0, -4],
                            [0, 0, -0.25]],
                           [[2, 2, 0],
                            [0.25, 0.25, 0]],
                           [[2, 1, 2],
                            [0.66666666, 0.33333333, 0.66666666]],
                           ])
    def testNormDiv(self, v, expected):
        result = pyCGM.normDiv(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)
    '''

    @parameterized.expand([[[[1, 0, 0], [0, 1.0, -0.0], [0, 0.0, 1.0]],
                            [[1.0, 0, 0.0], [0, 1, 0], [-0.0, 0, 1.0]],
                            [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
                           [[[1]], [[1]], [[1]]],
                           [[[2], [1]], [[1, 2]], [[2, 4], [1, 2]]],
                           [[[1, 2, 0], [0, 1, 2]], [[2, 1], [1, 4]], [[4, 9], [1, 4]]],
                           [[[11,12,13],[14,15,16]],
                            [[1, 2], [3, 4], [5, 6]],
                            [[112, 148], [139, 184]]],
                           [[[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]], [[58, 64], [139, 154]]]
                           ])
    def testMatrixmult(self, A, B, expected):
        result = pyCGM.matrixmult(A, B)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @parameterized.expand([[0.0, 0.0, 180, [[-1.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0],
                                            [0.0, 0.0, 1.0]]],
                           [0, 0, 0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                           [90, 0, 0, [[1, 0, 0], [0, 0, -1], [0, 1, 0]]],
                           [0, 135, 0, [[-0.70710678, 0, 0.70710678], [0, 1, 0], [-0.70710678, 0, -0.70710678]]],
                           [0, 0, -60, [[0.5, 0.8660254, 0], [-0.8660254, 0.5, 0], [0, 0, 1]]],
                           [90, 0, 90, [[0, -1, 0], [0, 0, -1], [1, 0, 0]]],
                           [0, 150, -30, [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]],
                           [90, 180, -90, [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]],
                           ])
    def testRotmat(self, x, y, z, expected):
        result = pyCGM.rotmat(x, y, z)
        np.testing.assert_almost_equal(result, expected, rounding_precision)