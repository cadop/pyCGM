import unittest
import pyCGM_Single.pyCGM as pyCGM
import numpy as np

rounding_precision = 8

class TestUpperBodyAxis(unittest.TestCase):

    def testFindWandMarker(self):
        frame = {'RSHO': [428.88476562, 270.552948, 1500.73010254],
                 'LSHO': [68.24668121, 269.01049805, 1510.1072998]}
        thorax = [[[256.23991128535846, 365.30496976939753, 1459.662169500559],
                   [257.1435863244796, 364.21960599061947, 1459.588978712983],
                   [256.0843053658035, 364.32180498523223, 1458.6575930699294]],
                  [256.149810236564, 364.3090603933987, 1459.6553639290375]]
        expected = [[255.92550222678443, 364.3226950497605, 1460.6297868417887],
                    [256.42380097331767, 364.27770361353487, 1460.6165849382387]]

        result = pyCGM.findwandmarker(frame, thorax)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def testCross(self):
        a = [0.13232936, 0.98562946, -0.10499292]
        b = [-0.99119134, 0.13101088, -0.01938735]
        expected = [-0.005353527183234709, 0.10663358915485248, 0.994283972218527]

        result = pyCGM.cross(a, b)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def testNorm2d(self):
        v = [-9944.089508486479, -20189.20612828088, 150.42955108569652]
        expected = 22505.812344655435

        result = pyCGM.norm2d(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def testNorm3d(self):
        v = [-212.5847168, 28.09841919, -4.15808105]
        expected = np.array(214.47394390603984)

        result = pyCGM.norm3d(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def testNormDiv(self):
        v = [1.44928201, 1.94301493, 2.49204956]
        expected = [0.11991375545853512, 0.16076527243190078, 0.20619245907039865]

        result = pyCGM.normDiv(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def testMatrixmult(self):
        A = [[1, 0, 0], [0, 1.0, -0.0], [0, 0.0, 1.0]]
        B = [[1.0, 0, 0.0], [0, 1, 0], [-0.0, 0, 1.0]]
        expected = [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        result = pyCGM.matrixmult(A, B)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def testRotmat(self):
        x = 0.0
        y = 0.0
        z = 180
        expected = [[-1.0, -1.2246467991473532e-16, 0.0],
                    [1.2246467991473532e-16, -1.0, 0.0],
                    [0.0, 0.0, 1.0]]

        result = pyCGM.rotmat(x, y, z)
        np.testing.assert_almost_equal(result, expected, rounding_precision)