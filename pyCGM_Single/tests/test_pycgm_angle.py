import unittest
import pyCGM_Single.pyCGM as pyCGM
import numpy as np
from parameterized import parameterized

rounding_precision = 8

class TestPycgmAngle(unittest.TestCase):

    sqrt3div2 = 0.86602540378
    sqrt2div2 = 0.70710678118
    yCoor30Degree = 0.577350269
    yCoor60Degree = 1.732050808
    coor_000 = [0, 0, 0]
    coor_100 = [1, 0, 0]
    coor_010 = [0, 1, 0]
    coor_001 = [0, 0, 1]
    coor_neg100 = [-1, 0, 0]
    coor_neg010 = [0, -1, 0]
    coor_neg001 = [0, 0, -1]
    testcase_a = [[coor_100, coor_000, coor_000],
                  [coor_000, coor_000, coor_100],
                  [90.0, 0.0, 0.0]]
    testcase_b = [[coor_100, coor_000, coor_100],
                  [coor_000, coor_000, [sqrt2div2, 0, 0]],
                  [45.0, 0.0, 0]]
    testcase_c = [[coor_neg100, coor_000, coor_100],
                  [coor_000, coor_000, [sqrt2div2, 0, 0]],
                  [-45.0, 0.0, 0]]
    testcase_d = [[coor_neg100, coor_000, coor_000],
                  [coor_000, coor_000, coor_100],
                  [-90.0, 0.0, 0.0]]
    testcase_e = [[coor_000, coor_neg100, coor_000],
                  [coor_000, coor_000, coor_100],
                  [0.0, 90.0, 0.0]]

    @parameterized.expand([[90, 0, 0, [0, 90, 0]],
                           [30, 0, 0, [0, 30, 0]],
                           [-30, 0, 0, [0, -30, 0]],
                           [120, 0, 0, [0, 120, 0]],
                           [-120, 0, 0, [0, -120, 0]],
                           [180, 0, 0, [0, 180, 0]],
                           [0, 90, 0, [90, 0, 0]],
                           [0, 30, 0, [30, 0, 0]],
                           [0, -30, 0, [-30, 0, 0]],
                           [0, 120, 0, [60, -180, -180]],
                           [0, -120, 0, [-60, -180, -180]],
                           [0, 180, 0, [0, -180, -180]],
                           [0, 0, 90, [0, 0, 90]],
                           [0, 0, 30, [0, 0, 30]],
                           [0, 0, -30, [0, 0, -30]],
                           [0, 0, 120, [0, 0, 120]],
                           [0, 0, -120, [0, 0, -120]],
                           [0, 0, 180, [0, 0, 180]],
                           [150, 30, 0, [30, 150, 0]],
                           [45, 0, 60, [0, 45, 60]],
                           [0, 90, 120, [90, 0, 120]],
                           [135, 45, 90, [45, 135, 90]]])
    def test_getangle_sho2(self, xRot, yRot, zRot, expected):
        axisP = pyCGM.rotmat(xRot, yRot, zRot)
        axisD = pyCGM.rotmat(0, 0, 0)
        result = pyCGM.getangle_sho(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

        result_nparray = pyCGM.getangle_sho(np.array(axisP), np.array(axisD))
        np.testing.assert_almost_equal(result_nparray, expected, rounding_precision)

    @parameterized.expand([
        [[coor_000, coor_000, coor_000],
         [coor_000, coor_000, coor_000],
         [0.0, 0.0, 0.0]],
        #Test from running sample data
        [[[0.13232935635315357104, 0.98562945782504129966, -0.10499292030749529658],
          [-0.99119134439010281312, 0.13101087562301927392, -0.01938734831355759525],
          [-0.00535352718324588750, 0.10663358915485332545, 0.99428397221845443710]],
         [[0.09010104879445179904, 0.99590937599884910014, 0.00680557152145411237],
          [0.99377608791559168822, -0.08945440277921079542, -0.06638521605464120512],
          [-0.06550487076049194002, 0.01274459183355247660, -0.99777085910818641423]],
         [-6.4797057790916615, -2.893068979100172, -4.638276625836626]],
        # Tests for the x angle, 90 degrees to -90 degrees
        [[coor_000, coor_000, coor_100],
         [coor_neg100, coor_000, coor_000],
         [90.0, 0.0, 0.0]],
         [[coor_000, coor_000, coor_001],
          [[0, 0, -sqrt3div2], coor_000, coor_000],
          [60.0, 0.0, 0.0]],
        [[coor_000, coor_000, coor_100],
         [[-sqrt2div2, 0, 0], coor_000, coor_000],
         [45.0, 0.0, 0.0]],
        [[coor_000, coor_000, coor_010],
         [[0, -0.5, 0], coor_000, coor_000],
         [30.0, 0.0, 0.0]],
        [[coor_000, coor_000, coor_001],
         [[0, 0, 0.5], coor_000, coor_000],
         [-30.0, 0.0, 0.0]],
        [[coor_000, coor_000, coor_010],
         [[0, sqrt2div2, 0], coor_000, coor_000],
         [-45.0, 0.0, 0.0]],
        [[coor_000, coor_000, coor_100],
         [[sqrt3div2, 0, 0], coor_000, coor_000],
         [-60.0, 0.0, 0.0]],
        [[coor_000, coor_000, coor_010],
         [coor_010, coor_000, coor_000],
         [-90.0, 0.0, 0.0]],
        # Tests for the y angle, 90 degrees to -90 degrees
        [[coor_neg100, coor_000, coor_000],
         [coor_000, coor_100, coor_000],
        [0.0, 90.0, 0.0]],
        [[coor_010, coor_000, coor_000],
         [coor_000, [0, -sqrt3div2, 0], coor_000],
         [0.0, 60.0, 0.0]],
        [[coor_001, coor_000, coor_000],
         [coor_000, [0, 0, -sqrt2div2], coor_000],
         [0.0, 45.0, 0.0]],
        [[coor_010, coor_000, coor_000],
         [coor_000, [0, -0.5, 0], coor_000],
         [0.0, 30.0, 0.0]],
        [[coor_001, coor_000, coor_000],
         [coor_000, [0, 0, 0.5], coor_000],
         [0.0, -30.0, 0.0]],
        [[[sqrt2div2, 0, 0], coor_000, coor_000],
         [coor_000, coor_100, coor_000],
         [0.0, -45.0, 0.0]],
        [[coor_001, coor_000, coor_000],
         [coor_000, [0, 0, sqrt3div2], coor_000],
         [0.0, -60.0, 0.0]],
        [[coor_010, coor_000, coor_000],
         [coor_000, coor_010, coor_000],
         [0.0, -90.0, 0.0]],
        # Tests for the z angle, 90 degrees to -90 degrees
        [[coor_000, coor_000, coor_100],
         [coor_000, coor_100, coor_000],
         [0.0, 0.0, 90.0]],
        [[coor_000, coor_000, coor_010],
         [coor_000, [0, sqrt3div2, 0], coor_000],
         [0.0, 0.0, 60.0]],
        [[coor_000, coor_000, coor_001],
         [coor_000, [0, 0, sqrt2div2], coor_000],
         [0.0, 0.0, 45.0]],
        [[coor_000, coor_000, coor_100],
         [coor_000, [0.5, 0, 0], coor_000],
         [0.0, 0.0, 30.0]],
        [[coor_000, coor_000, coor_010],
         [coor_000, [0, -0.5, 0], coor_000],
         [0.0, 0.0, -30.0]],
        [[coor_000, coor_000, coor_010],
         [coor_000, [0, -sqrt2div2, 0], coor_000],
         [0.0, 0.0, -45.0]],
        [[coor_000, coor_000, coor_010],
         [coor_000, [0, -sqrt3div2, 0], coor_000],
         [0.0, 0.0, -60.0]],
        [[coor_000, coor_000, coor_100],
         [coor_000, coor_neg100, coor_000],
         [0.0, 0.0, -90.0]]])
    def test_getangle_spi(self, axisP, axisD, expected):
        result = pyCGM.getangle_spi(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

        result_nparray = pyCGM.getangle_spi(np.array(axisP), np.array(axisD))
        np.testing.assert_almost_equal(result_nparray, expected, rounding_precision)

    @parameterized.expand([
        [[coor_000, coor_000, coor_000],
         [coor_000, coor_000, coor_000],
         [0.0, 0.0, 0.0]],
        # Test from running sample data
        [[coor_100, coor_010, coor_001],
         [[0.13232935635315357103, 0.98562945782504129965, -0.10499292030749529658],
          [-0.99119134439010281312, 0.13101087562301927391, -0.01938734831355759524],
          [-0.00535352718324588749, 0.10663358915485332545, 0.99428397221845443709]],
         [-0.30849491450945404, -6.121292793370006, 7.5714311021517124]],
        # Tests for the x angle, 180 degrees to -150 degrees
        [[coor_000, coor_000, coor_neg100],
         [coor_000, coor_000, coor_100],
         [180.0, 0.0, 0.0]],
        [[[0, 0, yCoor30Degree], coor_000, coor_neg001],
         [coor_000, coor_000, coor_001],
         [150.0, 0.0, 0.0]],
        [[coor_010, coor_000, coor_neg010],
         [coor_000, coor_000, coor_010],
         [135.0, 0.0, 0.0]],
        [[[0, yCoor60Degree, 0], coor_000, coor_neg010],
         [coor_000, coor_000, coor_010],
         [120.0, 0.0, 0.0]],
        testcase_a,
        [[[0, yCoor60Degree, 0], coor_000, coor_010],
         [coor_000, coor_000, coor_010],
         [60.0, 0.0, 0.0]],
        testcase_b,
        [[[0, 0, yCoor30Degree], coor_000, coor_001],
         [coor_000, coor_000, coor_001],
         [30.0, 0.0, 0.0]],
        [[[0, -yCoor30Degree, 0], coor_000, coor_010],
         [coor_000, coor_000, coor_010],
         [-30.0, 0.0, 0.0]],
        testcase_c,
        [[[0, 0, -yCoor60Degree], coor_000, coor_001],
         [coor_000, coor_000, coor_001],
         [-60.0, 0.0, 0.0]],
        testcase_d,
        [[[0, 0, -yCoor60Degree], coor_000, coor_neg001],
         [coor_000, coor_000, coor_001],
         [-120.0, 0.0, 0.0]],
        [[coor_neg001, coor_000, coor_neg001],
         [coor_000, coor_000, coor_001],
         [-135.0, 0.0, 0.0]],
        [[[0, -yCoor30Degree, 0], coor_000, coor_neg010],
         [coor_000, coor_000, coor_010],
         [-150.0, 0.0, 0.0]],
        # Tests for the y angle, 90 degrees to -90 degrees
        testcase_e,
        [[coor_000, coor_010, coor_000],
         [coor_000, coor_000, [0, -sqrt3div2, 0]],
         [0.0, 60.0, 0.0]],
        [[coor_000, coor_001, coor_000],
         [coor_000, coor_000, [0, 0, -sqrt2div2]],
         [0.0, 45.0, 0.0]],
        [[coor_000, coor_001, coor_000],
         [coor_000, coor_000, [0, 0, -0.5]],
         [0.0, 30.0, 0.0]],
        [[coor_000, coor_010, coor_000],
         [coor_000, coor_000, [0, 0.5, 0]],
         [0.0, -30.0, 0.0]],
        [[coor_000, coor_100, coor_100],
         [coor_000, coor_000, [sqrt2div2, 0, 0]],
         [0.0, -45.0, 0]],
        [[coor_000, coor_010, coor_000],
         [coor_000, coor_000, [0, sqrt3div2, 0]],
         [0.0, -60.0, 0.0]],
        [[coor_000, coor_001, coor_000],
         [coor_000, coor_000, coor_001],
         [0.0, -90.0, 0.0]],
        # Tests for the z angle, 180 degrees to -150 degrees
        [[coor_000, coor_100, coor_000],
         [coor_neg100, coor_000, coor_000],
         [0.0, 0.0, 180.0]],
        [[coor_000, coor_001, coor_000],
         [coor_neg001, [0, 0, yCoor30Degree], coor_000],
         [0.0, 0.0, 150.0]],
        [[coor_000, coor_100, coor_000],
         [coor_neg100, coor_100, coor_000],
         [0.0, 0.0, 135.0]],
        [[coor_000, coor_010, coor_000],
         [coor_neg010, [0, yCoor60Degree, 0], coor_000],
         [0.0, 0.0, 120.0]],
        [[coor_000, coor_100, coor_000],
         [coor_000, coor_100, coor_000],
         [0.0, 0.0, 90.0]],
        [[coor_000, coor_010, coor_000],
         [coor_010, [0, yCoor60Degree, 0], coor_000],
         [0.0, 0.0, 60.0]],
        [[coor_000, coor_100, coor_000],
         [coor_100, coor_100, coor_000],
         [0.0, 0.0, 45.0]],
        [[coor_000, coor_001, coor_000],
         [coor_001, [0, 0, yCoor30Degree], coor_000],
         [0.0, 0.0, 30.0]],
        [[coor_000, coor_010, coor_000],
         [coor_010, [0, -yCoor30Degree, 0], coor_000],
         [0.0, 0.0, -30.0]],
        [[coor_000, coor_001, coor_000],
         [coor_001, coor_neg001, coor_000],
         [0.0, 0.0, -45.0]],
        [[coor_000, coor_001, coor_000],
         [coor_001, [0, 0, -yCoor60Degree], coor_000],
         [0.0, 0.0, -60.0]],
        [[coor_000, coor_neg100, coor_000],
         [coor_000, coor_100, coor_000],
         [0.0, 0.0, -90.0]],
        [[coor_000, coor_001, coor_000],
         [coor_neg001, [0, 0, -yCoor60Degree], coor_000],
         [0.0, 0.0, -120.0]],
        [[coor_000, coor_001, coor_000],
         [coor_neg001, coor_neg001, coor_000],
         [0.0, 0.0, -135.0]],
        [[coor_000, coor_010, coor_000],
         [coor_neg010, [0, -yCoor30Degree, 0], coor_000],
         [0.0, 0.0, -150.0]]])
    def test_getangle(self, axisP, axisD, expected):
        result = pyCGM.getangle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

        result_nparray = pyCGM.getangle(np.array(axisP), np.array(axisD))
        np.testing.assert_almost_equal(result_nparray, expected, rounding_precision)

    def test_getHeadangle(self):
        axisP = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        axisD = [[2.61438500e-02, 9.95119786e-01, -9.51478379e-02],
                 [-9.99658123e-01, 2.59900920e-02, -2.85510739e-03],
                 [-3.68272800e-04, 9.51899525e-02, 9.95459059e-01]]
        expected = [359.97880327072426, -5.462252836649474, -91.49608534396434]

        result = pyCGM.getHeadangle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

        result_nparray = pyCGM.getHeadangle(np.array(axisP), np.array(axisD))
        np.testing.assert_almost_equal(result_nparray, expected, rounding_precision)

    def test_getPelAngle(self):
        axisP = [[0.0464229, 0.99648672, 0.06970743],
                 [0.99734011, -0.04231089, -0.05935067],
                 [-0.05619277, 0.07227725, -0.99580037]]
        axisD = [[-0.18067218, -0.98329158, -0.02225371],
                 [0.71383942, -0.1155303, -0.69071415],
                 [0.67660243, -0.1406784, 0.7227854]]
        expected = np.array([-175.65183483, 39.63221918, -10.2668477])

        result = pyCGM.getPelangle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

        result_nparray = pyCGM.getPelangle(np.array(axisP), np.array(axisD))
        np.testing.assert_almost_equal(result_nparray, expected, rounding_precision)