import unittest
import pyCGM_Single.pyCGM as pyCGM
import numpy as np
from parameterized import parameterized

rounding_precision = 8

class TestPycgmAngle(unittest.TestCase):

    @parameterized.expand(
        [[0, 0, 0, [0, 0, 0]],
         # X rotations
         [90, 0, 0, [0, 90, 0]], [30, 0, 0, [0, 30, 0]], [-30, 0, 0, [0, -30, 0]], [120, 0, 0, [0, 120, 0]], [-120, 0, 0, [0, -120, 0]], [180, 0, 0, [0, 180, 0]],
         # Y rotations
         [0, 90, 0, [90, 0, 0]], [0, 30, 0, [30, 0, 0]], [0, -30, 0, [-30, 0, 0]], [0, 120, 0, [60, -180, -180]], [0, -120, 0, [-60, -180, -180]],[0, 180, 0, [0, -180, -180]],
         #Z rotations
         [0, 0, 90, [0, 0, 90]], [0, 0, 30, [0, 0, 30]], [0, 0, -30, [0, 0, -30]], [0, 0, 120, [0, 0, 120]], [0, 0, -120, [0, 0, -120]], [0, 0, 180, [0, 0, 180]],
         # Multiple Rotations
         [150, 30, 0, [30, 150, 0]], [45, 0, 60, [0, 45, 60]], [0, 90, 120, [90, 0, 120]], [135, 45, 90, [45, 135, 90]]
         ])
    def test_getangle_sho(self, xRot, yRot, zRot, expected):
        axisP = pyCGM.rotmat(xRot, yRot, zRot)
        axisD = pyCGM.rotmat(0, 0, 0)
        result = pyCGM.getangle_sho(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getangle_sho_datatypes(self):
        axisD = pyCGM.rotmat(0, 0, 0)
        axisP_floats = pyCGM.rotmat(90, 0, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [0, 90, 90]

        result_int_list = pyCGM.getangle_sho(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        result_int_nparray = pyCGM.getangle_sho(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        result_float_list = pyCGM.getangle_sho(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        result_float_nparray = pyCGM.getangle_sho(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @parameterized.expand([
        [0, 0, 0, [0, 0, 0]],
        # X rotations
        [90, 0, 0, [0, 0, 90]], [30, 0, 0, [0, 0, 30]], [-30, 0, 0, [0, 0, -30]], [120, 0, 0, [0, 0, 60]], [-120, 0, 0, [0, 0, -60]], [180, 0, 0, [0, 0, 0]],
        # Y rotations
        [0, 90, 0, [90, 0, 0]], [0, 30, 0, [30, 0, 0]], [0, -30, 0, [-30, 0, 0]], [0, 120, 0, [60, 0, 0]], [0, -120, 0, [-60, 0, 0]], [0, 180, 0, [0, 0, 0]],
        # Z rotations
        [0, 0, 90, [0, 90, 0]], [0, 0, 30, [0, 30, 0]], [0, 0, -30, [0, -30, 0]], [0, 0, 120, [0, 60, 0]], [0, 0, -120, [0, -60, 0]], [0, 0, 180, [0, 0, 0]],
        # Multiple Rotations
        [150, 30, 0, [-30, 0, 30]], [45, 0, 60, [-40.89339465, 67.7923457, 20.70481105]], [0, 90, 120, [-90, 0, 60]], [135, 45, 90, [-54.73561032, 54.73561032, -30]]
    ])
    def test_getangle_spi(self, xRot, yRot, zRot, expected):
        axisP = pyCGM.rotmat(xRot, yRot, zRot)
        axisD = pyCGM.rotmat(0, 0, 0)
        result = pyCGM.getangle_spi(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getangle_spi_datatypes(self):
        axisD = pyCGM.rotmat(0, 0, 0)
        axisP_floats = pyCGM.rotmat(90, 0, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [-90, 90, 0]

        result_int_list = pyCGM.getangle_spi(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        result_int_nparray = pyCGM.getangle_spi(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        result_float_list = pyCGM.getangle_spi(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        result_float_nparray = pyCGM.getangle_spi(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @parameterized.expand([
        [0, 0, 0, [0, 0, 90]],
        # X rotations
        [90, 0, 0, [0, 90, 90]], [30, 0, 0, [0, 30, 90]], [-30, 0, 0, [0, -30, 90]], [120, 0, 0, [180, 60, -90]], [-120, 0, 0, [180, -60, -90]], [180, 0, 0, [180, 0, -90]],
        # Y rotations
        [0, 90, 0, [90, 0, 90]], [0, 30, 0, [30, 0, 90]], [0, -30, 0, [-30, 0, 90]], [0, 120, 0, [120, 0, 90]], [0, -120, 0, [-120, 0, 90]], [0, 180, 0, [180, 0, 90]],
        # Z rotations
        [0, 0, 90, [0, 0, 0]], [0, 0, 30, [0, 0, 60]], [0, 0, -30, [0, 0, 120]], [0, 0, 120, [0, 0, -30]], [0, 0, -120, [0, 0, -150]], [0, 0, 180, [0, 0, -90]],
        # Multiple Rotations
        [150, 30, 0, [146.30993247,  25.65890627, -73.89788625]], [45, 0, 60, [0, 45, 30]], [0, 90, 120, [90, 0, -30]], [135, 45, 90, [125.26438968, 30, -144.73561032]]
    ])
    def test_getangle(self, xRot, yRot, zRot, expected):
        axisP = pyCGM.rotmat(xRot, yRot, zRot)
        axisD = pyCGM.rotmat(0, 0, 0)
        result = pyCGM.getangle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getangle_datatypes(self):
        axisD = pyCGM.rotmat(0, 0, 0)
        axisP_floats = pyCGM.rotmat(90, 0, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [0, 90, 0]

        result_int_list = pyCGM.getangle(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        result_int_nparray = pyCGM.getangle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        result_float_list = pyCGM.getangle(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        result_float_nparray = pyCGM.getangle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @parameterized.expand([
        [0, 0, 0, [0, 0, -180]], [90, 0, 90, [0, 90, -90]],
        # X rotations
        [90, 0, 0, [0, 90, -180]], [30, 0, 0, [0, 30, -180]], [-30, 0, 0, [0, -30, -180]], [120, 0, 0, [180, 60, 0]], [-120, 0, 0, [180, -60, 0]], [180, 0, 0, [180, 0, 0]],
        # Y rotations
        [0, 90, 0, [90, 0, -180]], [0, 30, 0, [30, 0, -180]], [0, -30, 0, [330, 0, -180]], [0, 120, 0, [120, 0, -180]], [0, -120, 0, [240, 0, -180]], [0, 180, 0, [180, 0, -180]],
        # Z rotations
        [0, 0, 90, [0, 0, -90]], [0, 0, 30, [0, 0, -150]], [0, 0, -30, [0, 0, -210]], [0, 0, 120, [0, 0, -60]], [0, 0, -120, [0, 0, -300]], [0, 0, 180, [0, 0, 0]],
        # Multiple Rotations
        [150, 30, 0, [146.30993247,  25.65890627, -16.10211375]], [45, 0, 60, [0, 45, -120]], [0, 90, 120, [90, 0, -60]], [135, 45, 90, [125.26438968, 30, 54.73561032]]
    ])
    def test_getHeadangle(self, xRot, yRot, zRot, expected):
        axisP = pyCGM.rotmat(xRot, yRot, zRot)
        axisD = pyCGM.rotmat(0, 0, 0)
        result = pyCGM.getHeadangle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getHeadangle_datatypes(self):
        axisD = pyCGM.rotmat(0, 0, 0)
        axisP_floats = pyCGM.rotmat(90, 90, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [90, 0, 0]

        result_int_list = pyCGM.getHeadangle(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        result_int_nparray = pyCGM.getHeadangle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        result_float_list = pyCGM.getHeadangle(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        result_float_nparray = pyCGM.getHeadangle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @parameterized.expand([
        [0, 0, 0, [0, 0, 0]],
        # X rotations
        [90, 0, 0, [0, -90, 0]], [30, 0, 0, [0, -30, 0]], [-30, 0, 0, [0, 30, 0]], [120, 0, 0, [180, -60, 180]], [-120, 0, 0, [180, 60, 180]], [180, 0, 0, [180, 0, 180]],
        # Y rotations
        [0, 90, 0, [90, 0, 0]], [0, 30, 0, [30, 0, 0]], [0, -30, 0, [-30, 0, 0]], [0, 120, 0, [120, 0, 0]], [0, -120, 0, [-120, 0, -0]], [0, 180, 0, [180, 0, 0]],
        # Z rotations
        [0, 0, 90, [0, 0, 90]], [0, 0, 30, [0, 0, 30]], [0, 0, -30, [0, 0, -30]], [0, 0, 120, [0, 0, 120]], [0, 0, -120, [0, 0, -120]], [0, 0, 180, [0, 0, 180]],
        # Multiple Rotations
        [150, 30, 0, [146.30993247,  -25.65890627, 163.89788625]], [45, 0, 60, [0, -45, 60]], [0, 90, 120, [90, 0, 120]], [135, 45, 90, [125.26438968, -30, -125.26438968]]
    ])
    def test_getPelangle(self, xRot, yRot, zRot, expected):
        axisP = pyCGM.rotmat(xRot, yRot, zRot)
        axisD = pyCGM.rotmat(0, 0, 0)
        result = pyCGM.getPelangle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getPelangle_datatypes(self):
        axisD = pyCGM.rotmat(0, 0, 0)
        axisP_floats = pyCGM.rotmat(90, 90, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [90, 0, 180]

        result_int_list = pyCGM.getPelangle(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        result_int_nparray = pyCGM.getPelangle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        result_float_list = pyCGM.getPelangle(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        result_float_nparray = pyCGM.getPelangle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)
