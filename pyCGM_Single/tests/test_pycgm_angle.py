import unittest
import pyCGM_Single.pyCGM as pyCGM
import numpy as np
<<<<<<< HEAD
import pytest

rounding_precision = 6

class TestPycgmAngle():
    """
    This class tests the functions used for getting angles in pyCGM.py:
    getangle_sho
    getangle_spi
    getangle
    getHeadangle
    getPelangle
    """

    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected"], [
        (0, 0, 0, [0, 0, 0]),
        # X rotations
        (90, 0, 0, [0, 90, 0]), (30, 0, 0, [0, 30, 0]), (-30, 0, 0, [0, -30, 0]), (120, 0, 0, [0, 120, 0]),
        (-120, 0, 0, [0, -120, 0]), (180, 0, 0, [0, 180, 0]),
        # Y rotations
        (0, 90, 0, [90, 0, 0]), (0, 30, 0, [30, 0, 0]), (0, -30, 0, [-30, 0, 0]), (0, 120, 0, [60, -180, -180]),
        (0, -120, 0, [-60, -180, -180]), (0, 180, 0, [0, -180, -180]),
        # Z rotations
        (0, 0, 90, [0, 0, 90]), (0, 0, 30, [0, 0, 30]), (0, 0, -30, [0, 0, -30]), (0, 0, 120, [0, 0, 120]),
        (0, 0, -120, [0, 0, -120]), (0, 0, 180, [0, 0, 180]),
        # Multiple Rotations
        (150, 30, 0, [30, 150, 0]), (45, 0, 60, [0, 45, 60]), (0, 90, 120, [90, 0, 120]), (135, 45, 90, [45, 135, 90])
    ])
    def test_getangle_sho(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the getangle_sho function in pyCGM.py,
        defined as getangle_sho(axisP,axisD) where axisP is the proximal axis and axisD is the dorsal axis

        This test calls pyCGM.rotmat() to create axisP with an x, y, and z rotation defined in the parameters.
        It then calls pyCGM.getangle_sho() with axisP and axisD, which was created with no rotation in the
        x, y or z direction. This result is then compared to the expected result. The results from this test will
        be in the YXZ order, meaning that a parameter with an inputed x rotation will have a result with the same
        angle in the y direction. The only exception to this is a 120, -120, or 180 degree Y rotation. These will end
        up with a 60, -60, and 0 degree angle in the X direction respectively, and with a -180 degree
        angle in the y and z direction.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given in testcase
        axisP = pyCGM.rotmat(xRot, yRot, zRot)
        axisD = pyCGM.rotmat(0, 0, 0)
        result = pyCGM.getangle_sho(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getangle_sho_datatypes(self):
        """
        This test provides coverage of the getangle_sho function in pyCGM.py, defined as getangle_sho(axisP,axisD).
        It checks that the resulting output from calling getangle_sho is correct for a list of ints, a numpy array of
        ints, a list of floats, and a numpy array of floats.
        """
        axisD = pyCGM.rotmat(0, 0, 0)
        axisP_floats = pyCGM.rotmat(90, 0, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [0, 90, 90]

        # Check that calling getangle_sho on a list of ints yields the expected results
        result_int_list = pyCGM.getangle_sho(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling getangle_sho on a numpy array of ints yields the expected results
        result_int_nparray = pyCGM.getangle_sho(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling getangle_sho on a list of floats yields the expected results
        result_float_list = pyCGM.getangle_sho(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling getangle_sho on a numpy array of floats yields the expected results
        result_float_nparray = pyCGM.getangle_sho(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected"], [
        (0, 0, 0, [0, 0, 0]),
        # X rotations
        (90, 0, 0, [0, 0, 90]), (30, 0, 0, [0, 0, 30]), (-30, 0, 0, [0, 0, -30]), (120, 0, 0, [0, 0, 60]), (-120, 0, 0, [0, 0, -60]), (180, 0, 0, [0, 0, 0]),
        # Y rotations
        (0, 90, 0, [90, 0, 0]), (0, 30, 0, [30, 0, 0]), (0, -30, 0, [-30, 0, 0]), (0, 120, 0, [60, 0, 0]), (0, -120, 0, [-60, 0, 0]), (0, 180, 0, [0, 0, 0]),
        # Z rotations
        (0, 0, 90, [0, 90, 0]), (0, 0, 30, [0, 30, 0]), (0, 0, -30, [0, -30, 0]), (0, 0, 120, [0, 60, 0]), (0, 0, -120, [0, -60, 0]), (0, 0, 180, [0, 0, 0]),
        # Multiple Rotations
        (150, 30, 0, [-30, 0, 30]), (45, 0, 60, [-40.89339465, 67.7923457, 20.70481105]), (0, 90, 120, [-90, 0, 60]), (135, 45, 90, [-54.73561032, 54.73561032, -30])
    ])
    def test_getangle_spi(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the getangle_spi function in pyCGM.py,
        defined as getangle_spi(axisP,axisD) where axisP is the proximal axis and axisD is the dorsal axis

        This test calls pyCGM.rotmat() to create axisP with an x, y, and z rotation defined in the parameters.
        It then calls pyCGM.getangle_spi() with axisP and axisD, which was created with no rotation in the
        x, y or z direction. This result is then compared to the expected result. The results from this test will
        be in the YZX order, meaning that a parameter with an inputed x rotation will have a result with the same
        angle in the z direction. The only exception to this is a 120, -120, or 180 degree Y rotation. The exception
        to this is that 120, -120, and 180 degree rotations end up with 60, -60, and 0 degree angles respectively.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given in testcase
        axisP = pyCGM.rotmat(xRot, yRot, zRot)
        axisD = pyCGM.rotmat(0, 0, 0)
        result = pyCGM.getangle_spi(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getangle_spi_datatypes(self):
        """
        This test provides coverage of the getangle_spi function in pyCGM.py, defined as getangle_spi(axisP,axisD).
        It checks that the resulting output from calling getangle_spi is correct for a list of ints, a numpy array of
        ints, a list of floats, and a numpy array of floats.
        """
        axisD = pyCGM.rotmat(0, 0, 0)
        axisP_floats = pyCGM.rotmat(90, 0, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [-90, 90, 0]

        # Check that calling getangle_spi on a list of ints yields the expected results
        result_int_list = pyCGM.getangle_spi(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling getangle_spi on a numpy array of ints yields the expected results
        result_int_nparray = pyCGM.getangle_spi(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling getangle_spi on a list of floats yields the expected results
        result_float_list = pyCGM.getangle_spi(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling getangle_spi on a numpy array of floats yields the expected results
        result_float_nparray = pyCGM.getangle_spi(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected"], [
        (0, 0, 0, [0, 0, 90]),
        # X rotations
        (90, 0, 0, [0, 90, 90]), (30, 0, 0, [0, 30, 90]), (-30, 0, 0, [0, -30, 90]), (120, 0, 0, [180, 60, -90]), (-120, 0, 0, [180, -60, -90]), (180, 0, 0, [180, 0, -90]),
        # Y rotations
        (0, 90, 0, [90, 0, 90]), (0, 30, 0, [30, 0, 90]), (0, -30, 0, [-30, 0, 90]), (0, 120, 0, [120, 0, 90]), (0, -120, 0, [-120, 0, 90]), (0, 180, 0, [180, 0, 90]),
        # Z rotations
        (0, 0, 90, [0, 0, 0]), (0, 0, 30, [0, 0, 60]), (0, 0, -30, [0, 0, 120]), (0, 0, 120, [0, 0, -30]), (0, 0, -120, [0, 0, -150]), (0, 0, 180, [0, 0, -90]),
        # Multiple Rotations
        (150, 30, 0, [146.30993247, 25.65890627, -73.89788625]), (45, 0, 60, [0, 45, 30]), (0, 90, 120, [90, 0, -30]), (135, 45, 90, [125.26438968, 30, -144.73561032])
    ])
    def test_getangle(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the getangle function in pyCGM.py,
        defined as getangle(axisP,axisD) where axisP is the proximal axis and axisD is the dorsal axis

        This test calls pyCGM.rotmat() to create axisP with an x, y, and z rotation defined in the parameters.
        It then calls pyCGM.getangle() with axisP and axisD, which was created with no rotation in the x, y or z
        direction. This result is then compared to the expected result. The results from this test will be in the
        YXZ order, meaning that a parameter with an inputed x rotation will have a result with the same angle in
        the z direction. There is also an additional 90 degree angle in the z direction if there was no z rotation.
        If there was a z rotation than there will be a different angle in the z direction. A z rotation of 90, 30, -30,
        120, -120, 180 degrees results in a 0, 60, 120, -30, -150, -90 degree angle in the z direction respectively.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given in testcase
        axisP = pyCGM.rotmat(xRot, yRot, zRot)
        axisD = pyCGM.rotmat(0, 0, 0)
        result = pyCGM.getangle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getangle_datatypes(self):
        """
        This test provides coverage of the getangle function in pyCGM.py, defined as getangle(axisP,axisD).
        It checks that the resulting output from calling getangle is correct for a list of ints, a numpy array of
        ints, a list of floats, and a numpy array of floats.
        """
        axisD = pyCGM.rotmat(0, 0, 0)
        axisP_floats = pyCGM.rotmat(90, 0, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [0, 90, 0]

        # Check that calling getangle on a list of ints yields the expected results
        result_int_list = pyCGM.getangle(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling getangle on a numpy array of ints yields the expected results
        result_int_nparray = pyCGM.getangle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling getangle on a list of floats yields the expected results
        result_float_list = pyCGM.getangle(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling getangle on a numpy array of floats yields the expected results
        result_float_nparray = pyCGM.getangle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected"], [
        (0, 0, 0, [0, 0, -180]),
        # X rotations
        (90, 0, 0, [0, 90, -180]), (30, 0, 0, [0, 30, -180]), (-30, 0, 0, [0, -30, -180]), (120, 0, 0, [180, 60, 0]), (-120, 0, 0, [180, -60, 0]), (180, 0, 0, [180, 0, 0]),
        # Y rotations
        (0, 90, 0, [90, 0, -180]), (0, 30, 0, [30, 0, -180]), (0, -30, 0, [330, 0, -180]), (0, 120, 0, [120, 0, -180]), (0, -120, 0, [240, 0, -180]), (0, 180, 0, [180, 0, -180]),
        # Z rotations
        (0, 0, 90, [0, 0, -90]), (0, 0, 30, [0, 0, -150]), (0, 0, -30, [0, 0, -210]), (0, 0, 120, [0, 0, -60]), (0, 0, -120, [0, 0, -300]), (0, 0, 180, [0, 0, 0]),
        # Multiple Rotations
        (150, 30, 0, [146.30993247, 25.65890627, -16.10211375]), (45, 0, 60, [0, 45, -120]), (0, 90, 120, [90, 0, -60]), (135, 45, 90, [125.26438968, 30, 54.73561032])
    ])
    def test_getHeadangle(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the getHeadangle function in pyCGM.py,
        defined as getHeadangle(axisP,axisD) where axisP is the proximal axis and axisD is the dorsal axis

        This test calls pyCGM.rotmat() to create axisP with an x, y, and z rotation defined in the parameters.
        It then calls pyCGM.getHeadangle() with axisP and axisD, which was created with no rotation in the x, y or z
        direction. This result is then compared to the expected result. The results from this test will be in the
        YXZ order, meaning that a parameter with an inputed x rotation will have a result with the same angle in
        the z direction. There is also an additional -180 degree angle in the z direction if there was no z rotation.
        If there was a z rotation than there will be a different angle in the z direction. A z rotation of 90, 30, -30,
        120, -120, 180 degrees results in a -90, -150, -210, -60, -300, 0 degree angle in the z direction respectively.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given in testcase
        axisP = pyCGM.rotmat(xRot, yRot, zRot)
        axisD = pyCGM.rotmat(0, 0, 0)
        result = pyCGM.getHeadangle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getHeadangle_datatypes(self):
        """
        This test provides coverage of the getHeadangle function in pyCGM.py, defined as getHeadangle(axisP,axisD).
        It checks that the resulting output from calling getHeadangle is correct for a list of ints, a numpy array of
        ints, a list of floats, and a numpy array of floats.
        """
        axisD = pyCGM.rotmat(0, 0, 0)
        axisP_floats = pyCGM.rotmat(90, 90, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [90, 0, 0]

        # Check that calling getHeadangle on a list of ints yields the expected results
        result_int_list = pyCGM.getHeadangle(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling getHeadangle on a numpy array of ints yields the expected results
        result_int_nparray = pyCGM.getHeadangle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling getHeadangle on a list of floats yields the expected results
        result_float_list = pyCGM.getHeadangle(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling getHeadangle on a numpy array of floats yields the expected results
        result_float_nparray = pyCGM.getHeadangle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected"], [
        (0, 0, 0, [0, 0, 0]),
        # X rotations
        (90, 0, 0, [0, -90, 0]), (30, 0, 0, [0, -30, 0]), (-30, 0, 0, [0, 30, 0]), (120, 0, 0, [180, -60, 180]), (-120, 0, 0, [180, 60, 180]), (180, 0, 0, [180, 0, 180]),
        # Y rotations
        (0, 90, 0, [90, 0, 0]), (0, 30, 0, [30, 0, 0]), (0, -30, 0, [-30, 0, 0]), (0, 120, 0, [120, 0, 0]), (0, -120, 0, [-120, 0, -0]), (0, 180, 0, [180, 0, 0]),
        # Z rotations
        (0, 0, 90, [0, 0, 90]), (0, 0, 30, [0, 0, 30]), (0, 0, -30, [0, 0, -30]), (0, 0, 120, [0, 0, 120]), (0, 0, -120, [0, 0, -120]), (0, 0, 180, [0, 0, 180]),
        # Multiple Rotations
        (150, 30, 0, [146.30993247, -25.65890627, 163.89788625]), (45, 0, 60, [0, -45, 60]), (0, 90, 120, [90, 0, 120]), (135, 45, 90, [125.26438968, -30, -125.26438968])
    ])
    def test_getPelangle(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the getPelangle function in pyCGM.py,
        defined as getPelangle(axisP,axisD) where axisP is the proximal axis and axisD is the dorsal axis

        This test calls pyCGM.rotmat() to create axisP with an x, y, and z rotation defined in the parameters.
        It then calls pyCGM.getHeadangle() with axisP and axisD, which was created with no rotation in the x, y or z
        direction. This result is then compared to the expected result. The results from this test will be in the
        YXZ order, meaning that a parameter with an inputed x rotation will have a result with the same angle in
        the z direction. The exception to this is x rotations. An x rotation of 90, 30, -30, 120, -120, 180
        degrees results in a -90, -30, 30, -6, 60, 0 degree angle in the y direction respectively. A x rotation or
        120, -120, or 180 also results in a 180 degree rotation in the x and z angles.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given in testcase
        axisP = pyCGM.rotmat(xRot, yRot, zRot)
        axisD = pyCGM.rotmat(0, 0, 0)
        result = pyCGM.getPelangle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getPelangle_datatypes(self):
        """
        This test provides coverage of the getPelangle function in pyCGM.py, defined as getPelangle(axisP,axisD).
        It checks that the resulting output from calling getPelangle is correct for a list of ints, a numpy array of
        ints, a list of floats, and a numpy array of floats.
        """
        axisD = pyCGM.rotmat(0, 0, 0)
        axisP_floats = pyCGM.rotmat(90, 90, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [90, 0, 180]

        # Check that calling getPelangle on a list of ints yields the expected results
        result_int_list = pyCGM.getPelangle(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling getPelangle on a numpy array of ints yields the expected results
        result_int_nparray = pyCGM.getPelangle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling getPelangle on a list of floats yields the expected results
        result_float_list = pyCGM.getPelangle(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling getPelangle on a numpy array of floats yields the expected results
        result_float_nparray = pyCGM.getPelangle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)
=======
from parameterized import parameterized

rounding_precision = 8

class TestPycgmAngle(unittest.TestCase):

    testcase_a = [[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
                  [90.0, 0.0, 0.0]]
    testcase_b = [[[1, 0, 0], [0, 0, 0], [1, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0.70710678118, 0, 0]],
                  [45.0, 0.0, 0]]
    testcase_c = [[[0, 0, 0], [-1, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
                  [0.0, 90.0, 0.0]]
    testcase_d = [[[0, 0, 0], [1, 0, 0], [1, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0.70710678118, 0, 0]],
                  [0.0, -45.0, 0]]
    axisP_a = [[-1, 0, 0], [0, 0, 0], [0, 0, 0]]
    axisD_a = [[0, 0, 0], [1, 0, 0], [0, 0, 0]]
    axisP_b = [[0.70710678118, 0, 0], [0, 0, 0], [0, 0, 0]]
    axisD_b = [[1, 0, 0], [1, 0, 0], [0, 0, 0]]

    @parameterized.expand([
        [[[0.09010104879445179904, 0.99590937599884910014, 0.00680557152145411237],
          [0.99377608791559168822, -0.08945440277921079542, -0.06638521605464120512],
          [-0.06550487076049194002, 0.01274459183355247660, -0.99777085910818641423]],
         [[0.14362551354236074985, -0.98963110439615320502, -0.00141034441480769601],
          [0.68540404438920177199, 0.09844473917718232769, 0.72147760135931093828],
          [-0.71385783444225126004, -0.10458924677044478813, 0.69243633762630452111]],
         [-9.42569150165887, 130.86647058885387, -170.3887751198432]],
        testcase_a,
        testcase_b,
        testcase_c,
        testcase_d,
        [axisP_a,
         axisD_a,
         [0.0, 0.0, 90.0]],
        [axisP_b,
         axisD_b,
         [0.0, 0.0, -45.0]]])
    def test_getangle_sho(self, axisP, axisD, expected):
        result = pyCGM.getangle_sho(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @parameterized.expand([
        [[[0.13232935635315357104, 0.98562945782504129966, -0.10499292030749529658],
          [-0.99119134439010281312, 0.13101087562301927392, -0.01938734831355759525],
          [-0.00535352718324588750, 0.10663358915485332545, 0.99428397221845443710]],
         [[0.09010104879445179904, 0.99590937599884910014, 0.00680557152145411237],
          [0.99377608791559168822, -0.08945440277921079542, -0.06638521605464120512],
          [-0.06550487076049194002, 0.01274459183355247660, -0.99777085910818641423]],
         [-6.4797057790916615, -2.893068979100172, -4.638276625836626]],
        [[[0, 0, 0], [0, 0, 0], [1, 0, 0]],
         [[-1, 0, 0], [0, 0, 0], [0, 0, 0]],
         [90.0, 0.0, 0.0]],
        [[[0, 0, 0], [0, 0, 0], [0, 1, 0]],
         [[0, 0.70710678118, 0], [0, 0, 0], [0, 0, 0]],
         [-45.0, 0.0, 0.0]],
        [axisP_a,
         axisD_a,
         [0.0, 90.0, 0.0]],
        [axisP_b,
         axisD_b,
         [0.0, -45.0, 0.0]],
        [[[0, 0, 0], [0, 0, 0], [1, 0, 0]],
         [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
         [0.0, 0.0, 90.0]],
        [[[0, 0, 0], [0, 0, 0], [0, 1, 0]],
         [[0, 0, 0], [0, -0.70710678118, 0], [0, 0, 0]],
         [0.0, 0.0, -45.0]]])
    def test_getangle_spi(self, axisP, axisD, expected):
        result = pyCGM.getangle_spi(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @parameterized.expand([
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         [[0.13232935635315357103, 0.98562945782504129965, -0.10499292030749529658],
          [-0.99119134439010281312, 0.13101087562301927391, -0.01938734831355759524],
          [-0.00535352718324588749, 0.10663358915485332545, 0.99428397221845443709]],
         [-0.30849491450945404, -6.121292793370006, 7.5714311021517124]],
        testcase_a,
        testcase_b,
        testcase_c,
        testcase_d,
        [[[0, 0, 0], [1, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
         [0.0, 0.0, 90.0]],
        [[[0, 0, 0], [-0.70710678118, 0, 0], [0, 0, 0]],
         [[-1, 0, 0], [-1, 0, 0], [0, 0, 0]],
         [0.0, 0.0, 45.0]]])
    def test_getangle(self, axisP, axisD, expected):
        result = pyCGM.getangle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)
>>>>>>> c5c44c0... Update test_pycgm_angle.py
