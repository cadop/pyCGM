import unittest
import pyCGM_Single.pyCGM as pyCGM
import numpy as np
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
        defined as getangle_sho(axisP,axisD) where axisP is the proximal axis and axisD is the distal axis.

        getangle_sho takes in as input two axes, axisP and axisD, and returns in degrees, the Euler angle
        rotations required to rotate axisP to axisD as a list [alpha, beta, gamma]. getangle_sho uses the XYZ
        order Euler rotations to calculate the angles. The rotation matrix is obtained by directly comparing
        the vectors in axisP to those in axisD through dot products between different components
        of each axis. axisP and axisD each have 3 components to their axis, x, y, and z. 
        The angles are calculated as follows:
        
        \[ \alpha = \arcsin{(axisD_{z} \cdot axisP_{x})} \]
        \[ \beta = \arctan2{(-(axisD_{z} \cdot axisP_{y}), axisD_{z} \cdot axisP_{z})} \]
        \[ \gamma = \arctan2{(-(axisD_{y} \cdot axisP_{x}), axisD_{x} \cdot axisP_{x})} \]

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
        defined as getangle_spi(axisP,axisD) where axisP is the proximal axis and axisD is the distal axis

        getangle_spi takes in as input two axes, axisP and axisD, and returns in degrees, the Euler angle
        rotations required to rotate axisP to axisD as a list [beta, gamma, alpha]. getangle_spi uses the XZX
        order of Euler rotations to calculate the angles. The rotation matrix is obtained by directly comparing
        the vectors in axisP to those in axisD through dot products between different components
        of each axis. axisP and axisD each have 3 components to their axis, x, y, and z. 
        The angles are calculated as follows:
        
        \[ alpha = \arcsin{(axisD_{y} \cdot axisP_{z})} \]
        \[ gamma = \arcsin{(-(axisD_{y} \cdot axisP_{x}) / \cos{\alpha})} \]
        \[ beta = \arcsin{(-(axisD_{x} \cdot axisP_{z}) / \cos{\alpha})} \]

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
        defined as getangle(axisP,axisD) where axisP is the proximal axis and axisD is the distal axis

        getangle takes in as input two axes, axisP and axisD, and returns in degrees, the Euler angle
        rotations required to rotate axisP to axisD as a list [beta, alpha, gamma]. getangle uses the YXZ
        order of Euler rotations to calculate the angles. The rotation matrix is obtained by directly comparing
        the vectors in axisP to those in axisD through dot products between different components
        of each axis. axisP and axisD each have 3 components to their axis, x, y, and z. Since arcsin
        is being used, the function checks wether the angle alpha is between -pi/2 and pi/2.
        The angles are calculated as follows:

        \[ \alpha = \arcsin{(-axisD_{z} \cdot axisP_{y})} \]

        If alpha is between -pi/2 and pi/2

        \[ \beta = \arctan2{((axisD_{z} \cdot axisP_{x}), axisD_{z} \cdot axisP_{z})} \]
        \[ \gamma = \arctan2{((axisD_{y} \cdot axisP_{y}), axisD_{x} \cdot axisP_{y})} \]

        Otherwise

        \[ \beta = \arctan2{(-(axisD_{z} \cdot axisP_{x}), axisD_{z} \cdot axisP_{z})} \]
        \[ \gamma = \arctan2{(-(axisD_{y} \cdot axisP_{y}), axisD_{x} \cdot axisP_{y})} \]

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
        defined as getHeadangle(axisP,axisD) where axisP is the proximal axis and axisD is the distal axis

        getHeadangle takes in as input two axes, axisP and axisD, and returns in degrees, the Euler angle
        rotations required to rotate axisP to axisD as a list [alpha, beta, gamma]. getHeadangle uses the YXZ
        order of Euler rotations to calculate the angles. The rotation matrix is obtained by directly comparing
        the vectors in axisP to those in axisD through dot products between different components
        of each axis. axisP and axisD each have 3 components to their axis, x, y, and z. 
        The angles are calculated as follows:

        \[ \beta = \arctan2{((axisD_{z} \cdot axisP_{y}), \sqrt{(axisD_{x} \cdot axisP_{y})^2 + (axisD_{y} \cdot axisP_{y})^2}}) \]

        \[ \alpha = \arctan2{(-(axisD_{z} \cdot axisP_{x}), axisD_{z} \cdot axisP_{z})} \]
        \[ \gamma = \arctan2{(-(axisD_{x} \cdot axisP_{y}), axisD_{y} \cdot axisP_{y})} \]

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
        defined as getPelangle(axisP,axisD) where axisP is the proximal axis and axisD is the distal axis

        getPelangle takes in as input two axes, axisP and axisD, and returns in degrees, the Euler angle
        rotations required to rotate axisP to axisD as a list [alpha, beta, gamma]. getPelangle uses the YXZ
        order of Euler rotations to calculate the angles. The rotation matrix is obtained by directly comparing
        the vectors in axisP to those in axisD through dot products between different components
        of each axis. axisP and axisD each have 3 components to their axis, x, y, and z. 
        The angles are calculated as follows:

        \[ \beta = \arctan2{((axisD_{z} \cdot axisP_{y}), \sqrt{(axisD_{z} \cdot axisP_{x})^2 + (axisD_{z} \cdot axisP_{z})^2}}) \]

        \[ \alpha = \arctan2{((axisD_{z} \cdot axisP_{x}), axisD_{z} \cdot axisP_{z})} \]
        \[ \gamma = \arctan2{((axisD_{x} \cdot axisP_{y}), axisD_{y} \cdot axisP_{y})} \]

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