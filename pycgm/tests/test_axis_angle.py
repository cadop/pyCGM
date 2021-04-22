import unittest
import pycgm.axis as axis
import numpy as np
import pytest

rounding_precision = 6

class TestPycgmAngle():
    """
    This class tests the functions used for getting angles in axis.py:
    get_angle_sho
    get_angle_spi
    get_angle
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
    def test_get_shoulder_angle(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the get_shoulder_angle function in axis.py,
        defined as get_shoulder_angle(axisP,axisD) where axisP is the proximal axis and axisD is the distal axis.

        get_shoulder_angle takes in as input two axes, axisP and axisD, and returns in degrees, the Euler angle
        rotations required to rotate axisP to axisD as a list [alpha, beta, gamma]. get_shoulder_angle uses the XYZ
        order Euler rotations to calculate the angles. The rotation matrix is obtained by directly comparing
        the vectors in axisP to those in axisD through dot products between different components
        of each axis. axisP and axisD each have 3 components to their axis, x, y, and z. 
        The angles are calculated as follows:

        .. math::
            \[ \alpha = \arcsin{(axisD_{z} \cdot axisP_{x})} \]

            \[ \beta = \arctan2{(-(axisD_{z} \cdot axisP_{y}), axisD_{z} \cdot axisP_{z})} \]

            \[ \gamma = \arctan2{(-(axisD_{y} \cdot axisP_{x}), axisD_{x} \cdot axisP_{x})} \]

        This test calls axis.rotmat() to create axisP with an x, y, and z rotation defined in the parameters.
        It then calls axis.get_shoulder_angle() with axisP and axisD, which was created with no rotation in the
        x, y or z direction. This result is then compared to the expected result. The results from this test will
        be in the YXZ order, meaning that a parameter with an inputed x rotation will have a result with the same
        angle in the y direction. The only exception to this is a 120, -120, or 180 degree Y rotation. These will end
        up with a 60, -60, and 0 degree angle in the X direction respectively, and with a -180 degree
        angle in the y and z direction.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given in testcase
        axisP = axis.rotmat(xRot, yRot, zRot)
        axisD = axis.rotmat(0, 0, 0)
        result = axis.get_shoulder_angle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_get_shoulder_angle_datatypes(self):
        """
        This test provides coverage of the get_shoulder_angle function in axis.py, defined as get_shoulder_angle(axisP,axisD).
        It checks that the resulting output from calling get_shoulder_angle is correct for a list of ints, a numpy array of
        ints, a list of floats, and a numpy array of floats.
        """
        axisD = axis.rotmat(0, 0, 0)
        axisP_floats = axis.rotmat(90, 0, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [0, 90, 90]

        # Check that calling get_shoulder_angle on a list of ints yields the expected results
        result_int_list = axis.get_shoulder_angle(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling get_shoulder_angle on a numpy array of ints yields the expected results
        result_int_nparray = axis.get_shoulder_angle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling get_shoulder_angle on a list of floats yields the expected results
        result_float_list = axis.get_shoulder_angle(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling get_shoulder_angle on a numpy array of floats yields the expected results
        result_float_nparray = axis.get_shoulder_angle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)