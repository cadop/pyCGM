#!/usr/bin/python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from refactor.pycgm import StaticCGM, CGM

rounding_precision = 6

class TestStaticCGMAxis:
    """
    This class tests the utility functions in pycgm.StaticCGM:
        iad_calculation
        ankle_angle_calc
    """

    @pytest.mark.parametrize(["rasi", "lasi", "expected"], [
        (np.array([0, 0, 0]), np.array([0, 0, 0]), 0),
        (np.array([1, 0, 0]), np.array([0, 0, 0]), 1),
        (np.array([0, 0, 0]), np.array([2, 0, 0]), 2),
        (np.array([0, 1, 0]), np.array([0, 1, 0]), 0),
        (np.array([4, 0, 0]), np.array([2, 0, 0]), 2),
        (np.array([4, 0, 0]), np.array([-2, 0, 0]), 6),
        (np.array([0, 2, 1]), np.array([0, 4, 1]), 2),
        (np.array([-5, 3, 0]), np.array([0, 3, 0]), 5),
        (np.array([0, 3, -6]), np.array([0, 2, -5]), 1.4142135623730951),
        (np.array([-6, 4, 0]), np.array([0, 6, -8]), 10.198039027185569),
        (np.array([7, 2, -6]), np.array([3, -7, 2]), 12.68857754044952),
        # Testing that when frame is composed of lists of ints
        ([7, 2, -6], [3, -7, 2], 12.68857754044952),
        # Testing that when frame is composed of numpy arrays of ints
        (np.array([7, 2, -6], dtype='int'), np.array([3, -7, 2], dtype='int'), 12.68857754044952),
        # Testing that when frame is composed of lists of floats
        ([7.0, 2.0, -6.0], [3.0, -7.0, 2.0], 12.68857754044952),
        # Testing that when frame is composed ofe numpy arrays of floats
        (np.array([7.0, 2.0, -6.0], dtype='float'), np.array([3.0, -7.0, 2.0], dtype='float'), 12.68857754044952)
    ])
    def test_iad_calculation(self, rasi, lasi, expected):
        """
        This test provides coverage of the StaticCGM.iad_calculation, 
        defined as iad_calculation(rasi, lasi), where rasi and lasi are the
        arrays representing the positions of the RASI and LASI markers.

        Given the markers RASI and LASI, the Inter ASIS Distance is defined as:
        .. math::
            InterASISDist = \sqrt{(RASI_x-LASI_x)^2 + (RASI_y-LASI_y)^2 + (RASI_z-LASI_z)^2}
        where :math:`RASI_x` is the x-coordinate of the RASI marker

        This unit test ensures that:
        - the distance is measured correctly when some coordinates are the same, all coordinates are the same, and all
        coordinates are different
        - the distance is measured correctly given positive, negative and zero values
        - the resulting output is correct when frame is composed of lists of ints, numpy arrays of ints, lists of
        floats, and numpy arrays of floats.
        """
        result = StaticCGM.iad_calculation(rasi, lasi)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected_results"], [
        (0, 0, 0, [0, 0, 0]),
        # X rotations
        (90, 0, 0, [-1.570796, 0, 0]), (30, 0, 0, [-0.523599, 0, 0]), (-30, 0, 0, [0.523599, 0, 0]),
        (120, 0, 0, [-1.047198, 0, 0]), (-120, 0, 0, [1.047198, 0, 0]), (180, 0, 0, [0, 0, 0]),
        # Y rotations
        (0, 90, 0, [0, -1.570796, 0]), (0, 30, 0, [0, -0.523599, 0]), (0, -30, 0, [0, 0.523599, 0]),
        (0, 120, 0, [0, 1.047198, 0]), (0, -120, 0, [0, -1.047198, 0]), (0, 180, 0, [0, 0, 0]),
        # Z rotations
        (0, 0, 90, [0, 0, -1.570796]), (0, 0, 30, [0, 0, -0.523599]), (0, 0, -30, [0, 0, 0.523599]),
        (0, 0, 120, [0, 0, 1.047198]), (0, 0, -120, [0, 0, -1.047198]), (0, 0, 180, [0, 0, 0]),
        # Multiple Rotations
        (150, 30, 0, [-0.447832,  0.588003,  0.281035]), (45, 0, 60, [-0.785398, -0, -1.047198]),
        (0, 90, 120, [0, -1.570796,  1.047198]), (135, 45, 90, [-0.523599,  0.955317, -0.955317])])
    def test_ankle_angle_calc(self, xRot, yRot, zRot, expected_results):
        """
        This test provides coverage of the ankle_angle_calc method in StaticCGM in the file pycgm.py, it is defined as ankle_angle_calc(axis_p, axis_d)

        This test takes 3 parameters:
        axis_p: the unit vector of axis_p, the position of the proximal axis
        axis_d: the unit vector of axis_d, the position of the distal axis
        expected_results: the expected result from calling ankle_angle_calc on axis_p and axis_d. This returns the x, y, z angles
        from a XYZ Euler angle calculation from a rotational matrix. This rotational matrix is calculated by matrix
        multiplication of axis_d and the inverse of axis_p. This angle is in radians, not degrees.

        The x, y, and z angles are defined as:
        .. math::
            \[ x = \arctan{\frac{M[2][1]}{\sqrt{M[2][0]^2 + M[2][2]^2}}} \]
            \[ y = \arctan{\frac{-M[2][0]}{M[2][2]}} \]
            \[ z = \arctan{\frac{-M[0][1]}{M[1][1]}} \]
        where M is the rotation matrix produced from multiplying axis_d and :math:`axis_p^{-1}`

        This test ensures that:
        - A rotation in one axis will only effect the resulting angle in the corresponding axes
        - A rotation in multiple axes can effect the angles in other axes due to XYZ order
        """
        # Create axis_p as a rotational matrix using the x, y, and z rotations given
        # The rotational matrix will be created using CGM's rotation_matrix.
        axis_p = CGM.rotation_matrix(xRot, yRot, zRot)
        axis_d = CGM.rotation_matrix(0, 0, 0)
        result = StaticCGM.ankle_angle_calc(axis_p, axis_d)
        np.testing.assert_almost_equal(result, expected_results, rounding_precision)