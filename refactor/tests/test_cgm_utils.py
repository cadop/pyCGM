#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from refactor.pycgm import CGM

rounding_precision = 8


class TestCGMUtils():
    """
    This class tests the utils functions in the class CGM in pycgm.py:
    rotation_matrix
    """

    @pytest.mark.parametrize(["x", "y", "z", "expected"], [
        (0.0, 0.0, 180, [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        (0, 0, 0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        (90, 0, 0, [[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        (0, 135, 0, [[-0.70710678, 0, 0.70710678], [0, 1, 0], [-0.70710678, 0, -0.70710678]]),
        (0, 0, -60, [[0.5, 0.8660254, 0], [-0.8660254, 0.5, 0], [0, 0, 1]]),
        (90, 0, 90, [[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
        (0, 150, -30, [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]),
        (90, 180, -90, [[0, -1, 0], [0, 0, 1], [-1, 0, 0]])])
    def test_rotation_matrix(self, x, y, z, expected):
        """
        This test provides coverage of the rotation_matrix function in the class CGM in pycgm.py, defined as
        rotation_matrix(x, y, z)

        This test takes 4 parameters:
        x, y, z : float, optional
            Angle, which will be converted to radians, in each respective axis to describe the rotations.
            The default is 0 for each unspecified angle.
        expected : array
            A 3x3 ndarray which can bbe used to perform a rotation about the x, y, z axes.
        """
        result = CGM.rotation_matrix(x, y, z)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_rotation_matrix_datatypes(self):
        """
        This test provides coverage of the rotation_matrix function in the class CGM in pycgm.py, defined as
        rotation_matrix(x, y, z)

        This test checks that the resulting output from calling rotation_matrix is correct when called with ints or
        floats.
        """
        result_int = CGM.rotation_matrix(0, 150, -30)
        result_float = CGM.rotation_matrix(0.0, 150.0, -30.0)
        expected = [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]

        np.testing.assert_almost_equal(result_int, expected, rounding_precision)
        np.testing.assert_almost_equal(result_float, expected, rounding_precision)

    @pytest.mark.parametrize("axis_vectors, expected", [
        (np.array([[1.2, 2.3, 3.4], [7.7, 7.7, 7.7], [1.1, 1.1, 1.1], [3.0, 4.2, 5.7]]),
         np.array([[6.5, 5.4, 4.3], [-0.1, -1.2, -2.3], [1.8, 1.9, 2.3]])),
        (np.array([[-1.2, 2.313, -32.4], [7.7, -7.7, 7.7], [1.123, 15.1, -1.111], [3.03, 4.23, -52.7]]),
         np.array([[8.9, -10.013,  40.1], [2.323, 12.787, 31.289], [4.23, 1.917, -20.3]])),
        (np.array([[1, 1, 1], [-1, -2, -3], [4, 5, 6], [10, 100, 1000]]),
         np.array([[-2, -3, -4], [3, 4, 5], [9, 99, 999]])),
         #Test that when origin is [0, 0, 0], the axes do not change
        (np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]),
         np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])),
         #Test no error is raised when input is a list
        ([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
         [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    ])
    def test_subtract_origin(self, axis_vectors, expected):
        """
        This function tests CGM.subtract_origin(axis_vectors), where
        axis_vectors is an array of 4 1x3 arrays giving the origin point,
        x-axis, y-axis, and z-axis. CGM.subtract_origin subtracts the origin
        point from all of the x, y, and z axis vectors and returns them in a numpy array.

        We test for floats, ints, positive and negative numbers, and the case
        where the origin is at [0, 0, 0]. We test to make sure the result is
        returned as a numpy array.
        """
        result = CGM.subtract_origin(axis_vectors)
        #Ensure that the result is a numpy array
        assert isinstance(result, np.ndarray)
        np.testing.assert_almost_equal(result, expected, rounding_precision)
    