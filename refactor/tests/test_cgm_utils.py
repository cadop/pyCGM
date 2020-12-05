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

    nan_3d = np.array([np.nan, np.nan, np.nan])
    rand_coor = np.array([np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)])

    @pytest.mark.parametrize(["x", "y", "z", "expected"], [
        (0.0, 0.0, 180, [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        (0, 0, 0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        (90, 0, 0, [[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        (0, 135, 0, [[-0.70710678, 0, 0.70710678], [0, 1, 0], [-0.70710678, 0, -0.70710678]]),
        (0, 0, -60, [[0.5, 0.8660254, 0], [-0.8660254, 0.5, 0], [0, 0, 1]]),
        (90, 0, 90, [[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
        (0, 150, -30, [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]),
        (90, 180, -90, [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
        # Testing when x, y, z values are ints
        (0, 150, -30, [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]),
        # Testing when x, y, z values are floats
        (0.0, 150.0, -30.0, [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]),
    ])
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

    @pytest.mark.parametrize("axis_vectors, expected", [
        (np.array([[1.2, 2.3, 3.4], [7.7, 7.7, 7.7], [1.1, 1.1, 1.1], [3.0, 4.2, 5.7]]),
         np.array([[6.5, 5.4, 4.3], [-0.1, -1.2, -2.3], [1.8, 1.9, 2.3]])),
        (np.array([[-1.2, 2.313, -32.4], [7.7, -7.7, 7.7], [1.123, 15.1, -1.111], [3.03, 4.23, -52.7]]),
         np.array([[8.9, -10.013, 40.1], [2.323, 12.787, 31.289], [4.23, 1.917, -20.3]])),
        (np.array([[1, 1, 1], [-1, -2, -3], [4, 5, 6], [10, 100, 1000]]),
         np.array([[-2, -3, -4], [3, 4, 5], [9, 99, 999]])),
        # Test that when origin is [0, 0, 0], the axes do not change
        (np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]),
         np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])),
        # Test no error is raised when input is a list
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
        # Ensure that the result is a numpy array
        assert isinstance(result, np.ndarray)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["rsho", "lsho", "thorax_axis", "expected"], [
        ([428.88476562, 270.552948, 1500.73010254], [68.24668121, 269.01049805, 1510.1072998], np.array(
            [[256.149810236564, 364.3090603933987, 1459.6553639290375],
             [256.23991128535846, 365.30496976939753, 1459.662169500559], rand_coor, rand_coor]),
         [[255.92550222678443, 364.3226950497605, 1460.6297868417887],
          [256.42380097331767, 364.27770361353487, 1460.6165849382387]]),
        ([0, 0, 1], [0, 1, 0], np.array([[0, 0, 0], [1, 0, 0], rand_coor, rand_coor]), [[0, 1, 0], [0, 0, 1]]),
        ([0, 1, 1], [1, 1, 1], np.array([[0, 0, 0], [1, 0, 0], rand_coor, rand_coor]),
         [[0, 0.70710678, -0.70710678], [0, -0.70710678, 0.70710678]]),
        ([0, 1, 1], [1, 1, 1], np.array([[-1, 0, 0], [1, 0, 0], rand_coor, rand_coor]),
         [[-1, 0.70710678, -0.70710678], [-1, -0.70710678, 0.70710678]]),
        ([1, 2, 1], [2, 1, 2], np.array([[0, 0, 0], [1, 0, 0], rand_coor, rand_coor]),
         [[0, 0.4472136, -0.89442719], [0, -0.89442719, 0.4472136]]),
        ([1, 2, 1], [2, 2, 2], np.array([[0, 0, 0], [1, 0, 0], rand_coor, rand_coor]),
         [[0, 0.4472136, -0.89442719], [0, -0.70710678, 0.70710678]]),
        ([1, 2, 2], [2, 1, 2], np.array([[0, 0, 0], [1, 0, 0], rand_coor, rand_coor]),
         [[0, 0.70710678, -0.70710678], [0, -0.89442719, 0.4472136]]),
        ([1, 1, 1], [1, 1, 1], np.array([[0, 0, 0], [1, 0, 1], rand_coor, rand_coor]),
         [[0.70710678, 0, -0.70710678], [-0.70710678, 0, 0.70710678]]),
        ([1, 1, 1], [1, 1, 1], np.array([[0, 0, 1], [1, 0, 1], rand_coor, rand_coor]), [[0, 0, 0], [0, 0, 2]]),
        ([0, 1, 0], [0, 0, -1], np.array([[0, 0, 0], [0, 3, 4], rand_coor, rand_coor]), [[1, 0, 0], [-1, 0, 0]]),
        ([1, 0, 0], [0, 1, 0], np.array([[0, 0, 0], [7, 0, 24], rand_coor, rand_coor]), [[0, -1, 0], [-0.96, 0, 0.28]]),
        ([1, 0, 0], [0, 0, 1], np.array([[8, 0, 0], [8, 0, 6], rand_coor, rand_coor]), [[8, 1, 0], [8, -1, 0]])])
    def test_wand_marker(self, rsho, lsho, thorax_axis, expected):
        """
        This test provides coverage of the wand_marker function in the class CGM in pycgm.py, defined as
        wand_marker(rsho, lsho, thorax_axis)

        This function takes 4 parameters:
        rsho, lsho : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        thorax_axis : ndarray
            A 4x3 ndarray that contains the thorax origin and the
            thorax x, y, and z axis components.
        expected : array
            A 2x3 ndarray containing the right wand marker x, y, and z positions and the left wand marker x, y,
            and z positions.

        The function takes in the xyz position of the Right Shoulder and Left Shoulder markers, as well as the thorax
        frame, which is a list of [ xyz axis vectors, origin ]. The wand marker position is returned as a 2x3 array
        containing the right wand marker x, y, z positions (1x3) followed by the left wand marker x, y, z positions
        (1x3). The thorax axis is provided in global coordinates, which are subtracted inside the function to define
        the unit vectors.

        For the Right and Left wand markers, the function performs the same calculation, with the difference being the
        corresponding sides marker. Each wand marker is defined as the cross product between the unit vector of the
        x axis of the thorax frame, and the unit vector from the thorax frame origin to the Shoulder marker.

        Given a marker SHO, representing the right (RSHO) or left (LSHO) shoulder markers and a thorax axis TH, the
        wand marker W is defined as:

        .. math::
            W_R = (RSHO-TH_o) \times TH_x
            W_L = TH_x \times (LSHO-TH_o)

        where :math:`TH_o` is the origin of the thorax axis, :math:`TH_x` is the x unit vector of the thorax axis.

        From this calculation, it should be clear that changing the thorax y and z vectors should not have an impact
        on the results.

        This unit test ensure that:
        - The right and left markers do not impact the wand marker calculations for one another
        - The function requires global positions
        - The thorax y and z axis do not change the results
        """
        result = CGM.wand_marker(rsho, lsho, thorax_axis)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["point", "start", "end", "expected"], [
        # Test from running sample data
        ([1, 2, 3], np.array([4, 5, 6]), [7, 8, 9], (5.19615242, [4, 5, 6], [1, 2, 3])),
        # Test with zeros for all params
        ([0, 0, 0], np.array([0, 0, 0]), [0, 0, 0], (np.nan, nan_3d, [0, 0, 0])),
        # Testing when values are added to point
        ([8, 4, 5], np.array([0, 0, 0]), [0, 0, 0], (np.nan, nan_3d, [8, 4, 5])),
        # Testing when values are added to start
        ([0, 0, 0], np.array([-3, -2, 4]), [0, 0, 0], (0.0, [0, 0, 0], [0, 0, 0])),
        # Testing when values are added to end
        ([0, 0, 0], np.array([0, 0, 0]), [-8, -3, 7], (0.0, [0, 0, 0], [0, 0, 0])),
        # Testing when values are added to point and start
        ([8, 4, 5], np.array([-3, -2, 4]), [0, 0, 0], (10.246950765959598, [0, 0, 0], [8, 4, 5])),
        # Testing when values are added to point and end
        ([8, 4, 5], np.array([0, 0, 0]), [-8, -3, 7], (10.246950765959598, [0, 0, 0], [8, 4, 5])),
        # Testing when values are added to start and end
        ([0, 0, 0], np.array([-3, -2, 4]), [-8, -3, 7], (5.385164807134504, [-3, -2, 4], [0, 0, 0])),
        # Testing when values are added to point, start and end
        ([8, 4, 5], np.array([-3, -2, 4]), [-8, -3, 7], (12.569805089976535, [-3, -2, 4], [8, 4, 5])),
        # Testing that when point and end are composed of lists of ints
        ([8, 4, 5], np.array([-3, -2, 4]), [-8, -3, 7], (12.569805089976535, [-3, -2, 4], [8, 4, 5])),
        # Testing that when point and end are composed of numpy arrays of ints
        (np.array([8, 4, 5], dtype='int'), np.array([-3, -2, 4], dtype='int'), np.array([-8, -3, 7], dtype='int'),
         (12.569805089976535, [-3, -2, 4], [8, 4, 5])),
        # Testing that when point and end are composed of lists of floats
        ([8.0, 4.0, 5.0], np.array([-3.0, -2.0, 4.0]), [-8.0, -3.0, 7.0], (12.569805089976535, [-3, -2, 4], [8, 4, 5])),
        # Testing that when point and end are composed of numpy arrays of floats
        (np.array([8.0, 4.0, 5.0], dtype='float'), np.array([-3.0, -2.0, 4.0], dtype='float'),
         np.array([-8.0, -3.0, 7.0], dtype='float'), (12.569805089976535, [-3, -2, 4], [8, 4, 5]))])
    def test_point_to_line(self, point, start, end, expected):
        """
        This test provides coverage of the point_to_line function in the class CGM in pycgm.py, defined as
        point_to_line(point, start, end)

        This test takes 4 parameters:
        point, start, end : ndarray
            1x3 numpy arrays representing the XYZ coordinates of a point.
            `point` is a point not on the line.
            `start` and `end` form a line.
        expected (dist, nearest, point) : tuple
            `dist` is the closest distance from the point to the line.
            `nearest` is the closest point on the line from `point`.
            It is represented as a 1x3 array.
            `point` is the original point not on the line.

        This test is checking to make sure the distance from the point `point` to the line formed by the points
        `start` and `end` is calculated correctly.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when point and end are composed of lists of ints, numpy arrays of ints,
        lists of floats, and numpy arrays of floats. start was kept as numpy arrays as lists would cause errors from
        lines like the following in pycgm.py as lists cannot be added together:
        line_vector = end - start
        """
        result = CGM.point_to_line(point, start, end)
        np.testing.assert_equal(len(result), 3)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

    @pytest.mark.parametrize(["lhjc", "rhjc", "axis", "expected"], [
        # Test from running sample data
        ([308, 322, 937], np.array([182, 339, 935]), [rand_coor, rand_coor, rand_coor, [251, 391, 1033]],
         [[245, 330.5, 936], [271.06445299, 371.10239489, 1043.26924277]]),
        # Test with zeros for all params
        ([0, 0, 0], np.array([0, 0, 0]), [rand_coor, rand_coor, rand_coor, [0, 0, 0]], [[0, 0, 0], nan_3d]),
        # Testing when values are added to lhjc
        ([7, 9, -2], np.array([0, 0, 0]), [rand_coor, rand_coor, rand_coor, [0, 0, 0]], [[3.5, 4.5, -1], nan_3d]),
        # Testing when values are added to rhjc
        ([0, 0, 0], np.array([8, 4, 5]), [rand_coor, rand_coor, rand_coor, [0, 0, 0]], [[4, 2, 2.5], nan_3d]),
        # Testing when values are added to axis
        ([0, 0, 0], np.array([0, 0, 0]), [rand_coor, rand_coor, rand_coor, [5, -8, 4]], [[0, 0, 0], [0, 0, 0]]),
        # Testing when values are added to lhjc and rhjc
        ([7, 9, -2], np.array([8, 4, 5]), [rand_coor, rand_coor, rand_coor, [0, 0, 0]], [[7.5, 6.5, 1.5], nan_3d]),
        # Testing when values are added to lhjc and axis
        ([7, 9, -2], np.array([0, 0, 0]), [rand_coor, rand_coor, rand_coor, [5, -8, 4]],
         [[3.5, 4.5, -1], [8.72479779, -3.85967646, 3.17983823]]),
        # Testing when values are added to rhjc and axis
        ([0, 0, 0], np.array([8, 4, 5]), [rand_coor, rand_coor, rand_coor, [5, -8, 4]],
         [[4, 2, 2.5], [8.625, -5.4, 6.2]]),
        # Testing when values are added to lhjc, rhjc, and axis
        ([7, 9, -2], np.array([8, 4, 5]), [rand_coor, rand_coor, rand_coor, [5, -8, 4]],
         [[7.5, 6.5, 1.5], [11.40883843, 0.24585852, 4.62707074]]),
        # Testing that when lhjc and axis are composed of lists of ints
        ([7, 9, -2], np.array([8, 4, 5]), [rand_coor, rand_coor, rand_coor, [5, -8, 4]],
         [[7.5, 6.5, 1.5], [11.40883843, 0.24585852, 4.62707074]]),
        # Testing that when lhjc and axis are composed of numpy arrays of ints
        (np.array([7, 9, -2], dtype='int'), np.array([8, 4, 5], dtype='int'),
         np.array([rand_coor, rand_coor, rand_coor, [5, -8, 4]], dtype='int'),
         [[7.5, 6.5, 1.5], [11.40883843, 0.24585852, 4.62707074]]),
        # Testing that when lhjc and axis are composed of lists of floats
        ([7.0, 9.0, -2.0], np.array([8.0, 4.0, 5.0]), [rand_coor, rand_coor, rand_coor, [5.0, -8.0, 4.0]],
         [[7.5, 6.5, 1.5], [11.40883843, 0.24585852, 4.62707074]]),
        # Testing that when lhjc and axis are composed of numpy arrays of floats
        (np.array([7.0, 9.0, -2.0], dtype='int'), np.array([8.0, 4.0, 5.0], dtype='float'),
         np.array([rand_coor, rand_coor, rand_coor, [5.0, -8.0, 4.0]], dtype='float'),
         [[7.5, 6.5, 1.5], [11.40883843, 0.24585852, 4.62707074]])])
    def test_find_l5(self, lhjc, rhjc, axis, expected):
        """
        This test provides coverage of the find_l5 function in the class CGM in pycgm.py, defined as find_l5(lhjc,
        rhjc, axis)

        This test takes 4 parameters:
        lhjc, rhjc : ndarray
            1x3 ndarray giving the XYZ coordinates of the LHJC and RHJC
            markers respectively.
        axis : ndarray
            Numpy array containing 4 1x3 arrays of pelvis or thorax origin, x-axis, y-axis,
            and z-axis. Only the z-axis affects the estimated L5 result.
        expected: ndarray
            `mid_hip` is a 1x3 ndarray giving the XYZ coordinates of the middle of the LHJC and RHJC markers. `l5` is
            a 1x3 ndarray giving the estimated XYZ coordinates of the L5 marker.

        This test is checking to make sure the L5 marker position is calculated correctly given the pelvis or thorax
        axis.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when lhjc and axis are composed of lists of ints, numpy arrays of ints,
        lists of floats, and numpy arrays of floats. rhjc was kept as numpy arrays as lists would cause errors from
        lines like the following in pycgm.py as lists cannot be added together:
        mid_hip = (lhjc + rhjc) / 2.0
        """
        result = CGM.find_l5(lhjc, rhjc, axis)
        np.testing.assert_almost_equal(result, expected, rounding_precision)
