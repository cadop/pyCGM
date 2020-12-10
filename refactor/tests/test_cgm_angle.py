#!/usr/bin/python
# -*- coding: utf-8 -*-

from mock import patch
import numpy as np
import pytest

from refactor.pycgm import CGM

rounding_precision = 8


class TestCGMLowerBodyAngle:
    """
    This class tests the lower body angle functions in the class CGM in pycgm.py:
    pelvis_angle_calc
    hip_angle_calc
    knee_angle_calc
    ankle_angle_calc
    foot_angle_calc
    """

    nan_3d = [np.nan, np.nan, np.nan]
    rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]

    @pytest.mark.parametrize(["global_axis", "pelvis_axis", "mock_return_val", "expected_mock_args", "expected"], [
        # Test from running sample data
        (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
         np.array([[251.60830688, 391.74131774, 1032.89349365], [251.74063624, 392.72694720, 1032.78850073],
                   [250.61711554, 391.87232862, 1032.87410630], [251.60295335, 391.84795133, 1033.88777762]]),
         [-0.30849508, -6.12129284, 7.57143134],
         [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          [[0.13232936, 0.98562946, -0.10499292], [-0.99119134, 0.13101088, -0.01938735],
           [-0.00535353, 0.10663359, 0.99428397]]],
         [-0.30849508, - 6.12129284, 7.57143134]),
        # Test with zeros for all params
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [0, 0, 0],
         np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
         np.array([0, 0, 0])),
        # Testing when values are added to global_axis
        (np.array([[3, 8, 6], [6, -5, -9], [0, -8, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [0, 0, 0],
         np.array([[[3, 8, 6], [6, -5, -9], [0, -8, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
         np.array([0, 0, 0])),
        # Testing when values are added to pelvis origin
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[3, 1, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [0, 0, 0],
         np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-3, -1, -2], [-3, -1, -2], [-3, -1, -2]]]),
         np.array([0, 0, 0])),
        # Testing when values are added to pelvis x axis
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [-6, -3, -7], [0, 0, 0], [0, 0, 0]]),
         [0, 0, 0],
         np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-6, -3, -7], [0, 0, 0], [0, 0, 0]]]),
         np.array([0, 0, 0])),
        # Testing when values are added to pelvis y axis
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [4, -8, 7], [0, 0, 0]]),
         [0, 0, 0],
         np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [4, -8, 7], [0, 0, 0]]]),
         np.array([0, 0, 0])),
        # Testing when values are added to pelvis z axis
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [5, -7, -9]]),
         [0, 0, 0],
         np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [5, -7, -9]]]),
         np.array([0, 0, 0])),
        # Testing when values are added to pelvis x, y, z axes
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [-6, -3, -7], [4, -8, 7], [5, -7, -9]]),
         [0, 0, 0],
         np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-6, -3, -7], [4, -8, 7], [5, -7, -9]]]),
         np.array([0, 0, 0])),
        # Testing when values are added to pelvis_axis
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[3, 1, 2], [-6, -3, -7], [4, -8, 7], [5, -7, -9]]),
         [0, 0, 0],
         np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-9, -4, -9], [1, -9, 5], [2, -8, -11]]]),
         np.array([0, 0, 0])),
        # Testing when values are added to mock_return_val
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [6, 2, -7],
         np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
         np.array([6, 2, -7])),
        # Testing when values are added to global_axis and pelvis_axis
        (np.array([[3, 8, 6], [6, -5, -9], [0, -8, 0]]),
         np.array([[3, 1, 2], [-6, -3, -7], [4, -8, 7], [5, -7, -9]]),
         [0, 0, 0],
         np.array([[[3, 8, 6], [6, -5, -9], [0, -8, 0]], [[-9, -4, -9], [1, -9, 5], [2, -8, -11]]]),
         np.array([0, 0, 0])),
        # Testing when values are added to global_axis, pelvis_axis, and mock_return_val
        (np.array([[3, 8, 6], [6, -5, -9], [0, -8, 0]]),
         np.array([[3, 1, 2], [-6, -3, -7], [4, -8, 7], [5, -7, -9]]),
         [6, 2, -7],
         np.array([[[3, 8, 6], [6, -5, -9], [0, -8, 0]], [[-9, -4, -9], [1, -9, 5], [2, -8, -11]]]),
         np.array([6, 2, -7])),
        # Testing that when global_axis and pelvis_axis are composed of lists of ints
        ([[3, 8, 6], [6, -5, -9], [0, -8, 0]],
         [[3, 1, 2], [-6, -3, -7], [4, -8, 7], [5, -7, -9]],
         [6, 2, -7],
         np.array([[[3, 8, 6], [6, -5, -9], [0, -8, 0]], [[-9, -4, -9], [1, -9, 5], [2, -8, -11]]]),
         np.array([6, 2, -7])),
        # Testing that when global_axis and pelvis_axis are composed of numpy arrays of ints
        (np.array([[3, 8, 6], [6, -5, -9], [0, -8, 0]], dtype='int'),
         np.array([[3, 1, 2], [-6, -3, -7], [4, -8, 7], [5, -7, -9]], dtype='int'),
         [6, 2, -7],
         np.array([[[3, 8, 6], [6, -5, -9], [0, -8, 0]], [[-9, -4, -9], [1, -9, 5], [2, -8, -11]]]),
         np.array([6, 2, -7])),
        # Testing that when global_axis and pelvis_axis are composed of lists of floats
        ([[3.0, 8.0, 6.0], [6.0, -5.0, -9.0], [0.0, -8.0, 0.0]],
         [[3.0, 1.0, 2.0], [-6.0, -3.0, -7.0], [4.0, -8.0, 7.0], [5.0, -7.0, -9.0]],
         [6, 2, -7],
         np.array([[[3, 8, 6], [6, -5, -9], [0, -8, 0]], [[-9, -4, -9], [1, -9, 5], [2, -8, -11]]]),
         np.array([6, 2, -7])),
        # Testing that when global_axis and pelvis_axis are composed of numpy arrays of floats
        (np.array([[3, 8, 6], [6, -5, -9], [0, -8, 0]], dtype='float'),
         np.array([[3, 1, 2], [-6, -3, -7], [4, -8, 7], [5, -7, -9]], dtype='float'),
         [6, 2, -7],
         np.array([[[3, 8, 6], [6, -5, -9], [0, -8, 0]], [[-9, -4, -9], [1, -9, 5], [2, -8, -11]]]),
         np.array([6, 2, -7]))])
    def test_pelvis_angle_calc(self, global_axis, pelvis_axis, mock_return_val, expected_mock_args, expected):
        """
        This test provides coverage of the pelvis_angle_calc function in the class CGM in pycgm.py, defined as
        pelvis_angle_calc(global_axis, pelvis_axis)

        This test takes 5 parameters:
        global_axis : ndarray
            A 3x3 ndarray representing the global coordinate system.
        pelvis_axis : ndarray
            A 4x3 ndarray containing the origin and three unit vectors of the pelvis axis.
        mock_return_val : list
            The value to be returned by the mock for get_angle
        expected_mock_args : list
            The expected arguments used to call the mocked function, get_angle
        expected : array
            A 1x3 ndarray containing the flexion, abduction, and rotation angles of the pelvis.

        This test is checking to make sure the global pelvis angle is calculated correctly given the input
        parameters. This tests mocks get_angle to make sure the correct parameters are being passed into it given the
        parameters passed into pelvis_angle_calc, expected_mock_args, and to also ensure that pelvis_angle_calc
        returns the correct value considering the return value of get_angle, mock_return_val.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when global_axis and pelvis_axis are composed of lists of ints,
        numpy arrays of ints, list of floats, and numpy arrays of floats.
        """
        with patch.object(CGM, 'get_angle', return_value=mock_return_val) as mock_get_angle:
            result = CGM.pelvis_angle_calc(global_axis, pelvis_axis)

        # Asserting that there was only 1 call to get_angle
        np.testing.assert_equal(mock_get_angle.call_count, 1)

        # Asserting that the correct params were sent in the call to get_angle
        np.testing.assert_almost_equal(expected_mock_args, mock_get_angle.call_args_list[0][0], rounding_precision)

        # Asserting that pelvis_angle_calc returned the correct result given the return value given by mocked get_angle
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["hip_axis", "knee_axis", "mock_return_val", "expected_mock_args", "expected"], [
        # Test from running sample data
        (np.array(
            [nan_3d, nan_3d, [245.47574167, 331.11787135, 936.75939593], [245.60807102, 332.10350081, 936.65440301],
             [244.48455032, 331.24888223, 936.74000858], [245.47038814, 331.22450494, 937.75367990]]),
         np.array([[364.17774613, 292.17051731, 515.19181496], [364.61959153, 293.06758353, 515.18513093],
                   [363.29019771, 292.60656648, 515.04309095], [364.04724540, 292.24216263, 516.18067111],
                   [143.55478579, 279.90370346, 524.78408753], [143.65611281, 280.88685896, 524.63197541],
                   [142.56434499, 280.01777942, 524.86163553], [143.64837986, 280.04650380, 525.76940383]]),
         [[-2.91422854, -6.86706805, 108.82100186], [2.86020432, 5.34565068, 88.19743763]],
         [[[[0.13232935, 0.98562946, -0.10499292], [-0.99119135, 0.13101088, -0.01938735],
            [-0.00535353, 0.10663359, 0.99428397]],
           [[0.4418454, 0.89706622, -0.00668403], [-0.88754842, 0.43604917, -0.14872401],
            [-0.13050073, 0.07164532, 0.98885615]]], [
              [[0.13232935, 0.98562946, -0.10499292], [-0.99119135, 0.13101088, -0.01938735],
               [-0.00535353, 0.10663359, 0.99428397]],
              [[0.10132702, 0.9831555, -0.15211212], [-0.9904408, 0.11407596, 0.077548],
               [0.09359407, 0.14280034, 0.9853163]]]],
         [[2.91422854, -6.86706805, -18.82100186], [-2.86020432, -5.34565068, -1.80256237]]),
        # Test with zeros for all params
        (np.array([rand_coor, rand_coor, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to hip origin
        (np.array([rand_coor, rand_coor, [-6, -2, -9], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[6, 2, 9], [6, 2, 9], [6, 2, 9]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[6, 2, 9], [6, 2, 9], [6, 2, 9]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to hip x, y, z axes
        (np.array([rand_coor, rand_coor, [0, 0, 0], [3, 4, 1], [-9, 1, -1], [-2, -5, 4]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[3, 4, 1], [-9, 1, -1], [-2, -5, 4]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[3, 4, 1], [-9, 1, -1], [-2, -5, 4]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to hip_axis
        (np.array([rand_coor, rand_coor, [-6, -2, -9], [3, 4, 1], [-9, 1, -1], [-2, -5, 4]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to right knee origin and x, y, z axes
        (np.array([rand_coor, rand_coor, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[5, -4, -3], [1, -6, 3], [8, -7, 1], [-9, 3, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-4, -2, 6], [3, -3, 4], [-14, 7, 2]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to left knee origin and x, y, z axes
        (np.array([rand_coor, rand_coor, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [-7, -8, -1], [-1, 8, 9], [-7, 5, -3], [2, 0, -9]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[6, 16, 10], [0, 13, -2], [9, 8, -8]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to knee_axis
        (np.array([rand_coor, rand_coor, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array(
             [[5, -4, -3], [1, -6, 3], [8, -7, 1], [-9, 3, -1], [-7, -8, -1], [-1, 8, 9], [-7, 5, -3], [2, 0, -9]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-4, -2, 6], [3, -3, 4], [-14, 7, 2]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[6, 16, 10], [0, 13, -2], [9, 8, -8]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to mock_return_val
        (np.array([rand_coor, rand_coor, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[1, -45, 145], [-29, -165, 157]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[-1, -45, -55], [29, 165, 67]]),
        # Testing when values are added to hip_axis and knee_axis
        (np.array([rand_coor, rand_coor, [-6, -2, -9], [3, 4, 1], [-9, 1, -1], [-2, -5, 4]]),
         np.array(
             [[5, -4, -3], [1, -6, 3], [8, -7, 1], [-9, 3, -1], [-7, -8, -1], [-1, 8, 9], [-7, 5, -3], [2, 0, -9]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[-4, -2, 6], [3, -3, 4], [-14, 7, 2]]],
          [[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[6, 16, 10], [0, 13, -2], [9, 8, -8]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to hip_axis, knee_axis, and mock_return_val
        (np.array([rand_coor, rand_coor, [-6, -2, -9], [3, 4, 1], [-9, 1, -1], [-2, -5, 4]]),
         np.array(
             [[5, -4, -3], [1, -6, 3], [8, -7, 1], [-9, 3, -1], [-7, -8, -1], [-1, 8, 9], [-7, 5, -3], [2, 0, -9]]),
         [[1, -45, 145], [-29, -165, 157]],
         [[[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[-4, -2, 6], [3, -3, 4], [-14, 7, 2]]],
          [[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[6, 16, 10], [0, 13, -2], [9, 8, -8]]]],
         [[-1, -45, -55], [29, 165, 67]]),
        # Testing that when hip_axis and knee_axis are composed of lists of ints
        ([rand_coor, rand_coor, [-6, -2, -9], [3, 4, 1], [-9, 1, -1], [-2, -5, 4]],
         [[5, -4, -3], [1, -6, 3], [8, -7, 1], [-9, 3, -1], [-7, -8, -1], [-1, 8, 9], [-7, 5, -3], [2, 0, -9]],
         [[1, -45, 145], [-29, -165, 157]],
         [[[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[-4, -2, 6], [3, -3, 4], [-14, 7, 2]]],
          [[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[6, 16, 10], [0, 13, -2], [9, 8, -8]]]],
         [[-1, -45, -55], [29, 165, 67]]),
        # Testing that when hip_axis and knee_axis are composed of numpy arrays of ints
        (np.array([rand_coor, rand_coor, [-6, -2, -9], [3, 4, 1], [-9, 1, -1], [-2, -5, 4]], dtype='int'),
         np.array([[5, -4, -3], [1, -6, 3], [8, -7, 1], [-9, 3, -1], [-7, -8, -1], [-1, 8, 9], [-7, 5, -3], [2, 0, -9]],
                  dtype='int'),
         [[1, -45, 145], [-29, -165, 157]],
         [[[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[-4, -2, 6], [3, -3, 4], [-14, 7, 2]]],
          [[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[6, 16, 10], [0, 13, -2], [9, 8, -8]]]],
         [[-1, -45, -55], [29, 165, 67]]),
        # Testing that when hip_axis and knee_axis are composed of lists of floats
        ([rand_coor, rand_coor, [-6.0, -2.0, -9.0], [3.0, 4.0, 1.0], [-9.0, 1.0, -1.0], [-2.0, -5.0, 4.0]],
         [[5.0, -4.0, -3.0], [1.0, -6.0, 3.0], [8.0, -7.0, 1.0], [-9.0, 3.0, -1.0], [-7.0, -8.0, -1.0],
          [-1.0, 8.0, 9.0], [-7.0, 5.0, -3.0], [2.0, 0.0, -9.0]],
         [[1, -45, 145], [-29, -165, 157]],
         [[[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[-4, -2, 6], [3, -3, 4], [-14, 7, 2]]],
          [[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[6, 16, 10], [0, 13, -2], [9, 8, -8]]]],
         [[-1, -45, -55], [29, 165, 67]]),
        # Testing that when hip_axis and knee_axis are composed of numpy arrays of floats
        (np.array([rand_coor, rand_coor, [-6.0, -2.0, -9.0], [3.0, 4.0, 1.0], [-9.0, 1.0, -1.0], [-2.0, -5.0, 4.0]],
                  dtype='float'),
         np.array([[5.0, -4.0, -3.0], [1.0, -6.0, 3.0], [8.0, -7.0, 1.0], [-9.0, 3.0, -1.0], [-7.0, -8.0, -1.0],
                   [-1.0, 8.0, 9.0], [-7.0, 5.0, -3.0], [2.0, 0.0, -9.0]], dtype='float'),
         [[1, -45, 145], [-29, -165, 157]],
         [[[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[-4, -2, 6], [3, -3, 4], [-14, 7, 2]]],
          [[[9, 6, 10], [-3, 3, 8], [4, -3, 13]], [[6, 16, 10], [0, 13, -2], [9, 8, -8]]]],
         [[-1, -45, -55], [29, 165, 67]])])
    def test_hip_angle_calc(self, hip_axis, knee_axis, mock_return_val, expected_mock_args, expected):
        """
        This test provides coverage of the hip_angle_calc function in the class CGM in pycgm.py, defined as
        hip_angle_calc(hip_axis, knee_axis)

        This test takes 5 parameters:
        hip_axis : ndarray
            A 6x3 ndarray containing the right and left hip joint centers, the hip origin, and the hip unit vectors.
        knee_axis : ndarray
            An 8x3 ndarray containing the right knee origin, right knee unit vectors, left knee origin, and left knee
            unit vectors.
        mock_return_val : list
            The value to be returned by the mock for get_angle
        expected_mock_args : list
            The expected arguments used to call the mocked function, get_angle
        expected : array
            A 2x3 ndarray containing the flexion, abduction, and rotation angles of the right and left hip.

        This test is checking to make sure the hip angle is calculated correctly given the input parameters. This
        tests mocks get_angle to make sure the correct parameters are being passed into it given the parameters
        passed into hip_angle_calc, expected_mock_args, and to also ensure that hip_angle_calc returns the correct
        value considering the return value of get_angle, mock_return_val.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when hip_axis and knee_axis are composed of lists of ints, numpy arrays of
        ints, list of floats, and numpy arrays of floats.
        """
        with patch.object(CGM, 'get_angle', side_effect=mock_return_val) as mock_get_angle:
            result = CGM.hip_angle_calc(hip_axis, knee_axis)

        # Asserting that there was only 1 call to get_angle
        np.testing.assert_equal(mock_get_angle.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to get_angle
        np.testing.assert_almost_equal(expected_mock_args[0], mock_get_angle.call_args_list[0][0], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to get_angle
        np.testing.assert_almost_equal(expected_mock_args[1], mock_get_angle.call_args_list[1][0], rounding_precision)

        # Asserting that hip_angle_calc returned the correct result given the return value given by mocked get_angle
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["knee_axis", "ankle_axis", "mock_return_val", "expected_mock_args", "expected"], [
        # Test from running sample data
        (np.array([[364.17774614, 292.17051722, 515.19181496], [364.61959153, 293.06758353, 515.18513093],
                   [363.29019771, 292.60656648, 515.04309095], [364.04724541, 292.24216264, 516.18067112],
                   [143.55478579, 279.90370346, 524.78408753], [143.65611282, 280.88685896, 524.63197541],
                   [142.56434499, 280.01777943, 524.86163553], [143.64837987, 280.04650381, 525.76940383]]),
         np.array([[393.76181608, 247.67829633, 87.73775041], [394.48171575, 248.37201348, 87.71536800],
                   [393.07114384, 248.39110006, 87.61575574], [393.69314056, 247.78157916, 88.73002876],
                   [98.74901939, 219.46930221, 80.63068160], [98.47494966, 220.42553803, 80.52821783],
                   [97.79246671, 219.20927275, 80.76255901], [98.84848169, 219.60345781, 81.61663775]]),
         [[3.19436865, 2.38341045, 109.47591616], [-0.45848726, 0.3866728, 68.12419149]],
         [[[[0.44184539, 0.89706631, -0.00668403], [-0.88754843, 0.43604926, -0.14872401],
            [-0.13050073, 0.07164542, 0.98885616]],
           [[0.71989967, 0.69371715, -0.02238241], [-0.69067224, 0.71280373, -0.12199467],
            [-0.06867552, 0.10328283, 0.99227835]]], [
              [[0.10132703, 0.9831555, -0.15211212], [-0.9904408, 0.11407597, 0.077548],
               [0.09359408, 0.14280035, 0.9853163]],
              [[-0.27406973, 0.95623582, -0.10246377], [-0.95655268, -0.26002946, 0.13187741],
               [0.0994623, 0.1341556, 0.98595615]]]],
         [[3.19436865, 2.38341045, -19.47591616], [-0.45848726, -0.3866728, -21.87580851]]),
        # Test with zeros for all params
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to right knee origin and x, y, z axes
        (np.array([[6, 5, -7], [9, 8, -3], [3, 5, -3], [-8, 3, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[3, 3, 4], [-3, 0, 4], [-14, -2, 6]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to left knee origin and x, y, z axes
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 4, 3], [5, 6, -5], [1, -7, 3], [0, -4, 4]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[5, 2, -8], [1, -11, 0], [0, -8, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to knee_axis
        (np.array([[6, 5, -7], [9, 8, -3], [3, 5, -3], [-8, 3, -1], [0, 4, 3], [5, 6, -5], [1, -7, 3], [0, -4, 4]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[3, 3, 4], [-3, 0, 4], [-14, -2, 6]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[5, 2, -8], [1, -11, 0], [0, -8, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to right ankle origin and x, y, z axes
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 4, 1], [5, 8, 9], [0, 2, 5], [-9, 0, -8], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[5, 4, 8], [0, -2, 4], [-9, -4, -9]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to left ankle origin and x, y, z axes
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [4, 1, 0], [9, 4, 8], [4, -4, -4], [-5, -3, -5]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[5, 3, 8], [0, -5, -4], [-9, -4, -5]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to ankle_axis
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 4, 1], [5, 8, 9], [0, 2, 5], [-9, 0, -8], [4, 1, 0], [9, 4, 8], [4, -4, -4], [-5, -3, -5]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[5, 4, 8], [0, -2, 4], [-9, -4, -9]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[5, 3, 8], [0, -5, -4], [-9, -4, -5]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to mock_return_val
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[66, -109, 121], [-56, 175, -32]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[66, -109, -31], [-56, -175, -122]]),
        # Testing when values are added to knee_axis and ankle_axis
        (np.array([[6, 5, -7], [9, 8, -3], [3, 5, -3], [-8, 3, -1], [0, 4, 3], [5, 6, -5], [1, -7, 3], [0, -4, 4]]),
         np.array([[0, 4, 1], [5, 8, 9], [0, 2, 5], [-9, 0, -8], [4, 1, 0], [9, 4, 8], [4, -4, -4], [-5, -3, -5]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[3, 3, 4], [-3, 0, 4], [-14, -2, 6]], [[5, 4, 8], [0, -2, 4], [-9, -4, -9]]],
          [[[5, 2, -8], [1, -11, 0], [0, -8, 1]], [[5, 3, 8], [0, -5, -4], [-9, -4, -5]]]],
         [[0, 0, 90], [0, 0, -90]]),
        # Testing when values are added to knee_axis, ankle_axis, and mock_return_val
        (np.array([[6, 5, -7], [9, 8, -3], [3, 5, -3], [-8, 3, -1], [0, 4, 3], [5, 6, -5], [1, -7, 3], [0, -4, 4]]),
         np.array([[0, 4, 1], [5, 8, 9], [0, 2, 5], [-9, 0, -8], [4, 1, 0], [9, 4, 8], [4, -4, -4], [-5, -3, -5]]),
         [[66, -109, 121], [-56, 175, -32]],
         [[[[3, 3, 4], [-3, 0, 4], [-14, -2, 6]], [[5, 4, 8], [0, -2, 4], [-9, -4, -9]]],
          [[[5, 2, -8], [1, -11, 0], [0, -8, 1]], [[5, 3, 8], [0, -5, -4], [-9, -4, -5]]]],
         [[66, -109, -31], [-56, -175, -122]]),
        # Testing that when knee_axis and ankle_axis are composed of lists of ints
        ([[6, 5, -7], [9, 8, -3], [3, 5, -3], [-8, 3, -1], [0, 4, 3], [5, 6, -5], [1, -7, 3], [0, -4, 4]],
         [[0, 4, 1], [5, 8, 9], [0, 2, 5], [-9, 0, -8], [4, 1, 0], [9, 4, 8], [4, -4, -4], [-5, -3, -5]],
         [[66, -109, 121], [-56, 175, -32]],
         [[[[3, 3, 4], [-3, 0, 4], [-14, -2, 6]], [[5, 4, 8], [0, -2, 4], [-9, -4, -9]]],
          [[[5, 2, -8], [1, -11, 0], [0, -8, 1]], [[5, 3, 8], [0, -5, -4], [-9, -4, -5]]]],
         [[66, -109, -31], [-56, -175, -122]]),
        # Testing that when knee_axis and ankle_axis are composed of numpy arrays of ints
        (np.array([[6, 5, -7], [9, 8, -3], [3, 5, -3], [-8, 3, -1], [0, 4, 3], [5, 6, -5], [1, -7, 3], [0, -4, 4]],
                  dtype='int'),
         np.array([[0, 4, 1], [5, 8, 9], [0, 2, 5], [-9, 0, -8], [4, 1, 0], [9, 4, 8], [4, -4, -4], [-5, -3, -5]],
                  dtype='int'),
         [[66, -109, 121], [-56, 175, -32]],
         [[[[3, 3, 4], [-3, 0, 4], [-14, -2, 6]], [[5, 4, 8], [0, -2, 4], [-9, -4, -9]]],
          [[[5, 2, -8], [1, -11, 0], [0, -8, 1]], [[5, 3, 8], [0, -5, -4], [-9, -4, -5]]]],
         [[66, -109, -31], [-56, -175, -122]]),
        # Testing that when knee_axis and ankle_axis are composed of lists of floats
        ([[6.0, 5.0, -7.0], [9.0, 8.0, -3.0], [3.0, 5.0, -3.0], [-8.0, 3.0, -1.0], [0.0, 4.0, 3.0], [5.0, 6.0, -5.0],
          [1.0, -7.0, 3.0], [0.0, -4.0, 4.0]],
         [[0.0, 4.0, 1.0], [5.0, 8.0, 9.0], [0.0, 2.0, 5.0], [-9.0, 0.0, -8.0], [4.0, 1.0, 0.0], [9.0, 4.0, 8.0],
          [4.0, -4.0, -4.0], [-5.0, -3.0, -5.0]],
         [[66, -109, 121], [-56, 175, -32]],
         [[[[3, 3, 4], [-3, 0, 4], [-14, -2, 6]], [[5, 4, 8], [0, -2, 4], [-9, -4, -9]]],
          [[[5, 2, -8], [1, -11, 0], [0, -8, 1]], [[5, 3, 8], [0, -5, -4], [-9, -4, -5]]]],
         [[66, -109, -31], [-56, -175, -122]]),
        # Testing that when knee_axis and ankle_axis are composed of numpy arrays of floats
        (np.array(
            [[6.0, 5.0, -7.0], [9.0, 8.0, -3.0], [3.0, 5.0, -3.0], [-8.0, 3.0, -1.0], [0.0, 4.0, 3.0], [5.0, 6.0, -5.0],
             [1.0, -7.0, 3.0], [0.0, -4.0, 4.0]], dtype='float'),
         np.array(
             [[0.0, 4.0, 1.0], [5.0, 8.0, 9.0], [0.0, 2.0, 5.0], [-9.0, 0.0, -8.0], [4.0, 1.0, 0.0], [9.0, 4.0, 8.0],
              [4.0, -4.0, -4.0], [-5.0, -3.0, -5.0]], dtype='float'),
         [[66, -109, 121], [-56, 175, -32]],
         [[[[3, 3, 4], [-3, 0, 4], [-14, -2, 6]], [[5, 4, 8], [0, -2, 4], [-9, -4, -9]]],
          [[[5, 2, -8], [1, -11, 0], [0, -8, 1]], [[5, 3, 8], [0, -5, -4], [-9, -4, -5]]]],
         [[66, -109, -31], [-56, -175, -122]])])
    def test_knee_angle_calc(self, knee_axis, ankle_axis, mock_return_val, expected_mock_args, expected):
        """
        This test provides coverage of the knee_angle_calc function in the class CGM in pycgm.py, defined as
        knee_angle_calc(knee_axis, ankle_axis)

        This test takes 5 parameters:
        knee_axis : ndarray
            An 8x3 ndarray containing the right knee origin, right knee unit vectors, left knee origin, and left knee
            unit vectors.
        ankle_axis : ndarray
            An 8x3 ndarray containing the right ankle origin, right ankle unit vectors, left ankle origin,
            and left ankle unit vectors.
        mock_return_val : list
            The value to be returned by the mock for get_angle
        expected_mock_args : list
            The expected arguments used to call the mocked function, get_angle
        expected : array
            A 2x3 ndarray containing the flexion, abduction, and rotation angles of the right and left ankle.

        This test is checking to make sure the knee angle is calculated correctly given the input parameters. This
        tests mocks get_angle to make sure the correct parameters are being passed into it given the parameters
        passed into knee_angle_calc, expected_mock_args, and to also ensure that knee_angle_calc returns the correct
        value considering the return value of get_angle, mock_return_val.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when knee_axis and ankle_axis are composed of lists of ints, numpy arrays
        of ints, list of floats, and numpy arrays of floats.
        """
        with patch.object(CGM, 'get_angle', side_effect=mock_return_val) as mock_get_angle:
            result = CGM.knee_angle_calc(knee_axis, ankle_axis)

        # Asserting that there was only 1 call to get_angle
        np.testing.assert_equal(mock_get_angle.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to get_angle
        np.testing.assert_almost_equal(expected_mock_args[0], mock_get_angle.call_args_list[0][0], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to get_angle
        np.testing.assert_almost_equal(expected_mock_args[1], mock_get_angle.call_args_list[1][0], rounding_precision)

        # Asserting that knee_angle_calc returned the correct result given the return value given by mocked get_angle
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["ankle_axis", "foot_axis", "mock_return_val", "expected_mock_args", "expected"], [
        # Test from running sample data
        (np.array([[393.76181608, 247.67829633, 87.73775041], [394.48171575, 248.37201348, 87.71536800],
                   [393.07114384, 248.39110006, 87.61575574], [393.69314056, 247.78157916, 88.73002876],
                   [98.74901939, 219.46930221, 80.63068160], [98.47494966, 220.42553803, 80.52821783],
                   [97.79246671, 219.20927275, 80.76255901], [98.84848169, 219.60345781, 81.61663775]]),
         np.array([[442.81997681, 381.62280273, 42.66047668], [442.84624127, 381.65130240, 43.65972538],
                   [441.87735056, 381.95630350, 42.67574106], [442.48716163, 380.68048378, 42.69610044],
                   [39.43652725, 382.44522095, 41.78911591], [39.56652626, 382.50901000, 42.77857597],
                   [38.49313328, 382.14606841, 41.93234850], [39.74166342, 381.49315020, 41.81040458]]),
         [[-92.50533765, 26.4981019, 97.68822002], [-94.38467038, 2.37873795, 90.59929708]],
         [[[[0.71989967, 0.69371715, -0.02238241], [-0.69067224, 0.71280373, -0.12199467],
            [-0.06867552, 0.10328283, 0.99227835]],
           [[0.02626446, 0.02849967, 0.9992487], [-0.94262625, 0.33350077, 0.01526438],
            [-0.33281518, -0.94231895, 0.03562376]]], [
              [[-0.27406973, 0.95623582, -0.10246377], [-0.95655268, -0.26002946, 0.13187741],
               [0.0994623, 0.1341556, 0.98595615]],
              [[0.12999901, 0.06378905, 0.98946006], [-0.94339397, -0.29915254, 0.14323259],
               [0.30513617, -0.95207075, 0.02128867]]]],
         [[2.50533765, -7.68822002, 26.4981019], [4.38467038, 0.59929708, -2.37873795]]),
        # Test with zeros for all params
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[-90, 90, 0], [-90, -90, 0]]),
        # Testing when values are added to right ankle origin and x, y, z axes
        (np.array([[2, 3, 0], [6, -6, -5], [1, -1, 4], [2, 1, -3], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[4, -9, -5], [-1, -4, 4], [0, -2, -3]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[-90, 90, 0], [-90, -90, 0]]),
        # Testing when values are added to left ankle origin and x, y, z axes
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [5, 0, -2], [6, 9, -3], [-7, -2, 6], [6, -5, -7]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[1, 9, -1], [-12, -2, 8], [1, -5, -5]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[-90, 90, 0], [-90, -90, 0]]),
        # Testing when values are added to ankle_axis
        (np.array([[2, 3, 0], [6, -6, -5], [1, -1, 4], [2, 1, -3], [5, 0, -2], [6, 9, -3], [-7, -2, 6], [6, -5, -7]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[4, -9, -5], [-1, -4, 4], [0, -2, -3]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[1, 9, -1], [-12, -2, 8], [1, -5, -5]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[-90, 90, 0], [-90, -90, 0]]),
        # Testing when values are added to right foot origin and x, y, z axes
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[5, -4, -1], [-9, 6, 2], [6, 9, -6], [8, -5, 6], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-14, 10, 3], [1, 13, -5], [3, -1, 7]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[-90, 90, 0], [-90, -90, 0]]),
        # Testing when values are added to left foot origin and x, y, z axes
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [-1, 7, 6], [4, -5, 7], [-3, -2, 6], [2, 7, 7]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[5, -12, 1], [-2, -9, 0], [3, 0, 1]]]],
         [[-90, 90, 0], [-90, -90, 0]]),
        # Testing when values are added to foot_axis
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[5, -4, -1], [-9, 6, 2], [6, 9, -6], [8, -5, 6], [-1, 7, 6], [4, -5, 7], [-3, -2, 6], [2, 7, 7]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-14, 10, 3], [1, 13, -5], [3, -1, 7]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[5, -12, 1], [-2, -9, 0], [3, 0, 1]]]],
         [[-90, 90, 0], [-90, -90, 0]]),
        # Testing when values are added to mock_return_val
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[-131, 8, 27], [27, -47, 81]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[41, 63, 8], [-117, -9, 47]]),
        # Testing when values are added to ankle_axis and foot_axis
        (np.array([[2, 3, 0], [6, -6, -5], [1, -1, 4], [2, 1, -3], [5, 0, -2], [6, 9, -3], [-7, -2, 6], [6, -5, -7]]),
         np.array([[5, -4, -1], [-9, 6, 2], [6, 9, -6], [8, -5, 6], [-1, 7, 6], [4, -5, 7], [-3, -2, 6], [2, 7, 7]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[4, -9, -5], [-1, -4, 4], [0, -2, -3]], [[-14, 10, 3], [1, 13, -5], [3, -1, 7]]],
          [[[1, 9, -1], [-12, -2, 8], [1, -5, -5]], [[5, -12, 1], [-2, -9, 0], [3, 0, 1]]]],
         [[-90, 90, 0], [-90, -90, 0]]),
        # Testing when values are added to ankle_axis, foot_axis, and mock_return_val
        (np.array([[2, 3, 0], [6, -6, -5], [1, -1, 4], [2, 1, -3], [5, 0, -2], [6, 9, -3], [-7, -2, 6], [6, -5, -7]]),
         np.array([[5, -4, -1], [-9, 6, 2], [6, 9, -6], [8, -5, 6], [-1, 7, 6], [4, -5, 7], [-3, -2, 6], [2, 7, 7]]),
         [[-131, 8, 27], [27, -47, 81]],
         [[[[4, -9, -5], [-1, -4, 4], [0, -2, -3]], [[-14, 10, 3], [1, 13, -5], [3, -1, 7]]],
          [[[1, 9, -1], [-12, -2, 8], [1, -5, -5]], [[5, -12, 1], [-2, -9, 0], [3, 0, 1]]]],
         [[41, 63, 8], [-117, -9, 47]]),
        # Testing that when ankle_axis and foot_axis are composed of lists of ints
        ([[2, 3, 0], [6, -6, -5], [1, -1, 4], [2, 1, -3], [5, 0, -2], [6, 9, -3], [-7, -2, 6], [6, -5, -7]],
         [[5, -4, -1], [-9, 6, 2], [6, 9, -6], [8, -5, 6], [-1, 7, 6], [4, -5, 7], [-3, -2, 6], [2, 7, 7]],
         [[-131, 8, 27], [27, -47, 81]],
         [[[[4, -9, -5], [-1, -4, 4], [0, -2, -3]], [[-14, 10, 3], [1, 13, -5], [3, -1, 7]]],
          [[[1, 9, -1], [-12, -2, 8], [1, -5, -5]], [[5, -12, 1], [-2, -9, 0], [3, 0, 1]]]],
         [[41, 63, 8], [-117, -9, 47]]),
        # Testing that when ankle_axis and foot_axis are composed of numpy arrays of ints
        (np.array([[2, 3, 0], [6, -6, -5], [1, -1, 4], [2, 1, -3], [5, 0, -2], [6, 9, -3], [-7, -2, 6], [6, -5, -7]],
                  dtype='int'),
         np.array([[5, -4, -1], [-9, 6, 2], [6, 9, -6], [8, -5, 6], [-1, 7, 6], [4, -5, 7], [-3, -2, 6], [2, 7, 7]],
                  dtype='int'),
         [[-131, 8, 27], [27, -47, 81]],
         [[[[4, -9, -5], [-1, -4, 4], [0, -2, -3]], [[-14, 10, 3], [1, 13, -5], [3, -1, 7]]],
          [[[1, 9, -1], [-12, -2, 8], [1, -5, -5]], [[5, -12, 1], [-2, -9, 0], [3, 0, 1]]]],
         [[41, 63, 8], [-117, -9, 47]]),
        # Testing that when ankle_axis and foot_axis are composed of lists of floats
        ([[2.0, 3.0, 0.0], [6.0, -6.0, -5.0], [1.0, -1.0, 4.0], [2.0, 1.0, -3.0], [5.0, 0.0, -2.0], [6.0, 9.0, -3.0],
          [-7.0, -2.0, 6.0], [6.0, -5.0, -7.0]],
         [[5.0, -4.0, -1.0], [-9.0, 6.0, 2.0], [6.0, 9.0, -6.0], [8.0, -5.0, 6.0], [-1.0, 7.0, 6.0], [4.0, -5.0, 7.0],
          [-3.0, -2.0, 6.0], [2.0, 7.0, 7.0]],
         [[-131, 8, 27], [27, -47, 81]],
         [[[[4, -9, -5], [-1, -4, 4], [0, -2, -3]], [[-14, 10, 3], [1, 13, -5], [3, -1, 7]]],
          [[[1, 9, -1], [-12, -2, 8], [1, -5, -5]], [[5, -12, 1], [-2, -9, 0], [3, 0, 1]]]],
         [[41, 63, 8], [-117, -9, 47]]),
        # Testing that when ankle_axis and foot_axis are composed of numpy arrays of floats
        (np.array(
            [[2.0, 3.0, 0.0], [6.0, -6.0, -5.0], [1.0, -1.0, 4.0], [2.0, 1.0, -3.0], [5.0, 0.0, -2.0], [6.0, 9.0, -3.0],
             [-7.0, -2.0, 6.0], [6.0, -5.0, -7.0]], dtype='float'),
         np.array([[5.0, -4.0, -1.0], [-9.0, 6.0, 2.0], [6.0, 9.0, -6.0], [8.0, -5.0, 6.0], [-1.0, 7.0, 6.0],
                   [4.0, -5.0, 7.0], [-3.0, -2.0, 6.0], [2.0, 7.0, 7.0]], dtype='float'),
         [[-131, 8, 27], [27, -47, 81]],
         [[[[4, -9, -5], [-1, -4, 4], [0, -2, -3]], [[-14, 10, 3], [1, 13, -5], [3, -1, 7]]],
          [[[1, 9, -1], [-12, -2, 8], [1, -5, -5]], [[5, -12, 1], [-2, -9, 0], [3, 0, 1]]]],
         [[41, 63, 8], [-117, -9, 47]])])
    def test_ankle_angle_calc(self, ankle_axis, foot_axis, mock_return_val, expected_mock_args, expected):
        """
        This test provides coverage of the ankle_angle_calc function in the class CGM in pycgm.py, defined as
        ankle_angle_calc(ankle_axis, foot_axis)

        This test takes 5 parameters:
        ankle_axis : ndarray
            An 8x3 ndarray containing the right ankle origin, right ankle unit vectors, left ankle origin,
            and left ankle unit vectors.
        foot_axis : ndarray
            An 8x3 ndarray containing the right foot origin, right foot unit vectors, left foot origin, and left foot
            unit vectors.
        mock_return_val : list
            The value to be returned by the mock for get_angle
        expected_mock_args : list
            The expected arguments used to call the mocked function, get_angle
        expected : ndarray
            A 2x3 ndarray containing the flexion, abduction, and rotation angles of the right and left ankle.

        This test is checking to make sure the ankle angle is calculated correctly given the input parameters. This
        tests mocks get_angle to make sure the correct parameters are being passed into it given the parameters
        passed into ankle_angle_calc, expected_mock_args, and to also ensure that ankle_angle_calc returns the
        correct value considering the return value of get_angle, mock_return_val.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when ankle_axis and foot_axis are composed of lists of ints, numpy arrays
        of ints, list of floats, and numpy arrays of floats.
        """
        with patch.object(CGM, 'get_angle', side_effect=mock_return_val) as mock_get_angle:
            result = CGM.ankle_angle_calc(ankle_axis, foot_axis)

        # Asserting that there was only 1 call to get_angle
        np.testing.assert_equal(mock_get_angle.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to get_angle
        np.testing.assert_almost_equal(expected_mock_args[0], mock_get_angle.call_args_list[0][0], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to get_angle
        np.testing.assert_almost_equal(expected_mock_args[1], mock_get_angle.call_args_list[1][0], rounding_precision)

        # Asserting that ankle_angle_calc returned the correct result given the return value given by mocked get_angle
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["global_axis", "foot_axis", "mock_return_val", "expected_mock_args", "expected"], [
        # Test from running sample data
        (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
         np.array([[442.81997681, 381.62280273, 42.66047668], [442.84624127, 381.65130240, 43.65972538],
                   [441.87735056, 381.95630350, 42.67574106], [442.48716163, 380.68048378, 42.69610044],
                   [39.43652725, 382.44522095, 41.78911591], [39.56652626, 382.50901000, 42.77857597],
                   [38.49313328, 382.14606841, 41.93234850], [39.74166342, 381.49315020, 41.81040458]]),
         [[-83.89045512, 70.44471323, 85.11559381], [86.00906822, 72.18901201, -77.96294971]],
         [[[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
           [[0.02626446, 0.02849967, 0.9992487], [-0.94262625, 0.33350077, 0.01526438],
            [-0.33281518, -0.94231895, 0.03562376]]], [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                                       [[0.12999901, 0.06378905, 0.98946006],
                                                        [-0.94339397, -0.29915254, 0.14323259],
                                                        [0.30513617, -0.95207075, 0.02128867]]]],
         [[-83.89045512, -4.88440619, 70.44471323], [86.00906822, 167.96294971, -72.18901201]]),
        # Test with zeros for all params
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, -90, 0], [0, 90, 0]]),
        # Testing when values are added to global_axis
        (np.array([[6, -4, -8], [-1, 2, 9], [-1, -3, -3]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, -90, 0], [0, 90, 0]]),
        # Testing when values are added to right foot origin and x, y, z axes
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[6, 2, 8], [8, 9, -2], [-7, -9, -9], [-8, -3, -3], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[2, 7, -10], [-13, -11, -17], [-14, -5, -11]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[0, -90, 0], [0, 90, 0]]),
        # Testing when values are added to left foot origin and x, y, z axes
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [2, -3, -9], [1, -6, -6], [-2, -8, 3], [2, 9, 2]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-1, -3, 3], [-4, -5, 12], [0, 12, 11]]]],
         [[0, -90, 0], [0, 90, 0]]),
        # Testing when values are added to foot_axis
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array(
             [[6, 2, 8], [8, 9, -2], [-7, -9, -9], [-8, -3, -3], [2, -3, -9], [1, -6, -6], [-2, -8, 3], [2, 9, 2]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[2, 7, -10], [-13, -11, -17], [-14, -5, -11]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-1, -3, 3], [-4, -5, 12], [0, 12, 11]]]],
         [[0, -90, 0], [0, 90, 0]]),
        # Testing when values are added to mock_return_val
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         [[-68, -15, 29], [-142, 59, 118]],
         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
         [[-68, -61, -15], [-142, -28, -59]]),
        # Testing when values are added to global_axis and foot_axis
        (np.array([[6, -4, -8], [-1, 2, 9], [-1, -3, -3]]),
         np.array(
             [[6, 2, 8], [8, 9, -2], [-7, -9, -9], [-8, -3, -3], [2, -3, -9], [1, -6, -6], [-2, -8, 3], [2, 9, 2]]),
         [[0, 0, 0], [0, 0, 0]],
         [[[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[2, 7, -10], [-13, -11, -17], [-14, -5, -11]]],
          [[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[-1, -3, 3], [-4, -5, 12], [0, 12, 11]]]],
         [[0, -90, 0], [0, 90, 0]]),
        # Testing when values are added to global_axis, foot_axis, and mock_return_val
        (np.array([[6, -4, -8], [-1, 2, 9], [-1, -3, -3]]),
         np.array(
             [[6, 2, 8], [8, 9, -2], [-7, -9, -9], [-8, -3, -3], [2, -3, -9], [1, -6, -6], [-2, -8, 3], [2, 9, 2]]),
         [[-68, -15, 29], [-142, 59, 118]],
         [[[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[2, 7, -10], [-13, -11, -17], [-14, -5, -11]]],
          [[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[-1, -3, 3], [-4, -5, 12], [0, 12, 11]]]],
         [[-68, -61, -15], [-142, -28, -59]]),
        # Testing that when global_axis and foot_axis are composed of lists of ints
        ([[6, -4, -8], [-1, 2, 9], [-1, -3, -3]],
         [[6, 2, 8], [8, 9, -2], [-7, -9, -9], [-8, -3, -3], [2, -3, -9], [1, -6, -6], [-2, -8, 3], [2, 9, 2]],
         [[-68, -15, 29], [-142, 59, 118]],
         [[[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[2, 7, -10], [-13, -11, -17], [-14, -5, -11]]],
          [[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[-1, -3, 3], [-4, -5, 12], [0, 12, 11]]]],
         [[-68, -61, -15], [-142, -28, -59]]),
        # Testing that when global_axis and foot_axis are composed of numpy arrays of ints
        (np.array([[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], dtype='int'),
         np.array([[6, 2, 8], [8, 9, -2], [-7, -9, -9], [-8, -3, -3], [2, -3, -9], [1, -6, -6], [-2, -8, 3], [2, 9, 2]],
                  dtype='int'),
         [[-68, -15, 29], [-142, 59, 118]],
         [[[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[2, 7, -10], [-13, -11, -17], [-14, -5, -11]]],
          [[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[-1, -3, 3], [-4, -5, 12], [0, 12, 11]]]],
         [[-68, -61, -15], [-142, -28, -59]]),
        # Testing that when global_axis and foot_axis are composed of lists of floats
        ([[6.0, -4.0, -8.0], [-1.0, 2.0, 9.0], [-1.0, -3.0, -3.0]],
         [[6.0, 2.0, 8.0], [8.0, 9.0, -2.0], [-7.0, -9.0, -9.0], [-8.0, -3.0, -3.0], [2.0, -3.0, -9.0],
          [1.0, -6.0, -6.0], [-2.0, -8.0, 3.0], [2.0, 9.0, 2.0]],
         [[-68, -15, 29], [-142, 59, 118]],
         [[[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[2, 7, -10], [-13, -11, -17], [-14, -5, -11]]],
          [[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[-1, -3, 3], [-4, -5, 12], [0, 12, 11]]]],
         [[-68, -61, -15], [-142, -28, -59]]),
        # Testing that when global_axis and foot_axis are composed of numpy arrays of floats
        (np.array([[6.0, -4.0, -8.0], [-1.0, 2.0, 9.0], [-1.0, -3.0, -3.0]], dtype='float'),
         np.array([[6.0, 2.0, 8.0], [8.0, 9.0, -2.0], [-7.0, -9.0, -9.0], [-8.0, -3.0, -3.0], [2.0, -3.0, -9.0],
                   [1.0, -6.0, -6.0], [-2.0, -8.0, 3.0], [2.0, 9.0, 2.0]], dtype='float'),
         [[-68, -15, 29], [-142, 59, 118]],
         [[[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[2, 7, -10], [-13, -11, -17], [-14, -5, -11]]],
          [[[6, -4, -8], [-1, 2, 9], [-1, -3, -3]], [[-1, -3, 3], [-4, -5, 12], [0, 12, 11]]]],
         [[-68, -61, -15], [-142, -28, -59]])])
    def test_foot_angle_calc(self, global_axis, foot_axis, mock_return_val, expected_mock_args, expected):
        """
        This test provides coverage of the foot_angle_calc function in the class CGM in pycgm.py, defined as
        foot_angle_calc(global_axis, foot_axis)

        This test takes 5 parameters:
        global_axis : ndarray
            A 3x3 ndarray representing the global coordinate system.
        foot_axis : ndarray
            An 8x3 ndarray containing the right foot origin, right foot unit vectors, left foot origin, and left foot
            unit vectors.
        mock_return_val : list
            The value to be returned by the mock for get_angle
        expected_mock_args : list
            The expected arguments used to call the mocked function, get_angle
        expected : ndarray
            A 2x3 ndarray containing the flexion, abduction, and rotation angles of the right and left foot.

        This test is checking to make sure the foot angle is calculated correctly given the input parameters. This
        tests mocks get_angle to make sure the correct parameters are being passed into it given the parameters
        passed into foot_angle_calc, expected_mock_args, and to also ensure that foot_angle_calc returns the correct
        value considering the return value of get_angle, mock_return_val.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when global_axis and foot_axis are composed of lists of ints, numpy arrays
        of ints, list of floats, and numpy arrays of floats.
        """
        with patch.object(CGM, 'get_angle', side_effect=mock_return_val) as mock_get_angle:
            result = CGM.foot_angle_calc(global_axis, foot_axis)

        # Asserting that there was only 1 call to get_angle
        np.testing.assert_equal(mock_get_angle.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to get_angle
        np.testing.assert_almost_equal(expected_mock_args[0], mock_get_angle.call_args_list[0][0], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to get_angle
        np.testing.assert_almost_equal(expected_mock_args[1], mock_get_angle.call_args_list[1][0], rounding_precision)

        # Asserting that foot_angle_calc returned the correct result given the return value given by mocked get_angle
        np.testing.assert_almost_equal(result, expected, rounding_precision)


class TestCGMUpperBodyAngle:
    """
    This class tests the upper body angle functions in the class CGM in pycgm.py:
    head_angle_calc
    """

    @pytest.mark.parametrize(["global_axis", "head_axis", "mock_return_val", "expected_mock_args", "expected"], [
        # Test from running sample data
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         [[99.5, 82.2, 1483.8], [100.2, 83.4, 1484.7], [98.1, 83.6, 1483.3], [99.8, 82.4, 1484.2]],
         [36.86989765, -6.19039946, -139.39870535],
         [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0.7, 1.2, 0.9], [-1.4, 1.4, - 0.5], [0.3, 0.2, 0.4]]],
         [-36.86989765, 6.19039946, -139.39870535]),
        # Test with zeros for all params
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [0, 0, 0],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
         [0, 0, 0]),
        # Testing when values are added to global_axis
        ([[-4, 5, 9], [6, -8, 0], [-9, 4, -4]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [0, 0, 0],
         [[[-4, 5, 9], [6, -8, 0], [-9, 4, -4]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
         [0, 0, 0]),
        # Testing when values are added to head origin
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[-5, 3, 4], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [0, 0, 0],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[5, -3, -4], [5, -3, -4], [5, -3, -4]]],
         [0, 0, 0]),
        # Testing when values are added to head x axis
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, -2, -1], [0, 0, 0], [0, 0, 0]],
         [0, 0, 0],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, -2, -1], [0, 0, 0], [0, 0, 0]]],
         [0, 0, 0]),
        # Testing when values are added to head y axis
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [1, 6, -2], [0, 0, 0]],
         [0, 0, 0],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, 6, -2], [0, 0, 0]]],
         [0, 0, 0]),
        # Testing when values are added to head z axis
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [2, 9, 0]],
         [0, 0, 0],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [2, 9, 0]]],
         [0, 0, 0]),
        # Testing when values are added to head x, y, z axes
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, -2, -1], [1, 6, -2], [2, 9, 0]],
         [0, 0, 0],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, -2, -1], [1, 6, -2], [2, 9, 0]]],
         [0, 0, 0]),
        # Testing when values are added to head_axis
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[-5, 3, 4], [0, -2, -1], [1, 6, -2], [2, 9, 0]],
         [0, 0, 0],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[5, -5, -5], [6, 3, -6], [7, 6, -4]]],
         [0, 0, 0]),
        # Testing when values are added to mock_return_val
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [-54, 71, 192],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
         [54, -71, 192]),
        # Testing when values are added to global_axis and head_axis
        ([[-4, 5, 9], [6, -8, 0], [-9, 4, -4]],
         [[-5, 3, 4], [0, -2, -1], [1, 6, -2], [2, 9, 0]],
         [0, 0, 0],
         [[[-4, 5, 9], [6, -8, 0], [-9, 4, -4]], [[5, -5, -5], [6, 3, -6], [7, 6, -4]]],
         [0, 0, 0]),
        # Testing when values are added to global_axis, head_axis, and mock_return_val
        ([[-4, 5, 9], [6, -8, 0], [-9, 4, -4]],
         [[-5, 3, 4], [0, -2, -1], [1, 6, -2], [2, 9, 0]],
         [-54, 71, 192],
         [[[-4, 5, 9], [6, -8, 0], [-9, 4, -4]], [[5, -5, -5], [6, 3, -6], [7, 6, -4]]],
         [54, -71, 192]),
        # Testing that when global_axis and head_axis are composed of lists of ints
        ([[-4, 5, 9], [6, -8, 0], [-9, 4, -4]],
         [[-5, 3, 4], [0, -2, -1], [1, 6, -2], [2, 9, 0]],
         [-54, 71, 192],
         [[[-4, 5, 9], [6, -8, 0], [-9, 4, -4]], [[5, -5, -5], [6, 3, -6], [7, 6, -4]]],
         [54, -71, 192]),
        # Testing that when global_axis and head_axis are composed of numpy arrays of ints
        (np.array([[-4, 5, 9], [6, -8, 0], [-9, 4, -4]], dtype='int'),
         np.array([[-5, 3, 4], [0, -2, -1], [1, 6, -2], [2, 9, 0]], dtype='int'),
         [-54, 71, 192],
         [[[-4, 5, 9], [6, -8, 0], [-9, 4, -4]], [[5, -5, -5], [6, 3, -6], [7, 6, -4]]],
         [54, -71, 192]),
        # Testing that when global_axis and head_axis are composed of lists of floats
        ([[-4.0, 5.0, 9.0], [6.0, -8.0, 0.0], [-9.0, 4.0, -4.0]],
         [[-5.0, 3.0, 4.0], [0.0, -2.0, -1.0], [1.0, 6.0, -2.0], [2.0, 9.0, 0.0]],
         [-54, 71, 192],
         [[[-4, 5, 9], [6, -8, 0], [-9, 4, -4]], [[5, -5, -5], [6, 3, -6], [7, 6, -4]]],
         [54, -71, 192]),
        # Testing that when global_axis and head_axis are composed of numpy arrays of floats
        (np.array([[-4.0, 5.0, 9.0], [6.0, -8.0, 0.0], [-9.0, 4.0, -4.0]], dtype='float'),
         np.array([[-5.0, 3.0, 4.0], [0.0, -2.0, -1.0], [1.0, 6.0, -2.0], [2.0, 9.0, 0.0]], dtype='float'),
         [-54, 71, 192],
         [[[-4, 5, 9], [6, -8, 0], [-9, 4, -4]], [[5, -5, -5], [6, 3, -6], [7, 6, -4]]],
         [54, -71, 192])])
    def test_head_angle_calc(self, global_axis, head_axis, mock_return_val, expected_mock_args, expected):
        """
        This test provides coverage of the head_angle_calc function in the class CGM in pycgm.py, defined as
        head_angle_calc(global_axis, head_axis)

        This test takes 5 parameters:
        global_axis : ndarray
            A 3x3 ndarray representing the global coordinate system.
        head_axis : ndarray
            A 4x3 ndarray containing the origin and three unit vectors of the head axis.
        mock_return_val : list
            The value to be returned by the mock for get_head_angle
        expected_mock_args : list
            The expected arguments used to call the mocked function, get_head_angle
        expected : array
            A 1x3 ndarray containing the flexion, abduction, and rotation angles of the head.

        This test is checking to make sure the global head angle is calculated correctly given the input parameters.
        This tests mocks get_head_angle to make sure the correct parameters are being passed into it given the
        parameters passed into head_angle_calc, expected_mock_args, and to also ensure that head_angle_calc returns
        the correct value considering the return value of get_head_angle, mock_return_val.

        This unit test ensures that:
        - the correct expected values are altered per parameter given.
        - the resulting output is correct when global_axis and head_axis are composed of lists of ints, numpy arrays
        of ints, list of floats, and numpy arrays of floats.
        """
        with patch.object(CGM, 'get_head_angle', return_value=mock_return_val) as mock_get_angle:
            result = CGM.head_angle_calc(global_axis, head_axis)

        # Asserting that there was only 1 call to get_head_angle
        np.testing.assert_equal(mock_get_angle.call_count, 1)

        # Asserting that the correct params were sent in the call to get_head_angle
        np.testing.assert_almost_equal(expected_mock_args, mock_get_angle.call_args_list[0][0], rounding_precision)

        # Asserting that head_angle_calc returned the correct result given the return value given by mocked
        # get_head_angle
        np.testing.assert_almost_equal(result, expected, rounding_precision)


class TestCGMAngleUtils():
    """
    This class tests the angle utils functions in the class CGM in pycgm.py:
    get_angle
    """

    @pytest.mark.parametrize(["x_rot", "y_rot", "z_rot", "expected"], [
        (0, 0, 0, [0, 0, 90]),
        # X rotations
        (90, 0, 0, [0, 90, 90]), (30, 0, 0, [0, 30, 90]), (-30, 0, 0, [0, -30, 90]), (120, 0, 0, [180, 60, -90]),
        (-120, 0, 0, [180, -60, -90]), (180, 0, 0, [180, 0, -90]),
        # Y rotations
        (0, 90, 0, [90, 0, 90]), (0, 30, 0, [30, 0, 90]), (0, -30, 0, [-30, 0, 90]), (0, 120, 0, [120, 0, 90]),
        (0, -120, 0, [-120, 0, 90]), (0, 180, 0, [180, 0, 90]),
        # Z rotations
        (0, 0, 90, [0, 0, 0]), (0, 0, 30, [0, 0, 60]), (0, 0, -30, [0, 0, 120]), (0, 0, 120, [0, 0, -30]),
        (0, 0, -120, [0, 0, -150]), (0, 0, 180, [0, 0, -90]),
        # Multiple Rotations
        (150, 30, 0, [146.30993247, 25.65890627, -73.89788625]), (45, 0, 60, [0, 45, 30]), (0, 90, 120, [90, 0, -30]),
        (135, 45, 90, [125.26438968, 30, -144.73561032])])
    def test_get_angle(self, x_rot, y_rot, z_rot, expected):
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

        .. math::
            \[ \alpha = \arcsin{(-axisD_{z} \cdot axisP_{y})} \]

        If alpha is between -pi/2 and pi/2

        .. math::
            \[ \beta = \arctan2{((axisD_{z} \cdot axisP_{x}), axisD_{z} \cdot axisP_{z})} \]

            \[ \gamma = \arctan2{((axisD_{y} \cdot axisP_{y}), axisD_{x} \cdot axisP_{y})} \]

        Otherwise

        .. math::
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
        axis_p = CGM.rotation_matrix(x_rot, y_rot, z_rot)
        axis_d = CGM.rotation_matrix(0, 0, 0)
        result = CGM.get_angle(axis_p, axis_d)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_get_angle_datatypes(self):
        """
        This test provides coverage of the getangle function in pyCGM.py, defined as getangle(axisP,axisD).
        It checks that the resulting output from calling getangle is correct for a list of ints, a numpy array of
        ints, a list of floats, and a numpy array of floats.
        """
        axis_d = CGM.rotation_matrix(0, 0, 0)
        axis_p_floats = CGM.rotation_matrix(90, 0, 90)
        axis_p_ints = [[int(y) for y in x] for x in axis_p_floats]
        expected = [0, 90, 0]

        # Check that calling getangle on a list of ints yields the expected results
        result_int_list = CGM.get_angle(axis_p_ints, axis_d)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling getangle on a numpy array of ints yields the expected results
        result_int_nparray = CGM.get_angle(np.array(axis_p_ints, dtype='int'), np.array(axis_d, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling getangle on a list of floats yields the expected results
        result_float_list = CGM.get_angle(axis_p_floats, axis_d)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling getangle on a numpy array of floats yields the expected results
        result_float_nparray = CGM.get_angle(np.array(axis_p_floats, dtype='float'), np.array(axis_d, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)
