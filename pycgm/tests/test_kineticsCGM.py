#!/usr/bin/python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from pycgm.pycgm import CGM
import os

rounding_precision = 8


class TestKineticsCGM:

    @pytest.mark.parametrize("lhjc, rhjc, axis, expected", [
        (np.array([282.57097863, 139.43231855, 435.52900012]),
         np.array([208.38050472, 122.80342417, 437.98979061]),
         np.array([[151.60830688, 291.74131775, 832.89349365],
                   [251.74063624, 392.72694721, 1032.78850073],
                   [250.61711554, 391.87232862, 1032.8741063],
                   [251.60295336, 391.84795134, 1033.88777762]]),
         ([[245.47574167, 131.11787136, 436.75939536],
           [261.0890402, 155.43411628, 500.91761881]])),
        (np.array([984.96369008, 161.72241084, 714.78280362]),
         np.array([-570.727107, 409.48579719, 387.17336605]),
         np.array([[-553.90052549, -327.14438741, -4.58459872],
                   [586.81782059, 994.852335, -164.15032491],
                   [367.53692416, -193.11814502, 141.95648112],
                   [814.64795266, 681.51439276, 87.63894117]]),
         ([[207.11829154, 285.60410402, 550.97808484],
           [1344.7944079, 1237.35589455, 673.3680447]])),
        (np.array([624.42435686, 746.90148656, -603.71552902]),
         np.array([-651.87182756, -493.94862894, 640.38910712]),
         np.array([[691.47208667, 395.90359428, 273.23978111],
                   [711.02920886, -701.16459687, 532.55441473],
                   [-229.76970935, -650.15236712, 359.70108366],
                   [222.81186893, 536.56366268, 386.21334066]]),
         ([[-13.72373535, 126.47642881, 18.33678905],
           [627.86028973, 1671.50486946, 1130.43333413]])),
        # Test that changing origin, x, or y axes doesn't affect output
        (np.array([624.42435686, 746.90148656, -603.71552902]),
         np.array([-651.87182756, -493.94862894, 640.38910712]),
         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [222.81186893, 536.56366268, 386.21334066]]),
         ([[-13.72373535, 126.47642881, 18.33678905],
           [627.86028973, 1671.50486946, 1130.43333413]]))
    ])
    def test_find_l5(self, lhjc, rhjc, axis, expected):
        """
        This function tests CGM.find(lhjc, rhjc, axis),
        where lhjc and rhjc are numpy arrays that give the position of the
        LHJC and RHJC markers, and pelvis_axis gives the arrays of the pelvis
        or thorax origin,x-axis, y-axis, and z-axis.

        This test ensures that changing the origin, x-axis, or y-axis components
        of the axis array does not affect the output.
        """
        result = CGM.find_l5(lhjc, rhjc, axis)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize("point, start, end, expected", [
        (np.array([1.1, -2.24, 31.32]), np.array([4, 5.3, -6]), np.array([2.14, 12.52, 13.2]),
         (23.393879541452716, [2.14, 12.52, 13.2], [1.1, -2.24, 31.32])),
        (np.array([35.83977741, 61.57074759, 68.44530267]), np.array([74.67790922, 14.29054848, -26.04736139]),
         np.array([0.56489944, -16.12960177, 63.33083103]),
         (76.7407926, [23.82194806, -6.58360012, 35.28348857], [35.83977741, 61.57074759, 68.44530267])),
        (np.array([23.90166027, 88.64089564, 49.65111862]), np.array([48.50606388, -75.41062664, 75.31899688]),
         np.array([-34.87278229, 25.60601135, 78.81218762]),
         (90.98461233, [-34.87278229, 25.60601135, 78.81218762], [23.90166027, 88.64089564, 49.65111862])),
        (np.array([687.84935299, -545.36574903, 668.52916292]), np.array([-39.73733639, 854.80603373, 19.05056745]),
         np.array([84.09259043, 617.95544147, 501.49109559]),
         (1321.26459747, [84.09259043, 617.95544147, 501.49109559], [687.84935299, -545.36574903, 668.52916292])),
        (np.array([660.95556608, 329.67656854, -142.68363472]), np.array([773.43109446, 253.42967266, 455.42278696]),
         np.array([233.66307152, 432.69607959, 590.12473739]),
         (613.34788275, [773.43109446, 253.42967266, 455.42278696], [660.95556608, 329.67656854, -142.68363472]))
    ])
    def test_point_to_line(self, point, start, end, expected):
        """
        This function tests CGM.point_to_line(point, start, end), where point,
        start, and end are 1x3 numpy arrays representing the XYZ coordinates of a point.

        We test cases where the inputs are numpy arrays, negative numbers, ints, and floats.

        CGM.point_to_line() returns dist, nearest, point as a tuple, where dist is the
        distance from the point to the line, nearest is the nearest point on the line, and
        point is the original point.
        """
        result_dist, result_nearest, result_point = CGM.point_to_line(point, start, end)
        expected_dist, expected_nearest, expected_point = expected
        np.testing.assert_almost_equal(result_dist, expected_dist, rounding_precision)
        np.testing.assert_almost_equal(result_nearest, expected_nearest, rounding_precision)
        np.testing.assert_almost_equal(result_point, expected_point, rounding_precision)

    def test_point_to_line_exceptions(self):
        """
        Ensure that the inputs to CGM.point_to_line cannot be python lists.
        """
        point, start, end = [1, 2, 3], [4, 5, 6], [0, 0, 0]
        with pytest.raises(Exception):
            CGM.point_to_line(point, start, end)
