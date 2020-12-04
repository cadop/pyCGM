#!/usr/bin/python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from refactor.pycgm import StaticCGM

rounding_precision = 8
class TestStaticCGMAxis:
    """
    This class tests the axis functions in pycgm.StaticCGM:
        static_calculation_head
    """

    @pytest.mark.parametrize(["head_axis", "expected"], [
        # Test from running sample data
        ([[244.89547729492188, 325.0578918457031, 1730.1619873046875],
          [244.87227957886893, 326.0240255639856, 1730.4189843948805],
          [243.89575702706503, 325.0366593474616, 1730.1515677531293],
          [244.89086730509763, 324.80072493605866, 1731.1283433097797]],
          0.25992807335420975),
        # Test with zeros for all params
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         np.nan),
        # Testing when values are added to head[0][0]
        ([[0, 0, 0], [-1, 8, 9], [0, 0, 0], [0, 0, 0]],
         1.5707963267948966),
        # Testing when values are added to head[0][1]
        ([[0, 0, 0], [0, 0, 0], [7, 5, 7], [0, 0, 0]],
         np.nan),
        # Testing when values are added to head[0][2]
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [3, -6, -2]],
         0.0),
        # Testing when values are added to head[0]
        ([[0, 0, 0], [-1, 8, 9], [7, 5, 7], [3, -6, -2]],
         -1.3521273809209546),
        # Testing when values are added to head[1]
        ([[-4, 7, 8], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         0.7853981633974483),
        # Testing when values are added to head
        ([[-4, 7, 8], [-1, 8, 9], [7, 5, 7], [3, -6, -2]],
         -0.09966865249116204),
        # Testing that when head is composed of lists of ints
        ([[-4, 7, 8], [-1, 8, 9], [7, 5, 7], [3, -6, -2]],
         -0.09966865249116204),
        # Testing that when head is composed of numpy arrays of ints
        (np.array([[-4, 7, 8], [-1, 8, 9], [7, 5, 7], [3, -6, -2]], dtype='int'),
         -0.09966865249116204),
        # Testing that when head is composed of lists of floats
        ([[-4.0, 7.0, 8.0], [-1.0, 8.0, 9.0], [7.0, 5.0, 7.0], [3.0, -6.0, -2.0]],
         -0.09966865249116204),
        # Testing that when head is composed of numpy arrays of floats
        (np.array([[-4.0, 7.0, 8.0], [-1.0, 8.0, 9.0], [7.0, 5.0, 7.0], [3.0, -6.0, -2.0]], dtype='float'),
         -0.09966865249116204)])
    def test_static_calculation_head(self, head_axis, expected):
        """
        This test provides coverage of the static_calculation_head function in 
        pycgm.StaticCGM, defined as static_calculation_head(head_axis)

        This function first calculates the x, y, z axes of the head by subtracting the given head origin from the head
        axes. It then calculates the offset angle using the global axis as the proximal axis, and the 
        head axis as the distal axis using inverse tangent.

        This test ensures that:
        - the head axis and the head origin both have an effect on the final offset angle
        - the resulting output is correct when head is composed of lists of ints, numpy arrays of ints, lists of
        floats, and numpy arrays of floats.
        """
        result = StaticCGM.static_calculation_head(head_axis)
        np.testing.assert_almost_equal(result, expected, rounding_precision)
