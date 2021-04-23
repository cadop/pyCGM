import pycgm.axis as axis
import pytest
import numpy as np

rounding_precision = 6

class TestAxisUtils():
    """
    This class tests the lower body axis functions in axis.py:
    find_joint_center
    """

    @pytest.mark.parametrize(["p_a", "p_b", "p_c", "delta", "expected"], [
        # Test from running sample data
        ([426.50338745, 262.65310669, 673.66247559],
         [308.38050472, 322.80342417, 937.98979061],
         [416.98687744, 266.22558594, 524.04089355],
         59.5,
         [364.17774614, 292.17051722, 515.19181496]),
        # Testing with basic value in a and c
        ([1, 0, 0], [0, 0, 0], [0, 0, 1], 0.0, [0, 0, 1]),
        # Testing with value in a and basic value in c
        ([-7, 1, 2], [0, 0, 0], [0, 0, 1], 0.0, [0, 0, 1]),
        #  Testing with value in b and basic value in c
        ([0, 0, 0], [1, 4, 3], [0, 0, 1], 0.0, [0, 0, 1]),
        #  Testing with value in a and b and basic value in c
        ([-7, 1, 2], [1, 4, 3], [0, 0, 1], 0.0, [0, 0, 1]),
        #  Testing with value in a, b, and c
        ([-7, 1, 2], [1, 4, 3], [3, 2, -8], 0.0, [3, 2, -8]),
        # Testing with value in a, b, c and delta of 1
        ([-7, 1, 2], [1, 4, 3], [3, 2, -8], 1.0, [3.91271, 2.361115, -7.808801]),
        # Testing with value in a, b, c and delta of 20
        ([-7, 1, 2], [1, 4, 3], [3, 2, -8], 10.0, [5.867777, 5.195449, 1.031332]),
        # Testing that when a, b, and c are lists of ints and delta is an int
        ([-7, 1, 2], [1, 4, 3], [3, 2, -8], 10, [5.867777, 5.195449, 1.031332]),
        # Testing that when a, b, and c are numpy arrays of ints and delta is an int
        (np.array([-7, 1, 2], dtype='int'), np.array([1, 4, 3], dtype='int'), np.array([3, 2, -8], dtype='int'),
         10, [5.867777, 5.195449, 1.031332]),
        # Testing that when a, b, and c are lists of floats and delta is a float
        ([-7.0, 1.0, 2.0], [1.0, 4.0, 3.0], [3.0, 2.0, -8.0], 10.0, [5.867777, 5.195449, 1.031332]),
        # Testing that when a, b, and c are numpy arrays of floats and delta is a float
        (np.array([-7.0, 1.0, 2.0], dtype='float'), np.array([1.0, 4.0, 3.0], dtype='float'),
         np.array([3.0, 2.0, -8.0], dtype='float'), 10.0, [5.867777, 5.195449, 1.031332])])

    def test_find_joint_center(self, p_a, p_b, p_c, delta, expected):
        """
        This test provides coverage of the find_joint_center function in axis.py, defined as 
        find_joint_center(p_a, p_b, p_c, delta)

        This test takes 5 parameters:
        p_a: list markers of x,y,z position
        p_b: list markers of x,y,z position
        p_c: list markers of x,y,z position
        delta: length from marker to joint center, retrieved from subject measurement file
        expected: the expected result from calling find_joint_center on a, b, c, and delta

        A plane will be generated using the positions of three specified markers. 
        The plane will then calculate a joint center by rotating the vector of the plane around the rotating axis (the orthogonal vector).

        Lastly, it checks that the resulting output is correct when a, b, and c are lists of ints, numpy arrays of ints,
        lists of floats, and numpy arrays of floats and delta is an int or a float.
        """
        result = axis.find_joint_center(p_a, p_b, p_c, delta)
        np.testing.assert_almost_equal(result, expected, rounding_precision)