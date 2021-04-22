import pytest
import numpy as np
import pycgm.static as static

rounding_precision = 6


class TestPycgmStaticUtils():
    """
    This class tests the utils functions in static.py:
    iad_calculation
    """
    nan_3d = [np.nan, np.nan, np.nan]
    rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10),
                 np.random.randint(0, 10)]

    @pytest.mark.parametrize(["rasi", "lasi", "expected_results"], [
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
        # Testing that when markers are composed of lists of ints
        ([7, 2, -6], [3, -7, 2], 12.68857754044952),
        # Testing that when markers are composed of numpy arrays of ints
        (np.array([7, 2, -6], dtype='int'), np.array([3, -7, 2], dtype='int'),
         12.68857754044952),
        # Testing that when markers are composed of lists of floats
        ([7.0, 2.0, -6.0], [3.0, -7.0, 2.0], 12.68857754044952),
        # Testing that when markers are composed of numpy arrays of floats
        (np.array([7.0, 2.0, -6.0], dtype='float'),
         np.array([3.0, -7.0, 2.0], dtype='float'), 12.68857754044952)])
    def test_iad_calculation(self, rasi, lasi, expected_results):
        r"""
        This test provides coverage of the iad_calculation function in
        static.py, defined as iad_calculation(rasi, lasi).

        This test takes 2 parameters:
            frame: dictionary of marker lists
            expected_results: the expected result from calling IADcalculation
                              on frame. This is the Inter ASIS Distance (IAD),
                              or the distance between the two markers RASI and
                              LASI in frame.

        Given the markers RASI and LASI in frame, the Inter ASIS Distance is
        defined as:
        .. math::
            InterASISDist = \sqrt{(RASI_x-LASI_x)^2 + (RASI_y-LASI_y)^2 +
                            (RASI_z-LASI_z)^2}
        where :math:`RASI_x` is the x-coordinate of the RASI marker in frame

        This unit test ensures that:
        - the distance is measured correctly when some coordinates are the
          same, all coordinates are the same, and all coordinates are different
        - the distance is measured correctly given positive, negative and zero
          values
        - the resulting output is correct when frame is composed of lists of
          ints, numpy arrays of ints, lists of floats, and numpy arrays of
          floats.
        """
        result = static.iad_calculation(rasi, lasi)
        np.testing.assert_almost_equal(result, expected_results,
                                       rounding_precision)
