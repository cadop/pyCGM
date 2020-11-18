import pytest
import numpy as np
import os
import sys
import pyCGM_Single.pyCGM_Helpers as pyCGM_Helpers
import pyCGM_Single.pycgmStatic as pycgmStatic
import pyCGM_Single.pycgmIO as pycgmIO
import pyCGM_Single.pycgmCalc as pycgmCalc

rounding_precision = 6

class TestPycgmStaticUtils():
    """
    This class tests the utils functions in pycgmStatic.py:
    rotmat
    getDist
    getStatic
    average
    IADcalculation
    headoffCalc
    staticCalculation
    getankleangle
    norm2d
    norm3d
    normDiv
    matrixmult
    cross
    """
    nan_3d = [np.nan, np.nan, np.nan]
    rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]

    @pytest.mark.parametrize(["x", "y", "z", "expected"], [
        (0.0, 0.0, 180, [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        (0, 0, 0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        (90, 0, 0, [[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        (0, 135, 0, [[-0.70710678, 0, 0.70710678], [0, 1, 0], [-0.70710678, 0, -0.70710678]]),
        (0, 0, -60, [[0.5, 0.8660254, 0], [-0.8660254, 0.5, 0], [0, 0, 1]]),
        (90, 0, 90, [[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
        (0, 150, -30, [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]),
        (90, 180, -90, [[0, -1, 0], [0, 0, 1], [-1, 0, 0]])])
    def test_rotmat(self, x, y, z, expected):
        """
        This test provides coverage of the rotmat function in pycgmStatic.py, defined as rotmat(x, y, z)
        where x, y, and z are all floats that represent the angle of rotation in a particular dimension.

        This test takes 4 parameters:
        x: angle to be rotated in the x axis
        y: angle to be rotated in the y axis
        z: angle to be rotated in the z axis
        expected: the expected rotation matrix from calling rotmat on x, y, and z. This will be a transformation
        matrix that can be used to perform a rotation in the x, y, and z directions at the values inputted.
        """
        result = pycgmStatic.rotmat(x, y, z)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_rotmat_datatypes(self):
        """
        This test provides coverage of the rotmat function in pycgmStatic.py, defined as rotmat(x, y, z)
        where x, y, and z are all floats that represent the angle of rotation in a particular dimension.

        This test checks that the resulting output from calling rotmat is correct when called with ints or floats.
         """
        result_int = pycgmStatic.rotmat(0, 150, -30)
        result_float = pycgmStatic.rotmat(0.0, 150.0, -30.0)
        expected = [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]

        np.testing.assert_almost_equal(result_int, expected, rounding_precision)
        np.testing.assert_almost_equal(result_float, expected, rounding_precision)

    @pytest.mark.parametrize(["p0", "p1", "expected"], [
        ([0, 0, 0], [1, 0, 0], 1),
        ([0, 5, 0], [0, 0, 0], 5),
        ([0, 0, 0], [0, 0, -8], 8),
        ([0, 0, 0], [0, 3, 4], 5),
        ([8, 0, -3], [0, 0, 0], 8.54400374531753),
        ([0, 4, 0], [0, 0, 2], 4.47213595499958),
        ([1, 0, 3], [0, -2, 8], 5.477225575051661),
        ([3, 3, 0], [3, 0, 3], 4.242640687119285),
        ([7, 0, -4], [7, 0, 2], 6),
        ([7, -2, 5], [1, -4, 9], 7.483314773547883)])
    def test_getDist(self, p0, p1, expected):
        """
        This test provides coverage of the getDist function in pycgmStatic.py, defined as getDist(p0, p1)

        This test takes 3 parameters:
        p0: position of first x,y,z coordinate
        p1: position of second x,y,z coordinate
        expected: the expected result from calling getDist on p0 and p1. This will be the distance between p0 and p1
        """
        result = pycgmStatic.getDist(p0, p1)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getDist_datatypes(self):
        """
        This test provides coverage of the getDist function in pycgmStatic.py, defined as getDist(p0, p1)

        This test checks that the resulting output from calling getDist is correct when called with a list of ints,
        a numpy array of ints, a list of floats, and a numpy array of floats.
        """
        p0_int = [7, -2, 5]
        p1_int = [1, -4, 9]
        p0_float = [7.0, -2.0, 5.0]
        p1_float = [1.0, -4.0, 9.0]
        expected = 7.483314773547883

        # Check the calling getDist on a list of ints yields the expected results
        result_int_list = pycgmStatic.getDist(p0_int, p1_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling getDist on a numpy array of ints yields the expected results
        result_int_nparray = pycgmStatic.getDist(np.array(p0_int, dtype='int'), np.array(p1_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling getDist on a list of floats yields the expected results
        result_float_list = pycgmStatic.getDist(p0_float, p1_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling getDist on a numpy array of floats yields the expected results
        result_float_nparray = pycgmStatic.getDist(np.array(p0_float, dtype='float'), np.array(p1_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["list", "expected"], [
        ([0], 0),
        ([3], 3),
        ([-1], -1),
        ([1, 2], 1.5),
        ([-1, 3], 1),
        ([-3, 1], -1),
        ([-2, 0, 2], 0),
        ([1, 2, 3, 4, 5], 3),
        ([-1, -2, -3, -4, -5], -3),
        ([0.1, 0.2, 0.3, 0.4, 0.5], 0.3)])
    def test_average(self, list, expected):
        """
        This test provides coverage of the average function in pycgmStatic.py, defined as average(list)

        This test takes 2 parameters:
        list: list or array of values
        expected: the expected result from calling average on list. This will be the average of the values given in list
        """
        result = pycgmStatic.average(list)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_average_datatypes(self):
        """
        This test provides coverage of the average function in pycgmStatic.py, defined as average(list)

        This test checks that the resulting output from calling average is correct when called with a list of ints,
        a numpy array of ints, a list of floats, and a numpy array of floats.
        """
        list_int = [-1, -2, -3, -4, -5]
        list_float = [-1.0, -2.0, -3.0, -4.0, -5.0]
        expected = -3

        # Check the calling average on a list of ints yields the expected results
        result_int_list = pycgmStatic.average(list_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling average on a numpy array of ints yields the expected results
        result_int_nparray = pycgmStatic.average(np.array(list_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling average on a list of floats yields the expected results
        result_float_list = pycgmStatic.average(list_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling average on a numpy array of floats yields the expected results
        result_float_nparray = pycgmStatic.average(np.array(list_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["frame", "expected"], [
        ({'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0])}, 0),
        ({'RASI': np.array([1, 0, 0]), 'LASI': np.array([0, 0, 0])}, 1),
        ({'RASI': np.array([0, 0, 0]), 'LASI': np.array([2, 0, 0])}, 2),
        ({'RASI': np.array([0, 1, 0]), 'LASI': np.array([0, 1, 0])}, 0),
        ({'RASI': np.array([4, 0, 0]), 'LASI': np.array([2, 0, 0])}, 2),
        ({'RASI': np.array([4, 0, 0]), 'LASI': np.array([-2, 0, 0])}, 6),
        ({'RASI': np.array([0, 2, 1]), 'LASI': np.array([0, 4, 1])}, 2),
        ({'RASI': np.array([-5, 3, 0]), 'LASI': np.array([0, 3, 0])}, 5),
        ({'RASI': np.array([0, 3, -6]), 'LASI': np.array([0, 2, -5])}, 1.4142135623730951),
        ({'RASI': np.array([-6, 4, 0]), 'LASI': np.array([0, 6, -8])}, 10.198039027185569),
        ({'RASI': np.array([7, 2, -6]), 'LASI': np.array([3, -7, 2])}, 12.68857754044952),
        # Testing that when frame is composed of lists of ints
        ({'RASI': [7, 2, -6], 'LASI': [3, -7, 2]}, 12.68857754044952),
        # Testing that when frame is composed of numpy arrays of ints
        ({'RASI': np.array([7, 2, -6], dtype='int'), 'LASI': np.array([3, -7, 2], dtype='int')}, 12.68857754044952),
        # Testing that when frame is composed of lists of floats
        ({'RASI': [7.0, 2.0, -6.0], 'LASI': [3.0, -7.0, 2.0]}, 12.68857754044952),
        # Testing that when frame is composed ofe numpy arrays of floats
        ({'RASI': np.array([7.0, 2.0, -6.0], dtype='float'), 'LASI': np.array([3.0, -7.0, 2.0], dtype='float')}, 12.68857754044952)])
    def test_IADcalculation(self, frame, expected):
        """
        This test provides coverage of the IADcalculation function in pycgmStatic.py, defined as IADcalculation(frame)

        This test takes 2 parameters:
        frame: dictionary of marker lists
        expected: the expected result from calling IADcalculation on frame. This is the Inter ASIS Distance (IAD), or
        the distance between the two markers RASI and LASI in frame.

        This test checks that this distance between these two markers is calculated correctly for a variety of different
        coordinates. It also checks that the resulting output is correct when frame is composed of lists of ints,
        numpy arrays of ints, lists of floats, and numpy arrays of floats.
        """
        result = pycgmStatic.IADcalculation(frame)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected"], [
        (0, 0, 0, [0]),
        # X rotations
        (90, 0, 0, [0]), (30, 0, 0, [0]), (-30, 0, 0, [0]), (120, 0, 0, [0]), (-120, 0, 0, [0]), (180, 0, 0, [0]),
        # Y rotations
        (0, 90, 0, [-1.570796]), (0, 30, 0, [-0.523599]), (0, -30, 0, [0.523599]), (0, 120, 0, [1.047198]), (0, -120, 0, [-1.047198]), (0, 180, 0, [0]),
        # Z rotations
        (0, 0, 90, [0]), (0, 0, 30, [0]), (0, 0, -30, [0]), (0, 0, 120, [0]), (0, 0, -120, [0]), (0, 0, 180, [0]),
        # Multiple Rotations
        (150, 30, 0, [-0.523599]), (45, 0, 60, [0.713724]), (0, 90, 120, [1.570796]), (135, 45, 90, [-0.955317])])
    def test_headoffCalc(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the headoffCalc function in pycgmStatic.py, defined as headoffCalc(axisP, axisD)

        This test takes 3 parameters:
        axisP: the unit vector of axisP, the position of the proximal axis
        axisD: the unit vector of axisD, the position of the distal axis
        expected: the expected result from calling headoffCalc on axisP and axisD. This returns the y-rotation
        from a rotatinal matrix calculated by matrix multiplication of axisD x inverse of axisP. This angle is in
        radians, not degrees.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given
        axisP = pycgmStatic.rotmat(xRot, yRot, zRot)
        axisD = pycgmStatic.rotmat(0, 0, 0)
        result = pycgmStatic.headoffCalc(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_headoffCalc_datatypes(self):
        """
        This test provides coverage of the headoffCalc function in pycgmStatic.py, defined as headoffCalc(axisP, axisD)

        This test checks that the resulting output from calling headoffCalc is correct when called with a list of ints,
        a numpy array of ints, a list of floats, and a numpy array of floats.
        """
        axisD = pycgmStatic.rotmat(0, 0, 0)
        axisP_floats = pycgmStatic.rotmat(90, 90, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [1.570796]

        # Check that calling headoffCalc on a list of ints yields the expected results
        result_int_list = pycgmStatic.headoffCalc(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling headoffCalc on a numpy array of ints yields the expected results
        result_int_nparray = pycgmStatic.headoffCalc(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling headoffCalc on a list of floats yields the expected results
        result_float_list = pycgmStatic.headoffCalc(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling headoffCalc on a numpy array of floats yields the expected results
        result_float_nparray = pycgmStatic.headoffCalc(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["frame", "ankle_JC", "flat_foot", "vsk", "expected"], [
        # Test from running sample data
        ({'RTOE': np.array([433.33508301, 354.97229004, 44.27765274]),
          'LTOE': np.array([31.77310181, 331.23657227, 42.15322876]),
          'RHEE': np.array([381.88534546, 148.47607422, 49.99120331]),
          'LHEE': np.array([122.18766785, 138.55477905, 46.29433441])},
         [np.array([397.45738291, 217.50712216, 87.83068433]), np.array([112.28082818, 175.83265027, 80.98477997]),
          [[np.array(rand_coor), np.array([396.73749179, 218.18875543, 87.69979179]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([111.34886681, 175.49163538, 81.10789314]), np.array(rand_coor)]]],
         False,
         {},
         [[-0.015688860223839234, 0.2703999495115947, -0.15237642705642993],
          [0.009550866847196991, 0.20242596489042683, -0.019420801722458948]]),
        # Testing with zeros for all params
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0]),
          'RHEE': np.array([0, 0, 0]), 'LHEE': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         False,
         {},
         [nan_3d, nan_3d]),
        # Testing with values for frame
        ({'RTOE': np.array([1, -4, -1]), 'LTOE': np.array([1, -1, -5]),
          'RHEE': np.array([1, -7, -4]), 'LHEE': np.array([-6, -5, 1])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         False,
         {},
         [nan_3d, nan_3d]),
        # Testing with values for ankle_JC
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0]),
          'RHEE': np.array([0, 0, 0]), 'LHEE': np.array([0, 0, 0])},
         [np.array([3, 3, -2]), np.array([5, 9, -4]),
          [[np.array(rand_coor), np.array([6, -9, 9]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([-5, 6, 9]), np.array(rand_coor)]]],
         False,
         {},
         [nan_3d, nan_3d]),
        # Testing with values for vsk
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0]),
          'RHEE': np.array([0, 0, 0]), 'LHEE': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         False,
         {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
         [nan_3d, nan_3d]),
        # Testing with values for frame and ankle_JC
        ({'RTOE': np.array([1, -4, -1]), 'LTOE': np.array([1, -1, -5]),
          'RHEE': np.array([1, -7, -4]), 'LHEE': np.array([-6, -5, 1])},
         [np.array([3, 3, -2]), np.array([5, 9, -4]),
          [[np.array(rand_coor), np.array([6, -9, 9]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([-5, 6, 9]), np.array(rand_coor)]]],
         False,
         {},
         [[-0.590828, -0.802097, -0.554384], [0.955226, 0.156745, 0.166848]]),
        # Testing with values for frame, ankle_JC, and vsk
        ({'RTOE': np.array([1, -4, -1]), 'LTOE': np.array([1, -1, -5]),
          'RHEE': np.array([1, -7, -4]), 'LHEE': np.array([-6, -5, 1])},
         [np.array([3, 3, -2]), np.array([5, 9, -4]),
          [[np.array(rand_coor), np.array([6, -9, 9]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([-5, 6, 9]), np.array(rand_coor)]]],
         False,
         {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
         [[-0.590828, -0.802097, -0.554384], [0.955226, 0.156745, 0.166848]]),
        # Testing with values for frame, ankle_JC, vsk and flat_foot = True
        ({'RTOE': np.array([1, -4, -1]), 'LTOE': np.array([1, -1, -5]),
          'RHEE': np.array([1, -7, -4]), 'LHEE': np.array([-6, -5, 1])},
         [np.array([3, 3, -2]), np.array([5, 9, -4]),
          [[np.array(rand_coor), np.array([6, -9, 9]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([-5, 6, 9]), np.array(rand_coor)]]],
         True,
         {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
         [[0.041042018208567545, -0.3065439019577841, -0.3106927663413161], [0.39326377295256626, 0.5657243847333632, 0.2128595189127902]]),
        # Testing that when frame and ankle_JC are composed of lists of ints and vsk values are ints
        ({'RTOE': [1, -4, -1], 'LTOE': [1, -1, -5], 'RHEE': [1, -7, -4], 'LHEE': [-6, -5, 1]},
         [[3, 3, -2], [5, 9, -4],
          [[rand_coor, [6, -9, 9], rand_coor],
           [rand_coor, [-5, 6, 9], rand_coor]]],
         True,
         {'RightSoleDelta': 1, 'LeftSoleDelta': -1},
         [[0.041042018208567545, -0.30654390195778408, -0.30110158620693045],
          [00.3932637729525662, 0.56572438473336295, 0.22802611517428609]]),
        # Testing that when frame and ankle_JC are composed of numpy arrays of ints and vsk values are ints
        ({'RTOE': np.array([1, -4, -1], dtype='int'), 'LTOE': np.array([1, -1, -5], dtype='int'),
          'RHEE': np.array([1, -7, -4], dtype='int'), 'LHEE': np.array([-6, -5, 1], dtype='int')},
         [np.array([3, 3, -2], dtype='int'), np.array([5, 9, -4], dtype='int'),
          [np.array([rand_coor, [6, -9, 9], rand_coor], dtype='int'),
           np.array([rand_coor, [-5, 6, 9], rand_coor], dtype='int')]],
         True,
         {'RightSoleDelta': 1, 'LeftSoleDelta': -1},
         [[0.041042018208567545, -0.30654390195778408, -0.30110158620693045],
          [00.3932637729525662, 0.56572438473336295, 0.22802611517428609]]),
        # Testing that when frame and ankle_JC are composed of lists of floats and vsk values are floats
        ({'RTOE': [1.0, -4.0, -1.0], 'LTOE': [1.0, -1.0, -5.0], 'RHEE': [1.0, -7.0, -4.0], 'LHEE': [-6.0, -5.0, 1.0]},
         [[3.0, 3.0, -2.0], [5.0, 9.0, -4.0],
          [[rand_coor, [6.0, -9.0, 9.0], rand_coor],
           [rand_coor, [-5.0, 6.0, 9.0], rand_coor]]],
         True,
         {'RightSoleDelta': 1.0, 'LeftSoleDelta': -1.0},
         [[0.041042018208567545, -0.30654390195778408, -0.30110158620693045],
          [00.3932637729525662, 0.56572438473336295, 0.22802611517428609]]),
        # Testing that when frame and ankle_JC are composed of numpy arrays of floats and vsk values are floats
        ({'RTOE': np.array([1.0, -4.0, -1.0], dtype='float'), 'LTOE': np.array([1.0, -1.0, -5.0], dtype='float'),
          'RHEE': np.array([1.0, -7.0, -4.0], dtype='float'), 'LHEE': np.array([-6.0, -5.0, 1.0], dtype='float')},
         [np.array([3.0, 3.0, -2.0], dtype='float'), np.array([5.0, 9.0, -4.0], dtype='float'),
          [np.array([rand_coor, [6.0, -9.0, 9.0], rand_coor], dtype='float'),
           np.array([rand_coor, [-5.0, 6.0, 9.0], rand_coor], dtype='float')]],
         True,
         {'RightSoleDelta': 1.0, 'LeftSoleDelta': -1.0},
         [[0.041042018208567545, -0.30654390195778408, -0.30110158620693045],
          [00.3932637729525662, 0.56572438473336295, 0.22802611517428609]])])
    def test_staticCalculation(self, frame, ankle_JC, flat_foot, vsk, expected):
        """
        This test provides coverage of the staticCalculation function in pycgmStatic.py, defined as staticCalculation(frame, ankle_JC, knee_JC, flat_foot, vsk)

        This test takes 5 parameters:
        frame: dictionary of marker lists
        ankle_JC: array containing the x,y,z axes marker positions of the ankle joint center
        flat_foot: boolean indicating if the feet are flat or not
        vsk: dictionary containing subject measurements from a VSK file
        expected: the expected result from calling staticCalculation on frame, ankle_JC, flat_foot, and vsk

        This test is checking to make sure the static angle function is calculated correctly given the input parameters.
        The test checks to see that the correct values in expected are updated per each input parameter added:
        When values are only added to frame, ankle_JC, or vsk, expected is not updated.
        When values are added to frame and ankle_JC, expected should be updated.
        When flat_foot is set to True, expected should be updated.

        Lastly, it checks that the resulting output is correct when frame and ankle_JC are composed of lists of ints,
        numpy arrays of ints, lists of floats, and numpy arrays of floats and when the vsk values are ints and floats.
        """
        result = pycgmStatic.staticCalculation(frame, ankle_JC, None, flat_foot, vsk)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected"], [
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
    def test_getankleangle(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the getankleangle function in pycgmStatic.py, defined as getankleangle(axisP, axisD)

        This test takes 3 parameters:
        axisP: the unit vector of axisP, the position of the proximal axis
        axisD: the unit vector of axisD, the position of the distal axis
        expected: the expected result from calling getankleangle on axisP and axisD

        This test calls pycgmStatic.rotmat() to create axisP with an x, y, and z rotation defined in the parameters.
        It then calls pycgmStatic.getankleangle() with axisP and axisD, which was created with no rotation in the x, y
        or z direction. This result is then compared to the expected result. The results from this test will be in the
        XYZ order. This output is in radians and not degrees.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given
        axisP = pycgmStatic.rotmat(xRot, yRot, zRot)
        axisD = pycgmStatic.rotmat(0, 0, 0)
        result = pycgmStatic.getankleangle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getankleangle_datatypes(self):
        """
        This test provides coverage of the getankleangle function in pycgmStatic.py, defined as getankleangle(axisP, axisD)

        This test checks that the resulting output from calling getankleangle is correct when called with a list of ints,
        a numpy array of ints, a list of floats, and a numpy array of floats.
        """
        axisD = pycgmStatic.rotmat(0, 0, 0)
        axisP_floats = pycgmStatic.rotmat(90, 90, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [0, -1.570796, 0]

        # Check that calling getankleangle on a list of ints yields the expected results
        result_int_list = pycgmStatic.getankleangle(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling getankleangle on a numpy array of ints yields the expected results
        result_int_nparray = pycgmStatic.getankleangle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling getankleangle on a list of floats yields the expected results
        result_float_list = pycgmStatic.getankleangle(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling getankleangle on a numpy array of floats yields the expected results
        result_float_nparray = pycgmStatic.getankleangle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["v", "expected"], [
        ([-9944.089508486479, -20189.20612828088, 150.42955108569652], 22505.812344655435),
        ([0, 0, 0], 0),
        ([2, 0, 0], 2),
        ([0, 0, -1], 1),
        ([0, 3, 4], 5),
        ([-3, 0, 4], 5),
        ([6, -8, 0], 10),
        ([-5, 0, -12], 13),
        ([1, -1, np.sqrt(2)], 2)])
    def test_norm2d(self, v, expected):
        """
        This test provides coverage of the norm2d function in pycgmStatic.py, defined as norm2d(v) where v is a 3D vector.

        This test takes 2 parameters:
        v: 3D vector
        expected: the expected result from calling norm2d on v. This will be the value of the normalization of vector v.
        """
        result = pycgmStatic.norm2d(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_norm2d_datatypes(self):
        """
        This test provides coverage of the norm2d function in pycgmStatic.py, defined as norm2d(v) where v is a 3D vector.

        This test checks that the resulting output from calling norm2d is correct when called with a list of ints, a
        numpy array of ints, a list of floats, and a numpy array of floats.
        """
        v_int = [6, 0, -8]
        v_float = [6.0, 0, -8.0]
        expected = 10

        # Check the calling norm2d on a list of ints yields the expected results
        result_int_list = pycgmStatic.norm2d(v_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling norm2d on a numpy array of ints yields the expected results
        result_int_nparray = pycgmStatic.norm2d(np.array(v_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling norm2d on a list of floats yields the expected results
        result_float_list = pycgmStatic.norm2d(v_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling norm2d on a numpy array of floats yields the expected results
        result_float_nparray = pycgmStatic.norm2d(np.array(v_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["v", "expected"], [
        ([-212.5847168, 28.09841919, -4.15808105], np.array(214.47394390603984)),
        ([0, 0, 0], np.array(0)),
        ([2, 0, 0], np.array(2)),
        ([0, 0, -1], np.array(1)),
        ([0, 3, 4], np.array(5)),
        ([-3, 0, 4], np.array(5)),
        ([-6, 8, 0], np.array(10)),
        ([-5, 0, -12], np.array(13)),
        ([1, -1, np.sqrt(2)], np.array(2))])
    def test_norm3d(self, v, expected):
        """
        This test provides coverage of the norm3d function in pycgmStatic.py, defined as norm3d(v) where v is a 3D vector.

        This test takes 2 parameters:
        v: 3D vector
        expected: the expected result from calling norm3d on v. This will be the normalization of the vector v,
        inside of a numpy array.
        """
        result = pycgmStatic.norm3d(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_norm3d_datatypes(self):
        """
        This test provides coverage of the norm3d function in pycgmStatic.py, defined as norm3d(v) where v is a 3D vector.

        This test checks that the resulting output from calling norm3d is correct when called with a list of ints, a
        numpy array of ints, a list of floats, and a numpy array of floats.
        """
        v_int = [-6, 0, 8]
        v_float = [-6.0, 0, 8.0]
        expected = np.array(10)

        # Check the calling norm3d on a list of ints yields the expected results
        result_int_list = pycgmStatic.norm3d(v_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling norm3d on a numpy array of ints yields the expected results
        result_int_nparray = pycgmStatic.norm3d(np.array(v_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling norm3d on a list of floats yields the expected results
        result_float_list = pycgmStatic.norm3d(v_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling norm3d on a numpy array of floats yields the expected results
        result_float_nparray = pycgmStatic.norm3d(np.array(v_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["v", "expected"], [
        ([-212.5847168, 28.09841919, -4.15808105], np.array([-4.62150006e-03,  6.10847515e-04, -9.03948887e-05])),
        ([0, 0, 0], np.array([np.nan, np.nan, np.nan])),
        ([2, 0, 0], np.array([0.5, 0, 0])),
        ([0, 0, -1], np.array([0, 0, -1])),
        ([0, 3, 4], np.array([0, 0.12, 0.16])),
        ([-3, 0, 4], np.array([-0.12, 0, 0.16])),
        ([-6, 8, 0], np.array([-0.06, 0.08, 0])),
        ([-5, 0, -12], np.array([-0.0295858, 0, -0.07100592])),
        ([1, -1, np.sqrt(2)], np.array([0.25, -0.25, 0.35355339]))
    ])
    def test_normDiv(self, v, expected):
        """
        This test provides coverage of the normDiv function in pycgmStatic.py, defined as normDiv(v) where v is a 3D vector.
        This test takes 2 parameters:
        v: 3D vector
        expected: the expected result from calling norm3d on v. This function returns the wrong result. It is supposed
        to return the normalization division, but in the function it divides the vector by the normalization twice.
        """
        result = pycgmStatic.normDiv(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_normDiv_datatypes(self):
        """
        This test provides coverage of the normDiv function in pycgmStatic.py, defined as normDiv(v) where v is a 3D vector.
        This test checks that the resulting output from calling normDiv is correct when called with a list of ints, a
        numpy array of ints, a list of floats, and a numpy array of floats.
        """
        v_int = [-6, 0, 8]
        v_float = [-6.0, 0, 8.0]
        expected = np.array([-0.06, 0, 0.08])

        # Check the calling normDiv on a list of ints yields the expected results
        result_int_list = pycgmStatic.normDiv(v_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling normDiv on a numpy array of ints yields the expected results
        result_int_nparray = pycgmStatic.normDiv(np.array(v_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling normDiv on a list of floats yields the expected results
        result_float_list = pycgmStatic.normDiv(v_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling normDiv on a numpy array of floats yields the expected results
        result_float_nparray = pycgmStatic.normDiv(np.array(v_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["A", "B", "expected"], [
        ([[1, 0, 0], [0, 1.0, -0.0], [0, 0.0, 1.0]], [[1.0, 0, 0.0], [0, 1, 0], [-0.0, 0, 1.0]], [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ([[1]], [[1]], [[1]]),
        # Invalid matrix dimensions
        ([[1, 2]], [[1]], [[1]]),
        ([[2], [1]], [[1, 2]], [[2, 4], [1, 2]]),
        # Invalid matrix dimensions
        ([[1, 2, 0], [0, 1, 2]], [[2, 1], [1, 4]], [[4, 9], [1, 4]]),
        ([[11, 12, 13], [14, 15, 16]], [[1, 2], [3, 4], [5, 6]], [[112, 148], [139, 184]]),
        ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]], [[58, 64], [139, 154]])])
    def test_matrixmult(self, A, B, expected):
        """
        This test provides coverage of the matrixmult function in pyCGM.py, defined as matrixmult(a, b)
        where a and b are both lists that represent a matrix to be multiplied.

        This test takes 3 parameters:
        A: a matrix, 2D array format
        B: a matrix, 2D array format
        expected: the expected matrix from calling matrixmult on A and B. This is the result of multiplying the two
        matrices A and B. It gives the correct result for multiplying two valid matrices, but still gives a result
        in some cases when the two matrices can't be multiplied. For two matrices to be multiplied, len(A[0]) need to
        be equal to len(B), but this function gives an output even when this isn't true
        """
        result = pycgmStatic.matrixmult(A, B)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_matrixmult_datatypes(self):
        """
        This test provides coverage of the matrixmult function in pycgmStatic.py, defined as matrixmult(a, b)
        where a and b are both lists that represent a matrix to be multiplied.

        This test checks that the resulting output from calling matrixmult is correct when called with a list of ints,
        a numpy array of ints, a list of floats, and a numpy array of floats.
        """
        A_int = [[1, 2, 0], [0, 1, 2]]
        B_int = [[2, 1], [1, 4]]
        A_float = [[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]]
        B_float = [[2.0, 1.0], [1.0, 4.0]]
        expected = [[4, 9], [1, 4]]

        # Check the calling matrixmult on a list of ints yields the expected results
        result_int_list = pycgmStatic.matrixmult(A_int, B_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling matrixmult on a numpy array of ints yields the expected results
        result_int_nparray = pycgmStatic.matrixmult(np.array(A_int, dtype='int'), np.array(B_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling matrixmult on a list of floats yields the expected results
        result_float_list = pycgmStatic.matrixmult(A_float, B_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling matrixmult on a numpy array of floats yields the expected results
        result_float_nparray = pycgmStatic.matrixmult(np.array(A_float, dtype='float'), np.array(B_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["a", "b", "expected"], [
        ([0.13232936, 0.98562946, -0.10499292], [-0.99119134, 0.13101088, -0.01938735],
         [-0.005353527183234709, 0.10663358915485248, 0.994283972218527]),
        ([0, 0, 0], [0, 0, 0], [0, 0, 0]),
        ([1, 1, 1], [1, 1, 1], [0, 0, 0]),
        ([0, 0, -2], [0, 4, 0], [8, 0, 0]),
        ([0, 0, 4], [-0.5, 0, 0], [0, -2, 0]),
        ([-1.5, 0, 0], [0, 4, 0], [0, 0, -6]),
        ([1, 0, 1], [0, 1, 0], [-1, 0, 1]),
        ([1, 2, 3], [3, 2, 1], [-4, 8, -4]),
        ([-2, 3, 1], [4, -1, 5], [16, 14, -10])])
    def test_cross(self, a, b, expected):
        """
        This test provides coverage of the cross function in pycgmStatic.py, defined as cross(a, b) where a and b are both 3D vectors.

        This test takes 3 parameters:
        a: 3D vector
        b: 3D vector
        expected: the expected result from calling cross on a and b. This result is the cross product of the vectors
        a and b.
        """
        result = pycgmStatic.cross(a, b)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_cross_datatypes(self):
        """
        This test provides coverage of the cross function in pycgmStatic.py, defined as cross(a, b) where a and b are both 3D vectors.

        This test checks that the resulting output from calling cross is correct when called with a list of ints, a numpy
        array of ints, a list of floats, and a numpy array of floats.
        """
        A_int = [-2, 3, 1]
        A_float = [-2.0, 3.0, 1.0]
        B_int = [4, -1, 5]
        B_float = [4.0, -1.0, 5.0]
        expected = [16, 14, -10]

        # Check the calling cross on a list of ints yields the expected results
        result_int_list = pycgmStatic.cross(A_int, B_int)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check the calling cross on a numpy array of ints yields the expected results
        result_int_nparray = pycgmStatic.cross(np.array(A_int, dtype='int'), np.array(B_int, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check the calling cross on a list of floats yields the expected results
        result_float_list = pycgmStatic.cross(A_float, B_float)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check the calling cross on a numpy array of floats yields the expected results
        result_float_nparray = pycgmStatic.cross(np.array(A_float, dtype='float'), np.array(B_float, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

class TestPycgmStaticGetStatic():
    """
    This class tests the getStatic function in pycgmStatic.py:
    """

    @classmethod
    def setup_class(cls):
        """
        Called once for all tests. Loads filenames to be used for testing getStatic() from SampleData/ROM/.
        """
        cwd = os.getcwd()
        if (cwd.split(os.sep)[-1] == "pyCGM_Single"):
            parent = os.path.dirname(cwd)
            os.chdir(parent)
        cls.cwd = os.getcwd()

        dynamic_trial, static_trial, vsk_file, _, _ = pyCGM_Helpers.getfilenames(x=2)
        cls.motion_data = pycgmIO.loadData(os.path.join(cls.cwd, static_trial))
        cls.vsk_data_original = pycgmIO.loadVSK(os.path.join(cls.cwd, vsk_file), dict=False)
        cls.vsk_data = cls.vsk_data_original.copy()

    def setup_method(self):
        """
        Called once before all tests in TestPycgmStaticGetStatic. Resets the vsk_data dictionary to its original state
        as returned from pycgmIO.loadVSK from the file SampleData/ROM/Sample_SM.vsk.
        """
        self.vsk_data = self.vsk_data_original.copy()

    @pytest.mark.parametrize("key", [
        ('LeftLegLength'),
        ('RightLegLength'),
        ('Bodymass'),
        ('LeftAsisTrocanterDistance'),
        ('InterAsisDistance'),
        ('LeftTibialTorsion'),
        ('RightTibialTorsion'),
        ('LeftShoulderOffset'),
        ('RightShoulderOffset'),
        ('LeftElbowWidth'),
        ('RightElbowWidth'),
        ('LeftWristWidth'),
        ('RightWristWidth'),
        ('LeftHandThickness'),
        ('RightHandThickness')])
    def test_getStatic_required_markers(self, key):
        """
        This function tests that an exception is raised when removing given keys from the vsk_data dictionary. All
        of the tested markers are required to exist in the vsk_data dictionary to run pycgmStatic.getStatic(), so
        deleting those keys should raise an exception.
        """
        del self.vsk_data[key]
        with pytest.raises(Exception):
            pycgmStatic.getStatic(self.motion_data, self.vsk_data)

    @pytest.mark.parametrize("key", [
        ('RightKneeWidth'),
        ('LeftKneeWidth'),
        ('RightAnkleWidth'),
        ('LeftAnkleWidth')])
    def test_getStatic_zero_markers(self, key):
        """
        This function tests that when deleting given keys from the vsk_data dictionary, the value in the resulting
        calSM is 0. All of the tested markers are set to 0 if they don't exist in pycgmStatic.getStatic(), so
        deleting these keys should not raise an exception.
        """
        del self.vsk_data[key]
        result = pycgmStatic.getStatic(self.motion_data, self.vsk_data)
        np.testing.assert_almost_equal(result[key], 0, rounding_precision)

    @pytest.mark.parametrize(["key", "keyVal"], [
        ('Bodymass', 95),
        ('InterAsisDistance', 28),
        ('RightKneeWidth', -11),
        ('LeftKneeWidth', 30),
        ('RightAnkleWidth', -41),
        ('LeftAnkleWidth', -5),
        ('LeftTibialTorsion', 28),
        ('RightTibialTorsion', -37),
        ('LeftShoulderOffset', 48),
        ('RightShoulderOffset', 93),
        ('LeftElbowWidth', -10),
        ('RightElbowWidth', -4),
        ('LeftWristWidth', 71),
        ('RightWristWidth', 9),
        ('LeftHandThickness', 15),
        ('RightHandThickness', -21),
        ('InterAsisDistance', -10)])
    def test_getStatic_marker_assignment(self, key, keyVal):
        """
        This function tests that when assigning a value to the given keys from the vsk_data dictionary, the value in
        the resulting calSM corresponds. All of the tested markers are assigned to calSM in pycgmStatic.getStatic()
        """
        self.vsk_data[key] = keyVal
        result = pycgmStatic.getStatic(self.motion_data, self.vsk_data)
        np.testing.assert_almost_equal(result[key], keyVal, rounding_precision)

    @pytest.mark.parametrize(["LeftLegLength", "RightLegLength", "MeanLegLengthExpected"], [
        (0, 0, 0),
        (0, 40, 20),
        (-34, 0, -17),
        (-15, 15, 0),
        (5, 46, 25.5)])
    def test_getStatic_MeanLegLength(self, LeftLegLength, RightLegLength, MeanLegLengthExpected):
        """
        This function tests that pycgmStatic.getStatic() properly calculates calSM['MeanLegLength'] by averaging the
        values in vsk_data['LeftLegLength'] and vsk_data['RightLegLength']
        """
        self.vsk_data['LeftLegLength'] = LeftLegLength
        self.vsk_data['RightLegLength'] = RightLegLength
        result = pycgmStatic.getStatic(self.motion_data, self.vsk_data)
        np.testing.assert_almost_equal(result['MeanLegLength'], MeanLegLengthExpected, rounding_precision)

    @pytest.mark.parametrize(["leftVal", "rightVal", "leftExpected", "rightExpected"], [
        # Test where left and right are 0
        (0, 0, 72.512, 72.512),
        # Test where left is 0
        (85, 0, 72.512, 72.512),
        # Test where right is 0
        (0, 61, 72.512, 72.512),
        # Test where left and right aren't 0
        (85, 61, 85, 61)])
    def test_getStatic_AsisToTrocanterMeasure(self, leftVal, rightVal, leftExpected, rightExpected):
        """
        Tests that if LeftAsisTrocanterDistance or RightAsisTrocanterDistance are 0 in the input vsk dictionary, their
        corresponding values in calSM will be calculated from LeftLegLength and RightLegLength, but if they both have
        values than they will be assigned from the vsk dictionary
        """
        self.vsk_data['LeftAsisTrocanterDistance'] = leftVal
        self.vsk_data['RightAsisTrocanterDistance'] = rightVal
        result = pycgmStatic.getStatic(self.motion_data, self.vsk_data)
        np.testing.assert_almost_equal(result['L_AsisToTrocanterMeasure'], leftExpected, rounding_precision)
        np.testing.assert_almost_equal(result['R_AsisToTrocanterMeasure'], rightExpected, rounding_precision)

    def test_getStatic_InterAsisDistance(self):
        """
        This function tests that when pycgmStatic.getStatic() is called with vsk_data['InterAsisDistance'] is 0, that
        the value for calSM['InterAsisDistance'] is calculated from motionData
        """
        self.vsk_data['InterAsisDistance'] = 0
        result = pycgmStatic.getStatic(self.motion_data, self.vsk_data)
        np.testing.assert_almost_equal(result['InterAsisDistance'], 215.9094195515741, rounding_precision)

    def test_gen(self):
        """
        This function tests that the correct values are returned in calSM['RightStaticRotOff'],
        calSM['RightStaticPlantFlex'], calSM['LeftStaticRotOff'], calSM['LeftStaticPlantFlex'], and calSM['HeadOffset'].
        All of these values are calculated from calls to staticCalculation and staticCalculationHead for every frame
        in motionData
        """
        result = pycgmStatic.getStatic(self.motion_data, self.vsk_data)
        np.testing.assert_almost_equal(result['RightStaticRotOff'], 0.015683497632642041, rounding_precision)
        np.testing.assert_almost_equal(result['RightStaticPlantFlex'], 0.27024179070027576, rounding_precision)
        np.testing.assert_almost_equal(result['LeftStaticRotOff'], 0.0094029102924030206, rounding_precision)
        np.testing.assert_almost_equal(result['LeftStaticPlantFlex'], 0.20251085737834015, rounding_precision)
        np.testing.assert_almost_equal(result['HeadOffset'], 0.25719904693106527, rounding_precision)