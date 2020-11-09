import pytest
import pyCGM_Single.pycgmStatic as pycgmStatic
import numpy as np

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
        expected: the expected result from calling rotmat on x, y, and z
        """
        result = pycgmStatic.rotmat(x, y, z)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_rotmat_datatypes(self):
        """
        This test provides coverage of the rotmat function in pycgmStatic.py, defined as rotmat(x, y, z)
        where x, y, and z are all floats that represent the angle of rotation in a particular dimension.

        This test checks that the resulting output from calling rotmat is correct for different input data types.
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
        expected: the expected result from calling getDist on p0 and p1
        """
        result = pycgmStatic.getDist(p0, p1)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    # getstatic
    '''
    @pytest.mark.parametrize(["motionData", "vsk", "flat_foot", "GCS", "expected"], [
        ({}, 
         {},
         [],
         [],
         []),
    ])
    def test_getstatic(self, motionData, vsk,flat_foot,GCS, expected):
        """
        This test provides coverage of the getstatic function in pycgmStatic.py, defined as getstatic(p0, p1)

        This test takes 3 parameters:
        p0: position of first x,y,z coordinate
        p1: position of second x,y,z coordinate
        expected: the expected result from calling getstatic on p0 and p1
        """
        result = pycgmStatic.getstatic(motionData, vsk, flat_foot, ,GCS)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    '''

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
        expected: the expected result from calling average on list
        """
        result = pycgmStatic.average(list)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

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
        ({'RASI': np.array([7, 2, -6]), 'LASI': np.array([3, -7, 2])}, 12.68857754044952)])
    def test_IADcalculation(self, frame, expected):
        """
        This test provides coverage of the IADcalculation function in pycgmStatic.py, defined as IADcalculation(frame)

        This test takes 2 parameters:
        frame: dictionary of marker lists
        expected: the expected result from calling IADcalculation on frame
        """
        result = pycgmStatic.IADcalculation(frame)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    '''
    @pytest.mark.parametrize(["axisP", "axisD", "expected"], [
        ([[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
         [[-0.02319771605294818, 0.9661337182824923, 0.2569970901929537],
          [-0.9997202678568442, -0.021232498241545272, -0.010419551558243256],
          [-0.004609989824245986, -0.257166909644468, 0.966356005092166]],
         0.25992807335420975),
    ])
    def test_headoffCalc(self, axisP, axisD, expected):
        """
        This test provides coverage of the headoffCalc function in pycgmStatic.py, defined as headoffCalc(axisP, axisD)

        This test takes 3 parameters:
        axisP: the unit vector of axisP, the position of the proximal axis
        axisD: the unit vector of axisD, the position of the distal axis
        expected: the expected result from calling headoffCalc on axisP and axis D
        """
        result = pycgmStatic.headoffCalc(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)
    '''

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
        expected: the expected result from calling headoffCalc on axisP and axisD
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given
        axisP = pycgmStatic.rotmat(xRot, yRot, zRot)
        axisD = pycgmStatic.rotmat(0, 0, 0)
        result = pycgmStatic.headoffCalc(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

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
         [[-0.590828, -0.802097, -0.554384], [ 0.955226,  0.156745,  0.166848]]),
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
         [[0.041042018208567545, -0.3065439019577841, -0.3106927663413161], [0.39326377295256626, 0.5657243847333632, 0.2128595189127902]])])
    def test_staticCalculation(self, frame, ankle_JC, flat_foot, vsk, expected):
        """
        This test provides coverage of the staticCalculation function in pycgmStatic.py, defined as staticCalculation(frame, ankle_JC, knee_JC, flat_foot, vsk)

        This test takes 5 parameters:
        frame: dictionary of marker lists
        ankle_JC: array containing the x,y,z axes marker positions of the ankle joint center
        flat_foot: boolean indicating if the feet are flat or not
        vsk: dictionary containing subject measurements from a VSK file
        expected: the expected result from calling staticCalculation on frame, ankle_JC, flat_foot, and vsk
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
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given
        axisP = pycgmStatic.rotmat(xRot, yRot, zRot)
        axisD = pycgmStatic.rotmat(0, 0, 0)
        result = pycgmStatic.getankleangle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

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
        expected: the expected result from calling norm2d on v
        """
        result = pycgmStatic.norm2d(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_norm2d_datatypes(self):
        """
        This test provides coverage of the norm2d function in pycgmStatic.py, defined as norm2d(v) where v is a 3D vector.

        This test checks that the resulting output from calling norm2d is correct for different input data types.
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
        expected: the expected result from calling norm3d on v
        """
        result = pycgmStatic.norm3d(v)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_norm3d_datatypes(self):
        """
        This test provides coverage of the norm3d function in pycgmStatic.py, defined as norm3d(v) where v is a 3D vector.

        This test checks that the resulting output from calling norm3d is correct for different input data types.
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

    @pytest.mark.parametrize(["A", "B", "expected"], [
        ([[1, 0, 0], [0, 1.0, -0.0], [0, 0.0, 1.0]], [[1.0, 0, 0.0], [0, 1, 0], [-0.0, 0, 1.0]], [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ([[1]], [[1]], [[1]]),
        ([[2], [1]], [[1, 2]], [[2, 4], [1, 2]]),
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
        expected: the expected result from calling matrixmult on A and B
        """
        result = pycgmStatic.matrixmult(A, B)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_matrixmult_datatypes(self):
        """
        This test provides coverage of the matrixmult function in pycgmStatic.py, defined as matrixmult(a, b)
        where a and b are both lists that represent a matrix to be multiplied.

        This test checks that the resulting output from calling matrixmult is correct for different input data types.
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
        expected: the expected result from calling cross on a and b
        """
        result = pycgmStatic.cross(a, b)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_cross_datatypes(self):
        """
        This test provides coverage of the cross function in pycgmStatic.py, defined as cross(a, b) where a and b are both 3D vectors.

        This test checks that the resulting output from calling cross is correct for different input data types.
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