import unittest
import pyCGM_Single.pycgmStatic as pycgmStatic
import numpy as np

rounding_precision = 6

class TestPycgmStaticUtils(unittest.TestCase):
    def test_cross(self):
        """
        This test provides coverage of the cross function in pycgmStatic.py, defined as cross(a, b) where a and b are both 3D vectors.
        The list testcases consists of lists that each represent a different test case. Each test case consists of
        the two 3D arrays, a and b, and the expected result from calling cross on these parameters.
        The first part of this test iterates over testcases and checks to ensure the resulting output from
        calling cross matches the expected output for each test case.
        The second part of this test checks that the resulting output from calling cross is correct for
        different input data types.
        """
        testcases = [[[0.13232936, 0.98562946, -0.10499292], [-0.99119134, 0.13101088, -0.01938735],
                      [-0.005353527183234709, 0.10663358915485248, 0.994283972218527]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                     [[0, 0, -2], [0, 4, 0], [8, 0, 0]],
                     [[0, 0, 4], [-0.5, 0, 0], [0, -2, 0]],
                     [[-1.5, 0, 0], [0, 4, 0], [0, 0, -6]],
                     [[1, 0, 1], [0, 1, 0], [-1, 0, 1]],
                     [[1, 2, 3], [3, 2, 1], [-4, 8, -4]],
                     [[-2, 3, 1], [4, -1, 5], [16, 14, -10]]]

        for testcase in testcases:
            # Call cross(a, b) with the parameters given from each index in testcases.
            result = pycgmStatic.cross(testcase[0], testcase[1])
            np.testing.assert_almost_equal(result, testcase[2], rounding_precision)

        # Initialization for testing data types
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

    def test_norm2d(self):
        """
        This test provides coverage of the norm2d function in pycgmStatic.py, defined as norm2d(v) where v is a 3D vector.
        The list testcases consists of lists that each represent a different test case. Each test case consists of
        a 3D array, v, and the expected result from calling norm2d on this parameter.
        The first part of this test iterates over testcases and checks to ensure the resulting output from
        calling norm2d matches the expected output for each test case.
        The second part of this test checks that the resulting output from calling norm2d is correct for
        different input data types.
        """
        testcases = [[[-9944.089508486479, -20189.20612828088, 150.42955108569652], 22505.812344655435],
                     [[0, 0, 0], 0],
                     [[2, 0, 0], 2],
                     [[0, 0, -1], 1],
                     [[0, 3, 4], 5],
                     [[-3, 0, 4], 5],
                     [[6, -8, 0], 10],
                     [[-5, 0, -12], 13],
                     [[1, -1, np.sqrt(2)], 2]]

        for testcase in testcases:
            # Call norm2d(v) with the parameter given from each index in testcases.
            result = pycgmStatic.norm2d(testcase[0])
            np.testing.assert_almost_equal(result, testcase[1], rounding_precision)

        # Initialization for testing data types
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

    def test_norm3d(self):
        """
        This test provides coverage of the norm3d function in pycgmStatic.py, defined as norm3d(v) where v is a 3D vector.
        The list testcases consists of lists that each represent a different test case. Each test case consists of
        a 3D array, v, and the expected result from calling norm3d on this parameter.
        The first part of this test iterates over testcases and checks to ensure the resulting output from
        calling norm3d matches the expected output for each test case.
        The second part of this test checks that the resulting output from calling norm3d is correct for
        different input data types.
        """
        testcases = [[[-212.5847168, 28.09841919, -4.15808105], np.array(214.47394390603984)],
                     [[0, 0, 0], np.array(0)],
                     [[2, 0, 0], np.array(2)],
                     [[0, 0, -1], np.array(1)],
                     [[0, 3, 4], np.array(5)],
                     [[-3, 0, 4], np.array(5)],
                     [[-6, 8, 0], np.array(10)],
                     [[-5, 0, -12], np.array(13)],
                     [[1, -1, np.sqrt(2)], np.array(2)]]

        for testcase in testcases:
            # Call norm3d(v) with the parameter given from each index in testcases.
            result = pycgmStatic.norm3d(testcase[0])
            np.testing.assert_almost_equal(result, testcase[1], rounding_precision)

        # Initialization for testing data types
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

    def test_matrixmult(self):
        """
        This test provides coverage of the matrixmult function in pycgmStatic.py, defined as matrixmult(a, b)
        where a and b are both lists that represent a matrix to be multiplied.
        The list testcases consists of lists that each represent a different test case. Each test case consists of
        two lists, a and b, and the expected result from calling matrixmult on this parameter.
        The first part of this test iterates over testcases and checks to ensure the resulting output from
        calling matrixmult matches the expected output for each test case.
        The second part of this test checks that the resulting output from calling matrixmult is correct for
        different input data types.
        """
        testcases = [[[[1, 0, 0], [0, 1.0, -0.0], [0, 0.0, 1.0]], [[1.0, 0, 0.0], [0, 1, 0], [-0.0, 0, 1.0]],
                      [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
                     [[[1]], [[1]], [[1]]],
                     [[[2], [1]], [[1, 2]], [[2, 4], [1, 2]]],
                     [[[1, 2, 0], [0, 1, 2]], [[2, 1], [1, 4]], [[4, 9], [1, 4]]],
                     [[[11, 12, 13], [14, 15, 16]], [[1, 2], [3, 4], [5, 6]], [[112, 148], [139, 184]]],
                     [[[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]], [[58, 64], [139, 154]]]]

        for testcase in testcases:
            # Call matrixmult(a, b) with the parameters given from each index in testcases.
            result = pycgmStatic.matrixmult(testcase[0], testcase[1])
            np.testing.assert_almost_equal(result, testcase[2], rounding_precision)

        # Initialization for testing data types
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

    def test_rotmat(self):
        """
        This test provides coverage of the rotmat function in pycgmStatic.py, defined as rotmat(x, y, z)
        where x, y, and z are all floats that represent the angle of rotation in a particular dimension.
        The list testcases consists of lists that each represent a different test case. Each test case consists of
        three floats, x, y, and z, and the expected result from calling rotmat on this parameter.
        The first part of this test iterates over testcases and checks to ensure the resulting output from
        calling rotmat matches the expected output for each test case.
        The second part of this test checks that the resulting output from calling rotmat is correct for
        different input data types.
        """
        testcases = [[0.0, 0.0, 180, [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]],
                     [0, 0, 0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                     [90, 0, 0, [[1, 0, 0], [0, 0, -1], [0, 1, 0]]],
                     [0, 135, 0, [[-0.70710678, 0, 0.70710678], [0, 1, 0], [-0.70710678, 0, -0.70710678]]],
                     [0, 0, -60, [[0.5, 0.8660254, 0], [-0.8660254, 0.5, 0], [0, 0, 1]]],
                     [90, 0, 90, [[0, -1, 0], [0, 0, -1], [1, 0, 0]]],
                     [0, 150, -30, [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]],
                     [90, 180, -90, [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]]]

        for testcase in testcases:
            # Call rotmat(x, y, z) with the parameters given from each index in testcases.
            result = pycgmStatic.rotmat(testcase[0], testcase[1], testcase[2])
            np.testing.assert_almost_equal(result, testcase[3], rounding_precision)

        # Initialization for testing data types
        result_int = pycgmStatic.rotmat(0, 150, -30)
        result_float = pycgmStatic.rotmat(0.0, 150.0, -30.0)
        expected = [[-0.75, -0.4330127, 0.5], [-0.5, 0.8660254, 0], [-0.4330127, -0.25, -0.8660254]]

        # Check that calling rotmat yields the expected results when called with ints
        np.testing.assert_almost_equal(result_int, expected, rounding_precision)

        # Check that calling rotmat yields the expected results when called with floats
        np.testing.assert_almost_equal(result_float, expected, rounding_precision)

    def test_average(self):
        testcases = [
            [[0], 0],
            [[3], 3],
            [[-1], -1],
            [[1, 2], 1.5],
            [[-1, 3], 1],
            [[-3, 1], -1],
            [[-2, 0, 2], 0],
            [[1, 2, 3, 4, 5], 3],
            [[-1, -2, -3, -4, -5], -3],
            [[0.1, 0.2, 0.3, 0.4, 0.5], 0.3],
        ]
        for testcase in testcases:
            result = pycgmStatic.average(testcase[0])
            np.testing.assert_almost_equal(result, testcase[1], rounding_precision)

    def test_getDist(self):
        testcases = [
            [[0, 0, 0], [1, 0, 0], 1],
        ]
        for testcase in testcases:
            result = pycgmStatic.getDist(testcase[0], testcase[1])
            np.testing.assert_almost_equal(result, testcase[2], rounding_precision)

    def test_IADcalculation(self):
        testcases = [
            [{'RASI': np.array([0, 0, 0]), 'LASI': np.array([1, 0, 0])}, 1],
        ]
        for testcase in testcases:
            result = pycgmStatic.IADcalculation(testcase[0])
            np.testing.assert_almost_equal(result, testcase[1], rounding_precision)

    def test_headoffCalc(self):
        testcases = [
            [[[0, 1, 0], [-1, 0, 0], [0, 0, 1]], [[-0.02319771605294818, 0.9661337182824923, 0.2569970901929537], [-0.9997202678568442, -0.021232498241545272, -0.010419551558243256], [-0.004609989824245986, -0.257166909644468, 0.966356005092166]], 0.25992807335420975],
        ]
        for testcase in testcases:
            result = pycgmStatic.headoffCalc(testcase[0], testcase[1])
            np.testing.assert_almost_equal(result, testcase[2], rounding_precision)

    def test_getankleangle(self):
        testcases = [
            [[[0.09057882, 0.27923024, 0.95594244], [-0.966178, 0.25735564, 0.01637524], [-0.24144471, -0.92509381, 0.29309708]], [[-0.12471782, 0.05847945, 0.99046737], [-0.96230839, 0.23602646, -0.13510763], [-0.24167752, -0.9699854, 0.02683856]], [-0.015688860223839234, 0.2703999495115947, -0.15237642705642993]],
        ]
        for testcase in testcases:
            result = pycgmStatic.getankleangle(testcase[0], testcase[1])
            np.testing.assert_almost_equal(result, testcase[2], rounding_precision)

    def test_staticCalculation(self):
        testcases = [
            [{'RTOE': np.array([433.33508301, 354.97229004,  44.27765274]),  'LTOE': np.array([ 31.77310181, 331.23657227,  42.15322876]), 'RHEE': np.array([381.88534546, 148.47607422,  49.99120331]), 'LHEE': np.array([122.18766785, 138.55477905,  46.29433441]),},
             [np.array([397.45738291, 217.50712216, 87.83068433]), np.array([112.28082818, 175.83265027, 80.98477997]), [[np.array([398.14685839, 218.23110187, 87.8088449]), np.array([396.73749179, 218.18875543, 87.69979179]), np.array([397.37750585, 217.61309136, 88.82184031])], [np.array([111.92715492, 176.76246715, 80.88301651]), np.array([111.34886681, 175.49163538, 81.10789314]), np.array([112.36059802, 175.97103172, 81.97194123])]]],
             False, {}, [[-0.015688860223839234, 0.2703999495115947, -0.15237642705642993], [0.009550866847196991, 0.20242596489042683, -0.019420801722458948]]],
        ]
        for testcase in testcases:
            result = pycgmStatic.staticCalculation(testcase[0], testcase[1], None, testcase[2], testcase[3])
            np.testing.assert_almost_equal(result, testcase[4], rounding_precision)