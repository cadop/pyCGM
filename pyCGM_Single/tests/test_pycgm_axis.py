from unittest.mock import patch
import pyCGM_Single.pyCGM as pyCGM
import pytest
import numpy as np

rounding_precision = 6

class TestUpperBodyAxis():
    """
    This class tests the upper body axis functions in pyCGM.py:
    headJC
    thoraxJC
    findshoulderJC
    shoulderAxisCalc
    elbowJointCenter
    wristJointCenter
    handJointCenter
    """
    nan_3d = [np.nan, np.nan, np.nan]
    rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]

    @pytest.mark.parametrize(["frame", "vsk", "expected"], [
        # Test from running sample data
        ({'LFHD': np.array([184.55158997, 409.68713379, 1721.34289551]), 'RFHD': np.array([325.82983398, 402.55450439, 1722.49816895]), 'LBHD': np.array([197.8621521 , 251.28889465, 1696.90197754]), 'RBHD': np.array([304.39898682, 242.91339111, 1694.97497559])},
         {'HeadOffset': 0.2571990469310653},
         [[[255.21685582510975, 407.11593887758056, 1721.8253843887082], [254.19105385179665, 406.146809183757, 1721.9176771191715], [255.19034370229795, 406.2160090443217, 1722.9159912851449]], [255.19071197509766, 406.1208190917969, 1721.9205322265625]]),
        # Basic test with a variance of 1 in the x and y dimensions of the markers
        ({'LFHD': np.array([1, 1, 0]), 'RFHD': np.array([0, 1, 0]), 'LBHD': np.array([1, 0, 0]), 'RBHD': np.array([0, 0, 0])},
         {'HeadOffset': 0.0},
         [[[0.5, 2, 0], [1.5, 1, 0], [0.5, 1, -1]], [0.5, 1, 0]]),
        # Setting the markers so there's no variance in the x-dimension
        ({'LFHD': np.array([0, 1, 0]), 'RFHD': np.array([0, 1, 0]), 'LBHD': np.array([0, 0, 0]), 'RBHD': np.array([0, 0, 0])},
         {'HeadOffset': 0.0},
         [[nan_3d, nan_3d, nan_3d], [0, 1, 0]]),
        # Setting the markers so there's no variance in the y-dimension
        ({'LFHD': np.array([1, 0, 0]), 'RFHD': np.array([0, 0, 0]), 'LBHD': np.array([1, 0, 0]), 'RBHD': np.array([0, 0, 0])},
         {'HeadOffset': 0.0},
         [[nan_3d, nan_3d, nan_3d], [0.5, 0, 0]]),
        # Setting each marker in a different xy quadrant
        ({'LFHD': np.array([-1, 1, 0]), 'RFHD': np.array([1, 1, 0]), 'LBHD': np.array([-1, -1, 0]), 'RBHD': np.array([1, -1, 0])},
         {'HeadOffset': 0.0},
         [[[0, 2, 0], [-1, 1, 0], [0, 1, 1]], [0, 1, 0]]),
        # Setting values of the markers so that midpoints will be on diagonals
        ({'LFHD': np.array([-2, 1, 0]), 'RFHD': np.array([1, 2, 0]), 'LBHD': np.array([-1, -2, 0]), 'RBHD': np.array([2, -1, 0])},
         {'HeadOffset': 0.0},
         [[[-0.81622777, 2.4486833 ,  0], [-1.4486833, 1.18377223, 0], [-0.5, 1.5,  1]], [-0.5, 1.5, 0]]),
        # Adding the value of 1 in the z dimension for all 4 markers
        ({'LFHD': np.array([1, 1, 1]), 'RFHD': np.array([0, 1, 1]), 'LBHD': np.array([1, 0, 1]), 'RBHD': np.array([0, 0, 1])},
         {'HeadOffset': 0.0},
         [[[0.5, 2, 1], [1.5, 1, 1], [0.5, 1, 0]], [0.5, 1, 1]]),
        # Setting the z dimension value higher for LFHD and LBHD
        ({'LFHD': np.array([1, 1, 2]), 'RFHD': np.array([0, 1, 1]), 'LBHD': np.array([1, 0, 2]), 'RBHD': np.array([0, 0, 1])},
         {'HeadOffset': 0.0},
         [[[0.5, 2, 1.5], [1.20710678, 1, 2.20710678], [1.20710678, 1, 0.79289322]], [0.5, 1, 1.5]]),
        # Setting the z dimension value higher for LFHD and RFHD
        ({'LFHD': np.array([1, 1, 2]), 'RFHD': np.array([0, 1, 2]), 'LBHD': np.array([1, 0, 1]), 'RBHD': np.array([0, 0, 1])},
         {'HeadOffset': 0.0},
         [[[0.5, 1.70710678, 2.70710678], [1.5, 1, 2], [0.5, 1.70710678, 1.29289322]], [0.5, 1, 2]]),
        # Adding a value for HeadOffset
        ({'LFHD': np.array([1, 1, 0]), 'RFHD': np.array([0, 1, 0]), 'LBHD': np.array([1, 0, 0]), 'RBHD': np.array([0, 0, 0])},
         {'HeadOffset': 0.5},
         [[[0.5, 1.87758256, 0.47942554], [1.5, 1, 0], [0.5, 1.47942554, -0.87758256]], [0.5, 1, 0]])])
    def testHeadJC(self, frame, vsk, expected):
        """
        This test provides coverage of the headJC function in pyCGM.py, defined as headJC(frame, vsk)

        This test takes 3 parameters:
        frame: dictionary of marker lists
        vsk: dictionary containing subject measurements from a VSK file
        expected: the expected result from calling headJC on frame and vsk

        This test is checking to make sure the head joint center and head joint axis are calculated correctly given
        the 4 coordinates given in frame. This includes testing when there is no variance in the coordinates,
        when the coordinates are in different quadrants, when the midpoints will be on diagonals, and when the z
        dimension is variable. It also checks to see the difference when a value is set for HeadOffSet in vsk.
        """
        result = pyCGM.headJC(frame, vsk)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)

    @pytest.mark.parametrize(["frame", "expected"], [
        # Test from running sample data
        ({'C7': np.array([251.22619629, 229.75683594, 1533.77624512]), 'T10': np.array([228.64323425, 192.32041931, 1279.6418457]), 'CLAV': np.array([256.78051758, 371.28042603, 1459.70300293]), 'STRN': np.array([251.67492676, 414.10391235, 1292.08508301])},
        [[[256.23991128535846, 365.30496976939753, 1459.662169500559], [257.1435863244796, 364.21960599061947, 1459.588978712983], [256.0843053658035, 364.32180498523223, 1458.6575930699294]], [256.149810236564, 364.3090603933987, 1459.6553639290375]]),
        # Basic test with a variance of 1 in the x and y dimensions of the markers
        ({'C7': np.array([1, 1, 0]), 'T10': np.array([0, 1, 0]), 'CLAV': np.array([1, 0, 0]), 'STRN': np.array([0, 0, 0])},
         [[[1, 6, 0], [1, 7, 1], [0, 7, 0]], [1, 7, 0]]),
        # Setting the markers so there's no variance in the x-dimension
        ({'C7': np.array([0, 1, 0]), 'T10': np.array([0, 1, 0]), 'CLAV': np.array([0, 0, 0]), 'STRN': np.array([0, 0, 0])},
         [[nan_3d, nan_3d, nan_3d], nan_3d]),
        # Setting the markers so there's no variance in the y-dimension
        ({'C7': np.array([1, 0, 0]), 'T10': np.array([0, 0, 0]), 'CLAV': np.array([1, 0, 0]), 'STRN': np.array([0, 0, 0])},
         [[nan_3d, nan_3d, nan_3d], nan_3d]),
        # Setting each marker in a different xy quadrant
        ({'C7': np.array([-1, 1, 0]), 'T10': np.array([1, 1, 0]), 'CLAV': np.array([-1, -1, 0]), 'STRN': np.array([1, -1, 0])},
         [[[-1, 5, 0], [-1, 6, -1], [0, 6, 0]], [-1, 6, 0]]),
        # Setting values of the markers so that midpoints will be on diagonals
        ({'C7': np.array([-2, 1, 0]), 'T10': np.array([1, 2, 0]), 'CLAV': np.array([-1, -2, 0]), 'STRN': np.array([2, -1, 0])},
         [[[-2.8973666, 3.69209979, 0], [-3.21359436, 4.64078309, -1], [-2.26491106, 4.95701085, 0]], [-3.21359436, 4.64078309, 0]]),
        # Adding the value of 1 in the z dimension for all 4 markers
        ({'C7': np.array([1, 1, 1]), 'T10': np.array([0, 1, 1]), 'CLAV': np.array([1, 0, 1]), 'STRN': np.array([0, 0, 1])},
         [[[1, 6, 1], [1, 7, 2], [0, 7, 1]], [1, 7, 1]]),
        # Setting the z dimension value higher for C7 and CLAV
        ({'C7': np.array([1, 1, 2]), 'T10': np.array([0, 1, 1]), 'CLAV': np.array([1, 0, 2]), 'STRN': np.array([0, 0, 1])},
         [[[1, 6, 2], [0.29289322, 7, 2.70710678], [0.29289322, 7, 1.29289322]], [1, 7, 2]]),
        # Setting the z dimension value higher for C7 and T10
        ({'C7': np.array([1, 1, 2]), 'T10': np.array([0, 1, 2]), 'CLAV': np.array([1, 0, 1]), 'STRN': np.array([0, 0, 1])},
         [[[1, 4.24264069, 5.24264069], [1, 4.24264069, 6.65685425], [0, 4.94974747, 5.94974747]], [1, 4.94974747, 5.94974747]])])
    def testThoraxJC(self, frame, expected):
        """
        This test provides coverage of the thoraxJC function in pyCGM.py, defined as thoraxJC(frame)

        This test takes 2 parameters:
        frame: dictionary of marker lists
        expected: the expected result from calling thoraxJC on frame

        This test is checking to make sure the thorax joint center and thorax joint axis are calculated correctly given
        the 4 coordinates given in frame. This includes testing when there is no variance in the coordinates,
        when the coordinates are in different quadrants, when the midpoints will be on diagonals, and when the z
        dimension is variable.
        """
        result = pyCGM.thoraxJC(frame)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)

    @pytest.mark.parametrize(["frame", "thorax", "wand", "vsk", "expected_args"], [
        # Test from running sample data
        ({'RSHO': np.array([428.88476562, 270.552948, 1500.73010254]), 'LSHO': np.array([68.24668121, 269.01049805, 1510.1072998])},
         [[[256.23991128535846, 365.30496976939753, 1459.662169500559], [257.1435863244796, 364.21960599061947, 1459.588978712983], [256.0843053658035, 364.32180498523223, 1458.6575930699294]], [256.149810236564, 364.3090603933987, 1459.6553639290375]],
         [[255.92550222678443, 364.3226950497605, 1460.6297868417887], [256.42380097331767, 364.27770361353487, 1460.6165849382387]],
         {'RightShoulderOffset': 40.0, 'LeftShoulderOffset': 40.0},
         [[[255.92550222678443, 364.3226950497605, 1460.6297868417887], [256.149810236564, 364.3090603933987, 1459.6553639290375], np.array([ 428.88476562,  270.552948  , 1500.73010254]), 47.0],
         [[256.42380097331767, 364.27770361353487, 1460.6165849382387], [256.149810236564, 364.3090603933987, 1459.6553639290375], np.array([68.24668121, 269.01049805, 1510.1072998]), 47.0]]),
        # Basic test with zeros for all params
        ({'RSHO': np.array([0, 0, 0]), 'LSHO': np.array([0, 0, 0])},
         [[rand_coor, rand_coor, rand_coor], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0]],
         {'RightShoulderOffset': 0.0, 'LeftShoulderOffset': 0.0},
         [[[0, 0, 0], [0, 0, 0], np.array([0, 0, 0]), 7.0],
         [[0, 0, 0], [0, 0, 0], np.array([0, 0, 0]), 7.0]]),
        # Testing when values are added to RSHO and LSHO
        ({'RSHO': np.array([2, -1, 3]), 'LSHO': np.array([-3, 1, 2])},
         [[rand_coor, rand_coor, rand_coor], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0]],
         {'RightShoulderOffset': 0.0, 'LeftShoulderOffset': 0.0},
         [[[0, 0, 0], [0, 0, 0], np.array([2, -1, 3]), 7.0],
         [[0, 0, 0], [0, 0, 0], np.array([-3, 1, 2]), 7.0]]),
        # Testing when a value is added to thorax_origin
        ({'RSHO': np.array([0, 0, 0]), 'LSHO': np.array([0, 0, 0])},
         [[rand_coor, rand_coor, rand_coor], [5, -2, 7]],
         [[0, 0, 0], [0, 0, 0]],
         {'RightShoulderOffset': 0.0, 'LeftShoulderOffset': 0.0},
         [[[0, 0, 0], [5, -2, 7], np.array([0, 0, 0]), 7.0],
         [[0, 0, 0], [5, -2, 7], np.array([0, 0, 0]), 7.0]]),
        # Testing when a value is added to wand
        ({'RSHO': np.array([0, 0, 0]), 'LSHO': np.array([0, 0, 0])},
         [[rand_coor, rand_coor, rand_coor], [0, 0, 0]],
         [[2, 6, -4], [-3, 5, 2]],
         {'RightShoulderOffset': 0.0, 'LeftShoulderOffset': 0.0},
         [[[2, 6, -4], [0, 0, 0], np.array([0, 0, 0]), 7.0],
         [[-3, 5, 2], [0, 0, 0], np.array([0, 0, 0]), 7.0]]),
        # Testing when values are added to RightShoulderOffset and LeftShoulderOffset
        ({'RSHO': np.array([0, 0, 0]), 'LSHO': np.array([0, 0, 0])},
         [[rand_coor, rand_coor, rand_coor], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0]],
         {'RightShoulderOffset': 20.0, 'LeftShoulderOffset': -20.0},
         [[[0, 0, 0], [0, 0, 0], np.array([0, 0, 0]), 27.0],
         [[0, 0, 0], [0, 0, 0], np.array([0, 0, 0]), -13.0]]),
        # Adding when values are added to all params
        ({'RSHO': np.array([3, -5, 2]), 'LSHO': np.array([-7, 3 , 9])},
         [[rand_coor, rand_coor, rand_coor], [-1, -9, -5]],
         [[-7, -1, 5], [5, -9, 2]],
         {'RightShoulderOffset': -6.0, 'LeftShoulderOffset': 42.0},
         [[[-7, -1, 5], [-1, -9, -5], np.array([3, -5, 2]), 1.0],
         [[5, -9, 2], [-1, -9, -5], np.array([-7, 3 , 9]), 49.0]])])
    def testFindShoulderJC(self, frame, thorax, wand, vsk, expected_args):
        """
        This test provides coverage of the findshoulderJC function in pyCGM.py, defined as findshoulderJC(frame, thorax, wand, vsk)

        This test takes 5 parameters:
        frame: dictionary of marker lists
        thorax: array containing several x,y,z markers for the thorax
        wand: array containing two x,y,z markers for wand
        vsk: dictionary containing subject measurements from a VSK file
        expected_args: the expected arguments used to call the mocked function, findJointC

        This test is checking to make sure the shoulder joint center is calculated correctly given the input parameters.
        This tests mocks findJointC to make sure the correct parameters are being passed into it given the parameters
        passed into findshoulderJC. The test checks to see that the correct values in expected_args are updated per
        each input parameter added:
        When values are added to frame, expected_args[0][2] and expected_args[1][2] should be updated
        When values are added to thorax, expected_args[0][1] and expected_args[1][1] should be updated
        When values are added to wand, expected_args[0][0] and expected_args[1][0] should be updated
        When values are added to vsk, expected_args[0][3] and expected_args[1][3] should be updated
        """
        rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]
        with patch.object(pyCGM, 'findJointC', return_value=rand_coor) as mock_findJointC:
            result = pyCGM.findshoulderJC(frame, thorax, wand, vsk)

        # Asserting that there were only 2 calls to findJointC
        np.testing.assert_equal(mock_findJointC.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to findJointC
        np.testing.assert_almost_equal(expected_args[0][0], mock_findJointC.call_args_list[0][0][0], rounding_precision)
        np.testing.assert_almost_equal(expected_args[0][1], mock_findJointC.call_args_list[0][0][1], rounding_precision)
        np.testing.assert_almost_equal(expected_args[0][2], mock_findJointC.call_args_list[0][0][2], rounding_precision)
        np.testing.assert_almost_equal(expected_args[0][3], mock_findJointC.call_args_list[0][0][3], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to findJointC
        np.testing.assert_almost_equal(expected_args[1][0], mock_findJointC.call_args_list[1][0][0], rounding_precision)
        np.testing.assert_almost_equal(expected_args[1][1], mock_findJointC.call_args_list[1][0][1], rounding_precision)
        np.testing.assert_almost_equal(expected_args[1][2], mock_findJointC.call_args_list[1][0][2], rounding_precision)
        np.testing.assert_almost_equal(expected_args[1][3], mock_findJointC.call_args_list[1][0][3], rounding_precision)

        # Asserting that findShoulderJC returned the correct result given the return value given by mocked findJointC
        np.testing.assert_almost_equal(result[0], rand_coor, rounding_precision)
        np.testing.assert_almost_equal(result[1], rand_coor, rounding_precision)

    @pytest.mark.parametrize(["thorax", "shoulderJC", "wand", "expected"], [
        # Test from running sample data
        ([[[256.23991128535846, 365.30496976939753, 1459.662169500559], [257.1435863244796, 364.21960599061947, 1459.588978712983], [256.0843053658035, 364.32180498523223, 1458.6575930699294]],  [256.149810236564, 364.3090603933987, 1459.6553639290375]],
         [np.array([429.66951995, 275.06718615, 1453.95397813]), np.array([64.51952734, 274.93442161, 1463.6313334 ])],
         [[255.92550222678443, 364.3226950497605, 1460.6297868417887], [256.42380097331767, 364.27770361353487, 1460.6165849382387]],
         [[np.array([429.66951995, 275.06718615, 1453.95397813]), np.array([64.51952734, 274.93442161, 1463.6313334 ])],
          [[[430.12731330596756, 275.9513661907463, 1454.0469882869343], [429.6862168456729, 275.1632337671314, 1452.9587414419757],  [428.78061812142147, 275.5243518770602, 1453.9831850281803]],
           [[64.10400324869988, 275.83192826468195, 1463.7790545425955],  [64.59882848203122, 274.80838068265837, 1464.620183745389],  [65.42564601518438, 275.3570272042577, 1463.6125331307376]]]]),
        # Test with zeros for all params
        ([[rand_coor, rand_coor, rand_coor], [0, 0, 0]],
         [np.array([0, 0, 0]),  np.array([0, 0, 0])],
         [[0, 0, 0], [0, 0, 0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when adding values in thorax but zeros for all other params
        ([[rand_coor, rand_coor, rand_coor], [8, 2, -6]],
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[0, 0, 0], [0, 0, 0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[nan_3d, nan_3d, [0.78446454, 0.19611614, -0.58834841]],
           [nan_3d, nan_3d, [0.78446454, 0.19611614, -0.58834841]]]]),
        # Testing when adding values in shoulderJC but zeros for all other params
        ([[rand_coor, rand_coor, rand_coor], [0, 0, 0]],
         [np.array([1, 5, -3]), np.array([0, -9, 2])],
         [[0, 0, 0], [0, 0, 0]],
         [[np.array([1, 5, -3]), np.array([0, -9, 2])],
          [[nan_3d, nan_3d, [0.830969149054, 4.154845745271, -2.4929074471]],
           [nan_3d, nan_3d, [0.0, -8.02381293981, 1.783069542181]]]]),
        # Testing when adding values in wand but zeros for all other params
        ([[rand_coor, rand_coor, rand_coor], [0, 0, 0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[1, 0, -7], [-3, 5, 3]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when adding values to thorax and shoulderJC
        ([[rand_coor, rand_coor, rand_coor], [8, 2, -6]],
         [np.array([1, 5, -3]), np.array([0, -9, 2])],
         [[0, 0, 0], [0, 0, 0]],
         [[np.array([1, 5, -3]), np.array([0, -9, 2])],
          [[[0.50428457, 4.62821343, -3.78488277], [1.15140320, 5.85290468, -3.49963055], [1.85518611, 4.63349167, -3.36650833]],
           [[-0.5611251741, -9.179560055, 1.191979749], [-0.65430149, -8.305871473, 2.3001252440], [0.5069794004, -8.302903324, 1.493020599]]]]),
        # Testing when adding values to thorax and wand
        ([[rand_coor, rand_coor, rand_coor], [8, 2, -6]],
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[1, 0, -7], [-3, 5, 3]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[[-0.269430125, 0.96225044, -0.03849001], [0.55859, 0.18871284, 0.80769095], [0.78446454, 0.19611614, -0.58834841]],
           [[-0.6130824329, 0.10218040549, -0.7833831087], [-0.09351638899, 0.9752423423, 0.20039226212], [0.7844645405, 0.19611613513, -0.5883484054]]]]),
        # Testing when adding values to shoulderJC and wand
        ([[rand_coor, rand_coor, rand_coor], [0, 0, 0]],
         [np.array([1, 5, -3]), np.array([0, -9, 2])],
         [[1, 0, -7], [-3, 5, 3]],
         [[np.array([1, 5, -3]), np.array([0, -9, 2])],
          [[[1.98367400, 4.88758011, -2.85947514], [0.93824211, 5.52256679, -2.14964131], [0.83096915, 4.15484575, -2.49290745]],
           [[-0.80094836, -9.12988352, 1.41552417], [-0.59873343, -8.82624991, 2.78187543], [0.0, -8.02381294, 1.78306954]]]]),
        # Testing when adding values to thorax, shoulderJC and wand
        ([[rand_coor, rand_coor, rand_coor], [8, 2, -6]],
         [np.array([1, 5, -3]), np.array([0, -9, 2])],
         [[1, 0, -7], [-3, 5, 3]],
         [[np.array([1, 5, -3]), np.array([0, -9, 2])],
          [[[0.93321781, 5.62330046, -3.77912558], [1.51400083, 5.69077360, -2.49143833], [1.85518611, 4.63349167, -3.36650833]],
           [[-0.64460664, -9.08385127, 1.24009787], [-0.57223612, -8.287942994, 2.40684228], [0.50697940, -8.30290332, 1.4930206]]]])])
    def testShoulderAxisCalc(self, thorax, shoulderJC, wand, expected):
        """
        This test provides coverage of the shoulderAxisCalc function in pyCGM.py, defined as shoulderAxisCalc(frame, thorax, shoulderJC, wand)

        This test takes 4 parameters:
        thorax: array containing several x,y,z markers for the thorax
        shoulderJC: array containing x,y,z position of the shoulder joint center
        wand: array containing two x,y,z markers for wand
        expected: the expected result from calling shoulderAxisCalc on thorax, shoulderJC, and wand

        This test is checking to make sure the shoulder joint axis is calculated correctly given the input parameters.
        The test checks to see that the correct values in expected are updated per each input parameter added:
        When values are added to thorax, expected[1][0][2] and expected[1][1][2] should be updated
        When values are added to shoulderJC, expected[0], expected[1][0][2] and expected[1][1][2] should be updated
        When values are only added to wand, no values in expected should be updated
        If values are added to any two parameters, then all values in expected[1] should be updated
        """
        result = pyCGM.shoulderAxisCalc(None, thorax, shoulderJC, wand)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)

    @pytest.mark.parametrize(["frame", "thorax", "shoulderJC", "vsk", "mockReturnVal", "expectedMockArgs", "expected"], [
        # Test from running sample data
        ({'RSHO': np.array([428.88476562, 270.552948, 1500.73010254]), 'LSHO': np.array([68.24668121, 269.01049805, 1510.1072998]),
          'RELB': np.array([658.90338135, 326.07580566, 1285.28515625]), 'LELB': np.array([-156.32162476, 335.2583313, 1287.39916992]),
          'RWRA': np.array([ 776.51898193, 495.68103027, 1108.38464355]), 'RWRB': np.array([ 830.9072876 , 436.75341797, 1119.11901855]),
          'LWRA': np.array([-249.28146362, 525.32977295, 1117.09057617]), 'LWRB': np.array([-311.77532959, 477.22512817, 1125.1619873 ])},
         [[rand_coor, [257.1435863244796, 364.21960599061947, 1459.588978712983], rand_coor],
          [256.149810236564, 364.3090603933987, 1459.6553639290375]],
         [np.array([429.66951995, 275.06718615, 1453.95397813]), np.array([64.51952734, 274.93442161, 1463.6313334])],
         {'RightElbowWidth': 74.0, 'LeftElbowWidth': 74.0, 'RightWristWidth': 55.0, 'LeftWristWidth': 55.0},
         [[633.66707588, 304.95542115, 1256.07799541], [-129.16952219, 316.8671644, 1258.06440717]],
         [[[429.7839232523795, 96.8248244295684, 904.5644429627703], [429.66951995, 275.06718615, 1453.95397813], [658.90338135, 326.07580566, 1285.28515625], -44.0],
          [[-409.6146956013004, 530.6280208729519, 1671.682014527917], [64.51952734, 274.93442161, 1463.6313334], [-156.32162476, 335.2583313, 1287.39916992], 44.0]],
         [[np.array([633.66707587, 304.95542115, 1256.07799541]),
           np.array([-129.16952218, 316.8671644, 1258.06440717])],
          [[[633.8107013869995, 303.96579004975194, 1256.07658506845], [634.3524799178464, 305.0538658933253, 1256.799473014224], [632.9532180390149, 304.85083190737765, 1256.770431750491]],
           [[-129.32391792749496, 315.8807291324946, 1258.008662931836], [-128.45117135279028, 316.79382333592827, 1257.3726028780698], [-128.49119037560908, 316.72030884193634, 1258.7843373067021]]],
          [[793.3281430325068, 451.2913478825204, 1084.4325513020426], [-272.4594189740742, 485.801522109477, 1091.3666238350822]]]),
        # Test with zeros for all params
        ({'RSHO': np.array([0, 0, 0]), 'LSHO': np.array([0, 0, 0]), 'RELB': np.array([0, 0, 0]), 'LELB': np.array([0, 0, 0]),
          'RWRA': np.array([0, 0, 0]), 'RWRB': np.array([0, 0, 0]), 'LWRA': np.array([0, 0, 0]), 'LWRB': np.array([0, 0, 0])},
         [[rand_coor, [0, 0, 0], rand_coor], [0, 0, 0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
         [[0, 0, 0], [0, 0, 0]],
         [[nan_3d, [0, 0, 0], [0, 0, 0], -7.0], [nan_3d, [0, 0, 0], [0, 0, 0], 7.0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])], [[nan_3d, nan_3d, nan_3d], [nan_3d, nan_3d, nan_3d]], [nan_3d, nan_3d]]),
        # Testing when values are added to frame
        ({'RSHO': np.array([9, -7, -6]), 'LSHO': np.array([3, -8, 5]), 'RELB': np.array([-9, 1, -4]), 'LELB': np.array([-4, 1, -6]),
          'RWRA': np.array([2, -3, 9]), 'RWRB': np.array([-4, -2, -7]), 'LWRA': np.array([-9, 1, -1]), 'LWRB': np.array([-3, -4, -9])},
         [[rand_coor, [0, 0, 0], rand_coor], [0, 0, 0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
         [[0, 0, 0], [0, 0, 0]],
         [[[149.87576359540907, -228.48721408225754, -418.8422716102348], [0, 0, 0], [-9, 1, -4], -7.0], [[282.73117218166414, -326.69276820761615, -251.76957615571214], [0, 0, 0], [-4, 1, -6], 7.0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]],
          [[4.7413281, -5.7386979, -1.35541665], [-4.96790631, 4.69256216, -8.09628108]]]),
        # Testing when values are added to thorax
        ({'RSHO': np.array([0, 0, 0]), 'LSHO': np.array([0, 0, 0]), 'RELB': np.array([0, 0, 0]), 'LELB': np.array([0, 0, 0]),
          'RWRA': np.array([0, 0, 0]), 'RWRB': np.array([0, 0, 0]), 'LWRA': np.array([0, 0, 0]), 'LWRB': np.array([0, 0, 0])},
         [[rand_coor, [-9, 5, -5], rand_coor], [-5, -2, -3]],
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
         [[0, 0, 0], [0, 0, 0]],
         [[nan_3d, [0, 0, 0], [0, 0, 0], -7.0],
          [nan_3d, [0, 0, 0], [0, 0, 0], 7.0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])], [[nan_3d, nan_3d, nan_3d], [nan_3d, nan_3d, nan_3d]], [nan_3d, nan_3d]]),
        # Testing when values are added to shoulderJC
        ({'RSHO': np.array([0, 0, 0]), 'LSHO': np.array([0, 0, 0]), 'RELB': np.array([0, 0, 0]), 'LELB': np.array([0, 0, 0]),
          'RWRA': np.array([0, 0, 0]), 'RWRB': np.array([0, 0, 0]), 'LWRA': np.array([0, 0, 0]), 'LWRB': np.array([0, 0, 0])},
         [[rand_coor, [0, 0, 0], rand_coor], [0, 0, 0]],
         [np.array([-2, -8, -3]), np.array([5, -3, 2])],
         {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
         [[0, 0, 0], [0, 0, 0]],
         [[nan_3d, [-2, -8, -3], [0, 0, 0], -7.0],
          [nan_3d, [5, -3, 2], [0, 0, 0], 7.0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[nan_3d, nan_3d, [-0.2279211529192759, -0.9116846116771036, -0.3418817293789138]],
           [nan_3d, nan_3d, [0.8111071056538127, -0.48666426339228763, 0.3244428422615251]]],
          [nan_3d, nan_3d]]),
        # Testing when values are added to vsk
        ({'RSHO': np.array([0, 0, 0]), 'LSHO': np.array([0, 0, 0]), 'RELB': np.array([0, 0, 0]), 'LELB': np.array([0, 0, 0]),
          'RWRA': np.array([0, 0, 0]), 'RWRB': np.array([0, 0, 0]), 'LWRA': np.array([0, 0, 0]), 'LWRB': np.array([0, 0, 0])},
         [[rand_coor, [0, 0, 0], rand_coor], [0, 0, 0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         {'RightElbowWidth': -38.0, 'LeftElbowWidth': 6.0, 'RightWristWidth': 47.0, 'LeftWristWidth': -7.0},
         [[0, 0, 0], [0, 0, 0]],
         [[nan_3d, [0, 0, 0], [0, 0, 0], 12.0],
          [nan_3d, [0, 0, 0], [0, 0, 0], 10.0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])], [[nan_3d, nan_3d, nan_3d], [nan_3d, nan_3d, nan_3d]], [nan_3d, nan_3d]]),
        # Testing when values are added to mockReturnVal
        ({'RSHO': np.array([0, 0, 0]), 'LSHO': np.array([0, 0, 0]), 'RELB': np.array([0, 0, 0]), 'LELB': np.array([0, 0, 0]),
          'RWRA': np.array([0, 0, 0]), 'RWRB': np.array([0, 0, 0]), 'LWRA': np.array([0, 0, 0]), 'LWRB': np.array([0, 0, 0])},
         [[rand_coor, [0, 0, 0], rand_coor], [0, 0, 0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
         [[5, 4, -4], [6, 3, 5]],
         [[nan_3d, [0, 0, 0], [0, 0, 0], -7.0],
          [nan_3d, [0, 0, 0], [0, 0, 0], 7.0]],
         [[np.array([5, 4, -4]), np.array([6, 3, 5])],
          [[nan_3d, nan_3d, [4.337733821467478, 3.4701870571739826, -3.4701870571739826]],
           [nan_3d, nan_3d, [5.2828628343993635, 2.6414314171996818, 4.4023856953328036]]],
          [nan_3d, nan_3d]]),
        # Testing when values are added to frame and thorax
        ({'RSHO': np.array([9, -7, -6]), 'LSHO': np.array([3, -8, 5]), 'RELB': np.array([-9, 1, -4]), 'LELB': np.array([-4, 1, -6]),
          'RWRA': np.array([2, -3, 9]), 'RWRB': np.array([-4, -2, -7]), 'LWRA': np.array([-9, 1, -1]), 'LWRB': np.array([-3, -4, -9])},
         [[rand_coor, [-9, 5, -5], rand_coor], [-5, -2, -3]],
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
         [[0, 0, 0], [0, 0, 0]],
         [[[149.87576359540907, -228.48721408225754, -418.8422716102348], [0, 0, 0], [-9, 1, -4], -7.0],
          [[282.73117218166414, -326.69276820761615, -251.76957615571214], [0, 0, 0], [-4, 1, -6], 7.0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])], [[nan_3d, nan_3d, nan_3d],  [nan_3d, nan_3d, nan_3d]],
          [[4.7413281, -5.7386979, -1.35541665], [-4.96790631, 4.69256216, -8.09628108]]]),
        # Testing when values are added to frame, thorax, and shoulderJC
        ({'RSHO': np.array([9, -7, -6]), 'LSHO': np.array([3, -8, 5]), 'RELB': np.array([-9, 1, -4]), 'LELB': np.array([-4, 1, -6]),
          'RWRA': np.array([2, -3, 9]), 'RWRB': np.array([-4, -2, -7]), 'LWRA': np.array([-9, 1, -1]), 'LWRB': np.array([-3, -4, -9])},
         [[rand_coor, [-9, 5, -5], rand_coor], [-5, -2, -3]],
         [np.array([-2, -8, -3]), np.array([5, -3, 2])],
         {'RightElbowWidth': 0.0, 'LeftElbowWidth': 0.0, 'RightWristWidth': 0.0, 'LeftWristWidth': 0.0},
         [[0, 0, 0], [0, 0, 0]],
         [[[-311.42865408643604, -195.76081109238007, 342.15327877363165], [-2, -8, -3], [-9, 1, -4], -7.0],
          [[183.9753004933977, -292.7114070209339, -364.32791656553934], [5, -3, 2], [-4, 1, -6], 7.0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[[-0.9661174276011973, 0.2554279765068226, -0.03706298561739535], [0.12111591199009825, 0.3218504585188577, -0.9390118307103527], [-0.2279211529192759, -0.9116846116771036, -0.3418817293789138]],
           [[-0.40160401780320154, -0.06011448807273248, 0.9138383123989052], [-0.4252287337918506, -0.8715182976051595, -0.24420561192811296], [0.8111071056538127, -0.48666426339228763, 0.3244428422615251]]],
          [[4.7413281, -5.7386979, -1.35541665], [-4.96790631, 4.69256216, -8.09628108]]]),
        # Testing when values are added to frame, thorax, shoulderJC, and vsk
        ({'RSHO': np.array([9, -7, -6]), 'LSHO': np.array([3, -8, 5]), 'RELB': np.array([-9, 1, -4]), 'LELB': np.array([-4, 1, -6]),
          'RWRA': np.array([2, -3, 9]), 'RWRB': np.array([-4, -2, -7]), 'LWRA': np.array([-9, 1, -1]), 'LWRB': np.array([-3, -4, -9])},
         [[rand_coor, [-9, 5, -5], rand_coor], [-5, -2, -3]],
         [np.array([-2, -8, -3]), np.array([5, -3, 2])],
         {'RightElbowWidth': -38.0, 'LeftElbowWidth': 6.0, 'RightWristWidth': 47.0, 'LeftWristWidth': -7.0},
         [[0, 0, 0], [0, 0, 0]],
         [[[-311.42865408643604, -195.76081109238007, 342.15327877363165], [-2, -8, -3], [-9, 1, -4], 12.0],
          [[183.9753004933977, -292.7114070209339, -364.32791656553934], [5, -3, 2], [-4, 1, -6], 10.0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[[-0.9685895544902782, 0.17643783885299713, 0.17522546605219314], [-0.09942948749879241, 0.3710806621909213, -0.9232621075099284], [-0.2279211529192759, -0.9116846116771036, -0.3418817293789138]],
           [[-0.10295276972565287, 0.4272479327059481, 0.8982538233730544], [-0.5757655689286321, -0.7619823480470763, 0.29644040025096585], [0.8111071056538127, -0.48666426339228763, 0.3244428422615251]]],
          [[24.015786700696715, -16.61146942090584, -9.262886851567883], [-5.48395315345786, 1.5962810792528397, -6.54814053962642]]]),
        # Testing when values are added to frame, thorax, shoulderJC, vsk and mockReturnVal
        ({'RSHO': np.array([9, -7, -6]), 'LSHO': np.array([3, -8, 5]), 'RELB': np.array([-9, 1, -4]), 'LELB': np.array([-4, 1, -6]),
          'RWRA': np.array([2, -3, 9]), 'RWRB': np.array([-4, -2, -7]), 'LWRA': np.array([-9, 1, -1]), 'LWRB': np.array([-3, -4, -9])},
         [[rand_coor, [-9, 5, -5], rand_coor], [-5, -2, -3]],
         [np.array([-2, -8, -3]), np.array([5, -3, 2])],
         {'RightElbowWidth': -38.0, 'LeftElbowWidth': 6.0, 'RightWristWidth': 47.0, 'LeftWristWidth': -7.0},
         [[5, 4, -4], [6, 3, 5]],
         [[[-311.42865408643604, -195.76081109238007, 342.15327877363165], [-2, -8, -3], [-9, 1, -4], 12.0],
          [[183.9753004933977, -292.7114070209339, -364.32791656553934], [5, -3, 2], [-4, 1, -6], 10.0]],
         [[np.array([5, 4, -4]), np.array([6, 3, 5])],
          [[[4.156741342815987, 4.506819397152288, -3.8209778344606247], [4.809375978699987, 4.029428853750657, -4.981221904092206], [4.4974292889675835, 3.138450209658714, -3.928204184138226]],
           [[6.726856988207308, 2.5997910101837682, 5.558132316896694], [5.329224487433077, 2.760784472038086, 5.702022893446135], [5.852558043845103, 2.1153482630706173, 4.557674131535308]]],
          [[17.14176226312361, -25.58951560761187, -7.246255574147096], [-5.726512929518925, 1.5474273567891164, -6.699526795132392]]])])
    def testElbowJointCenter(self, frame, thorax, shoulderJC, vsk, mockReturnVal, expectedMockArgs, expected):
        """
        This test provides coverage of the elbowJointCenter function in pyCGM.py, defined as elbowJointCenter(frame, thorax, shoulderJC, wand, vsk)

        This test takes 7 parameters:
        frame: dictionary of marker lists
        thorax: array containing several x,y,z markers for the thorax
        shoulderJC: array containing x,y,z position of the shoulder joint center
        vsk: dictionary containing subject measurements from a VSK file
        mockReturnVal: the value to be returned by the mock for findJointC
        expectedMockArgs: the expected arguments used to call the mocked function, findJointC
        expected: the expected result from calling elbowJointCenter on frame, thorax, shoulderJC, and vsk

        This test is checking to make sure the elbow joint axis is calculated correctly given the input parameters.
        This tests mocks findJointC to make sure the correct parameters are being passed into it given the parameters
        passed into findshoulderJC, and also ensure that elbowJointCenter returns the correct value considering
        the return value of findJointC, mockReturnVal. The test checks to see that the correct values in expected_args
        and expected are updated per each input parameter added:
        When values are added to frame, expectedMockArgs[0][0], expectedMockArgs[0][2],  expectedMockArgs[1][0],
        expectedMockArgs[0][2], and expected[2] should be updated
        When values are added to thorax, nothing should be updated
        When values are added to shoulderJC, expectedMockArgs[0][1], expectedMockArgs[1][1], expected[1][0][2] and
        expected[1][1][2] should be updated
        When values are added to vsk, expectedMockArgs[0][3], and expectedMockArgs[1][3] should be updated
        When values are added to mockReturnVal, expected[0], expected[1][0][2] and expected[1][1][2] should be updated
        """
        with patch.object(pyCGM, 'findJointC', side_effect=mockReturnVal) as mock_findJointC:
            result = pyCGM.elbowJointCenter(frame, thorax, shoulderJC, None, vsk)

        # Asserting that there were only 2 calls to findJointC
        np.testing.assert_equal(mock_findJointC.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to findJointC
        np.testing.assert_almost_equal(expectedMockArgs[0][0], mock_findJointC.call_args_list[0][0][0], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][1], mock_findJointC.call_args_list[0][0][1], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][2], mock_findJointC.call_args_list[0][0][2], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][3], mock_findJointC.call_args_list[0][0][3], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to findJointC
        np.testing.assert_almost_equal(expectedMockArgs[1][0], mock_findJointC.call_args_list[1][0][0], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][1], mock_findJointC.call_args_list[1][0][1], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][2], mock_findJointC.call_args_list[1][0][2], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][3], mock_findJointC.call_args_list[1][0][3], rounding_precision)

        # Asserting that findShoulderJC returned the correct result given the return value given by mocked findJointC
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

    @pytest.mark.parametrize(["elbowJC", "expected"], [
        # Test from running sample data
        ([[np.array([633.667075873457, 304.955421154148, 1256.077995412865]), np.array([-129.169522182155, 316.867164398512, 1258.064407167222])],
          [[[633.8107013869995, 303.96579004975194, 1256.07658506845], [634.3524799178464, 305.0538658933253, 1256.799473014224],  [632.9532180390149, 304.85083190737765, 1256.770431750491]],
           [[-129.32391792749496, 315.8807291324946, 1258.008662931836], [-128.45117135279028, 316.79382333592827, 1257.3726028780698],  [-128.49119037560908, 316.72030884193634, 1258.7843373067021]]],
          [[793.3281430325068, 451.2913478825204, 1084.4325513020426],  [-272.4594189740742, 485.801522109477, 1091.3666238350822]]],
         [[[793.3281430325068, 451.2913478825204, 1084.4325513020426], [-272.4594189740742, 485.801522109477, 1091.3666238350822]],
          [[[793.771337279616, 450.4487918719012, 1084.1264823093322], [794.013547076896, 451.3897926216976, 1085.154028903402], [792.7503886251119, 450.761812234714, 1085.053672741407]],
           [[-272.92507281675125, 485.0120241803687, 1090.9667994752267], [-271.74106814470946, 485.7281810468936, 1090.6748195459295], [-271.9425644638384, 485.1921666233502, 1091.967911874857]]]]),
        # Test with zeros for all params
        ([[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[0, 0, 0], [0, 0, 0]]],
         [[[0, 0, 0], [0, 0, 0]],
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when adding values in elbowJC[0]
        ([[np.array([9, -5, 7]), np.array([-1, 6, 4])],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[0, 0, 0], [0, 0, 0]]],
         [[[0, 0, 0], [0, 0, 0]],
          [[nan_3d, [-0.7228974 ,  0.40160966, -0.56225353], nan_3d],
           [nan_3d, [0.13736056, -0.82416338, -0.54944226], nan_3d]]]),
        # Testing when adding values in elbowJC[1]
        ([[np.array([0, 0, 0]),  np.array([0, 0, 0])],
          [[[-3, -9, 6], [4, -5, 5], [-9, 7, 0]], [[4, -1, 0], [3, -5, 1], [0, -9, 7]]],
          [[0, 0, 0], [0, 0, 0]]],
         [[[0, 0, 0], [0, 0, 0]],
          [[nan_3d, [0.49236596, -0.61545745,  0.61545745], nan_3d],
           [nan_3d, [0.50709255, -0.84515425, 0.16903085], nan_3d]]]),
        # Testing when adding values in elbowJC[2]
        ([[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[6, -1, 5], [7, 6, 0]]],
         [[[6, -1, 5], [7, 6, 0]],
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when adding values in elbowJC[0] and elbowJC[1]
        ([[np.array([9, -5, 7]), np.array([-1, 6, 4])],
          [[[-3, -9, 6], [4, -5, 5], [-9, 7, 0]], [[4, -1, 0], [3, -5, 1], [0, -9, 7]]],
          [[0, 0, 0], [0, 0, 0]]],
         [[[0, 0, 0], [0, 0, 0]],
          [[[-0.31403715, 0.53386315, 0.78509287], [-0.92847669, 0, -0.37139068], [-0.1982718, -0.84557089, 0.49567949]],
           [[-0.81649658, -0.40824829, 0.40824829], [0.33104236, -0.91036648, -0.24828177], [0.47301616, -0.06757374, 0.87845859]]]]),
        # Testing when adding values in elbowJC[0] and elbowJC[2]
        ([[np.array([9, -5, 7]), np.array([-1, 6, 4])],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[6, -1, 5], [7, 6, 0]]],
         [[[6, -1, 5], [7, 6, 0]],
          [[[ 5.35300336, -1.10783277, 5.75482941], [5.2771026, -0.59839034, 4.43774647], [5.75748257, -1.90944036, 4.66220787]],
           [[6.60350884, 6.46257302, -0.79298232], [7.13736056, 5.17583662, -0.54944226], [6.09229584, 5.67322650, 0.26323421]]]]),
        # Testing when adding values in elbowJC[1] and elbowJC[2]
        ([[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[[-3, -9, 6], [4, -5, 5], [-9, 7, 0]], [[4, -1, 0], [3, -5, 1], [0, -9, 7]]],
          [[6, -1, 5], [7, 6, 0]]],
         [[[6, -1, 5], [7, 6, 0]],
          [[[ 6.58321184, -1.29160592, 4.2418246], [6.49236596, -1.61545745, 5.61545745], [5.35390426, -1.73224184, 4.78463475]],
           [[7.11153264, 5.86987859, -0.985205], [7.50709255, 5.15484575, 0.16903085], [6.14535527, 5.48155743, -0.02827869]]]]),
        # Testing when adding values in elbowJC
        ([[np.array([9, -5, 7]), np.array([-1, 6, 4])],
          [[[-3, -9, 6], [4, -5, 5], [-9, 7, 0]], [[4, -1, 0], [3, -5, 1], [0, -9, 7]]],
          [[6, -1, 5], [7, 6, 0]]],
         [[[6, -1, 5], [7, 6, 0]],
          [[[5.63485163, -0.81742581, 5.91287093], [ 5.07152331, -1, 4.62860932], [5.93219365, -1.98319208, 5.16951588]],
           [[6.55425751, 6.08104409, -0.89148499], [7.33104236, 5.08963352, -0.24828177], [6.16830018, 5.59421098, 0.37896]]]])])
    def testWristJointCenter(self, elbowJC, expected):
        """
        This test provides coverage of the wristJointCenter function in pyCGM.py, defined as wristJointCenter(frame, shoulderJC, wand, elbowJC)

        This test takes 2 parameters:
        elbowJC: array containing the x,y,z position of the elbow joint center
        expected: the expected result from calling wristJointCenter on elbowJC

        This test is checking to make sure the wrist joint axis is calculated correctly given the input parameters.
        The test checks to see that the correct values in expected are updated per each input parameter added:
        When values are only added to elbowJC[0] or elboxJC[1], expected[1][0][1] and expected[1][1][1] should be updated
        When values are only added to elbowJC[2], expected[0] should be updated
        When values are added to any two of elbowJC[0], elbowJC[1], and elbowJC[2], then all values in expected[1] should
        be updated
        """
        result = pyCGM.wristJointCenter(None, None, None, elbowJC)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)

    @pytest.mark.parametrize(["frame", "wristJC", "vsk", "mockReturnVal", "expectedMockArgs", "expected"], [
        # Test from running sample data
        ({'RWRA': np.array([ 776.51898193,  495.68103027, 1108.38464355]), 'RWRB': np.array([ 830.9072876 ,  436.75341797, 1119.11901855]), 'RFIN': np.array([ 863.71374512,  524.4475708 , 1074.54248047]),
          'LWRA': np.array([-249.28146362,  525.32977295, 1117.09057617]), 'LWRB': np.array([-311.77532959,  477.22512817, 1125.1619873 ]), 'LFIN': np.array([-326.65890503,  558.34338379, 1091.04284668])},
         [[[793.3281430325068, 451.2913478825204, 1084.4325513020426], [-272.4594189740742, 485.801522109477, 1091.3666238350822]], [[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]]],
         {'RightHandThickness': 34.0, 'LeftHandThickness': 34.0},
         [[-324.53477798, 551.88744289, 1068.02526837], [859.80614366, 517.28239823, 1051.97278945]],
         [[[-280.528396605, 501.27745056000003, 1121.126281735], [-272.4594189740742, 485.801522109477, 1091.3666238350822], [-326.65890503, 558.34338379, 1091.04284668], 24.0],
          [[803.713134765, 466.21722411999997, 1113.75183105], [793.3281430325068, 451.2913478825204, 1084.4325513020426], [863.71374512, 524.4475708, 1074.54248047], 24.0]],
         [[np.array([859.80614366, 517.28239823, 1051.97278944]), np.array([-324.53477798, 551.88744289, 1068.02526837])],
          [[[859.9567597867737, 517.5924123242138, 1052.9115152009197], [859.0797567344147, 517.9612045889317, 1051.8651606187454], [859.1355641971873, 516.6167307529585, 1052.300218811959]],
           [[-324.61994077156373, 552.1589330842497, 1068.9839343010813], [-325.3329318534787, 551.2929248618385, 1068.1227296356121], [-323.938374013488, 551.1305800350597, 1068.2925901317217]]]]),
        # Testing when values are added to wristJC
        ({'RWRA': np.array([0, 0, 0]), 'RWRB': np.array([0, 0, 0]), 'RFIN': np.array([0, 0, 0]),
          'LWRA': np.array([0, 0, 0]), 'LWRB': np.array([0, 0, 0]), 'LFIN': np.array([0, 0, 0])},
         [[[0, 4, 3], [9, 0, -6]], [[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]]],
         {'RightHandThickness': 0.0, 'LeftHandThickness': 0.0},
         [[0, 0, 0], [0, 0, 0]],
         [[[0, 0, 0], [9, 0, -6], [0, 0, 0], 7.0], [[0, 0, 0], [0, 4, 3], [0, 0, 0], 7.0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[nan_3d, nan_3d, [0, 0.8, 0.6]], [nan_3d, nan_3d, [0.8320502943378436, 0.0, -0.554700196225229]]]]),
        # Testing when values are added to wristJC, frame['RFIN'] and frame['LFIN']
        ({'RWRA': np.array([0, 0, 0]), 'RWRB': np.array([0, 0, 0]), 'RFIN': np.array([1, -9, 6]),
          'LWRA': np.array([0, 0, 0]), 'LWRB': np.array([0, 0, 0]), 'LFIN': np.array([-6, 3, 8])},
         [[[0, 4, 3], [9, 0, -6]], [[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]]],
         {'RightHandThickness': 0.0, 'LeftHandThickness': 0.0},
         [[0, 0, 0], [0, 0, 0]],
         [[[0, 0, 0], [9, 0, -6], [-6, 3, 8], 7.0], [[0, 0, 0], [0, 4, 3], [1, -9, 6], 7.0]],
         [[np.array([0, 0, 0]),  np.array([0, 0, 0])],
          [[nan_3d, nan_3d, [0, 0.8, 0.6]], [nan_3d, nan_3d, [0.8320502943378436, 0.0, -0.554700196225229]]]]),
        # Testing when values are added to wristJC, frame['RFIN'], frame['LFIN'], frame['RWRA'], and frame['LWRA']
        ({'RWRA': np.array([4, 7, 6]), 'RWRB': np.array([0, 0, 0]), 'RFIN': np.array([1, -9, 6]),
          'LWRA': np.array([-4, 5, 3]), 'LWRB': np.array([0, 0, 0]), 'LFIN': np.array([-6, 3, 8])},
         [[[0, 4, 3], [9, 0, -6]], [[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]]],
         {'RightHandThickness': 0.0, 'LeftHandThickness': 0.0},
         [[0, 0, 0], [0, 0, 0]],
         [[[-2.0, 2.5, 1.5], [9, 0, -6], [-6, 3, 8], 7.0], [[2.0, 3.5, 3.0], [0, 4, 3], [1, -9, 6], 7.0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[[-0.1483404529302446, -0.5933618117209785, 0.7911490822946381], [0.9889363528682976, -0.08900427175814675, 0.11867236234419568], [0, 0.8, 0.6]],
           [[0.5538487756217112, -0.05538487756217114, 0.8307731634325669], [-0.030722002451646625, -0.998465079678515, -0.04608300367746994], [0.8320502943378436, 0.0, -0.554700196225229]]]]),
        # Testing when values are added to frame and wristJC
        ({'RWRA': np.array([4, 7, 6]), 'RWRB': np.array([0, -5, 4]), 'RFIN': np.array([1, -9, 6]),
          'LWRA': np.array([-4, 5, 3]), 'LWRB': np.array([-3, 2, -7]), 'LFIN': np.array([-6, 3, 8])},
         [[[0, 4, 3], [9, 0, -6]], [[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]]],
         {'RightHandThickness': 0.0, 'LeftHandThickness': 0.0},
         [[0, 0, 0], [0, 0, 0]],
         [[[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 7.0], [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 7.0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[[0.813733471206735, -0.3487429162314579, 0.4649905549752772], [0.5812381937190965, 0.48824008272404085, -0.650986776965388], [0.0, 0.8, 0.6]],
           [[0.19988898139583083, -0.9328152465138774, 0.2998334720937463],  [-0.5174328002831333, -0.3603549859114678, -0.7761492004246999],  [0.8320502943378436, 0.0, -0.554700196225229]]]]),
        # Testing when values are added to frame, wristJC, and vsk
        ({'RWRA': np.array([4, 7, 6]), 'RWRB': np.array([0, -5, 4]), 'RFIN': np.array([1, -9, 6]),
          'LWRA': np.array([-4, 5, 3]), 'LWRB': np.array([-3, 2, -7]), 'LFIN': np.array([-6, 3, 8])},
         [[[0, 4, 3], [9, 0, -6]], [[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]]],
         {'RightHandThickness': 36.0, 'LeftHandThickness': -9.0},
         [[0, 0, 0], [0, 0, 0]],
         [[[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 2.5], [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 25.0]],
         [[np.array([0, 0, 0]), np.array([0, 0, 0])],
          [[[0.813733471206735, -0.3487429162314579, 0.4649905549752772], [0.5812381937190965, 0.48824008272404085, -0.650986776965388], [0.0, 0.8, 0.6]],
           [[0.19988898139583083, -0.9328152465138774, 0.2998334720937463], [-0.5174328002831333, -0.3603549859114678, -0.7761492004246999], [0.8320502943378436, 0.0, -0.554700196225229]]]]),
        # Testing when values are added to frame, wristJC, vsk and mockReturnVal
        ({'RWRA': np.array([4, 7, 6]), 'RWRB': np.array([0, -5, 4]), 'RFIN': np.array([1, -9, 6]),
          'LWRA': np.array([-4, 5, 3]), 'LWRB': np.array([-3, 2, -7]), 'LFIN': np.array([-6, 3, 8])},
         [[[0, 4, 3], [9, 0, -6]], [[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]]],
         {'RightHandThickness': 36.0, 'LeftHandThickness': -9.0},
         [[-6, 4, -4], [2, 8, 1]],
         [[[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 2.5], [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 25.0]],
         [[np.array([2, 8, 1]), np.array([-6, 4, -4])],
          [[[2.911684611677104, 7.658118270621086, 1.227921152919276], [1.9534757894800765, 8.465242105199236, 1.8839599998785472], [1.5917517095361369, 7.183503419072274, 1.4082482904638631]],
           [[-6.21615749183132, 3.059079153204844, -3.739339495144585], [-6.186838410896736, 3.777824759216273, -4.9569376001580645], [-5.04168515250009, 3.744449374000024, -4.127775312999988]]]])])
    def testHandJointCenter(self, frame, wristJC, vsk, mockReturnVal, expectedMockArgs, expected):
        """
        This test provides coverage of the handJointCenter function in pyCGM.py, defined as handJointCenter(frame, elbowJC, wristJC, vsk)

        This test takes 6 parameters:
        frame: dictionary of marker lists
        wristJC: array containing the x,y,z position of the wrist joint center
        vsk: dictionary containing subject measurements from a VSK file
        mockReturnVal: the value to be returned by the mock for findJointC
        expectedMockArgs: the expected arguments used to call the mocked function, findJointC
        expected: the expected result from calling handJointCenter on frame, wristJC, and vsk

        This test is checking to make sure the hand joint axis is calculated correctly given the input parameters.
        This tests mocks findJointC to make sure the correct parameters are being passed into it given the parameters
        passed into handJointCenter, and to also ensure that handJointCenter returns the correct value considering
        the return value of findJointC, mockReturnVal. The test checks to see that the correct values in expected_args
        and expected are updated per each input parameter added:
        When values are added to frame, expectedMockArgs[0][0], expectedMockArgs[0][2], expectedMockArgs[1][0],
        expectedMockArgs[1][2] and expected[1] should be updated
        When values are added to wristJC, expectedMockArgs[0][1], expectedMockArgs[1][1], expected[1][0][2],
        and expected[1][1][2] should be updated
        When values are added to vsk, expectedMockArgs[0][3] and expectedMockArgs[1][3] should be updated
        When values are added to mockReturnVal, expected[0] should be updated
        """
        with patch.object(pyCGM, 'findJointC', side_effect=mockReturnVal) as mock_findJointC:
            result = pyCGM.handJointCenter(frame, None, wristJC, vsk)

        # Asserting that there were only 2 calls to findJointC
        np.testing.assert_equal(mock_findJointC.call_count, 2)

        # Asserting that the correct params were sent in the 1st (left) call to findJointC
        np.testing.assert_almost_equal(expectedMockArgs[0][0], mock_findJointC.call_args_list[0][0][0], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][1], mock_findJointC.call_args_list[0][0][1], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][2], mock_findJointC.call_args_list[0][0][2], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][3], mock_findJointC.call_args_list[0][0][3], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (right) call to findJointC
        np.testing.assert_almost_equal(expectedMockArgs[1][0], mock_findJointC.call_args_list[1][0][0], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][1], mock_findJointC.call_args_list[1][0][1], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][2], mock_findJointC.call_args_list[1][0][2], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][3], mock_findJointC.call_args_list[1][0][3], rounding_precision)

        # Asserting that findShoulderJC returned the correct result given the return value given by mocked findJointC
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)

class TestLowerBodyAxis():
    """
    This class tests the lower body axis functions in pyCGM.py:
    pelvisJointCenter
    hipJointCenter
    hipAxisCenter
    kneeJointCenter
    ankleJointCenter
    footJointCenter
    """
    nan_3d = [np.nan, np.nan, np.nan]
    rand_int = np.random.randint(0, 10)
    rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]

    @pytest.mark.parametrize(["frame", "expected"], [
        # Test from running sample data
        ({'RASI': np.array([357.90066528, 377.69210815, 1034.97253418]), 'LASI': np.array([145.31594849, 405.79052734, 1030.81445312]),
          'RPSI': np.array([274.00466919, 205.64402771, 1051.76452637]), 'LPSI': np.array([189.15231323, 214.86122131, 1052.73486328])},
         [np.array([251.60830688, 391.74131775, 1032.89349365]),
          np.array([[251.74063624, 392.72694721, 1032.78850073], [250.61711554, 391.87232862, 1032.8741063], [251.60295336, 391.84795134, 1033.88777762]]),
          np.array([231.57849121, 210.25262451, 1052.24969482])]),
        # Test with zeros for all params
        ({'SACR': np.array([0, 0, 0]), 'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0]),
          'RPSI': np.array([0, 0, 0]), 'LPSI': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([nan_3d, nan_3d, nan_3d]), np.array([0, 0, 0])]),
        # Testing when adding values to frame['RASI'] and frame['LASI']
        ({'RASI': np.array([-6, 6, 3]), 'LASI': np.array([-7, -9, 1]), 'RPSI': np.array([0, 0, 0]), 'LPSI': np.array([0, 0, 0])},
         [np.array([-6.5, -1.5,  2.0]),
          np.array([[-7.44458106, -1.48072284, 2.32771179], [-6.56593805, -2.48907071, 1.86812391], [-6.17841206, -1.64617634, 2.93552855]]),
          np.array([0, 0, 0])]),
        # Testing when adding values to frame['RPSI'] and frame['LPSI']
        ({'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0]), 'RPSI': np.array([1, 0, -4]), 'LPSI': np.array([7, -2, 2])},
         [np.array([0, 0, 0]), np.array([nan_3d, nan_3d, nan_3d]), np.array([4., -1.0, -1.0])]),
        # Testing when adding values to frame['SACR']
        ({'SACR': np.array([-4, 8, -5]), 'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0]),
          'RPSI': np.array([0, 0, 0]), 'LPSI': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([nan_3d, nan_3d, nan_3d]), np.array([-4,  8, -5,])]),
        # Testing when adding values to frame['RASI'], frame['LASI'], frame['RPSI'] and frame['LPSI']
        ({'RASI': np.array([-6, 6, 3]), 'LASI': np.array([-7, -9, 1]), 'RPSI': np.array([1, 0, -4]), 'LPSI': np.array([7, -2, 2])},
         [np.array([-6.5, -1.5,  2.0]),
          np.array([[-7.45825845, -1.47407957, 2.28472598], [-6.56593805, -2.48907071, 1.86812391], [-6.22180416, -1.64514566, 2.9494945]]),
          np.array([4.0, -1.0, -1.0])]),
        # Testing when adding values to frame['SACR'], frame['RASI'] and frame['LASI']
        ({'SACR': np.array([-4, 8, -5]), 'RASI': np.array([-6, 6, 3]), 'LASI': np.array([-7, -9, 1]),
          'RPSI': np.array([0, 0, 0]), 'LPSI': np.array([0, 0, 0])},
         [np.array([-6.5, -1.5,  2.0]),
          np.array([[-6.72928306, -1.61360872, 2.96670695], [-6.56593805, -2.48907071, 1.86812391], [-5.52887619, -1.59397972, 2.21928602]]),
          np.array([-4, 8, -5])]),
        # Testing when adding values to frame['SACR'], frame['RPSI'] and frame['LPSI']
        ({'SACR': np.array([-4, 8, -5]), 'RASI': np.array([0, 0, 0]), 'LASI': np.array([0, 0, 0]),
          'RPSI': np.array([1, 0, -4]), 'LPSI': np.array([7, -2, 2])},
         [np.array([0, 0, 0]), np.array([nan_3d, nan_3d, nan_3d]), np.array([-4,  8, -5])]),
        # Testing when adding values to frame['SACR'], frame['RASI'], frame['LASI'], frame['RPSI'] and frame['LPSI']
        ({'SACR': np.array([-4, 8, -5]), 'RASI': np.array([-6, 6, 3]), 'LASI': np.array([-7, -9, 1]),
          'RPSI': np.array([1, 0, -4]), 'LPSI': np.array([7, -2, 2])},
         [np.array([-6.5, -1.5,  2.0]),
          np.array([[-6.72928306, -1.61360872, 2.96670695], [-6.56593805, -2.48907071, 1.86812391], [-5.52887619, -1.59397972,  2.21928602]]),
          np.array([-4,  8, -5])])])
    def testPelvisJointCenter(self, frame, expected):
        """
        This test provides coverage of the pelvisJointCenter function in pyCGM.py, defined as pelvisJointCenter(frame)

        This test takes 2 parameters:
        frame: dictionary of marker lists
        expected: the expected result from calling pelvisJointCenter on frame

        This test is checking to make sure the pelvis joint center and axis are calculated correctly given the input
        parameters. The test checks to see that the correct values in expected are updated per each input parameter added:
        When values are added to frame['RASI'] and frame['LASI'], expected[0] and expected[1] should be updated
        When values are added to frame['RPSI'] and frame['LPSI'], expected[2] should be updated
        When values are added to frame['SACR'], expected[2] should be updated, and expected[1] should also be updated
        if there are values for frame['RASI'] and frame['LASI']
        Values produced from frame['SACR'] takes precedent over frame['RPSI'] and frame['LPSI']
        """
        result = pyCGM.pelvisJointCenter(frame)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

    @pytest.mark.parametrize(["pel_origin", "pel_x", "pel_y", "pel_z", "vsk", "expected"], [
        # Test from running sample data
        ([251.608306884766, 391.741317749023, 1032.893493652344], [251.740636241119, 392.726947206848, 1032.788500732036], [250.617115540376, 391.872328624646, 1032.874106304030], [251.602953357582, 391.847951338178, 1033.887777624562],
         {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512, 'L_AsisToTrocanterMeasure': 72.512, 'InterAsisDistance': 215.908996582031},
         [[182.57097863, 339.43231855, 935.52900126], [308.38050472, 322.80342417, 937.98979061]]),
        # Basic test with zeros for all params
        ([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[0, 0, 0], [0, 0, 0]]),
        # Testing when values are added to pel_origin
        ([1, 0, -3], [0, 0, 0], [0, 0, 0], [0, 0, 0],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[-6.1387721, 0, 18.4163163], [8.53165418, 0, -25.59496255]]),
        # Testing when values are added to pel_x
        ([0, 0, 0], [-5, -3, -6], [0, 0, 0], [0, 0, 0],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[54.02442793, 32.41465676, 64.82931352], [54.02442793, 32.41465676, 64.82931352]]),
        # Testing when values are added to pel_y
        ([0, 0, 0], [0, 0, 0], [4, -1, 2], [0, 0, 0],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[29.34085257, -7.33521314, 14.67042628], [-29.34085257,   7.33521314, -14.67042628]]),
        # Testing when values are added to pel_z
        ([0, 0, 0], [0, 0, 0], [0, 0, 0], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[31.82533363, 84.86755635, 21.21688909], [31.82533363, 84.86755635, 21.21688909]]),
        # Test when values are added to pel_x, pel_y, and pel_z
        ([0, 0, 0], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[115.19061413, 109.94699997, 100.71662889], [56.508909  , 124.61742625,  71.37577632]]),
        # Test when values are added to pel_origin, pel_x, pel_y, and pel_z
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[109.05184203, 109.94699997, 119.13294518], [65.04056318, 124.61742625,  45.78081377]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[MeanLegLength]
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[100.88576753,  97.85280235, 106.39612748], [61.83654463, 110.86920998,  41.31408931]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[R_AsisToTrocanterMeasure]
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 0.0},
         [[109.05184203, 109.94699997, 119.13294518], [-57.09307697, 115.44008189,  14.36512267]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[L_AsisToTrocanterMeasure]
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0-7.0, 'InterAsisDistance': 0.0},
         [[73.42953032, 107.27027453, 109.97003528], [65.04056318, 124.61742625,  45.78081377]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and vsk[InterAsisDistance]
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 0.0, 'R_AsisToTrocanterMeasure': 0.0, 'L_AsisToTrocanterMeasure': 0.0, 'InterAsisDistance': 11.0},
         [[125.55184203, 104.44699997, 146.63294518], [48.54056318, 130.11742625,  18.28081377]]),
        # Test when values are added to pel_origin, pel_x, pel_y, pel_z, and all values in vsk
        ([1, 0, -3], [-5, -3, -6], [4, -1, 2], [3, 8, 2],
         {'MeanLegLength': 15.0, 'R_AsisToTrocanterMeasure': -24.0, 'L_AsisToTrocanterMeasure': -7.0, 'InterAsisDistance': 11.0},
         [[81.76345582,  89.67607691, 124.73321758], [-76.79709552, 107.19186562, -17.60160178]])])
    def testHipJointCenter(self, pel_origin, pel_x, pel_y, pel_z, vsk, expected):
        """
        This test provides coverage of the hipJointCenter function in pyCGM.py, defined as hipJointCenter(frame, pel_origin, pel_x, pel_y, pel_z, vsk)

        This test takes 6 parameters:
        pel_origin: array of x,y,z position of origin of the pelvis
        pel_x: array of x,y,z position of x-axis of the pelvis
        pel_y: array of x,y,z position of y-axis of the pelvis
        pel_z: array of x,y,z position of z-axis of the pelvis
        vsk: dictionary containing subject measurements from a VSK file
        expected: the expected result from calling hipJointCenter on pel_origin, pel_x, pel_y, pel_z, and vsk

        This test is checking to make sure the hip joint center is calculated correctly given the input parameters.
        The test checks to see that the correct values in expected are updated per each input parameter added. Any
        parameter that is added should change the value of every value in expected.
        """
        result = pyCGM.hipJointCenter(None, pel_origin, pel_x, pel_y, pel_z, vsk)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)

    @pytest.mark.parametrize(["l_hip_jc", "r_hip_jc", "pelvis_axis", "expected"], [
        # Test from running sample data
        ([182.57097863, 339.43231855, 935.52900126], [308.38050472, 322.80342417, 937.98979061],
         [np.array([251.60830688, 391.74131775, 1032.89349365]), np.array([[251.74063624, 392.72694721, 1032.78850073], [250.61711554, 391.87232862, 1032.8741063], [251.60295336, 391.84795134, 1033.88777762]]), np.array([231.57849121, 210.25262451, 1052.24969482])],
         [[245.47574167208043, 331.1178713574418, 936.7593959314677], [[245.60807102843359, 332.10350081526684, 936.6544030111602], [244.48455032769033, 331.2488822330648, 936.7400085831541], [245.47038814489719, 331.22450494659665, 937.7536799036861]]]),
        # Basic test with zeros for all params
        ([0, 0, 0], [0, 0, 0],
         [np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[0, 0, 0], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
        # Testing when values are added to l_hip_jc
        ([1, -3, 2], [0, 0, 0],
         [np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[0.5, -1.5, 1], [[0.5, -1.5, 1], [0.5, -1.5, 1], [0.5, -1.5, 1]]]),
        # Testing when values are added to r_hip_jc
        ([0, 0, 0], [-8, 1, 4],
         [np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[-4, 0.5, 2], [[-4, 0.5, 2], [-4, 0.5, 2], [-4, 0.5, 2]]]),
        # Testing when values are added to l_hip_jc and r_hip_jc
        ([8, -3, 7], [5, -2, -1],
         [np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[6.5, -2.5, 3], [[6.5, -2.5, 3], [6.5, -2.5, 3], [6.5, -2.5, 3]]]),
        # Testing when values are added to pelvis_axis[0]
        ([0, 0, 0], [0, 0, 0],
         [np.array([1, -3, 6]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(rand_coor)],
         [[0, 0, 0], [[-1, 3, -6], [-1, 3, -6], [-1, 3, -6]]]),
        # Testing when values are added to pelvis_axis[1]
        ([0, 0, 0], [0, 0, 0],
         [np.array([0, 0, 0]), np.array([[1, 0, 5], [-2, -7, -3], [9, -2, 7]]), np.array(rand_coor)],
         [[0, 0, 0], [[1, 0, 5], [-2, -7, -3], [9, -2, 7]]]),
        # Testing when values are added to pelvis_axis[0] and pelvis_axis[1]
        ([0, 0, 0], [0, 0, 0],
         [np.array([-3, 0, 5]), np.array([[-4, 5, -2], [0, 0, 0], [8, 5, -1]]), np.array(rand_coor)],
         [[0, 0, 0], [[-1, 5, -7], [3, 0, -5], [11, 5, -6]]]),
        # Testing when values are added to all params
        ([-5, 3, 8], [-3, -7, -1],
         [np.array([6, 3, 9]), np.array([[5, 4, -2], [0, 0, 0], [7, 2, 3]]), np.array(rand_coor)],
         [[-4, -2, 3.5], [[-5, -1, -7.5], [-10, -5, -5.5], [-3, -3, -2.5]]])])
    def testHipAxisCenter(self, l_hip_jc, r_hip_jc, pelvis_axis, expected):
        """
        This test provides coverage of the hipAxisCenter function in pyCGM.py, defined as hipAxisCenter(l_hip_jc, r_hip_jc, pelvis_axis)

        This test takes 4 parameters:
        l_hip_jc: array of left hip joint center x,y,z position
        r_hip_jc: array of right hip joint center x,y,z position
        pelvis_axis: array of pelvis origin and axis
        expected: the expected result from calling hipAxisCenter on l_hip_jc, r_hip_jc, and pelvis_axis

        This test is checking to make sure the hip axis center is calculated correctly given the input parameters.
        The test checks to see that the correct values in expected are updated per each input parameter added:
        When values are added to l_hip_jc or r_hip_jc, every value in expected should be updated
        When values are added to pelvis_axis, expected[1] should be updated
        """
        result = pyCGM.hipAxisCenter(l_hip_jc, r_hip_jc, pelvis_axis)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)

    @pytest.mark.parametrize(["frame", "hip_JC", "vsk", "mockReturnVal", "expectedMockArgs", "expected"], [
        # Test from running sample data
        ({'RTHI': np.array([426.50338745, 262.65310669, 673.66247559]),
          'LTHI': np.array([51.93867874, 320.01849365, 723.03186035]),
          'RKNE': np.array([416.98687744, 266.22558594, 524.04089355]),
          'LKNE': np.array([84.62355804, 286.69122314, 529.39819336])},
         [[182.57097863, 339.43231855, 935.52900126], [308.38050472, 322.80342417, 937.98979061]],
         {'RightKneeWidth': 105.0, 'LeftKneeWidth': 105.0},
         [np.array([364.17774614, 292.17051722, 515.19181496]), np.array([143.55478579, 279.90370346, 524.78408753])],
         [[[426.50338745, 262.65310669, 673.66247559], [308.38050472, 322.80342417, 937.98979061], [416.98687744, 266.22558594, 524.04089355], 59.5],
          [[51.93867874, 320.01849365, 723.03186035], [182.57097863, 339.43231855, 935.52900126], [84.62355804, 286.69122314, 529.39819336], 59.5]],
         [np.array([364.17774614, 292.17051722, 515.19181496]),
          np.array([143.55478579, 279.90370346, 524.78408753]),
          np.array([[[364.61959153, 293.06758353, 515.18513093], [363.29019771, 292.60656648, 515.04309095], [364.04724541, 292.24216264, 516.18067112]],
                    [[143.65611282, 280.88685896, 524.63197541], [142.56434499, 280.01777943, 524.86163553], [143.64837987, 280.04650381, 525.76940383]]])]),
        # Test with zeros for all params
        ({'RTHI': np.array([0, 0, 0]), 'LTHI': np.array([0, 0, 0]), 'RKNE': np.array([0, 0, 0]), 'LKNE': np.array([0, 0, 0])},
         [[0, 0, 0], [0, 0, 0]],
         {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing when values are added to frame
        ({'RTHI': np.array([1, 2, 4]), 'LTHI': np.array([-1, 0, 8]), 'RKNE': np.array([8, -4, 5]), 'LKNE': np.array([8, -8, 5])},
         [[0, 0, 0], [0, 0, 0]],
         {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[1, 2, 4], [0, 0, 0], [8, -4, 5], 7.0], [[-1, 0, 8], [0, 0, 0], [8, -8, 5], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing when values are added to hip_JC
        ({'RTHI': np.array([0, 0, 0]), 'LTHI': np.array([0, 0, 0]), 'RKNE': np.array([0, 0, 0]), 'LKNE': np.array([0, 0, 0])},
         [[-8, 8, -2], [1, -9, 2]],
         {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[0, 0, 0], [1, -9, 2], [0, 0, 0], 7.0], [[0, 0, 0], [-8, 8, -2], [0, 0, 0], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, [0.10783277, -0.97049496, 0.21566555]],
                    [nan_3d, nan_3d, [-0.69631062, 0.69631062, -0.17407766]]])]),
        # Testing when values are added to vsk
        ({'RTHI': np.array([0, 0, 0]), 'LTHI': np.array([0, 0, 0]), 'RKNE': np.array([0, 0, 0]), 'LKNE': np.array([0, 0, 0])},
         [[0, 0, 0], [0, 0, 0]],
         {'RightKneeWidth': 9.0, 'LeftKneeWidth': -6.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 11.5], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 4.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing when values are added to mockReturnVal
        ({'RTHI': np.array([0, 0, 0]), 'LTHI': np.array([0, 0, 0]), 'RKNE': np.array([0, 0, 0]), 'LKNE': np.array([0, 0, 0])},
         [[0, 0, 0], [0, 0, 0]],
         {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
         [np.array([-5, -5, -9]), np.array([3, -6, -5])],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
         [np.array([-5, -5, -9]), np.array([3, -6, -5]),
          np.array([[nan_3d, nan_3d, [-4.56314797, -4.56314797, -8.21366635]],
                    [nan_3d, nan_3d, [2.64143142, -5.28286283, -4.4023857]]])]),
        # Testing when values are added to frame and hip_JC
        ({'RTHI': np.array([1, 2, 4]), 'LTHI': np.array([-1, 0, 8]), 'RKNE': np.array([8, -4, 5]), 'LKNE': np.array([8, -8, 5])},
         [[-8, 8, -2], [1, -9, 2]],
         {'RightKneeWidth': 0.0, 'LeftKneeWidth': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 7.0], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[[-0.47319376, 0.14067923, 0.86965339], [-0.8743339, -0.19582873, -0.44406233], [0.10783277, -0.97049496, 0.21566555]],
                    [[-0.70710678, -0.70710678, 0.0], [-0.12309149, 0.12309149, 0.98473193], [-0.69631062, 0.69631062, -0.17407766]]])]),
        # Testing when values are added to frame, hip_JC, and vsk
        ({'RTHI': np.array([1, 2, 4]), 'LTHI': np.array([-1, 0, 8]), 'RKNE': np.array([8, -4, 5]),
          'LKNE': np.array([8, -8, 5])},
         [[-8, 8, -2], [1, -9, 2]],
         {'RightKneeWidth': 9.0, 'LeftKneeWidth': -6.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 11.5], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 4.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[[-0.47319376, 0.14067923, 0.86965339], [-0.8743339, -0.19582873, -0.44406233], [0.10783277, -0.97049496, 0.21566555]],
                    [[-0.70710678, -0.70710678, 0.0], [-0.12309149, 0.12309149, 0.98473193], [-0.69631062, 0.69631062, -0.17407766]]])]),
        # Testing when values are added to frame, hip_JC, vsk, and mockReturnVal
        ({'RTHI': np.array([1, 2, 4]), 'LTHI': np.array([-1, 0, 8]), 'RKNE': np.array([8, -4, 5]), 'LKNE': np.array([8, -8, 5])},
         [[-8, 8, -2], [1, -9, 2]],
         {'RightKneeWidth': 9.0, 'LeftKneeWidth': -6.0},
         [np.array([-5, -5, -9]), np.array([3, -6, -5])],
         [[[1, 2, 4], [1, -9, 2], [8, -4, 5], 11.5], [[-1, 0, 8], [-8, 8, -2], [8, -8, 5], 4.0]],
         [np.array([-5, -5, -9]), np.array([3, -6, -5]),
          np.array([[[-5.6293369, -4.4458078, -8.45520089], [-5.62916022, -5.77484544, -8.93858368], [-4.54382845, -5.30411437, -8.16368549]],
                    [[2.26301154, -6.63098327, -4.75770242], [3.2927155, -5.97483821, -4.04413154], [2.39076635, -5.22461171, -4.83384537]]])])])
    def testKneeJointCenter(self, frame, hip_JC, vsk, mockReturnVal, expectedMockArgs, expected):
        """
        This test provides coverage of the kneeJointCenter function in pyCGM.py, defined as kneeJointCenter(frame, hip_JC, delta, vsk)

        This test takes 6 parameters:
        frame: dictionary of marker lists
        hip_JC: array of hip_JC containing the x,y,z axes marker positions of the hip joint center
        vsk: dictionary containing subject measurements from a VSK file
        mockReturnVal: the value to be returned by the mock for findJointC
        expectedMockArgs: the expected arguments used to call the mocked function, findJointC
        expected: the expected result from calling kneeJointCenter on frame, hip_JC, vsk, and mockReturnVal

        This test is checking to make sure the knee joint center and axis are calculated correctly given the input
        parameters. This tests mocks findJointC to make sure the correct parameters are being passed into it given the
        parameters passed into kneeJointCenter, and to also ensure that kneeJointCenter returns the correct value considering
        the return value of findJointC, mockReturnVal. The test checks to see that the correct values in expected_args
        and expected are updated per each input parameter added:
        When values are added to frame, expectedMockArgs[0][0], expectedMockArgs[0][2], expectedMockArgs[1][0], and
        expectedMockArgs[1][2] should be updated
        When values are added to hip_JC, expectedMockArgs[0][1], expectedMockArgs[1][1], expected[2][0][2], and
        expected[2][1][2] should be updated, unless values are also added to frame, then expected[2] should be updated
        When values are added to vsk, expectedMockArgs[0][3], and expectedMockArgs[1][3] should be updated
        When values are added to mockReturnVal, expected[0], expected[2][0][2], and expected[2][1][2] should be updated
        """
        with patch.object(pyCGM, 'findJointC', side_effect=mockReturnVal) as mock_findJointC:
            result = pyCGM.kneeJointCenter(frame, hip_JC, None, vsk)

        # Asserting that there were only 2 calls to findJointC
        np.testing.assert_equal(mock_findJointC.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to findJointC
        np.testing.assert_almost_equal(expectedMockArgs[0][0], mock_findJointC.call_args_list[0][0][0], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][1], mock_findJointC.call_args_list[0][0][1], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][2], mock_findJointC.call_args_list[0][0][2], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][3], mock_findJointC.call_args_list[0][0][3], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to findJointC
        np.testing.assert_almost_equal(expectedMockArgs[1][0], mock_findJointC.call_args_list[1][0][0], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][1], mock_findJointC.call_args_list[1][0][1], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][2], mock_findJointC.call_args_list[1][0][2], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][3], mock_findJointC.call_args_list[1][0][3], rounding_precision)

        # Asserting that findShoulderJC returned the correct result given the return value given by mocked findJointC
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

    @pytest.mark.parametrize(["frame", "knee_JC", "vsk", "mockReturnVal", "expectedMockArgs", "expected"], [
        # Test from running sample data
        ({'RTIB': np.array([433.97537231, 211.93408203, 273.3008728 ]), 'LTIB': np.array([50.04016495, 235.90718079, 364.32226562]),
          'RANK': np.array([422.77005005, 217.74053955, 92.86152649]), 'LANK': np.array([58.57380676, 208.54806519, 86.16953278])},
         [np.array([364.17774614, 292.17051722, 515.19181496]), np.array([143.55478579, 279.90370346, 524.78408753]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': 70.0, 'LeftAnkleWidth': 70.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
         [np.array([393.76181608, 247.67829633,  87.73775041]), np.array([98.74901939, 219.46930221,  80.6306816])],
         [[[433.97537231, 211.93408203, 273.3008728 ], [364.17774614, 292.17051722, 515.19181496], [422.77005005, 217.74053955, 92.86152649], 42.0],
          [[50.04016495, 235.90718079, 364.32226562], [143.55478579, 279.90370346, 524.78408753], [58.57380676, 208.54806519, 86.16953278], 42.0]],
         [np.array([393.76181608, 247.67829633, 87.73775041]), np.array([98.74901939, 219.46930221, 80.6306816]),
          [[np.array([394.48171575, 248.37201348, 87.715368]),
            np.array([393.07114384, 248.39110006, 87.61575574]),
            np.array([393.69314056, 247.78157916, 88.73002876])],
           [np.array([98.47494966, 220.42553803, 80.52821783]),
            np.array([97.79246671, 219.20927275, 80.76255901]),
            np.array([98.84848169, 219.60345781, 81.61663775])]]]),
        # Test with zeros for all params
        ({'RTIB': np.array([0, 0, 0]), 'LTIB': np.array([0, 0, 0]), 'RANK': np.array([0, 0, 0]), 'LANK': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0],
          [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)],
           [np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)]]]),
        # Testing when values are added to frame
        ({'RTIB': np.array([-9, 6, -9]), 'LTIB': np.array([0, 2, -1]), 'RANK': np.array([1, 0, -5]),
          'LANK': np.array([2, -4, -5])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[-9, 6, -9], [0, 0, 0], [1, 0, -5], 7.0],
          [[0, 2, -1], [0, 0, 0], [2, -4, -5], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)],
           [np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)]]]),
        # Testing when values are added to knee_JC
        ({'RTIB': np.array([0, 0, 0]), 'LTIB': np.array([0, 0, 0]), 'RANK': np.array([0, 0, 0]), 'LANK': np.array([0, 0, 0])},
         [np.array([-7, 1, 2]), np.array([9, -8, 9]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[0, 0, 0], [-7, 1, 2], [0, 0, 0], 7.0],
          [[0, 0, 0], [9, -8, 9], [0, 0, 0], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array(nan_3d), np.array([-0.95257934, 0.13608276, 0.27216553])],
           [np.array(nan_3d), np.array(nan_3d), np.array([0.59867109, -0.53215208, 0.59867109])]]]),
        # Testing when values are added to vsk
        ({'RTIB': np.array([0, 0, 0]), 'LTIB': np.array([0, 0, 0]), 'RANK': np.array([0, 0, 0]), 'LANK': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': -38.0, 'LeftAnkleWidth': 18.0, 'RightTibialTorsion': 29.0, 'LeftTibialTorsion': -13.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0], -12.0],
          [[0, 0, 0], [0, 0, 0], [0, 0, 0], 16.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)],
           [np.array(nan_3d), np.array(nan_3d), np.array(nan_3d)]]]),
        # Testing when values are added to mockReturnVal
        ({'RTIB': np.array([0, 0, 0]), 'LTIB': np.array([0, 0, 0]), 'RANK': np.array([0, 0, 0]), 'LANK': np.array([0, 0, 0])},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
         [np.array([2, -5, 4]), np.array([8, -3, 1])],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0],
          [[0, 0, 0], [0, 0, 0], [0, 0, 0], 7.0]],
         [np.array([2, -5, 4]), np.array([8, -3, 1]),
          [[np.array(nan_3d), np.array(nan_3d), np.array([1.7018576 , -4.25464401, 3.40371521])],
           [np.array(nan_3d), np.array(nan_3d), np.array([7.07001889, -2.65125708, 0.88375236])]]]),
        # Testing when values are added to frame and knee_JC
        ({'RTIB': np.array([-9, 6, -9]), 'LTIB': np.array([0, 2, -1]), 'RANK': np.array([1, 0, -5]), 'LANK': np.array([2, -4, -5])},
         [np.array([-7, 1, 2]), np.array([9, -8, 9]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': 0.0, 'LeftAnkleWidth': 0.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], 7.0],
          [[0, 2, -1], [9, -8, 9], [2, -4, -5], 7.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array([-0.26726124, -0.80178373, -0.53452248]), np.array([0.14547859, -0.58191437, 0.80013226]), np.array([-0.95257934, 0.13608276, 0.27216553])],
           [np.array([0.79317435, 0.49803971, -0.35047239]), np.array([-0.11165737, 0.68466825, 0.72025136]), np.array([0.59867109, -0.53215208, 0.59867109])]]]),
        # Testing when values are added to frame, knee_JC, and vsk
        ({'RTIB': np.array([-9, 6, -9]), 'LTIB': np.array([0, 2, -1]), 'RANK': np.array([1, 0, -5]), 'LANK': np.array([2, -4, -5])},
         [np.array([-7, 1, 2]), np.array([9, -8, 9]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': -38.0, 'LeftAnkleWidth': 18.0, 'RightTibialTorsion': 29.0, 'LeftTibialTorsion': -13.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0])],
         [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0],
          [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array([-0.30428137, -0.41913816, -0.85541572]), np.array([-0.00233238, -0.89766624, 0.4406698]), np.array([-0.95257934, 0.13608276, 0.27216553])],
           [np.array([0.7477279, 0.63929183, -0.1794685]), np.array([-0.287221, 0.55508569, 0.7806305]), np.array([0.59867109, -0.53215208, 0.59867109])]]]),
        # Testing when values are added to frame, knee_JC, vsk and mockReturnVal
        ({'RTIB': np.array([-9, 6, -9]), 'LTIB': np.array([0, 2, -1]), 'RANK': np.array([1, 0, -5]), 'LANK': np.array([2, -4, -5])},
         [np.array([-7, 1, 2]), np.array([9, -8, 9]),
          np.array([[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]])],
         {'RightAnkleWidth': -38.0, 'LeftAnkleWidth': 18.0, 'RightTibialTorsion': 29.0, 'LeftTibialTorsion': -13.0},
         [np.array([2, -5, 4]), np.array([8, -3, 1])],
         [[[-9, 6, -9], [-7, 1, 2], [1, 0, -5], -12.0],
          [[0, 2, -1], [9, -8, 9], [2, -4, -5], 16.0]],
         [np.array([2, -5, 4]), np.array([8, -3, 1]),
          [[np.array([1.48891678, -5.83482493, 3.7953997 ]), np.array([1.73661348, -5.07447603, 4.96181124]), np.array([1.18181818, -4.45454545, 3.81818182])],
           [np.array([8.87317138, -2.54514024, 1.17514093]), np.array([7.52412119, -2.28213872, 1.50814815]), np.array([8.10540926, -3.52704628, 1.84327404])]]])])
    def testAnkleJointCenter(self, frame, knee_JC, vsk, mockReturnVal, expectedMockArgs, expected):
        """
        This test provides coverage of the ankleJointCenter function in pyCGM.py, defined as ankleJointCenter(frame, knee_JC, delta, vsk)

        This test takes 6 parameters:
        frame: dictionary of marker lists
        knee_JC: array of knee_JC each x,y,z position.
        vsk: dictionary containing subject measurements from a VSK file
        mockReturnVal: the value to be returned by the mock for findJointC
        expectedMockArgs: the expected arguments used to call the mocked function, findJointC
        expected: the expected result from calling ankleJointCenter on frame, knee_JC, vsk, and mockReturnVal

        This test is checking to make sure the ankle joint center and axis are calculated correctly given the input
        parameters. This tests mocks findJointC to make sure the correct parameters are being passed into it given the
        parameters passed into ankleJointCenter, and to also ensure that ankleJointCenter returns the correct value considering
        the return value of findJointC, mockReturnVal. The test checks to see that the correct values in expected_args
        and expected are updated per each input parameter added:
        When values are added to frame, expectedMockArgs[0][0], expectedMockArgs[0][2], expectedMockArgs[1][0], and
        expectedMockArgs[1][2] should be updated
        When values are added to knee_JC, expectedMockArgs[0][1], expectedMockArgs[1][1], expected[2][0][2], and
        expected[2][1][2] should be updated, unless values are also added to frame, then all of expected should be updated
        When values are added to vsk, expectedMockArgs[0][3], and expectedMockArgs[1][3] should be updated
        When values are added to mockReturnVal, expected[0], expected[2][0][2], and expected[2][1][2] should be updated
        """
        with patch.object(pyCGM, 'findJointC', side_effect=mockReturnVal) as mock_findJointC:
            result = pyCGM.ankleJointCenter(frame, knee_JC, None, vsk)

        # Asserting that there were only 2 calls to findJointC
        np.testing.assert_equal(mock_findJointC.call_count, 2)

        # Asserting that the correct params were sent in the 1st (right) call to findJointC
        np.testing.assert_almost_equal(expectedMockArgs[0][0], mock_findJointC.call_args_list[0][0][0], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][1], mock_findJointC.call_args_list[0][0][1], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][2], mock_findJointC.call_args_list[0][0][2], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[0][3], mock_findJointC.call_args_list[0][0][3], rounding_precision)

        # Asserting that the correct params were sent in the 2nd (left) call to findJointC
        np.testing.assert_almost_equal(expectedMockArgs[1][0], mock_findJointC.call_args_list[1][0][0], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][1], mock_findJointC.call_args_list[1][0][1], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][2], mock_findJointC.call_args_list[1][0][2], rounding_precision)
        np.testing.assert_almost_equal(expectedMockArgs[1][3], mock_findJointC.call_args_list[1][0][3], rounding_precision)

        # Asserting that findShoulderJC returned the correct result given the return value given by mocked findJointC
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

    @pytest.mark.parametrize(["frame", "vsk", "ankle_JC", "expected"], [
        # Test from running sample data
        ({'RTOE': np.array([442.81997681, 381.62280273, 42.66047668]), 'LTOE': np.array([39.43652725, 382.44522095, 41.78911591])},
         {'RightStaticRotOff': 0.015683497632642047, 'RightStaticPlantFlex': 0.2702417907002757, 'LeftStaticRotOff': 0.009402910292403022, 'LeftStaticPlantFlex': 0.20251085737834015},
         [np.array([393.76181608, 247.67829633, 87.73775041]), np.array([98.74901939, 219.46930221, 80.6306816]),
          [[np.array([394.48171575, 248.37201348, 87.715368]), np.array([393.07114384, 248.39110006, 87.61575574]), np.array([393.69314056, 247.78157916, 88.73002876])],
           [np.array([98.47494966, 220.42553803, 80.52821783]), np.array([97.79246671, 219.20927275, 80.76255901]), np.array([98.84848169, 219.60345781, 81.61663775])]]],
         [np.array([442.81997681, 381.62280273, 42.66047668]), np.array([39.43652725, 382.44522095, 41.78911591]),
          [[[442.8462412676692, 381.6513024007671, 43.65972537588915], [441.8773505621594, 381.95630350196393, 42.67574106247485], [442.48716163075153, 380.68048378251575, 42.69610043598381]],
           [[39.566526257915626, 382.50901000467115, 42.778575967950964], [38.493133283871245, 382.1460684058263, 41.932348504971834], [39.74166341694723, 381.493150197213, 41.81040458481808]]]]),
        # Test with zeros for all params
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when adding values in frame['RTOE'] and frame['LTOE']
        ({'RTOE': np.array([4, 0, -3]), 'LTOE': np.array([-1, 7, 2])},
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         [np.array([4, 0, -3]), np.array([-1, 7, 2]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when adding values in vsk
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         {'RightStaticRotOff': -12.0, 'RightStaticPlantFlex': 20.0, 'LeftStaticRotOff': 34.0, 'LeftStaticPlantFlex': -70.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when adding values to ankle_JC[0] and ankle_JC[1]
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         [np.array([-3, 5, 2]), np.array([2, 3, 9]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when adding values to ankle_JC[2]
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([-1, 0, 2]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([9, 3, -4]), np.array(rand_coor)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when adding values to ankle_JC
        ({'RTOE': np.array([0, 0, 0]), 'LTOE': np.array([0, 0, 0])},
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         [np.array([-3, 5, 2]), np.array([2, 3, 9]),
          [[np.array(rand_coor), np.array([-1, 0, 2]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([9, 3, -4]), np.array(rand_coor)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[[-0.84215192, -0.33686077, -0.42107596], [-0.23224564, -0.47815279,  0.8470135 ], [-0.48666426,  0.81110711,  0.32444284]],
           [[0.39230172, -0.89525264, 0.21123939], [0.89640737, 0.32059014, -0.30606502], [0.20628425, 0.30942637, 0.92827912]]]]),
        # Testing when adding values in frame and vsk
        ({'RTOE': np.array([4, 0, -3]), 'LTOE': np.array([-1, 7, 2])},
         {'RightStaticRotOff': -12.0, 'RightStaticPlantFlex': 20.0, 'LeftStaticRotOff': 34.0,'LeftStaticPlantFlex': -70.0},
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([0, 0, 0]), np.array(rand_coor)]]],
         [np.array([4, 0, -3]), np.array([-1, 7, 2]),
          [[nan_3d, nan_3d, nan_3d],
           [nan_3d, nan_3d, nan_3d]]]),
        # Testing when adding values in frame and thoraxJC
        ({'RTOE': np.array([4, 0, -3]), 'LTOE': np.array([-1, 7, 2])},
         {'RightStaticRotOff': 0.0, 'RightStaticPlantFlex': 0.0, 'LeftStaticRotOff': 0.0, 'LeftStaticPlantFlex': 0.0},
         [np.array([-3, 5, 2]), np.array([2, 3, 9]),
          [[np.array(rand_coor), np.array([-1, 0, 2]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([9, 3, -4]), np.array(rand_coor)]]],
         [np.array([4, 0, -3]), np.array([-1, 7, 2]),
          [[[3.31958618, -0.27216553, -3.68041382], [3.79484752, -0.82060994, -2.46660354], [3.29647353,  0.50251891, -2.49748109]],
           [[-1.49065338, 6.16966351, 1.73580203], [-0.20147784, 6.69287609, 1.48227684], [-0.65125708, 6.53500945, 2.81373347]]]]),
        # Testing when adding values in frame, vsk and thoraxJC
        ({'RTOE': np.array([4, 0, -3]), 'LTOE': np.array([-1, 7, 2])},
         {'RightStaticRotOff': -12.0, 'RightStaticPlantFlex': 20.0, 'LeftStaticRotOff': 34.0, 'LeftStaticPlantFlex': -70.0},
         [np.array([-3, 5, 2]), np.array([2, 3, 9]),
          [[np.array(rand_coor), np.array([-1, 0, 2]), np.array(rand_coor)],
           [np.array(rand_coor), np.array([9, 3, -4]), np.array(rand_coor)]]],
         [np.array([4, 0, -3]), np.array([-1, 7, 2]),
          [[[3.08005417, 0.34770638, -2.81889243], [4.00614173, -0.44911697, -2.10654814], [4.3919974, 0.82303962, -2.58897224]],
           [[-1.58062909, 6.83398388, 1.20293758], [-1.59355918, 7.75640754, 2.27483654], [-0.44272327, 7.63268181, 1.46226738]]]])])
    def testFootJointCenter(self, frame, vsk, ankle_JC, expected):
        """
        This test provides coverage of the footJointCenter function in pyCGM.py, defined as footJointCenter(frame, vsk, ankle_JC, knee_JC, delta)

        This test takes 4 parameters:
        frame: dictionary of marker lists
        vsk: dictionary containing subject measurements from a VSK file
        ankle_JC: array of ankle_JC containing the x,y,z axes marker positions of the ankle joint center
        expected: the expected result from calling footJointCenter on frame, knee_JC, vsk, and mockReturnVal

        This test is checking to make sure the foot joint center and axis are calculated correctly given the input
        parameters. The test checks to see that the correct values in expected are updated per each input parameter added:
        When values are added to frame, expected[0] and expected[1] should be updated
        When values are added to vsk, expected[2] should be updated as long as there are values for frame and ankle_JC
        When values are added to ankle_JC, expected[2] should be updated
        """
        result = pyCGM.footJointCenter(frame, vsk, ankle_JC, None, None)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2], rounding_precision)

class TestAxisUtils():
    """
    This class tests the lower body axis functions in pyCGM.py:
    findJointC
    JointAngleCalc
    """

    @pytest.mark.parametrize(["a", "b", "c", "delta", "expected"], [
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
        ([-7, 1, 2], [1, 4, 3], [3, 2, -8], 10.0, [5.867777, 5.195449, 1.031332])])
    def testfindJointC(self, a, b, c, delta, expected):
        """
        This test provides coverage of the findJointC function in pyCGM.py, defined as findJointC(a, b, c, delta)

        This test takes 5 parameters:
        a: list markers of x,y,z position
        b: list markers of x,y,z position
        c: list markers of x,y,z position
        delta: length from marker to joint center, retrieved from subject measurement file
        expected: the expected result from calling findJointC on a, b, c, and delta
        """
        result = pyCGM.findJointC(a, b, c, delta)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def testJointAngleCalc(self):
        """
        This test provides coverage of the JointAngleCalc function in pyCGM.py, defined as JointAngleCalc(frame, vsk)

        This test takes 3 parameters:
        frame: dictionary of marker lists
        vsk: dictionary containing subject measurements from a VSK file
        expected: the expected result from calling JointAngleCalc on frame and vsk
        """
        # Test from running sample data
        frame = {'LFHD': np.array([ 184.5515899658203,  409.6871337890625, 1721.3428955078125]),
                 'RFHD': np.array([ 325.829833984375  ,  402.55450439453125, 1722.4981689453125 ]),
                 'LBHD': np.array([ 197.86215209960938,  251.2888946533203 , 1696.9019775390625 ]),
                 'RBHD': np.array([ 304.39898681640625,  242.91339111328125, 1694.9749755859375 ]),
                 'C7': np.array([ 251.2261962890625,  229.7568359375   , 1533.7762451171875]),
                 'T10': np.array([ 228.6432342529297 ,  192.32041931152344, 1279.641845703125  ]),
                 'CLAV': np.array([ 256.780517578125 ,  371.2804260253906, 1459.7030029296875]),
                 'STRN': np.array([ 251.6749267578125,  414.1039123535156, 1292.0850830078125]),
                 'RBAK': np.array([ 337.792236328125 ,  154.8023223876953, 1395.8443603515625]),
                 'LSHO': np.array([  68.2466812133789,  269.010498046875 , 1510.1072998046875]),
                 'LELB': np.array([-156.32162475585938,  335.2583312988281 , 1287.399169921875  ]),
                 'LWRA': np.array([-249.28146362304688,  525.3297729492188 , 1117.090576171875  ]),
                 'LWRB': np.array([-311.77532958984375,  477.2251281738281 , 1125.1619873046875 ]),
                 'LFIN': np.array([-326.6589050292969,  558.3433837890625, 1091.0428466796875]),
                 'RSHO': np.array([ 428.884765625    ,  270.5529479980469, 1500.7301025390625]),
                 'RELB': np.array([ 658.9033813476562,  326.0758056640625, 1285.28515625     ]),
                 'RWRA': np.array([ 776.5189819335938,  495.6810302734375, 1108.3846435546875]),
                 'RWRB': np.array([ 830.9072875976562,  436.75341796875  , 1119.1190185546875]),
                 'RFIN': np.array([ 863.7137451171875,  524.4475708007812, 1074.54248046875  ]),
                 'LASI': np.array([ 145.31594848632812,  405.79052734375   , 1030.814453125     ]),
                 'RASI': np.array([ 357.9006652832031,  377.6921081542969, 1034.9725341796875]),
                 'LPSI': np.array([ 189.15231323242188,  214.86122131347656, 1052.73486328125   ]),
                 'RPSI': np.array([ 274.0046691894531 ,  205.64402770996094, 1051.7645263671875 ]),
                 'LTHI': np.array([ 51.93867874145508, 320.01849365234375, 723.0318603515625 ]),
                 'LKNE': np.array([ 84.6235580444336 , 286.69122314453125, 529.398193359375  ]),
                 'LTIB': np.array([ 50.040164947509766, 235.9071807861328  , 364.322265625      ]),
                 'LANK': np.array([ 58.57380676269531, 208.54806518554688,  86.1695327758789 ]),
                 'LHEE': np.array([105.30126953125   , 180.2130126953125 ,  47.15660858154297]),
                 'LTOE': np.array([ 39.436527252197266, 382.4452209472656  ,  41.78911590576172 ]),
                 'RTHI': np.array([426.5033874511719, 262.6531066894531, 673.6624755859375]),
                 'RKNE': np.array([416.98687744140625, 266.2255859375    , 524.0408935546875 ]),
                 'RTIB': np.array([433.9753723144531, 211.93408203125  , 273.3008728027344]),
                 'RANK': np.array([422.7700500488281 , 217.74053955078125,  92.86152648925781]),
                 'RHEE': np.array([374.0125732421875 , 181.5792999267578 ,  49.50960922241211]),
                 'RTOE': np.array([442.8199768066406 , 381.622802734375  ,  42.66047668457031]),
                 'CentreOfMass': np.array([ 253.8907012939453,  343.2032470703125, 1036.1832275390625]),
                 'CentreOfMassFloor': np.array([253.8907012939453, 343.2032470703125,   0.             ]),
                 'HEDO': np.array([ 250.3409423828125 ,  207.52053833007812, 1612.1177978515625 ]),
                 'HEDA': np.array([ 256.7513122558594,  458.5885009765625, 1653.1395263671875]),
                 'HEDL': np.array([  -4.0499267578125 ,  214.13446044921875, 1611.3912353515625 ]),
                 'HEDP': np.array([ 248.55796813964844,  166.53114318847656, 1863.2664794921875 ]),
                 'LCLO': np.array([  64.51952362060547,  274.9344482421875 , 1463.63134765625   ]),
                 'LCLA': np.array([ -23.357498168945312,  464.743408203125   , 1494.8720703125     ]),
                 'LCLL': np.array([  81.29048156738281,  248.27874755859375, 1672.758056640625  ]),
                 'LCLP': np.array([ 256.1497802734375,  364.3091125488281, 1459.6553955078125]),
                 'LFEO': np.array([143.55479431152344, 279.9037170410156 , 524.7841186523438 ]),
                 'LFEA': np.array([185.7947998046875 , 689.7478637695312 , 461.37371826171875]),
                 'LFEL': np.array([-269.32635498046875,  327.45831298828125,  557.1112060546875 ]),
                 'LFEP': np.array([182.57098388671875, 339.43231201171875, 935.5289916992188 ]),
                 'LFOO': np.array([ 40.889339447021484, 351.2103576660156  ,  66.45579528808594 ]),
                 'LFOA': np.array([ 67.14031219482422, 364.0914611816406 , 266.2594299316406 ]),
                 'LFOL': np.array([-149.612060546875 ,  290.8019104003906,   95.3790512084961]),
                 'LFOP': np.array([102.506103515625  , 158.95684814453125,  70.7547607421875 ]),
                 'LHNO': np.array([-376.61016845703125,  617.9734497070312 , 1044.68408203125   ]),
                 'LHNA': np.array([-391.4822082519531,  665.384033203125 , 1212.096923828125 ]),
                 'LHNL': np.array([-515.9927368164062,  514.152099609375 , 1061.703857421875 ]),
                 'LHNP': np.array([-272.45947265625   ,  485.80157470703125, 1091.36669921875   ]),
                 'LHUO': np.array([-129.16954040527344,  316.8671875       , 1258.064453125     ]),
                 'LHUA': np.array([-173.25531005859375,   35.203125        , 1242.1473388671875 ]),
                 'LHUL': np.array([  75.94642639160156,  295.9255676269531 , 1060.5284423828125 ]),
                 'LHUP': np.array([  64.51953125     ,  274.9344482421875, 1463.63134765625  ]),
                 'LRAO': np.array([-272.45941162109375,  485.8015441894531 , 1091.36669921875   ]),
                 'LRAA': np.array([-392.51806640625  ,  282.246826171875 ,  988.2807006835938]),
                 'LRAL': np.array([-87.24839782714844, 466.89215087890625, 913.0001220703125 ]),
                 'LRAP': np.array([-139.1998291015625 ,  328.69256591796875, 1246.3956298828125 ]),
                 'LTIO': np.array([ 98.7490234375    , 219.4693145751953 ,  80.63069152832031]),
                 'LTIA': np.array([-18.958290100097656, 630.1535034179688  ,  36.62455749511719 ]),
                 'LTIL': np.array([-312.0711975097656 ,  107.7920150756836 ,  137.26937866210938]),
                 'LTIP': np.array([141.46607971191406, 277.08642578125   , 504.07916259765625]),
                 'LTOO': np.array([ 20.35041618347168, 415.29486083984375,  65.0228271484375 ]),
                 'LTOA': np.array([ 29.100732803344727, 419.58856201171875 , 131.6240234375     ]),
                 'LTOL': np.array([-43.150047302246094, 395.15869140625    ,  74.66390991210938 ]),
                 'LTOP': np.array([ 40.889339447021484, 351.2103576660156  ,  66.4557876586914  ]),
                 'PELO': np.array([245.4757537841797, 331.1178283691406, 936.7594604492188]),
                 'PELA': np.array([262.2720031738281, 456.221435546875 , 923.4329833984375]),
                 'PELL': np.array([119.66619110107422, 347.7467346191406 , 934.2986450195312 ]),
                 'PELP': np.array([ 244.79623413085938,  344.6525573730469 , 1062.961669921875  ]),
                 'RCLO': np.array([ 429.6695556640625 ,  275.06719970703125, 1453.9541015625    ]),
                 'RCLA': np.array([ 519.033935546875 ,  447.6651611328125, 1472.1103515625   ]),
                 'RCLL': np.array([ 432.92889404296875,  293.8164367675781 , 1259.6771240234375 ]),
                 'RCLP': np.array([ 256.14984130859375,  364.30908203125   , 1459.655517578125  ]),
                 'RFEO': np.array([364.1777648925781, 292.1705322265625, 515.1918334960938]),
                 'RFEA': np.array([553.09423828125, 675.72265625   , 512.333984375  ]),
                 'RFEL': np.array([-15.30487060546875, 478.60882568359375, 451.60302734375   ]),
                 'RFEP': np.array([308.38055419921875, 322.80340576171875, 937.9898071289062 ]),
                 'RFOO': np.array([453.65185546875   , 378.307373046875  ,  69.25687408447266]),
                 'RFOA': np.array([458.9666748046875 , 384.07452392578125, 271.4610900878906 ]),
                 'RFOL': np.array([262.90557861328125, 445.7933349609375 ,  72.3457260131836 ]),
                 'RFOP': np.array([386.30462646484375, 187.62326049804688,  76.46563720703125]),
                 'RHNO': np.array([ 926.2841186523438,  583.2734985351562, 1019.5130004882812]),
                 'RHNA': np.array([ 956.1468505859375,  644.7401123046875, 1205.634521484375 ]),
                 'RHNL': np.array([782.26318359375  , 717.8606567382812, 998.1734008789062]),
                 'RHNP': np.array([ 793.328125       ,  451.2914123535156, 1084.4324951171875]),
                 'RHUO': np.array([ 633.6671142578125,  304.9554138183594, 1256.0780029296875]),
                 'RHUA': np.array([ 674.7107543945312,   22.1507568359375, 1255.675048828125 ]),
                 'RHUL': np.array([ 829.5335693359375 ,  333.08782958984375, 1462.2530517578125 ]),
                 'RHUP': np.array([ 429.66961669921875,  275.0671691894531 , 1453.9539794921875 ]),
                 'RRAO': np.array([ 793.3281860351562,  451.2913818359375, 1084.4326171875   ]),
                 'RRAA': np.array([ 907.2305908203125,  234.751708984375 , 1005.7719116210938]),
                 'RRAL': np.array([ 969.4793701171875,  476.592041015625 , 1269.8548583984375]),
                 'RRAP': np.array([ 644.8433837890625 ,  315.19891357421875, 1244.06298828125   ]),
                 'RTIO': np.array([393.7618408203125 , 247.67831420898438,  87.73775482177734]),
                 'RTIA': np.array([688.7626342773438 , 531.9500732421875 ,  78.56587982177734]),
                 'RTIL': np.array([110.73785400390625 , 539.7713012695312  ,  37.746742248535156]),
                 'RTIP': np.array([365.6199645996094, 290.0015869140625, 494.35400390625  ]),
                 'RTOO': np.array([476.1009521484375 , 441.8686828613281 ,  66.85396575927734]),
                 'RTOA': np.array([477.87249755859375, 443.7910461425781 , 134.25537109375   ]),
                 'RTOL': np.array([412.51885986328125, 464.364013671875  ,  67.883544921875  ]),
                 'RTOP': np.array([453.6518859863281 , 378.3073425292969 ,  69.25685119628906]),
                 'TRXO': np.array([ 256.14984130859375,  364.30908203125   , 1459.655517578125  ]),
                 'TRXA': np.array([ 274.47149658203125,  566.8231201171875 , 1461.0394287109375 ]),
                 'TRXL': np.array([ 458.23004150390625,  346.11895751953125, 1446.1563720703125 ]),
                 'TRXP': np.array([ 242.8296661376953 ,  366.90069580078125, 1256.762939453125  ]),
                 'LHipAngles': np.array([-2.860201120376587, -5.345648765563965, -1.802547931671143]),
                 'LKneeAngles': np.array([ -0.458484411239624,  -0.386675655841827, -21.875816345214844]),
                 'LAnkleAngles': np.array([ 4.384649753570557,  0.599297523498535, -2.378748655319214]),
                 'LAbsAnkleAngle': np.array([4.380864143371582, 0.               , 0.               ]),
                 'RHipAngles': np.array([  2.91422700881958 ,  -6.867066860198975, -18.82098960876465 ]),
                 'RKneeAngles': np.array([  3.194370031356812,   2.383407115936279, -19.475923538208008]),
                 'RAnkleAngles': np.array([ 2.50532603263855 , -7.68821382522583 , 26.498098373413086]),
                 'RAbsAnkleAngle': np.array([2.241997241973877, 0.               , 0.               ]),
                 'LShoulderAngles': np.array([ 4.602182388305664, 39.90544509887695 ,  0.748713552951813]),
                 'LElbowAngles': np.array([2.919420242309570e+01, 1.402363841100217e-15, 6.494637184821517e-13]),
                 'LWristAngles': np.array([ 12.894158363342285,  17.397676467895508, 128.74131774902344 ]),
                 'RShoulderAngles': np.array([ 9.425692558288574, 49.1335334777832  ,  9.611226081848145]),
                 'RElbowAngles': np.array([ 2.611592292785645e+01, -1.731600994895873e-15, -1.126373842287587e-12]),
                 'RWristAngles': np.array([  9.83027172088623 ,  16.793094635009766, 122.09709167480469 ]),
                 'LNeckAngles': np.array([-8.642641067504883,  3.937166929244995,  3.689980268478394]),
                 'RNeckAngles': np.array([-8.642641067504883, -3.937166929244995, -3.689980268478394]),
                 'LSpineAngles': np.array([-6.479706764221191, -4.638277053833008,  2.893069267272949]),
                 'RSpineAngles': np.array([-6.479706764221191,  4.638277053833008, -2.893069267272949]),
                 'LHeadAngles': np.array([ 9.269386291503906, -0.401442855596542,  1.443480491638184]),
                 'RHeadAngles': np.array([ 9.269386291503906,  0.401442855596542, -1.443480491638184]),
                 'LThoraxAngles': np.array([-0.731802940368652,  3.755841970443726,  5.180577754974365]),
                 'RThoraxAngles': np.array([-0.731802940368652, -3.755841970443726, -5.180577754974365]),
                 'LPelvisAngles': np.array([ 6.121381759643555, -0.306735992431641,  7.604328155517578]),
                 'RPelvisAngles': np.array([ 6.121381759643555,  0.306735992431641, -7.604328155517578]),
                 'LFootProgressAngles': np.array([-88.71904754638672,  -7.84589958190918, -17.7663631439209 ]),
                 'RFootProgressAngles': np.array([-87.83499908447266 ,   1.596025943756104, -19.43973731994629 ]),
                 'LAnklePower': np.array([np.nan, np.nan, np.nan]), 'RAnklePower': np.array([np.nan, np.nan, np.nan]),
                 'LKneePower': np.array([np.nan, np.nan, np.nan]), 'RKneePower': np.array([np.nan, np.nan, np.nan]),
                 'LHipPower': np.array([np.nan, np.nan, np.nan]), 'RHipPower': np.array([np.nan, np.nan, np.nan]),
                 'LWaistPower': np.array([np.nan, np.nan, np.nan]), 'RWaistPower': np.array([np.nan, np.nan, np.nan]),
                 'LNeckPower': np.array([np.nan, np.nan, np.nan]), 'RNeckPower': np.array([np.nan, np.nan, np.nan]),
                 'LShoulderPower': np.array([np.nan, np.nan, np.nan]), 'RShoulderPower': np.array([np.nan, np.nan, np.nan]),
                 'LElbowPower': np.array([np.nan, np.nan, np.nan]), 'RElbowPower': np.array([np.nan, np.nan, np.nan]),
                 'LWristPower': np.array([np.nan, np.nan, np.nan]), 'RWristPower': np.array([np.nan, np.nan, np.nan]),
                 'LGroundReactionForce': np.array([0.17760919, 0.09635437, 10.74801636]),
                 'RGroundReactionForce': np.array([np.nan, np.nan, np.nan]),
                 'LNormalisedGRF': np.array([-0.98220563, -1.81049132, 109.56183624]),
                 'RNormalisedGRF': np.array([np.nan, np.nan, np.nan]), 'LAnkleForce': np.array([np.nan, np.nan, np.nan]),
                 'RAnkleForce': np.array([np.nan, np.nan, np.nan]), 'LKneeForce': np.array([np.nan, np.nan, np.nan]),
                 'RKneeForce': np.array([np.nan, np.nan, np.nan]), 'LHipForce': np.array([np.nan, np.nan, np.nan]),
                 'RHipForce': np.array([np.nan, np.nan, np.nan]), 'LWaistForce': np.array([np.nan, np.nan, np.nan]),
                 'RWaistForce': np.array([np.nan, np.nan, np.nan]), 'LNeckForce': np.array([np.nan, np.nan, np.nan]),
                 'RNeckForce': np.array([np.nan, np.nan, np.nan]), 'LShoulderForce': np.array([np.nan, np.nan, np.nan]),
                 'RShoulderForce': np.array([np.nan, np.nan, np.nan]), 'LElbowForce': np.array([np.nan, np.nan, np.nan]),
                 'RElbowForce': np.array([np.nan, np.nan, np.nan]), 'LWristForce': np.array([np.nan, np.nan, np.nan]),
                 'RWristForce': np.array([np.nan, np.nan, np.nan]),
                 'LGroundReactionMoment': np.array([479.19515991, 1.15694141, 34.82278824]),
                 'RGroundReactionMoment': np.array([np.nan, np.nan, np.nan]),
                 'LAnkleMoment': np.array([np.nan, np.nan, np.nan]), 'RAnkleMoment': np.array([np.nan, np.nan, np.nan]),
                 'LKneeMoment': np.array([np.nan, np.nan, np.nan]), 'RKneeMoment': np.array([np.nan, np.nan, np.nan]),
                 'LHipMoment': np.array([np.nan, np.nan, np.nan]), 'RHipMoment': np.array([np.nan, np.nan, np.nan]),
                 'LWaistMoment': np.array([np.nan, np.nan, np.nan]), 'RWaistMoment': np.array([np.nan, np.nan, np.nan]),
                 'LNeckMoment': np.array([np.nan, np.nan, np.nan]), 'RNeckMoment': np.array([np.nan, np.nan, np.nan]),
                 'LShoulderMoment': np.array([np.nan, np.nan, np.nan]), 'RShoulderMoment': np.array([np.nan, np.nan, np.nan]),
                 'LElbowMoment': np.array([np.nan, np.nan, np.nan]), 'RElbowMoment': np.array([np.nan, np.nan, np.nan]),
                 'LWristMoment': np.array([np.nan, np.nan, np.nan]), 'RWristMoment': np.array([np.nan, np.nan, np.nan])}
        vsk = {'MeanLegLength': 940.0, 'Bodymass': 75.0, 'GCS': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
               'R_AsisToTrocanterMeasure': 72.512, 'L_AsisToTrocanterMeasure': 72.512,
               'InterAsisDistance': 215.908996582031, 'RightKneeWidth': 105.0, 'LeftKneeWidth': 105.0,
               'RightAnkleWidth': 70.0, 'LeftAnkleWidth': 70.0, 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0,
               'RightShoulderOffset': 40.0, 'LeftShoulderOffset': 40.0, 'RightElbowWidth': 74.0, 'LeftElbowWidth': 74.0,
               'RightWristWidth': 55.0, 'LeftWristWidth': 55.0, 'RightHandThickness': 34.0, 'LeftHandThickness': 34.0,
               'RightStaticRotOff': 0.015683497632642047, 'RightStaticPlantFlex': 0.2702417907002757,
               'LeftStaticRotOff': 0.009402910292403022, 'LeftStaticPlantFlex': 0.20251085737834015,
               'HeadOffset': 0.2571990469310653}
        expected = (np.array([-3.084949145094540e-01, -6.121292793370006e+00, 7.571431102151712e+00,  2.914222929716658e+00,
                              -6.867068980446340e+00, -1.882100070964313e+01, -2.860204550475336e+00, -5.345650918941786e+00,
                              -1.802561968633412e+00, 3.194368745140155e+00, 2.383410087509977e+00, -1.947591550139181e+01,
                              -4.584869655797333e-01, -3.866732130463932e-01, -2.187580804414196e+01,  2.505338413934339e+00,
                              -7.688220113637172e+00,  2.649810231950659e+01, 4.384670154323743e+00,  5.992969945839519e-01,
                              -2.378738006639775e+00, -8.389045650585302e+01, -4.884405611210539e+00,  7.044471355124547e+01,
                              8.600906645467649e+01,  1.679629484059455e+02, -7.218901201492267e+01,  2.119672927574356e-02,
                              5.462252836649474e+00, -9.149608534396434e+01, -3.756147383222924e+00,  7.302310925073288e-01,
                              2.648673619621234e+02,  6.093779646388498e+00, -3.937166516660525e+00, -3.689979861426082e+00,
                              -6.479705779091661e+00,  4.638276625836626e+00, -2.893068979100172e+00,  9.425691501658870e+00,
                              4.913352941114613e+01,  9.611224880156811e+00, 4.602181871876923e+00,  3.990544007068414e+01,
                              7.487134573309220e-01,  2.611592023895937e+01, 4.042485000140130e-12,  2.515321284590755e-12,
                              2.919419804190013e+01,  3.527235141743040e-12, -1.747935129969846e-12,  9.830271165266627e+00,
                              1.679309257731584e+01,  1.220970805137368e+02, 1.289415664327225e+01,  1.739767463438491e+01,
                              1.287413025779611e+02,  2.516083068847656e+02, 3.917413177490234e+02,  1.032893493652344e+03,
                              2.517406362411188e+02,  3.927269472068485e+02, 1.032788500732036e+03,  2.506171155403755e+02,
                              3.918723286246465e+02,  1.032874106304030e+03, 2.516029533575824e+02,  3.918479513381783e+02,
                              1.033887777624562e+03,  2.454757416720804e+02, 3.311178713574418e+02,  9.367593959314677e+02,
                              2.456080710284336e+02, 3.321035008152668e+02, 9.366544030111602e+02,  2.444845503276903e+02,
                              3.312488822330648e+02,  9.367400085831541e+02, 2.454703881448972e+02,  3.312245049465967e+02,
                              9.377536799036861e+02,  3.641777461389593e+02, 2.921705172185420e+02,  5.151918149640691e+02,
                              3.646195915323271e+02,  2.930675835327502e+02, 5.151851309329049e+02,  3.632901977137081e+02,
                              2.926065664837696e+02,  5.150430909541427e+02, 3.640472454064183e+02,  2.922421626385452e+02,
                              5.161806711177127e+02,  1.435547857935360e+02, 2.799037034643878e+02,  5.247840875332053e+02,
                              1.436561128167278e+02,  2.808868589643116e+02, 5.246319754122525e+02,  1.425643449910301e+02,
                              2.800177794253987e+02,  5.248616355302837e+02, 1.436483798697508e+02,  2.800465038078368e+02,
                              5.257694038330848e+02,  3.937618160814773e+02, 2.476782963272225e+02,  8.773775041198024e+01,
                              3.944817157501141e+02,  2.483720134846675e+02, 8.771536799559917e+01,  3.930711438441594e+02,
                              2.483911000599949e+02,  8.761575574347364e+01, 3.936931405567623e+02,  2.477815791622544e+02,
                              8.873002876420450e+01,  9.874901938986801e+01, 2.194693022064438e+02,  8.063068160484545e+01,
                              9.847494965871451e+01,  2.204255380335128e+02, 8.052821782834282e+01,  9.779246670826866e+01,
                              2.192092727535653e+02,  8.076255901367958e+01, 9.884848169323239e+01,  2.196034578126086e+02,
                              8.161663775171203e+01,  4.428199768066406e+02, 3.816228027343750e+02,  4.266047668457031e+01,
                              4.428462412676692e+02,  3.816513024007671e+02, 4.365972537588915e+01,  4.418773505621594e+02,
                              3.819563035019639e+02,  4.267574106247485e+01, 4.424871616307515e+02,  3.806804837825158e+02,
                              4.269610043598381e+01,  3.943652725219727e+01, 3.824452209472656e+02,  4.178911590576172e+01,
                              3.956652625791563e+01,  3.825090100046712e+02, 4.277857596795096e+01,  3.849313328387124e+01,
                              3.821460684058263e+02,  4.193234850497183e+01, 3.974166341694723e+01,  3.814931501972130e+02,
                              4.181040458481808e+01,  2.551907119750977e+02, 4.061208190917969e+02,  1.721920532226562e+03,
                              2.552168558251097e+02,  4.071159388775806e+02, 1.721825384388708e+03,  2.541910538517967e+02,
                              4.061468091837570e+02,  1.721917677119171e+03, 2.551903437022980e+02,  4.062160090443217e+02,
                              1.722915991285145e+03,  2.561498102365640e+02, 3.643090603933987e+02,  1.459655363929038e+03,
                              2.562399112853585e+02,  3.653049697693975e+02, 1.459662169500559e+03,  2.571435863244796e+02,
                              3.642196059906195e+02,  1.459588978712983e+03, 2.560843053658035e+02,  3.643218049852322e+02,
                              1.458657593069929e+03,  4.296695199452422e+02, 2.750671861469421e+02,  1.453953978131498e+03,
                              4.301273133059676e+02,  2.759513661907463e+02, 1.454046988286934e+03,  4.296862168456729e+02,
                              2.751632337671314e+02,  1.452958741441976e+03, 4.287806181214215e+02,  2.755243518770602e+02,
                              1.453983185028180e+03,  6.451952733569759e+01, 2.749344216095232e+02,  1.463631333396274e+03,
                              6.410400324869988e+01,  2.758319282646819e+02, 1.463779054542596e+03,  6.459882848203122e+01,
                              2.748083806826584e+02,  1.464620183745389e+03, 6.542564601518438e+01,  2.753570272042577e+02,
                              1.463612533130738e+03,  6.336670758734572e+02, 3.049554211541481e+02,  1.256077995412865e+03,
                              6.338107013869995e+02,  3.039657900497519e+02, 1.256076585068450e+03,  6.343524799178464e+02,
                              3.050538658933253e+02,  1.256799473014224e+03, 6.329532180390149e+02,  3.048508319073777e+02,
                              1.256770431750491e+03, -1.291695221821550e+02, 3.168671643985116e+02,  1.258064407167222e+03,
                              -1.293239179274950e+02,  3.158807291324946e+02, 1.258008662931836e+03, -1.284511713527903e+02,
                              3.167938233359283e+02,  1.257372602878070e+03, -1.284911903756091e+02,  3.167203088419363e+02,
                              1.258784337306702e+03,  7.933281430325068e+02, 4.512913478825204e+02,  1.084432551302043e+03,
                              7.937713372796160e+02,  4.504487918719012e+02, 1.084126482309332e+03,  7.940135470768960e+02,
                              4.513897926216976e+02,  1.085154028903402e+03, 7.927503886251119e+02,  4.507618122347140e+02,
                              1.085053672741407e+03, -2.724594189740742e+02, 4.858015221094770e+02,  1.091366623835082e+03,
                              -2.729250728167513e+02,  4.850120241803687e+02, 1.090966799475227e+03, -2.717410681447095e+02,
                              4.857281810468936e+02,  1.090674819545929e+03, -2.719425644638384e+02,  4.851921666233502e+02,
                              1.091967911874857e+03,  8.598061436603001e+02, 5.172823982323645e+02,  1.051972789444748e+03,
                              8.599567597867737e+02,  5.175924123242138e+02, 1.052911515200920e+03,  8.590797567344147e+02,
                              5.179612045889317e+02,  1.051865160618745e+03, 8.591355641971873e+02,  5.166167307529585e+02,
                              1.052300218811959e+03, -3.245347779766666e+02, 5.518874428934396e+02,  1.068025268366222e+03,
                              -3.246199407715637e+02,  5.521589330842497e+02, 1.068983934301081e+03, -3.253329318534787e+02,
                              5.512929248618385e+02,  1.068122729635612e+03, -3.239383740134880e+02,  5.511305800350597e+02,
                              1.068292590131722e+03]),
                    {'Pelvis_axis': [np.array([251.60830688476562, 391.74131774902344, 1032.8934936523438]),
                                     np.array([[251.74063624111878, 392.7269472068485, 1032.7885007320363],
                                            [250.61711554037552, 391.87232862464646, 1032.8741063040302],
                                            [251.60295335758238, 391.8479513381783, 1033.8877776245622]]),
                                     np.array([231.5784912109375, 210.25262451171875, 1052.2496948242188])],
                     'Thorax_axis': [[[256.23991128535846, 365.30496976939753, 1459.662169500559],
                                      [257.1435863244796, 364.21960599061947, 1459.588978712983],
                                      [256.0843053658035, 364.32180498523223, 1458.6575930699294]],
                                     [256.149810236564, 364.3090603933987, 1459.6553639290375]],
                     'Pelvis': np.array([251.60830688476562, 391.74131774902344, 1032.8934936523438]),
                     'RHip': np.array([308.38050471527424, 322.80342416841967, 937.9897906061211]),
                     'LHip': np.array([182.57097862888662, 339.43231854646393, 935.5290012568141]),
                     'RKnee': np.array([364.1777461389593, 292.170517218542, 515.1918149640691]),
                     'LKnee': np.array([143.55478579353598, 279.90370346438783, 524.7840875332053]),
                     'RAnkle': np.array([393.76181608147726, 247.6782963272225, 87.73775041198024]),
                     'LAnkle': np.array([98.74901938986801, 219.46930220644384, 80.63068160484545]),
                     'RFoot': np.array([442.8199768066406, 381.622802734375, 42.66047668457031]),
                     'LFoot': np.array([39.436527252197266, 382.4452209472656, 41.78911590576172]),
                     'RHEE': np.array([374.0125732421875, 181.5792999267578, 49.50960922241211]),
                     'LHEE': np.array([105.30126953125, 180.2130126953125, 47.15660858154297]),
                     'C7': np.array([251.2261962890625, 229.7568359375, 1533.7762451171875]),
                     'CLAV': np.array([256.780517578125, 371.2804260253906, 1459.7030029296875]),
                     'STRN': np.array([251.6749267578125, 414.1039123535156, 1292.0850830078125]),
                     'T10': np.array([228.6432342529297, 192.32041931152344, 1279.641845703125]),
                     'Front_Head': np.array([255.19071197509766, 406.1208190917969, 1721.9205322265625]),
                     'Back_Head': np.array([251.1305694580078, 247.10114288330078, 1695.9384765625]),
                     'Head': [255.19071197509766, 406.1208190917969, 1721.9205322265625],
                     'Thorax': [256.149810236564, 364.3090603933987, 1459.6553639290375],
                     'RShoulder': np.array([429.6695199452422, 275.0671861469421, 1453.9539781314984]),
                     'LShoulder': np.array([64.51952733569759, 274.9344216095232, 1463.631333396274]),
                     'RHumerus': np.array([633.6670758734572, 304.9554211541481, 1256.0779954128648]),
                     'LHumerus': np.array([-129.16952218215505, 316.86716439851165, 1258.0644071672225]),
                     'RRadius': [793.3281430325068, 451.2913478825204, 1084.4325513020426],
                     'LRadius': [-272.4594189740742, 485.801522109477, 1091.3666238350822],
                     'RHand': np.array([859.8061436603001, 517.2823982323645, 1051.9727894447476]),
                     'LHand': np.array([-324.53477797666665, 551.8874428934396, 1068.0252683662222])})

        result = pyCGM.JointAngleCalc(frame, vsk)
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1]['Pelvis_axis'][0], expected[1]['Pelvis_axis'][0], rounding_precision)
        np.testing.assert_almost_equal(result[1]['Pelvis_axis'][1], expected[1]['Pelvis_axis'][1], rounding_precision)
        np.testing.assert_almost_equal(result[1]['Pelvis_axis'][2], expected[1]['Pelvis_axis'][2], rounding_precision)
        np.testing.assert_almost_equal(result[1]['Thorax_axis'][0], expected[1]['Thorax_axis'][0], rounding_precision)
        np.testing.assert_almost_equal(result[1]['Thorax_axis'][1], expected[1]['Thorax_axis'][1], rounding_precision)
        np.testing.assert_almost_equal(result[1]['Pelvis'], expected[1]['Pelvis'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['RHip'], expected[1]['RHip'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['LHip'], expected[1]['LHip'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['RKnee'], expected[1]['RKnee'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['LKnee'], expected[1]['LKnee'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['RAnkle'], expected[1]['RAnkle'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['LAnkle'], expected[1]['LAnkle'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['RFoot'], expected[1]['RFoot'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['LFoot'], expected[1]['LFoot'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['RHEE'], expected[1]['RHEE'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['LHEE'], expected[1]['LHEE'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['C7'], expected[1]['C7'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['CLAV'], expected[1]['CLAV'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['STRN'], expected[1]['STRN'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['T10'], expected[1]['T10'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['Front_Head'], expected[1]['Front_Head'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['Back_Head'], expected[1]['Back_Head'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['Head'], expected[1]['Head'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['Thorax'], expected[1]['Thorax'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['RShoulder'], expected[1]['RShoulder'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['LShoulder'], expected[1]['LShoulder'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['RHumerus'], expected[1]['RHumerus'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['LHumerus'], expected[1]['LHumerus'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['RRadius'], expected[1]['RRadius'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['LRadius'], expected[1]['LRadius'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['RHand'], expected[1]['RHand'], rounding_precision)
        np.testing.assert_almost_equal(result[1]['LHand'], expected[1]['LHand'], rounding_precision)