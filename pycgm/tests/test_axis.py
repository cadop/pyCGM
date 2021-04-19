from mock import patch
import pyCGM_Single.pyCGM as pyCGM
import pytest
import numpy as np

rounding_precision = 6

class TestUpperBodyAxis():
    nan_3d = [np.nan, np.nan, np.nan]
    rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]

    @pytest.mark.parametrize(["rwra", "rwrb", "lwra", "lwrb", "rfin", "lfin", "wristJC", "vsk", "mockReturnVal", "expectedMockArgs", "expected"], [
        # Test from running sample data
        (np.array([ 776.51898193,  495.68103027, 1108.38464355]), 
        np.array([ 830.9072876 ,  436.75341797, 1119.11901855]), 
        np.array([ 863.71374512,  524.4475708 , 1074.54248047]),
        np.array([-249.28146362,  525.32977295, 1117.09057617]), 
        np.array([-311.77532959,  477.22512817, 1125.1619873 ]), 
        np.array([-326.65890503,  558.34338379, 1091.04284668]),

         [[[rand_coor, rand_coor, rand_coor, 793.3281430325068],
            [rand_coor, rand_coor, rand_coor, 451.2913478825204],
            [rand_coor, rand_coor, rand_coor, 1084.4325513020426],
            [0,0,0,1]],
            [[rand_coor, rand_coor, rand_coor, -272.4594189740742],
            [rand_coor, rand_coor, rand_coor, 485.801522109477],
            [rand_coor, rand_coor, rand_coor, 1091.3666238350822],
            [0,0,0,1]]], 

         {'RightHandThickness': 34.0, 'LeftHandThickness': 34.0},

         [[-324.53477798, 551.88744289, 1068.02526837], [859.80614366, 517.28239823, 1051.97278945]],

         [
          [[-280.528396605, 501.27745056000003, 1121.126281735], [-272.4594189740742, 485.801522109477, 1091.3666238350822], [-326.65890503, 558.34338379, 1091.04284668], 24.0],

          [[803.713134765, 466.21722411999997, 1113.75183105], [793.3281430325068, 451.2913478825204, 1084.4325513020426], [863.71374512, 524.4475708, 1074.54248047], 24.0]
         ],
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
           [[-6.21615749183132, 3.059079153204844, -3.739339495144585], [-6.186838410896736, 3.777824759216273, -4.9569376001580645], [-5.04168515250009, 3.744449374000024, -4.127775312999988]]]]),
        # Testing that when frame and wristJC are composed of lists of ints and vsk values are ints
        ({'RWRA': [4, 7, 6], 'RWRB': [0, -5, 4], 'RFIN': [1, -9, 6], 'LWRA': [-4, 5, 3], 'LWRB': [-3, 2, -7],
          'LFIN': [-6, 3, 8]},
         [[[0, 4, 3], [9, 0, -6]], [[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]]],
         {'RightHandThickness': 36, 'LeftHandThickness': -9},
         [[-6, 4, -4], [2, 8, 1]],
         [[[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 2.5], [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 25.0]],
         [[np.array([2, 8, 1]), np.array([-6, 4, -4])],
          [[[2.911684611677104, 7.658118270621086, 1.227921152919276], [1.9534757894800765, 8.465242105199236, 1.8839599998785472], [1.5917517095361369, 7.183503419072274, 1.4082482904638631]],
           [[-6.21615749183132, 3.059079153204844, -3.739339495144585], [-6.186838410896736, 3.777824759216273, -4.9569376001580645], [-5.04168515250009, 3.744449374000024, -4.127775312999988]]]]),
        # Testing that when frame and wristJC are composed of numpy arrays of ints and vsk values are ints
        ({'RWRA': np.array([4, 7, 6], dtype='int'), 'RWRB': np.array([0, -5, 4], dtype='int'),
          'RFIN': np.array([1, -9, 6], dtype='int'), 'LWRA': np.array([-4, 5, 3], dtype='int'),
          'LWRB': np.array([-3, 2, -7], dtype='int'), 'LFIN': np.array([-6, 3, 8], dtype='int')},
         [np.array([[0, 4, 3], [9, 0, -6]], dtype='int'),
          [[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]]],
         {'RightHandThickness': 36, 'LeftHandThickness': -9},
         [[-6, 4, -4], [2, 8, 1]],
         [[[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 2.5], [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 25.0]],
         [[np.array([2, 8, 1]), np.array([-6, 4, -4])],
          [[[2.911684611677104, 7.658118270621086, 1.227921152919276], [1.9534757894800765, 8.465242105199236, 1.8839599998785472], [1.5917517095361369, 7.183503419072274, 1.4082482904638631]],
           [[-6.21615749183132, 3.059079153204844, -3.739339495144585], [-6.186838410896736, 3.777824759216273, -4.9569376001580645], [-5.04168515250009, 3.744449374000024, -4.127775312999988]]]]),
        # Testing that when frame and wristJC are composed of lists of floats and vsk values are floats
        ({'RWRA': [4.0, 7.0, 6.0], 'RWRB': [0.0, -5.0, 4.0], 'RFIN': [1.0, -9.0, 6.0], 'LWRA': [-4.0, 5.0, 3.0],
          'LWRB': [-3.0, 2.0, -7.0], 'LFIN': [-6.0, 3.0, 8.0]},
         [[[0.0, 4.0, 3.0], [9.0, 0.0, -6.0]], [[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]]],
         {'RightHandThickness': 36.0, 'LeftHandThickness': -9.0},
         [[-6, 4, -4], [2, 8, 1]],
         [[[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 2.5], [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 25.0]],
         [[np.array([2, 8, 1]), np.array([-6, 4, -4])],
          [[[2.911684611677104, 7.658118270621086, 1.227921152919276], [1.9534757894800765, 8.465242105199236, 1.8839599998785472], [1.5917517095361369, 7.183503419072274, 1.4082482904638631]],
           [[-6.21615749183132, 3.059079153204844, -3.739339495144585], [-6.186838410896736, 3.777824759216273, -4.9569376001580645], [-5.04168515250009, 3.744449374000024, -4.127775312999988]]]]),
        # Testing that when frame and wristJC are composed of numpy arrays of floats and vsk values are floats
        ({'RWRA': np.array([4.0, 7.0, 6.0], dtype='float'), 'RWRB': np.array([0.0, -5.0, 4.0], dtype='float'),
          'RFIN': np.array([1.0, -9.0, 6.0], dtype='float'), 'LWRA': np.array([-4.0, 5.0, 3.0], dtype='float'),
          'LWRB': np.array([-3.0, 2.0, -7.0], dtype='float'), 'LFIN': np.array([-6.0, 3.0, 8.0], dtype='float')},
         [np.array([[0.0, 4.0, 3.0], [9.0, 0.0, -6.0]], dtype='float'),
          [[rand_coor, rand_coor, rand_coor], [rand_coor, rand_coor, rand_coor]]],
         {'RightHandThickness': 36.0, 'LeftHandThickness': -9.0},
         [[-6, 4, -4], [2, 8, 1]],
         [[[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 2.5], [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 25.0]],
         [[np.array([2, 8, 1]), np.array([-6, 4, -4])],
          [[[2.911684611677104, 7.658118270621086, 1.227921152919276], [1.9534757894800765, 8.465242105199236, 1.8839599998785472], [1.5917517095361369, 7.183503419072274, 1.4082482904638631]],
           [[-6.21615749183132, 3.059079153204844, -3.739339495144585], [-6.186838410896736, 3.777824759216273, -4.9569376001580645], [-5.04168515250009, 3.744449374000024, -4.127775312999988]]]])])
    def test_handJointCenter(self, frame, wristJC, vsk, mockReturnVal, expectedMockArgs, expected):
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
        the return value of findJointC, mockReturnVal. 

        Using RWRA, RWRB, LWRA, and LWRB from the given frame dictionary, 
        RWRI = (RWRA+RWRB)/2 
        LWRI = (LWRA+LWRB)/2
        aka the midpoints of the markers for each direction.

        LHND is calculated using the Rodriques' rotation formula with the LWRI, LWJC, and LFIN as reference points. The thickness of the left hand is also applied in the calculations. 
        The same can be said for the RHND, but with respective markers and measurements (aka RWRI, RWJC, and RFIN).
        z_axis = LWJC - LHND
        y-axis = LWRI - LRWA
        x-axis =  y-axis \cross z-axis 
        y-axis = z-axis \cross x-axis 

        This is for the handJC left axis, and is the same for the right axis but with the respective markers. 
        The origin for each direction is calculated by adding each axis to each HND marker. 

        Lastly, it checks that the resulting output is correct when frame and wristJC are composed of lists of ints,
        numpy arrays of ints, lists of floats, and numpy arrays of floats and vsk values are either an int or a float.
        wristJC cannot be a numpy array due to it not being shaped like a multi-dimensional array.
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