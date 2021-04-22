import pytest
import pycgm.static as static
import numpy as np

rounding_precision = 8


class TestPycgmStaticAxis():
    """
    This class tests the axis functions in static.py:
        calculate_head_angle
        pelvisJointCenter
        hipJointCenter
        hipAxisCenter
        kneeJointCenter
        ankleJointCenter
        foot_joint_center
        headJC
        uncorrect_footaxis
        rotaxis_footflat
        rotaxis_nonfootflat
        findJointC
    """
    nan_3d = [np.nan, np.nan, np.nan]
    rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10),
                 np.random.randint(0, 10)]

    @pytest.mark.parametrize(["head", "expected"], [
        # Test from running sample data
        ([[[244.87227957886893, 326.0240255639856, 1730.4189843948805],
           [243.89575702706503, 325.0366593474616, 1730.1515677531293],
           [244.89086730509763, 324.80072493605866, 1731.1283433097797]],
          [244.89547729492188, 325.0578918457031, 1730.1619873046875]],
         0.25992807335420975),
        # Test with zeros for all params
        ([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]],
         np.nan),
        # Testing when values are added to head[0][0]
        ([[[-1, 8, 9], [0, 0, 0], [0, 0, 0]], [0, 0, 0]],
         1.5707963267948966),
        # Testing when values are added to head[0][1]
        ([[[0, 0, 0], [7, 5, 7], [0, 0, 0]], [0, 0, 0]],
         np.nan),
        # Testing when values are added to head[0][2]
        ([[[0, 0, 0], [0, 0, 0], [3, -6, -2]], [0, 0, 0]],
         0.0),
        # Testing when values are added to head[0]
        ([[[-1, 8, 9], [7, 5, 7], [3, -6, -2]], [0, 0, 0]],
         -1.3521273809209546),
        # Testing when values are added to head[1]
        ([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [-4, 7, 8]],
         0.7853981633974483),
        # Testing when values are added to head
        ([[[-1, 8, 9], [7, 5, 7], [3, -6, -2]], [-4, 7, 8]],
         -0.09966865249116204),
        # Testing that when head is composed of lists of ints
        ([[[-1, 8, 9], [7, 5, 7], [3, -6, -2]], [-4, 7, 8]],
         -0.09966865249116204),
        # Testing that when head is composed of numpy arrays of ints
        ([np.array([[-1, 8, 9], [7, 5, 7], [3, -6, -2]], dtype='int'),
          np.array([-4, 7, 8], dtype='int')], -0.09966865249116204),
        # Testing that when head is composed of lists of floats
        ([[[-1.0, 8.0, 9.0], [7.0, 5.0, 7.0], [3.0, -6.0, -2.0]],
          [-4.0, 7.0, 8.0]], -0.09966865249116204),
        # Testing that when head is composed of numpy arrays of floats
        ([np.array([[-1.0, 8.0, 9.0], [7.0, 5.0, 7.0],
                    [3.0, -6.0, -2.0]], dtype='float'),
          np.array([-4.0, 7.0, 8.0], dtype='float')], -0.09966865249116204)])
    def test_calculate_head_angle(self, head, expected):
        """
        This test provides coverage of the staticCalculationHead function in
        static.py, defined as staticCalculationHead(frame, head).

        This test takes 2 parameters:
        head: array containing the head axis and head origin
        expected: the expected result from calling staticCalculationHead on
        head.

        This function first calculates the x, y, z axes of the head by
        subtracting the given head axes by the head origin. It then calls
        headoffCalc on this head axis and a global axis to find the head offset
        angles.

        This test ensures that:
        - the head axis and the head origin both have an effect on the final
          offset angle
        - the resulting output is correct when head is composed of lists of
          ints, numpy arrays of ints, lists of floats, and numpy arrays of
          floats.
       """
        result = static.calculate_head_angle(head)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["rtoe", "ltoe", "static_info",
                             "ankle_joint_center", "expected"],
                             [
        # Test from running sample data
        (np.array([442.81997681, 381.62280273, 42.66047668]),
         np.array([39.43652725, 382.44522095, 41.78911591]),
         [[0.03482194, 0.14879424, np.random.randint(0, 10)],
          [0.01139704, 0.02142806, np.random.randint(0, 10)]],
         [np.array([393.76181608, 247.67829633, 87.73775041]),
          np.array([98.74901939, 219.46930221, 80.6306816]),
          [[np.array(nan_3d),
            np.array([393.07114384, 248.39110006, 87.61575574]),
            np.array(nan_3d)],
           [np.array(nan_3d),
            np.array([97.79246671, 219.20927275, 80.76255901]),
            np.array(nan_3d)]]],
         [np.array([442.81997681, 381.62280273, 42.66047668]),
          np.array([39.43652725, 382.44522095, 41.78911591]),
          np.array([[[442.8881541, 381.76460597, 43.64802096],
                     [441.89515447, 382.00308979, 42.66971773],
                     [442.44573691, 380.70886969, 42.81754643]],
                    [[39.50785213, 382.67891581, 42.75880631],
                     [38.49231839, 382.14765966, 41.93027863],
                     [39.75805858, 381.51956227, 41.98854914]]])]),
        # Test with zeros for all params
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         [[0, 0, np.random.randint(0, 10)], [0, 0, np.random.randint(0, 10)]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values added to frame
        (np.array([-1, -1, -5]), np.array([-5, -6, 1]),
         [[0, 0, np.random.randint(0, 10)], [0, 0, np.random.randint(0, 10)]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([-1, -1, -5]), np.array([-5, -6, 1]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values added to static_info
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         [[-6, 7, np.random.randint(0, 10)],
          [2, -9, np.random.randint(0, 10)]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values added to ankle_JC
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         [[0, 0, np.random.randint(0, 10)], [0, 0, np.random.randint(0, 10)]],
         [np.array([6, 0, 3]), np.array([1, 4, -3]),
          [[np.array(nan_3d), np.array([-2, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([1, -6, 8]), np.array(nan_3d)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[[0.3713906763541037, 0.5570860145311556,
                      -0.7427813527082074],
                     [-0.24913643956121992, 0.8304547985373997,
                      0.49827287912243984],
                     [0.8944271909999159, 0.0, 0.4472135954999579]],
                    [[-0.6855829496241487, 0.538672317561831,
                      0.4897021068743917],
                     [0.701080937355391, 0.3073231506215415,
                      0.6434578466138523],
                     [0.19611613513818404, 0.7844645405527362,
                      -0.5883484054145521]]])]),
        # Testing with values added to frame and static_info
        (np.array([-1, -1, -5]), np.array([-5, -6, 1]),
         [[-6, 7, np.random.randint(0, 10)],
          [2, -9, np.random.randint(0, 10)]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          [[np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([0, 0, 0]), np.array(nan_3d)]]],
         [np.array([-1, -1, -5]), np.array([-5, -6, 1]),
          np.array([[nan_3d, nan_3d, nan_3d],
                    [nan_3d, nan_3d, nan_3d]])]),
        # Testing with values added to frame and ankle_JC
        (np.array([-1, -1, -5]), np.array([-5, -6, 1]),
         [[0, 0, np.random.randint(0, 10)], [0, 0, np.random.randint(0, 10)]],
         [np.array([6, 0, 3]), np.array([1, 4, -3]),
          [[np.array(nan_3d), np.array([-2, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([1, -6, 8]), np.array(nan_3d)]]],
         [np.array([-1, -1, -5]), np.array([-5, -6, 1]),
          np.array([[[-0.4764529245456802, -0.34134400184779123,
                      -5.540435690791556],
                     [-1.544126730072802, -0.25340750990010874,
                      -4.617213172448785],
                     [-0.3443899318928142, -0.9063414188418306,
                      -4.250731350734645]],
                    [[-5.617369411832039, -5.417908840272649,
                      1.5291737815703186],
                     [-4.3819280753253675, -6.057228881914318,
                      1.7840356822261547],
                     [-4.513335736607712, -5.188892894346187,
                      0.6755571577384749]]])]),
        # Testing with values added to static_info and ankle_JC
        (np.array([0, 0, 0]), np.array([0, 0, 0]),
         [[-6, 7, np.random.randint(0, 10)],
          [2, -9, np.random.randint(0, 10)]],
         [np.array([6, 0, 3]), np.array([1, 4, -3]),
          [[np.array(nan_3d), np.array([-2, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([1, -6, 8]), np.array(nan_3d)]]],
         [np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([[[0.8676189717605698, 0.41998838044559317,
                      -0.2661711481957037],
                     [-0.35944921047092726, 0.8996435491853136,
                      0.2478663944569317],
                     [0.3435601620283683, -0.11937857722363693,
                      0.9315123028533232]],
                    [[0.5438323231671144, -0.8140929502604927,
                      -0.20371321168453085],
                     [0.12764145145799288, 0.32016712879535714,
                      -0.9387228928222822],
                     [0.829429963377473, 0.48450560159311296,
                      0.27802923924749284]]])]),
        # Testing with values added to frame, static_info and ankle_JC
        (np.array([-1, -1, -5]), np.array([-5, -6, 1]),
         [[-6, 7, np.random.randint(0, 10)],
          [2, -9, np.random.randint(0, 10)]],
         [np.array([6, 0, 3]), np.array([1, 4, -3]),
          [[np.array(nan_3d), np.array([-2, 8, 5]), np.array(nan_3d)],
           [np.array(nan_3d), np.array([1, -6, 8]), np.array(nan_3d)]]],
         [np.array([-1, -1, -5]), np.array([-5, -6, 1]),
          np.array([[[-0.17456964188738444, -0.44190534702217665,
                      -4.915176169482615],
                     [-1.564451151846412, -0.1819624820720035,
                      -4.889503319319258],
                     [-1.0077214691178664, -1.139086223544123,
                      -4.009749828914483]],
                    [[-4.638059331793927, -6.864633064377841,
                      0.6515626072260268],
                     [-4.6226610672854616, -5.522323332954951,
                      0.2066272429566376],
                     [-4.147583269429562, -5.844325128086398,
                      1.4991503297587707]]])]),
        # Testing that when frame, static_info and ankle_JC are composed of
        # lists of ints
        ([-1, -1, -5], [-5, -6, 1],
         [[-6, 7, np.random.randint(0, 10)],
          [2, -9, np.random.randint(0, 10)]],
         [[6, 0, 3], [1, 4, -3],
          [[nan_3d, [-2, 8, 5], nan_3d],
           [nan_3d, [1, -6, 8], nan_3d]]],
         [np.array([-1, -1, -5]), np.array([-5, -6, 1]),
          np.array([[[-0.17456964188738444, -0.44190534702217665,
                      -4.915176169482615],
                     [-1.564451151846412, -0.1819624820720035,
                      -4.889503319319258],
                     [-1.0077214691178664, -1.139086223544123,
                      -4.009749828914483]],
                    [[-4.638059331793927, -6.864633064377841,
                      0.6515626072260268],
                     [-4.6226610672854616, -5.522323332954951,
                      0.2066272429566376],
                     [-4.147583269429562, -5.844325128086398,
                      1.4991503297587707]]])]),
        # Testing that when frame, static_info and ankle_JC are composed of
        # numpy arrays of ints
        (np.array([-1, -1, -5], dtype='int'),
         np.array([-5, -6, 1], dtype='int'),
         [np.array([-6, 7, np.random.randint(0, 10)], dtype='int'),
          np.array([2, -9, np.random.randint(0, 10)], dtype='int')],
         [np.array([6, 0, 3], dtype='int'), np.array([1, 4, -3], dtype='int'),
          [[np.array(nan_3d), np.array([-2, 8, 5], dtype='int'),
            np.array(nan_3d)],
           [np.array(nan_3d), np.array([1, -6, 8], dtype='int'),
            np.array(nan_3d)]]],
         [np.array([-1, -1, -5]), np.array([-5, -6, 1]),
          np.array([[[-0.17456964188738444, -0.44190534702217665,
                      -4.915176169482615],
                     [-1.564451151846412, -0.1819624820720035,
                      -4.889503319319258],
                     [-1.0077214691178664, -1.139086223544123,
                      -4.009749828914483]],
                    [[-4.638059331793927, -6.864633064377841,
                      0.6515626072260268],
                     [-4.6226610672854616, -5.522323332954951,
                      0.2066272429566376],
                     [-4.147583269429562, -5.844325128086398,
                      1.4991503297587707]]])]),
        # Testing that when frame, static_info and ankle_JC are composed of
        # lists of floats
        ([-1.0, -1.0, -5.0], [-5.0, -6.0, 1.0],
         [[-6.0, 7.0, np.random.randint(0, 10)],
          [2.0, -9.0, np.random.randint(0, 10)]],
         [[6.0, 0.0, 3.0], [1.0, 4.0, -3.0],
          [[nan_3d, [-2.0, 8.0, 5.0], nan_3d],
           [nan_3d, [1.0, -6.0, 8.0], nan_3d]]],
         [np.array([-1, -1, -5]), np.array([-5, -6, 1]),
          np.array([[[-0.17456964188738444, -0.44190534702217665,
                      -4.915176169482615],
                     [-1.564451151846412, -0.1819624820720035,
                      -4.889503319319258],
                     [-1.0077214691178664, -1.139086223544123,
                      -4.009749828914483]],
                    [[-4.638059331793927, -6.864633064377841,
                      0.6515626072260268],
                     [-4.6226610672854616, -5.522323332954951,
                      0.2066272429566376],
                     [-4.147583269429562, -5.844325128086398,
                      1.4991503297587707]]])]),
        # Testing that when frame, static_info and ankle_JC are composed of
        # numpy arrays of floats
        (np.array([-1.0, -1.0, -5.0], dtype='float'),
         np.array([-5.0, -6.0, 1.0], dtype='float'),
         [np.array([-6.0, 7.0, np.random.randint(0, 10)], dtype='float'),
          np.array([2.0, -9.0, np.random.randint(0, 10)], dtype='float')],
         [np.array([6.0, 0.0, 3.0], dtype='float'),
          np.array([1.0, 4.0, -3.0], dtype='float'),
          [[np.array(nan_3d), np.array([-2.0, 8.0, 5.0], dtype='float'),
            np.array(nan_3d)],
           [np.array(nan_3d), np.array([1.0, -6.0, 8.0], dtype='float'),
            np.array(nan_3d)]]],
         [np.array([-1, -1, -5]), np.array([-5, -6, 1]),
          np.array([[[-0.17456964188738444, -0.44190534702217665,
                      -4.915176169482615],
                     [-1.564451151846412, -0.1819624820720035,
                      -4.889503319319258],
                     [-1.0077214691178664, -1.139086223544123,
                      -4.009749828914483]],
                    [[-4.638059331793927, -6.864633064377841,
                      0.6515626072260268],
                     [-4.6226610672854616, -5.522323332954951,
                      0.2066272429566376],
                     [-4.147583269429562, -5.844325128086398,
                      1.4991503297587707]]])])])
    def test_foot_joint_center(self, rtoe, ltoe, static_info,
                               ankle_joint_center, expected):
        """
        This test provides coverage of the foot_joint_center function in
        static.py, defined as foot_joint_center(rtoe, ltoe, static_info,
        ankle_joint_center).

        This test takes 5 parameters:
            rtoe : array
                Array of marker data.
            ltoe : array
                Array of marker data.
            static_info : array
                An array containing offset angles.
            ankle_joint_center : array
                An array containing the x,y,z axes marker positions of the
                ankle joint center.
            expected: the expected result from calling footJointCenter on
                      frame, static_info, and ankle_JC

        The incorrect foot joint axes for both feet are calculated using the
        following calculations:
            z-axis = ankle joint center - TOE marker
            y-flex = ankle joint center flexion - ankle joint center
            x-axis = y-flex cross z-axis
            y-axis = z-axis cross x-axis
        Calculate the foot joint axis by rotating incorrect foot joint axes
        about offset angle.

        This test is checking to make sure the foot joint center and axis are
        calculated correctly given the input parameters. The test checks to see
        that the correct values in expected are updated per each input
        parameter added:
            When values are added to frame, expected[0] and expected[1] should
            be updated
            When values are added to vsk, expected[2] should be updated as long
            as there are values for frame and ankle_JC
            When values are added to ankle_JC, expected[2] should be updated
        """
        result = static.foot_joint_center(rtoe, ltoe, static_info,
                                          ankle_joint_center)
        np.testing.assert_almost_equal(result[0], expected[0],
                                       rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1],
                                       rounding_precision)
        np.testing.assert_almost_equal(result[2], expected[2],
                                       rounding_precision)
