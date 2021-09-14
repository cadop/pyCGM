from unittest import TestCase
import pytest
import pyCGM_Single.pycgmKinetics as pycgmKinetics
import numpy as np
import os

rounding_precision = 8

class Test_pycgmKinetics(TestCase):
    def test_f(self):
        """
        This test provides coverage of the f function in pycgmKinetics.py,
        defined as f(p, x), where p is a list of at least 2 elements and x 
        is a int or float.

        Only the first two elements of p is used, all excess elements are ignored in the calculation.

        Each index in accuracyTests is used as parameters for the function f 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, excessive elements, negative numbers, floats
        accuracyTests=[
            ([0,0],0),
            ([1,2],10),
            ([1,2,3,4,5],10),
            (np.array([1,2,3,4,5]),10),
            (np.array([142.10231243,23.40231521]),10),
            (np.array([-142.10231243,-23.40231521]),-10.421312),
            (np.array([0.10231243,3.40231521]),0.421312),
            (np.array([0.10231243,-3.40231521]),-0.421312),
            (np.array([-0.10231243,3.40231521]),-0.421312),
            (np.array([-0.10231243,-3.40231521]),-0.421312),
            (np.array([-0.10231243,-3.40231521,-8.10231243,-5.40231521,1]),-0.421312)
        ]
        accuracyResults=[
            0,
            12,
            12,
            12,
            1444.42543951,
            1457.49021854,
            3.44542066,
            -3.44542066,
            3.44542066,
            -3.35920975,
            -3.35920975
        ]
        for i in range(len(accuracyTests)):
            # Call f(p, x) with the variable given from each accuracyTests index.
            result = pycgmKinetics.f(accuracyTests[i][0],accuracyTests[i][1])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)

        # f([1,2], 2) should result in 4, test to make sure it does not result as anything else.
        self.assertFalse(pycgmKinetics.f([1,2], 2) != 4)
        
        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for f.
        exceptionTests=[([1],2), ([],2), ([1,2]), (2), ([]), ([1,2], "a"), (["a","b"], "c")]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                pycgmKinetics.f(e[0],e[1])

    def test_dot(self):
        """
        This test provides coverage of the dot function in pycgmKinetics.py,
        defined as dot(v, w), where v and w are 3-element lists.

        Each index in accuracyTests is used as parameters for the function dot 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, list and numpy array, negatives, and floats.
        accuracyTests=[
            ([0,0,0],[3,4,5]),
            ([1,2,3],[4,5,6]),
            ([1,2,3],[-4,-5,-6]),
            (np.array([1,2,3]),np.array([4,5,6])),
            (np.array([0.1,0.2,0.3]),np.array([4,5,6])),
            (np.array([240.92318213,160.32949124,429.2941023]),np.array([204.2931024,20.20142134,1.4293544])),
            ([366.87249488, 972.25566446, 519.54469762], [318.87916021, 624.44837115, 173.28031049]),
            ([143.73405485, 253.65432719, 497.53480618], [607.58442024, -836.1222914, 747.91240563]),
            ([918.86076151, 92.04884656, 568.38140393], [-186.24219938, -724.27298992, -155.58515366]),
            ([467.7365042, -788.74773579, 500.33205429], [649.06495926, 310.14934252, 853.05203014]),
            ([27.7365042, -78.74773579, 31.33205429], np.array([69.06495926, 30.14934252, 53.05203014]))
        ]
        accuracyResults=[
            0,
            32,
            -32,
            32,
            3.2,
            53071.44133720,
            814138.32560191,
            247356.98888589,
            -326230.85053223,
            485771.05802978,
            1203.65716212
        ]
        for i in range(len(accuracyTests)):
            # Call dot(v, w) with the variable given from each accuracyTests index.
            result = pycgmKinetics.dot(accuracyTests[i][0],accuracyTests[i][1])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)
        
        # dot((0,0,0],[3,4,5]) should result in 0, test to make sure it does not result as anything else.
        self.assertFalse(pycgmKinetics.dot([0,0,0],[3,4,5]) != 0)

        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for dot. 
        exceptionTests=[([]), ([1,2,3],[4,5]), ([],[]), ([1,2],[4,5,6]), ([1,2,"c"],[4,5,6]), (["a","b","c"], ["e","f","g"])]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                pycgmKinetics.dot(e[0],e[1])

    def test_length(self):
        """
        This test provides coverage of the length function in pycgmKinetics.py,
        defined as length(v), where v is a 3-element list.

        Each index in accuracyTests is used as parameters for the function length 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, floats, numpy arrays, and negatives
        accuracyTests=[
            ([0,0,0]),
            ([1,2,3]),
            ([1.1,2.2,3.3]),
            (np.array([1.1,2.2,3.3])),
            (np.array([-1.1,-2.2,-3.3])),
            (np.array([4.1,-5.2,6.3])),
            (np.array([20.1,-0.2,0])),
            (np.array([477.96370143, -997.67255536, 400.99490597])),
            (np.array([330.80492334, 608.46071522, 451.3237226])),
            (np.array([-256.41091237, 391.85451166, 679.8028365])),
            (np.array([197.08510663, 319.00331132, -195.89839035])),
            (np.array([910.42721331, 184.76837848, -67.24503815])),
            (np.array([313.91884245, -703.86347965, -831.19994848])),
            (np.array([710.57698646, 991.83524562, 781.3712082]))
        ]
        accuracyResults=[
            0.0,
            3.74165738,
            4.11582312,
            4.11582312,
            4.11582312,
            9.14002188,
            20.10099500,
            1176.68888930,
            826.64952782,
            825.486772034,
            423.06244365,
            931.41771487,
            1133.51761873,
            1448.86085361
        ]
        for i in range(len(accuracyTests)):
            # Call length(v) with the variable given from each accuracyTests index.
            result = pycgmKinetics.length(accuracyTests[i])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)
        
        # length([0,0,0]) should result in 0.0, test to make sure it does not result as anything else.
        self.assertFalse(pycgmKinetics.length([0,0,0]) != 0.0)

        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for length.
        exceptionTests=[([]), ([1]), ([1,2]), ([1,2,"c"]), (["a","b",3])]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                pycgmKinetics.length(e[0])
    
    def test_vector(self):
        """
        This test provides coverage of the vector function in pycgmKinetics.py,
        defined as vector(b, e), where b and e are 3-element lists.

        Each index in accuracyTests is used as parameters for the function vector 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, negatives, and floats 
        accuracyTests=[
            ([0,0,0],[1,2,3]),
            ([1,2,3],[1,2,3]),
            ([1,2,3],[4,5,6]),
            ([-1,-2,-3],[4,5,6]),
            ([-1.1,-2.2,-3.3],[4.4,5.5,6]),
            (np.array([-1,-2,-3]),np.array([4,5,6])),
            (np.array([871.13796878, 80.07048505, 81.7226316]), np.array([150.60899971, 439.55690306, -746.27742664])),
            (np.array([109.96296398, 278.68529143, 224.18342906]), np.array([28.90044238, -332.38141918, 625.15884162])),
            (np.array([261.89862662, 635.64883561, 335.23199233]), np.array([462.68440338, 329.95040901, 260.75626459])),
            (np.array([-822.76892296, -457.04755227, 64.67044766]), np.array([883.37510574, 599.45910665, 94.24813625])),
            (np.array([-723.03974742, -913.26790889, 95.50575378]), np.array([-322.89139623, 175.08781892, -954.38748492])),
            (np.array([602.28250216, 868.53946449, 666.82151334]), np.array([741.07723854, -37.57504097, 321.13189537])),
            (np.array([646.40999378, -633.96507365, -33.52275607]), np.array([479.73019807, 923.99114103, 2.18614984])),
            (np.array([647.8991296, 223.85365454, 954.78426745]), np.array([-547.48178332, 93.92166408, -809.79295556]))
        ]
        accuracyResults=[
            ([1,2,3]),
            ([0,0,0]),
            ([3,3,3]),
            ([5,7,9]),
            ([5.5, 7.7, 9.3]),
            ([5,7,9]),
            ([-720.52896907,  359.48641801, -828.00005824]),
            ([-81.0625216 , -611.06671061,  400.97541256]),
            ([200.78577676, -305.6984266 ,  -74.47572774]),
            ([1706.1440287 , 1056.50665892, 29.57768859]),
            ([400.14835119,  1088.35572781, -1049.8932387 ]),
            ([138.79473638, -906.11450546, -345.68961797]),
            ([-166.67979571, 1557.95621468, 35.70890591]),
            ([-1195.38091292, -129.93199046, -1764.57722301])
        ]
        for i in range(len(accuracyTests)):
            # Call vector(b, e) with the variables given from each accuracyTests index.
            result = pycgmKinetics.vector(accuracyTests[i][0],accuracyTests[i][1])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)
        
        # vector([1,2,3],[4,5,6]) should result in (3,3,3), test to make sure it does not result as anything else.
        self.assertFalse(pycgmKinetics.vector([1,2,3],[4,5,6]) != (3,3,3))

        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for vector.
        exceptionTests=[([]), ([],[]), ([1,2,3],[4,5]), ([1,2],[4,5,6]), (["a",2,3],[4,5,6])]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                pycgmKinetics.vector(e[0],e[1])

    def test_unit(self):
        """
        This test provides coverage of the unit function in pycgmKinetics.py,
        defined as unit(v), where v is a 3-element list.

        Each index in accuracyTests is used as parameters for the function unit 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, floats, and negatives 
        accuracyTests=[
            ([1,1,1]),
            ([1,2,3]),
            ([1.1,2.2,3.3]),
            (np.array([1.1,2.2,3.3])),
            (np.array([-1.1,-2.2,-3.3])),
            (np.array([4.1,-5.2,6.3])),
            (np.array([20.1,-0.2,0])),
            (np.array([477.96370143, -997.67255536, 400.99490597])),
            (np.array([330.80492334, 608.46071522, 451.3237226])),
            (np.array([-256.41091237, 391.85451166, 679.8028365])),
            (np.array([197.08510663, 319.00331132, -195.89839035])),
            (np.array([910.42721331, 184.76837848, -67.24503815])),
            (np.array([313.91884245, -703.86347965, -831.19994848])),
            (np.array([710.57698646, 991.83524562, 781.3712082]))
        ]
        accuracyResults=[
            ([0.57735027, 0.57735027, 0.57735027]),
            ([0.26726124, 0.53452248, 0.80178373]),
            ([0.26726124, 0.53452248, 0.80178373]),
            ([0.26726124, 0.53452248, 0.80178373]),
            ([-0.26726124, -0.53452248, -0.80178373]),
            ([ 0.44857661, -0.56892643,  0.68927625]),
            ([ 0.9999505 , -0.00994976,  0.00000001]),
            ([ 0.40619377, -0.84786435,  0.34078244]),
            ([0.40017554, 0.73605645, 0.54596744]),
            ([-0.31061783,  0.47469508,  0.82351754]),
            ([ 0.46585347,  0.75403363, -0.46304841]),
            ([ 0.97746392,  0.19837327, -0.07219643]),
            ([ 0.27694218, -0.62095504, -0.73329248]),
            ([0.49043839, 0.68456211, 0.53930038])
        ]
        for i in range(len(accuracyTests)):
            # Call unit(v) with the v given from each accuracyTests index.
            result = pycgmKinetics.unit(accuracyTests[i])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)

        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for unit.
        exceptionTests=[([]), ([1]), ([1,2]), ([1,2,"c"]), (["a","b",3])]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                pycgmKinetics.unit(e[0])
        
    def test_distance(self):
        """
        This test provides coverage of the distance function in pycgmKinetics.py,
        defined as distance(p0, p1), where p0 and p1 are 3-element lists describing x, y, z points.

        Each index in accuracyTests is used as parameters for the function distance 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, negatives, and floats.
        accuracyTests=[
            ([0,0,0],[1,2,3]),
            ([1,2,3],[1,2,3]),
            ([1,2,3],[4,5,6]),
            ([-1,-2,-3],[4,5,6]),
            ([-1.1,-2.2,-3.3],[4.4,5.5,6]),
            (np.array([-1,-2,-3]),np.array([4,5,6])),
            (np.array([871.13796878, 80.07048505, 81.7226316]), np.array([150.60899971, 439.55690306, -746.27742664])),
            (np.array([109.96296398, 278.68529143, 224.18342906]), np.array([28.90044238, -332.38141918, 625.15884162])),
            (np.array([261.89862662, 635.64883561, 335.23199233]), np.array([462.68440338, 329.95040901, 260.75626459])),
            (np.array([-822.76892296, -457.04755227, 64.67044766]), np.array([883.37510574, 599.45910665, 94.24813625])),
            (np.array([-723.03974742, -913.26790889, 95.50575378]), np.array([-322.89139623, 175.08781892, -954.38748492])),
            (np.array([602.28250216, 868.53946449, 666.82151334]), np.array([741.07723854, -37.57504097, 321.13189537])),
            (np.array([646.40999378, -633.96507365, -33.52275607]), np.array([479.73019807, 923.99114103, 2.18614984])),
            (np.array([647.8991296, 223.85365454, 954.78426745]), np.array([-547.48178332, 93.92166408, -809.79295556]))
        ]
        accuracyResults=[
            (3.74165739),
            (0.00000000),
            (5.19615242),
            (12.4498996),
            (13.26762978),
            (12.4498996),
            (1154.97903723),
            (735.36041415),
            (373.24668813),
            (2006.98993686),
            (1564.26107344),
            (979.6983147),
            (1567.25391916),
            (2135.31042827)
        ]
        for i in range(len(accuracyTests)):
            # Call distance(p0, p1) with the variables given from each accuracyTests index.
            result = pycgmKinetics.distance(accuracyTests[i][0],accuracyTests[i][1])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)
        
        # distance([1,2,3],[1,2,3]) should result in (0), test to make sure it does not result as anything else.
        self.assertFalse(pycgmKinetics.distance([1,2,3],[1,2,3]) != (0))

        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for distance.
        exceptionTests=[([]), ([],[]), ([1,2,3],[4,5]), ([1,2],[4,5,6]), (["a",2,3],[4,5,6])]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                pycgmKinetics.vector(e[0],e[1])
        
    def test_scale(self):
        """
        This test provides coverage of the scale function in pycgmKinetics.py,
        defined as scale(v, sc), where v is a 3-element list and sc is a int or float.

        Each index in accuracyTests is used as parameters for the function scale 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, negatives, and floats 
        accuracyTests=[
            ([1,2,3],0),
            ([1,2,3],2),
            ([1.1,2.2,3.3],2),
            ([6,4,24],5.0),
            ([22,5,-7],5.0),
            ([-7,-2,1.0],-5.4),
            (np.array([1,2,3]),5.329),
            (np.array([-2.0,24,34]),3.2502014),
            (np.array([101.53593091, 201.1530486, 56.44356634]), 5.47749966),
            (np.array([0.55224332, 6.41308177, 41.99237585]), 18.35221769),
            (np.array([80.99568691, 61.05185784, -55.67558577]), -26.29967607),
            (np.array([0.011070408, -0.023198581, 0.040790087]), 109.68173477),
        ]
        accuracyResults=[
            ([0, 0, 0]),
            ([2, 4, 6]),
            ([2.2, 4.4, 6.6]),
            ([ 30.0,  20.0, 120.0]),
            ([110.0,  25.0, -35.0]),
            ([37.8, 10.8, -5.4]),
            ([ 5.329, 10.658, 15.987]),
            ([ -6.5004028,  78.0048336, 110.5068476]),
            ([ 556.16302704, 1101.81575531,  309.16961544]),
            ([ 10.13488963, 117.69427271, 770.65322292]),
            ([-2130.1603288 , -1605.64408466,  1464.24987076]),
            ([ 1.21422155, -2.54446061,  4.4739275 ])
        ]
        for i in range(len(accuracyTests)):
            # Call scale(v, sc) with the variables given from each accuracyTests index.
            result = pycgmKinetics.scale(accuracyTests[i][0],accuracyTests[i][1])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)
        
        # scale([1,2,3],0) should result in (0, 0, 0), test to make sure it does not result as anything else.
        self.assertFalse(pycgmKinetics.scale([1,2,3],0) != (0, 0, 0))

        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for scale.
        exceptionTests=[([],4), ([1,2,3]), (4), ([1,2],4)]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                pycgmKinetics.scale(e[0],e[1])
    
    def test_add(self):
        """
        This test provides coverage of the add function in pycgmKinetics.py,
        defined as add(v, w), where v and w are 3-element lists.

        Each index in accuracyTests is used as parameters for the function scale 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, negatives, and floats 
        accuracyTests=[
            ([0,0,0],[1,2,3]),
            ([1,2,3],[1,2,3]),
            ([1,2,3],[4,5,6]),
            ([-1,-2,-3],[4,5,6]),
            ([-1.1,-2.2,-3.3],[4.4,5.5,6]),
            (np.array([-1,-2,-3]),np.array([4,5,6])),
            (np.array([871.13796878, 80.07048505, 81.7226316]), np.array([150.60899971, 439.55690306, -746.27742664])),
            (np.array([109.96296398, 278.68529143, 224.18342906]), np.array([28.90044238, -332.38141918, 625.15884162])),
            (np.array([261.89862662, 635.64883561, 335.23199233]), np.array([462.68440338, 329.95040901, 260.75626459])),
            (np.array([-822.76892296, -457.04755227, 64.67044766]), np.array([883.37510574, 599.45910665, 94.24813625])),
            (np.array([-723.03974742, -913.26790889, 95.50575378]), np.array([-322.89139623, 175.08781892, -954.38748492])),
            (np.array([602.28250216, 868.53946449, 666.82151334]), np.array([741.07723854, -37.57504097, 321.13189537])),
            (np.array([646.40999378, -633.96507365, -33.52275607]), np.array([479.73019807, 923.99114103, 2.18614984])),
            (np.array([647.8991296, 223.85365454, 954.78426745]), np.array([-547.48178332, 93.92166408, -809.79295556]))
        ]
        accuracyResults=[
            ([1, 2, 3]),
            ([2, 4, 6]),
            ([5, 7, 9]),
            ([3, 3, 3]),
            ([3.3, 3.3, 2.7]),
            ([3, 3, 3]),
            ([1021.74696849,  519.62738811, -664.55479504]),
            ([138.86340636, -53.69612775, 849.34227068]),
            ([724.58303000, 965.59924462, 595.98825692]),
            ([ 60.60618278, 142.41155438, 158.91858391]),
            ([-1045.93114365,  -738.18008997,  -858.88173114]),
            ([1343.3597407 ,  830.96442352,  987.95340871]),
            ([1126.14019185,  290.02606738,  -31.33660623]),
            ([100.41734628, 317.77531862, 144.99131189])
        ]
        for i in range(len(accuracyTests)):
            # Call add(v, w) with the variables given from each accuracyTests index.
            result = pycgmKinetics.add(accuracyTests[i][0],accuracyTests[i][1])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)
        
        # add([1,2,3],[0,0,0]) should result in (1, 2, 3), test to make sure it does not result as anything else.
        self.assertFalse(pycgmKinetics.add([1,2,3],[0,0,0]) != (1,2,3))

        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for add.
        exceptionTests=[([]), ([],[]), ([1,2,3],[4,5]), ([1,2],[4,5,6]), (["a",2,3],[4,5,6])]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                pycgmKinetics.add(e[0],e[1])
    
    def test_pnt2line(self):
        """
        This test provides coverage of the pnt2line function in pycgmKinetics.py,
        defined as pnt2line(pnt, start, end), where pnt, start, and end are 3-element lists.

        Each index in accuracyTests is used as parameters for the function pnt2line 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, list and numpy array, negatives, and floats 
        accuracyTests=[
            ([1,2,3],[4,5,6],[0,0,0]),
            (np.array([1.1,-2.24,31.32]), np.array([4,5.3,-6]), np.array([2.14,12.52,13.2])),
            (np.array([35.83977741, 61.57074759, 68.44530267]), np.array([74.67790922, 14.29054848, -26.04736139]), np.array([0.56489944, -16.12960177, 63.33083103])),
            (np.array([23.90166027, 88.64089564, 49.65111862]), np.array([48.50606388, -75.41062664, 75.31899688]), np.array([-34.87278229, 25.60601135, 78.81218762])),
            (np.array([687.84935299, -545.36574903, 668.52916292]), np.array([-39.73733639, 854.80603373, 19.05056745]), np.array([84.09259043, 617.95544147, 501.49109559])),
            (np.array([660.95556608, 329.67656854, -142.68363472]), np.array([773.43109446, 253.42967266, 455.42278696]), np.array([233.66307152, 432.69607959, 590.12473739]))
        ]
        accuracyResults=[
            ([0.83743579, ([1.66233766, 2.07792208, 2.49350649]), ([1, 2, 3])]),
            ([23.393879541452716, (2.14, 12.52, 13.2), ([ 1.1 , -2.24, 31.32])]),
            ([76.7407926, ([23.8219481, -6.5836001, 35.2834886]), ([35.8397774, 61.5707476, 68.4453027])]),
            ([90.98461233, ([-34.8727823,  25.6060113,  78.8121876]), ([23.9016603, 88.6408956, 49.6511186])]),
            ([1321.26459747, ([ 84.0925904, 617.9554415, 501.4910956]), ([ 687.849353 , -545.365749 ,  668.5291629])]),
            ([613.34788275, ([773.4310945, 253.4296727, 455.422787 ]), ([ 660.9555661,  329.6765685, -142.6836347])])
        ]
        for i in range(len(accuracyTests)):
            # Call pnt2line(pnt, start, end) with variables given from each index inaccuracyTests and round 
            # each variable in the 3-element returned list with a rounding precision of 8.
            pnt, start, end = accuracyTests[i]
            result = [np.around(arr,rounding_precision) for arr in pycgmKinetics.pnt2line(pnt, start, end)]
            expected = list(accuracyResults[i])
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expected[j])
        
        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for pnt2line.
        exceptionTests=[([]), ([],[]), ([],[],[]), ([1,2],[1,2],[1,2]), (["a",2,3],[4,5,6],[7,8,9])]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                pycgmKinetics.pnt2line(e[0],e[1],e[2])
    
    def test_calc_l5_pelvis(self):
        """
        This test provides coverage of the calc_l5_pelvis function in pycgmKinetics.py,
        defined as calc_l5_pelvis(rhip, lhip, pelvis_axis):

        Each index in test_parameters is used as parameters for the function calc_l5_pelvis 
        and the result is then checked to be equal with the same index in 
        test_results using 8 decimal point precision comparison.
        """
        # Test 3 different frames that contain different markers for LHip, RHip, and Pelvis_axis.

        test_parameters=[]

        rhip = np.array([208.38050472, 122.80342417, 437.98979061])
        lhip = np.array([282.57097863, 139.43231855, 435.52900012])
        pelvis_axis = np.array([[251.74063624, 392.72694721, 1032.78850073, 151.60830688],
                                [250.61711554, 391.87232862, 1032.8741063,  291.74131775],
                                [251.60295336, 391.84795134, 1033.88777762, 832.89349365],
                                [  0,            0,             0,            0         ]])
        test_parameters.append([rhip, lhip, pelvis_axis])

        rhip = np.array([-570.727107, 409.48579719, 387.17336605])
        lhip = np.array([984.96369008, 161.72241084, 714.78280362])
        pelvis_axis = np.array([[586.81782059,  994.852335,  -164.15032491,  553.90052549],
                                [367.53692416, -193.11814502, 141.95648112, -327.14438741],
                                [814.64795266,  681.51439276,  87.63894117,   -4.58459872],
                                [  0,             0,            0,             0         ]])
        test_parameters.append([rhip, lhip, pelvis_axis])

        rhip = np.array([-651.87182756, -493.94862894, 640.38910712])
        lhip = np.array([624.42435686, 746.90148656, -603.71552902])
        pelvis_axis = np.array([[ 711.02920886, -701.16459687, 532.55441473, 691.47208667],
                                [-229.76970935, -650.15236712, 359.70108366, 395.90359428],
                                [ 222.81186893,  536.56366268, 386.21334066, 273.23978111],
                                [   0,             0,            0,            0         ]])
        test_parameters.append([rhip, lhip, pelvis_axis])

        test_results=[
            ([[245.4757417, 131.1178714, 436.7593954],[261.0890402, 155.4341163, 500.9176188]]),
            ([[207.1182915, 285.604104 , 550.9780848],[1344.7944079, 1237.3558945,  673.3680447]]),
            ([[-13.7237354, 126.4764288,  18.3367891],[ 627.8602897, 1671.5048695, 1130.4333341]])
        ]
        for i in range(len(test_parameters)):
            # Call calc_l5_pelvis(frame) with each frame in test_parameters and round each variable in the 3-element returned list.
            result = [np.around(arr,rounding_precision) for arr in pycgmKinetics.calc_l5_pelvis(*test_parameters[i])]
            expected = list(test_results[i])
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expected[j])
    
    def test_findL5_Thorax(self):
        """
        This test provides coverage of the findL5_Thorax function in pycgmKinetics.py,
        defined as findL5_Thorax(frame), frame contains the markers: C7, RHip, LHip, Thorax_axis

        Each index in accuracyTests is used as parameters for the function findL5_Thorax 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test 3 different frames that contain different markers for C7, RHip, LHip, Thorax_axis.
        """
        This function tests 3 different frames.
        """
        accuracyTests=[]
        frame=dict()
        frame['Thorax_axis'] = [[[256.3454633226447, 365.7223958512035, 1461.920891187948], [257.26637166499415, 364.69602499862503, 1462.2347234647593], [256.1842731803127, 364.4328898435265, 1461.363045336319]], [256.2729542797522, 364.79605748807074, 1462.2905392309394]]
        frame['C7'] = np.array([226.78051758, 311.28042603, 1259.70300293])
        frame['LHip'] = np.array([262.38020472, 242.80342417, 521.98979061])
        frame['RHip'] = np.array([82.53097863, 239.43231855, 835.529000126])
        accuracyTests.append(frame)

        frame=dict()
        frame['Thorax_axis'] = [[[309.69280961, 700.32003143, 203.66124527], [1111.49874303, 377.00086678, -140.88485905], [917.9480966, 60.89883132, -342.22796426]], [-857.91982333, -869.67870489, 438.51780456]]
        frame['C7'] = np.array([921.981682, 643.5500819, 439.96382993])
        frame['LHip'] = np.array([179.35982654, 815.09778236, 737.19459299])
        frame['RHip'] = np.array([103.01680043, 333.88103831, 823.33260927])
        accuracyTests.append(frame)

        frame=dict()
        frame['Thorax_axis'] = [[[345.07821036, -746.40495016, -251.18652575], [499.41682335, 40.88439602, 507.51025588], [668.3596798, 1476.88140274, 783.47804105]], [1124.81785806, -776.6778811, 999.39015919]]
        frame['C7'] = np.array([537.68019187, 691.49433996, 246.01153709])
        frame['LHip'] = np.array([47.94211912, 338.95742186, 612.52743329])
        frame['RHip'] = np.array([402.57410142, -967.96374463, 575.63618514])
        accuracyTests.append(frame)

        accuracyResults=[
            ([228.5241582, 320.87776246, 998.59374786]),
            ([569.20914046, 602.88531664, 620.68955025]),
            ([690.41775396, 713.36498782, 1139.36061258])
        ]
        for i in range(len(accuracyTests)):
            # Call findL5_Thorax(frame) with each frame in accuracyTests and round each variable in the 3-element returned list.
            result = [np.around(arr,rounding_precision) for arr in pycgmKinetics.findL5_Thorax(accuracyTests[i])]
            expected = list(accuracyResults[i])
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expected[j])
    
    def test_getKinetics(self):
        """
        This test provides coverage of the getKinetics function in pycgmKinetics.py,
        defined as getKinetics(data, Bodymass), where data is an array of joint centers 
        and Bodymass is a float or int.

        This test uses helper functions to obtain the data variable (aka joint_centers).

        Each index in accuracyTests is used as parameters for the function getKinetics 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Testing is done by using 5 different bodymasses and the same joint_center obtained from the helper functions.
        from pyCGM_Single.pyCGM_Helpers import getfilenames
        from pyCGM_Single.pycgmIO import loadData, loadVSK
        from pyCGM_Single.pycgmStatic import getStatic
        from pyCGM_Single.pycgmCalc import calcAngles

        cwd = os.getcwd() + os.sep
        # Data is obtained from the sample files.
        dynamic_trial,static_trial,vsk_file,_,_ = getfilenames(2)
        motionData  = loadData(cwd+dynamic_trial)
        staticData = loadData(cwd+static_trial)
        vsk = loadVSK(cwd+vsk_file,dict=False)

        calSM = getStatic(staticData,vsk,flat_foot=False)
        _,joint_centers=calcAngles(motionData,start=None,end=None,vsk=calSM, splitAnglesAxis=False,formatData=False,returnjoints=True)

        accuracyTests=[]
        calSM['Bodymass']=5.0
        # This creates five individual assertions to check, all with the same joint_centers but different bodymasses.
        for i in range(5):
            accuracyTests.append((joint_centers,calSM['Bodymass']))
            calSM['Bodymass']+=35.75 #Increment the bodymass by a substantial amount each time.
        
        accuracyResults=[
            ([ 246.57466721,  313.55662383, 1026.56323492]),
            ([ 246.59137623,  313.6216639 , 1026.56440096]),
            ([ 246.60850798,  313.6856272 , 1026.56531282]),
            ([ 246.6260863 ,  313.74845693, 1026.56594554]),
            ([ 246.64410308,  313.81017167, 1026.5663452 ]),
        ]
        for i in range(len(accuracyResults)):
            # Call getKinetics(joint_centers,bodymass) and round each variable in the 3-element returned list to the 8th decimal precision.
            result = [np.around(arr,rounding_precision) for arr in pycgmKinetics.getKinetics(accuracyTests[i][0],accuracyTests[i][1])]

            # Compare the result with the values in the expected results, within a rounding precision of 8.
            np.testing.assert_almost_equal(result[i],accuracyResults[i], rounding_precision)

