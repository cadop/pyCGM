from unittest import TestCase
import pytest
import pycgm.kinetics as kinetics
import numpy as np
import os

rounding_precision = 8


class Test_kinetics(TestCase):
    def test_length(self):
        """
        This test provides coverage of the length function in kinetics.py,
        defined as length(v), where v is a 3-element list.

        Each index in accuracyTests is used as parameters for the function length
        and the result is then checked to be equal with the same index in
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, floats, numpy arrays, and negatives
        accuracyTests = [
            ([0, 0, 0]),
            ([1, 2, 3]),
            ([1.1, 2.2, 3.3]),
            (np.array([1.1, 2.2, 3.3])),
            (np.array([-1.1, -2.2, -3.3])),
            (np.array([4.1, -5.2, 6.3])),
            (np.array([20.1, -0.2, 0])),
            (np.array([477.96370143, -997.67255536, 400.99490597])),
            (np.array([330.80492334, 608.46071522, 451.3237226])),
            (np.array([-256.41091237, 391.85451166, 679.8028365])),
            (np.array([197.08510663, 319.00331132, -195.89839035])),
            (np.array([910.42721331, 184.76837848, -67.24503815])),
            (np.array([313.91884245, -703.86347965, -831.19994848])),
            (np.array([710.57698646, 991.83524562, 781.3712082]))
        ]
        accuracyResults = [
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
            result = kinetics.length(accuracyTests[i])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(
                result, expected, rounding_precision)

        # length([0,0,0]) should result in 0.0, test to make sure it does not result as anything else.
        self.assertFalse(kinetics.length([0, 0, 0]) != 0.0)

        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for length.
        exceptionTests = [([]), ([1]), ([1, 2]),
                          ([1, 2, "c"]), (["a", "b", 3])]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                kinetics.length(e[0])

    def test_unit(self):
        """
        This test provides coverage of the unit function in kinetics.py,
        defined as unit(v), where v is a 3-element list.

        Each index in accuracyTests is used as parameters for the function unit
        and the result is then checked to be equal with the same index in
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, floats, and negatives
        accuracyTests = [
            ([1, 1, 1]),
            ([1, 2, 3]),
            ([1.1, 2.2, 3.3]),
            (np.array([1.1, 2.2, 3.3])),
            (np.array([-1.1, -2.2, -3.3])),
            (np.array([4.1, -5.2, 6.3])),
            (np.array([20.1, -0.2, 0])),
            (np.array([477.96370143, -997.67255536, 400.99490597])),
            (np.array([330.80492334, 608.46071522, 451.3237226])),
            (np.array([-256.41091237, 391.85451166, 679.8028365])),
            (np.array([197.08510663, 319.00331132, -195.89839035])),
            (np.array([910.42721331, 184.76837848, -67.24503815])),
            (np.array([313.91884245, -703.86347965, -831.19994848])),
            (np.array([710.57698646, 991.83524562, 781.3712082]))
        ]
        accuracyResults = [
            ([0.57735027, 0.57735027, 0.57735027]),
            ([0.26726124, 0.53452248, 0.80178373]),
            ([0.26726124, 0.53452248, 0.80178373]),
            ([0.26726124, 0.53452248, 0.80178373]),
            ([-0.26726124, -0.53452248, -0.80178373]),
            ([0.44857661, -0.56892643,  0.68927625]),
            ([0.9999505, -0.00994976,  0.00000001]),
            ([0.40619377, -0.84786435,  0.34078244]),
            ([0.40017554, 0.73605645, 0.54596744]),
            ([-0.31061783,  0.47469508,  0.82351754]),
            ([0.46585347,  0.75403363, -0.46304841]),
            ([0.97746392,  0.19837327, -0.07219643]),
            ([0.27694218, -0.62095504, -0.73329248]),
            ([0.49043839, 0.68456211, 0.53930038])
        ]
        for i in range(len(accuracyTests)):
            # Call unit(v) with the v given from each accuracyTests index.
            result = kinetics.unit(accuracyTests[i])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(
                result, expected, rounding_precision)

        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for unit.
        exceptionTests = [([]), ([1]), ([1, 2]),
                          ([1, 2, "c"]), (["a", "b", 3])]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                kinetics.unit(e[0])

    def test_distance(self):
        """
        This test provides coverage of the distance function in kinetics.py,
        defined as distance(p0, p1), where p0 and p1 are 3-element lists describing x, y, z points.

        Each index in accuracyTests is used as parameters for the function distance
        and the result is then checked to be equal with the same index in
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, negatives, and floats.
        accuracyTests = [
            ([0, 0, 0], [1, 2, 3]),
            ([1, 2, 3], [1, 2, 3]),
            ([1, 2, 3], [4, 5, 6]),
            ([-1, -2, -3], [4, 5, 6]),
            ([-1.1, -2.2, -3.3], [4.4, 5.5, 6]),
            (np.array([-1, -2, -3]), np.array([4, 5, 6])),
            (np.array([871.13796878, 80.07048505, 81.7226316]),
             np.array([150.60899971, 439.55690306, -746.27742664])),
            (np.array([109.96296398, 278.68529143, 224.18342906]),
             np.array([28.90044238, -332.38141918, 625.15884162])),
            (np.array([261.89862662, 635.64883561, 335.23199233]),
             np.array([462.68440338, 329.95040901, 260.75626459])),
            (np.array([-822.76892296, -457.04755227, 64.67044766]),
             np.array([883.37510574, 599.45910665, 94.24813625])),
            (np.array([-723.03974742, -913.26790889, 95.50575378]),
             np.array([-322.89139623, 175.08781892, -954.38748492])),
            (np.array([602.28250216, 868.53946449, 666.82151334]),
             np.array([741.07723854, -37.57504097, 321.13189537])),
            (np.array([646.40999378, -633.96507365, -33.52275607]),
             np.array([479.73019807, 923.99114103, 2.18614984])),
            (np.array([647.8991296, 223.85365454, 954.78426745]),
             np.array([-547.48178332, 93.92166408, -809.79295556]))
        ]
        accuracyResults = [
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
            result = kinetics.distance(
                accuracyTests[i][0], accuracyTests[i][1])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(
                result, expected, rounding_precision)

        # distance([1,2,3],[1,2,3]) should result in (0), test to make sure it does not result as anything else.
        self.assertFalse(kinetics.distance([1, 2, 3], [1, 2, 3]) != (0))

        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for distance.
        exceptionTests = [([]), ([], []), ([1, 2, 3], [4, 5]),
                          ([1, 2], [4, 5, 6]), (["a", 2, 3], [4, 5, 6])]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                kinetics.vector(e[0], e[1])

    def test_scale(self):
        """
        This test provides coverage of the scale function in kinetics.py,
        defined as scale(v, sc), where v is a 3-element list and sc is a int or float.

        Each index in accuracyTests is used as parameters for the function scale
        and the result is then checked to be equal with the same index in
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, negatives, and floats
        accuracyTests = [
            ([1, 2, 3], 0),
            ([1, 2, 3], 2),
            ([1.1, 2.2, 3.3], 2),
            ([6, 4, 24], 5.0),
            ([22, 5, -7], 5.0),
            ([-7, -2, 1.0], -5.4),
            (np.array([1, 2, 3]), 5.329),
            (np.array([-2.0, 24, 34]), 3.2502014),
            (np.array([101.53593091, 201.1530486, 56.44356634]), 5.47749966),
            (np.array([0.55224332, 6.41308177, 41.99237585]), 18.35221769),
            (np.array([80.99568691, 61.05185784, -55.67558577]), -26.29967607),
            (np.array([0.011070408, -0.023198581, 0.040790087]), 109.68173477),
        ]
        accuracyResults = [
            ([0, 0, 0]),
            ([2, 4, 6]),
            ([2.2, 4.4, 6.6]),
            ([30.0,  20.0, 120.0]),
            ([110.0,  25.0, -35.0]),
            ([37.8, 10.8, -5.4]),
            ([5.329, 10.658, 15.987]),
            ([-6.5004028,  78.0048336, 110.5068476]),
            ([556.16302704, 1101.81575531,  309.16961544]),
            ([10.13488963, 117.69427271, 770.65322292]),
            ([-2130.1603288, -1605.64408466,  1464.24987076]),
            ([1.21422155, -2.54446061,  4.4739275])
        ]
        for i in range(len(accuracyTests)):
            # Call scale(v, sc) with the variables given from each accuracyTests index.
            result = kinetics.scale(
                accuracyTests[i][0], accuracyTests[i][1])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(
                result, expected, rounding_precision)

        # scale([1,2,3],0) should result in (0, 0, 0), test to make sure it does not result as anything else.
        self.assertFalse(kinetics.scale([1, 2, 3], 0) != (0, 0, 0))

        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for scale.
        exceptionTests = [([], 4), ([1, 2, 3]), (4), ([1, 2], 4)]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                kinetics.scale(e[0], e[1])

    def test_pnt_line(self):
        """
        This test provides coverage of the pnt2line function in kinetics.py,
        defined as pnt2line(pnt, start, end), where pnt, start, and end are 3-element lists.

        Each index in accuracyTests is used as parameters for the function pnt2line
        and the result is then checked to be equal with the same index in
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, list and numpy array, negatives, and floats
        accuracyTests = [
            ([1, 2, 3], [4, 5, 6], [0, 0, 0]),
            (np.array([1.1, -2.24, 31.32]), np.array([4, 5.3, -6]),
             np.array([2.14, 12.52, 13.2])),
            (np.array([35.83977741, 61.57074759, 68.44530267]), np.array(
                [74.67790922, 14.29054848, -26.04736139]), np.array([0.56489944, -16.12960177, 63.33083103])),
            (np.array([23.90166027, 88.64089564, 49.65111862]), np.array(
                [48.50606388, -75.41062664, 75.31899688]), np.array([-34.87278229, 25.60601135, 78.81218762])),
            (np.array([687.84935299, -545.36574903, 668.52916292]), np.array([-39.73733639,
             854.80603373, 19.05056745]), np.array([84.09259043, 617.95544147, 501.49109559])),
            (np.array([660.95556608, 329.67656854, -142.68363472]), np.array([773.43109446,
             253.42967266, 455.42278696]), np.array([233.66307152, 432.69607959, 590.12473739]))
        ]
        accuracyResults = [
            ([0.83743579, ([1.66233766, 2.07792208, 2.49350649]), ([1, 2, 3])]),
            ([23.393879541452716, (2.14, 12.52, 13.2), ([1.1, -2.24, 31.32])]),
            ([76.7407926, ([23.8219481, -6.5836001, 35.2834886]),
             ([35.8397774, 61.5707476, 68.4453027])]),
            ([90.98461233, ([-34.8727823,  25.6060113,  78.8121876]),
             ([23.9016603, 88.6408956, 49.6511186])]),
            ([1321.26459747, ([84.0925904, 617.9554415, 501.4910956]),
             ([687.849353, -545.365749,  668.5291629])]),
            ([613.34788275, ([773.4310945, 253.4296727, 455.422787]),
             ([660.9555661,  329.6765685, -142.6836347])])
        ]
        for i in range(len(accuracyTests)):
            # Call pnt2line(pnt, start, end) with variables given from each index inaccuracyTests and round
            # each variable in the 3-element returned list with a rounding precision of 8.
            pnt, start, end = accuracyTests[i]
            result = [np.around(arr, rounding_precision)
                      for arr in kinetics.pnt_line(pnt, start, end)]
            expected = list(accuracyResults[i])
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expected[j])

        # Test the following exceptions to make sure that they do appropriately raise errors when used as parameters for pnt2line.
        exceptionTests = [([]), ([], []), ([], [], []), ([1, 2], [1, 2], [
            1, 2]), (["a", 2, 3], [4, 5, 6], [7, 8, 9])]
        for e in exceptionTests:
            with self.assertRaises(Exception):
                kinetics.pnt_line(e[0], e[1], e[2])
