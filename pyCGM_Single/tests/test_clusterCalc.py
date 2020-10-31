from unittest import TestCase
import pytest
import pyCGM_Single.clusterCalc as clusterCalc
import numpy as np
import os

rounding_precision = 8

class Test_clusterCalc(TestCase):
    def test_normalize(self):
        """
        This test provides coverage of the normalize function in clusterCalc.py,
        defined as normalize(v) where v is a 1-d list.

        Each index in accuracyTests is used as parameters for the function unit 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, negative numbers, floats
        accuracyTests=[
            ([0,0,0]),
            ([1,2,3]),
            ([1.1,2.2,3.3]),
            (np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.71])),
            (np.array([-1.1,-2.2,-3.3])),
            (np.array([4.1,-5.2,6.3])),
            (np.array([20.1,-0.2,0])),
            (np.array([477.96370143, -997.67255536, -831.19994848, 400.99490597])),
            (np.array([330.80492334, 608.46071522, 451.3237226])),
            (np.array([-256.41091237, 391.85451166, 679.8028365, 8.46071522])),
            (np.array([-67.24503815, 197.08510663, 710.57698646, 319.00331132, -195.89839035, 31.19994848, 13.91884245])),
            (np.array([910.42721331, 184.76837848, -67.24503815, 608.46071522])),
            (np.array([313.91884245, -703.86347965, -831.19994848])),
            (np.array([710.57698646, 991.83524562, 781.3712082, 84.76837848]))
        ]
        accuracyResults=[
            ([0, 0, 0]),
            ([0.26726124, 0.53452248, 0.80178373]),
            ([0.26726124, 0.53452248, 0.80178373]),
            ([0.08447701, 0.16895402, 0.25343103, 0.33790804, 0.42238505, 0.50686206, 0.59210705]),
            ([-0.26726124, -0.53452248, -0.80178373]),
            ([ 0.44857661, -0.56892643,  0.68927625]),
            ([ 0.9999505 , -0.00994976,  0.00000001]),
            ([ 0.33176807, -0.69251262, -0.5769593 ,  0.27834186]),
            ([0.40017554, 0.73605645, 0.54596744]),
            ([-0.31060152,  0.47467015,  0.82347429,  0.01024883]),
            ([-0.08097754,  0.23733301,  0.85568809,  0.38414885, -0.23590395, 0.03757147,  0.01676129]),
            ([ 0.81832612,  0.16607675, -0.06044236,  0.54690731]),
            ([ 0.27694218, -0.62095504, -0.73329248]),
            ([0.48960114, 0.68339346, 0.53837971, 0.05840703])
        ]
        for i in range(len(accuracyTests)):
            # Call normalize(v) with the variable given from each accuracyTests index.
            result = clusterCalc.normalize(accuracyTests[i])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_getMarkerLocation(self):
        """
        This test provides coverage of the normalize function in clusterCalc.py,
        defined as getMarkerLocation(Pm,C) where Pm is a 3-element list/array,
        and C is an numpy array in the form of [origin, x_dir, y_dir].

        Each index in accuracyTests is used as parameters for the function unit 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, negative numbers, floats
        # The function is unittested 5 times, each time uses different Pm and C variables.
        accuracyTests=[]
        for i in range 
        accuracyResults=[
            ([0, 0, 0]),
            ([0.26726124, 0.53452248, 0.80178373]),
            ([0.26726124, 0.53452248, 0.80178373]),
            ([0.08447701, 0.16895402, 0.25343103, 0.33790804, 0.42238505, 0.50686206, 0.59210705]),
            ([-0.26726124, -0.53452248, -0.80178373]),
            ([ 0.44857661, -0.56892643,  0.68927625]),
            ([ 0.9999505 , -0.00994976,  0.00000001]),
            ([ 0.33176807, -0.69251262, -0.5769593 ,  0.27834186]),
            ([0.40017554, 0.73605645, 0.54596744]),
            ([-0.31060152,  0.47467015,  0.82347429,  0.01024883]),
            ([-0.08097754,  0.23733301,  0.85568809,  0.38414885, -0.23590395, 0.03757147,  0.01676129]),
            ([ 0.81832612,  0.16607675, -0.06044236,  0.54690731]),
            ([ 0.27694218, -0.62095504, -0.73329248]),
            ([0.48960114, 0.68339346, 0.53837971, 0.05840703])
        ]
        for i in range(len(accuracyTests)):
            # Call normalize(v) with the variable given from each accuracyTests index.
            result = clusterCalc.getMarkerLocation(accuracyTests[i])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)