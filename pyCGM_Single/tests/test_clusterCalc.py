from unittest import TestCase
import pytest
import pyCGM_Single.clusterCalc as clusterCalc
import numpy as np
import os

rounding_precision = 8

class Test_clusterCalc(TestCase):
    """
    This class tests the coverage of all functions in clusterCalc.py, 
    except for printMat.
    """
    def test_normalize(self):
        """
        This test provides coverage of the normalize function in clusterCalc.py,
        defined as normalize(v) where v is a 1-d list.

        Each index in accuracyTests is used as parameters for the function normalize 
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
        This test provides coverage of the getMarkerLocation function in clusterCalc.py,
        defined as getMarkerLocation(Pm,C) where Pm is a 3-element list/array,
        and C is an numpy array in the form of [origin, x_dir, y_dir].

        Each index in accuracyTests is used as parameters for the function getMarkerLocation 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, negative numbers, floats
        # The function is unittested 5 times, each time uses different Pm and C variables.
        accuracyTests=[]
        
        Pm = [420.53975659, 30.76040902, 555.49711768]
        C = [np.array([-343.59864907, 238.25329134, -755.16883877]), np.array([8.1286508, 495.13257337, 885.7371809]), np.array([384.38897987, 741.88310889, 289.56653492])]
        accuracyTests.append((Pm,C))
        
        Pm = [290.67647141, -887.27170397, -928.18965884]
        C = [np.array([975.77145761, 169.07228161, 714.73898307]), np.array([34.79840373, 437.98858319, 342.44994367]), np.array([386.38290967, 714.0373601, -254.71890944])]
        accuracyTests.append((Pm,C))

        Pm = [451.73055418, 25.29186874, 212.82059603]
        C = [np.array([750.37513208, 777.29644972, 814.25477338]), np.array([785.58183092, 45.27606372, 228.32835519]), np.array([-251.24340957, 71.99704479, -70.78517678]) ]
        accuracyTests.append((Pm,C))

        Pm = (np.array([198.67934839, 617.12145922, -942.60245177]))
        C = [np.array([-888.79518579, 677.00555294, 580.34056878]), np.array([-746.7046053, 365.85692077, 964.74398363]), np.array([488.51200254, 242.19485233, -875.4405979])]
        accuracyTests.append((Pm,C))

        Pm = (np.array([151.45958988, 228.60024976, 571.69254842]))
        C = [np.array([329.65958402, 338.27760766, 893.98328401]), np.array([185.96009811, 933.21745694, 203.23381269]), np.array([370.92610191, 763.93031647, -624.83623717])]
        accuracyTests.append((Pm,C))

        accuracyResults=[
            ([-550.7935827 ,  773.66338285, -330.72688418]),
            ([1198.80575306,  683.68559959, 1716.24804919]),
            ([800.20530287, 543.8441708 , 356.96134002]),
            ([-986.86909809, -247.71320943,  -44.9278578 ]),
            ([-213.42223861,  284.87178829,  486.68789429])
        ]
        for i in range(len(accuracyTests)):
            # Call getMarkerLocation(Pm,C) with the variables given from each accuracyTests index.
            result = clusterCalc.getMarkerLocation(accuracyTests[i][0],accuracyTests[i][1])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)
    
    def test_getStaticTransform(self):
        """
        This test provides coverage of the getStaticTransform function in clusterCalc.py,
        defined as getStaticTransform(p,C) where Pm is a 3-element list/array
        representing a marker and C is an numpy array in the form of [origin, x_dir, y_dir].

        Each index in accuracyTests is used as parameters for the function getStaticTransform 
        and the result is then checked to be equal with the same index in 
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test the following cases: lists, numpy arrays, negative numbers, floats
        # The function is unittested 5 times, each time uses different p and C variables.
        accuracyTests=[]
        
        p = [61.25716038, 819.60483461, 871.28563964]
        C = [np.array([-109.75683574, -703.39208609, 23.40503888]), np.array([8.1286508, 495.13257337, 885.7371809]), np.array([384.38897987, 741.88310889, 289.56653492])]
        accuracyTests.append((p,C))
        
        p = [-108.77877024, 164.72037283, -487.34574257]
        C = [np.array([-840.15274045, -477.4003232, 989.63441835]), np.array([34.79840373, 437.98858319, 342.44994367]), np.array([386.38290967, 714.0373601, -254.71890944])]
        accuracyTests.append((p,C))

        p = [172.60504672, 189.51963254, 714.76733718]
        C = [np.array([100.28243274, -308.77342489, 823.62871217]), np.array([785.58183092, 45.27606372, 228.32835519]), np.array([-251.24340957, 71.99704479, -70.78517678]) ]
        accuracyTests.append((p,C))

        p = (np.array([606.02393735, 905.3131133, -759.04662559]))
        C = [np.array([144.30930144, -843.7618657, 105.12793356]), np.array([-746.7046053, 365.85692077, 964.74398363]), np.array([488.51200254, 242.19485233, -875.4405979])]
        accuracyTests.append((p,C))

        p = (np.array([-973.61392617, 246.51405629, 558.66195333]))
        C = [np.array([763.69368715, 709.90434444, 650.91067694]), np.array([185.96009811, 933.21745694, 203.23381269]), np.array([370.92610191, 763.93031647, -624.83623717])]
        accuracyTests.append((p,C))

        accuracyResults=[
            ([1391.11399871,  396.16136313,   77.71905097]),
            ([-3890.74300865,  5496.20861951,    81.25476604]),
            ([233.03998819, 154.49967147, 395.87517602]),
            ([ 435.34624558, 1905.72175012, -305.53267563]),
            ([ 2083.22592722, -1072.83708766,  1139.02963846])
        ]
        for i in range(len(accuracyTests)):
            # Call getStaticTransform(p,C) with the variables given from each accuracyTests index.
            result = clusterCalc.getStaticTransform(accuracyTests[i][0],accuracyTests[i][1])
            expected = accuracyResults[i]
            np.testing.assert_almost_equal(result, expected, rounding_precision)
    
    def test_targetName(self):
        """
        This test provides coverage of the targetName function in clusterCalc.py,
        defined as targetName().

        The function provides a fixed arrangement of marker names.  
        """
        
        expected = ['C7', 'T10', 'CLAV', 'STRN', 'RBAK', 'LPSI', 'RPSI', 'RASI', 'LASI', 'SACR', 
                  'LKNE', 'LKNE', 'RKNE', 'RKNE', 'LANK', 'RANK', 'LHEE', 'RHEE', 'LTOE', 
                  'RTOE', 'LTHI', 'RTHI', 'LTIB', 'RTIB', 'RBHD', 'RFHD', 'LBHD', 'LFHD', 
                  'RELB', 'LELB']

        # Test to make sure the targetName function returns the correct arrangement of marker names.
        np.testing.assert_array_equal(expected, clusterCalc.targetName())
    
    def test_target_dict(self):
        """
        This test provides coverage of the target_dict function in clusterCalc.py,
        defined as target_dict().

        The function provides a dictionary mapping markers to segments.  
        """
        expected = {'LFHD': 'Head', 'LBHD': 'Head', 'RFHD': 'Head', 'RBHD': 'Head', 
                    'C7': 'Trunk', 'T10': 'Trunk', 'CLAV': 'Trunk', 'STRN': 'Trunk', 
                    'RBAK': 'Trunk', 'LPSI': 'Pelvis', 'RPSI': 'Pelvis', 'RASI': 'Pelvis', 
                    'LASI': 'Pelvis', 'SACR': 'Pelvis', 'LKNE': 'LThigh', 'RKNE': 'RThigh', 
                    'LANK': 'LShin', 'RANK': 'RShin', 'LHEE': 'LFoot', 'LTOE': 'LFoot', 
                    'RHEE': 'RFoot', 'RTOE': 'RFoot', 'LTHI': 'LThigh', 'RTHI': 'RThigh', 
                    'LTIB': 'LShin', 'RTIB': 'RShin', 'RELB': 'RHum', 'LELB': 'LHum'}

        # Test to make sure the target_dict function returns 
        # all the markers appropriately mapped to segments
        np.testing.assert_equal(expected, clusterCalc.target_dict())

    def test_segment_dict(self):
        """
        This test provides coverage of the segment_dict function in clusterCalc.py,
        defined as segment_dict().

        The function provides a dictionary mapping segments to marker names
        """
        expected = {'Head': ['RFHD', 'RBHD', 'LFHD', 'LBHD', 'REAR', 'LEAR'],  
                    'Trunk': ['C7', 'STRN', 'CLAV', 'T10', 'RBAK', 'RSHO', 'LSHO'], 
                    'Pelvis': ['SACR', 'RPSI', 'LPSI', 'LASI', 'RASI'], 
                    'RThigh': ['RTHI', 'RTH2', 'RTH3', 'RTH4'], 
                    'LThigh': ['LTHI', 'LTH2', 'LTH3', 'LTH4'], 
                    'RShin': ['RTIB', 'RSH2', 'RSH3', 'RSH4'], 
                    'LShin': ['LTIB', 'LSH2', 'LSH3', 'LSH4'], 
                    'RFoot': ['RLFT1', 'RFT2', 'RMFT3', 'RLUP'], 
                    'LFoot': ['LLFT1', 'LFT2', 'LMFT3', 'LLUP'], 
                    'RHum': ['RMELB', 'RSHO', 'RUPA'], 
                    'LHum': ['LMELB', 'LSHO', 'LUPA']}

        # Test to make sure the segment_dict function returns 
        # all the markers appropriately segments to marker names.
        np.testing.assert_equal(expected, clusterCalc.segment_dict())
