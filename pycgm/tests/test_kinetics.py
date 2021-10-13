from unittest import TestCase
import pytest
import pycgm.kinetics as kinetics
import numpy as np
import os

rounding_precision = 5


class Test_kinetics(TestCase):
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
            (np.array([35.83978, 61.57075, 68.44530]), np.array(
                [74.67791, 14.29055, -26.04736]), np.array([0.56490, -16.12960, 63.33083])),
            (np.array([23.90166, 88.64090, 49.65112]), np.array(
                [48.50606, -75.41062, 75.31900]), np.array([-34.87278, 25.60601, 78.81219])),
            (np.array([687.84935, -545.36575, 668.52916]), np.array([-39.73734,
             854.80603, 19.05057]), np.array([84.09259, 617.95544, 501.49110])),
            (np.array([660.95557, 329.67657, -142.68363]), np.array([773.43109,
             253.42967, 455.42279]), np.array([233.66307, 432.69608, 590.12474]))
        ]
        accuracyResults = [
            ([1.66234, 2.07792, 2.49351]),
            (2.14, 12.52, 13.2),
            ([23.82195, -6.58360, 35.28349]),
            ([-34.87278,  25.60601,  78.81219]),
            ([84.09259, 617.95544, 501.49110]),
            ([773.43109, 253.42967, 455.42279]),
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

    def test_find_L5(self):
        """
        This test provides coverage of the find_L5_pelvis function in kinetics.py,
        defined as find_L5_pelvis(frame), where frame contains the markers: LHip, RHip, and Pelvis_axis.

        Each index in accuracyTests is used as parameters for the function find_L5_pelvis
        and the result is then checked to be equal with the same index in
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test 3 different frames that contain different markers for LHip, RHip, and Pelvis_axis.
        accuracyTests = []
        frame = {}
        frame['axis'] = [[251.74064, 392.72695, 1032.78850, 0],
                         [250.61712, 391.87233, 1032.87411, 0],
                         [251.60295, 391.84795, 1033.88778, 0],
                         [0, 0, 0, 1]]
        frame['RHip'] = np.array([208.38050, 122.80342, 437.98979])
        frame['LHip'] = np.array([282.57098, 139.43232, 435.52900])
        accuracyTests.append(frame)

        frame = dict()
        frame['axis'] = [[586.81782, 994.85234, -164.15032, 0],
                         [367.53692, -193.11815, 141.95648, 0],
                         [814.64795, 681.51439, 87.63894, 0],
                         [0, 0, 0, 1]]
        frame['RHip'] = np.array([-570.72711, 409.48580, 387.17337])
        frame['LHip'] = np.array([984.963690, 161.72241, 714.78280])
        accuracyTests.append(frame)

        frame = dict()
        frame['axis'] = [[711.02921, -701.16460, 532.55441, 0],
                         [-229.76971, -650.15237, 359.70108, 0],
                         [222.81187, 536.56366, 386.21334, 0],
                         [0, 0, 0, 1]]
        frame['RHip'] = np.array([-651.87183, -493.94863, 640.38911])
        frame['LHip'] = np.array([624.42436, 746.90149, -603.71553])
        accuracyTests.append(frame)

        accuracyResults = [
            ([[245.47574, 131.11787, 436.75940],
              [261.08904, 155.43412, 500.91762]]),
            ([[207.11829, 285.60410, 550.97808],
              [1344.79441, 1237.35590,  673.36804]]),
            ([[-13.72374, 126.47643,  18.33679],
              [627.86030, 1671.50487, 1130.43334]])
        ]
        for i in range(len(accuracyTests)):
            # Call find_L5_pelvis(frame) with each frame in accuracyTests and round each variable in the 3-element returned list.
            result = [np.around(arr, rounding_precision)
                      for arr in kinetics.find_L5(accuracyTests[i])]
            expected = list(accuracyResults[i])
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expected[j])

    def test_get_kinetics(self):
        """
        This test provides coverage of the get_kinetics function in kinetics.py,
        defined as get_kinetics(data, Bodymass), where data is an array of joint centers
        and Bodymass is a float or int.

        This test uses helper functions to obtain the data variable (aka joint_centers).

        Each index in accuracyTests is used as parameters for the function get_kinetics
        and the result is then checked to be equal with the same index in
        accuracyResults using 8 decimal point precision comparison.
        """
        # Testing is done by using 5 different bodymasses and the same joint_center obtained from the helper functions.
        from pycgm.helpers import getfilenames
        from pycgm.IO import loadData, loadVSK
        from pycgm.pycgmStatic import getStatic
        from pycgm.calc import calcAngles

        cwd = os.getcwd() + os.sep
        # Data is obtained from the sample files.
        dynamic_trial, static_trial, vsk_file, _, _ = getfilenames(2)
        motionData = loadData(cwd+dynamic_trial)
        staticData = loadData(cwd+static_trial)
        vsk = loadVSK(cwd+vsk_file, dict=False)

        calSM = getStatic(staticData, vsk, flat_foot=False)
        _, joint_centers = calcAngles(motionData, start=None, end=None, vsk=calSM,
                                      splitAnglesAxis=False, formatData=False, returnjoints=True)

        accuracyTests = []
        calSM['Bodymass'] = 5.0
        # This creates five individual assertions to check, all with the same joint_centers but different bodymasses.
        for i in range(5):
            accuracyTests.append((joint_centers, calSM['Bodymass']))
            # Increment the bodymass by a substantial amount each time.
            calSM['Bodymass'] += 35.75

        accuracyResults = [
            ([253.21284,  318.98239, 1029.61172]),
            ([253.22968,  319.04709, 1029.61511]),
            ([253.24696,  319.11072, 1029.61835]),
            ([253.26470,  319.17322, 1029.62142]),
            ([253.28289,  319.23461, 1029.62437]),
        ]
        for i in range(len(accuracyResults)):
            # Call get_kinetics(joint_centers,bodymass) and round each variable in the 3-element returned list to the 8th decimal precision.
            result = [np.around(arr, rounding_precision) for arr in kinetics.get_kinetics(
                accuracyTests[i][0], accuracyTests[i][1])]

            # Compare the result with the values in the expected results, within a rounding precision of 8.
            np.testing.assert_almost_equal(
                result[i], accuracyResults[i], rounding_precision)
