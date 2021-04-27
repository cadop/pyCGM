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

    # def test_find_L5(self):
    #    """
    #    This test provides coverage of the find_L5_pelvis function in kinetics.py,
    #    defined as find_L5_pelvis(frame), where frame contains the markers: LHip, RHip, and Pelvis_axis.

    #    Each index in accuracyTests is used as parameters for the function find_L5_pelvis
    #    and the result is then checked to be equal with the same index in
    #    accuracyResults using 8 decimal point precision comparison.
    #    """
    #    # Test 3 different frames that contain different markers for LHip, RHip, and Pelvis_axis.
    #    accuracyTests = []
    #    frame = {}
    #    frame['axis'] = [[251.74063624, 392.72694721, 1032.78850073, 0],
    #                     [250.61711554, 391.87232862, 1032.8741063, 0],
    #                     [251.60295336, 391.84795134, 1033.88777762, 0],
    #                     [0, 0, 0, 1]]
    #    frame['RHip'] = np.array([208.38050472, 122.80342417, 437.98979061])
    #    frame['LHip'] = np.array([282.57097863, 139.43231855, 435.52900012])
    #    accuracyTests.append(frame)

    #    frame = dict()
    #    frame['axis'] = [[586.81782059, 994.852335, -164.15032491, 0],
    #                     [367.53692416, -193.11814502, 141.95648112, 0],
    #                     [814.64795266, 681.51439276, 87.63894117, 0],
    #                     [0, 0, 0, 1]]
    #    frame['RHip'] = np.array([-570.727107, 409.48579719, 387.17336605])
    #    frame['LHip'] = np.array([984.96369008, 161.72241084, 714.78280362])
    #    accuracyTests.append(frame)

    #    frame = dict()
    #    frame['axis'] = [[711.02920886, -701.16459687, 532.55441473, 0],
    #                     [-229.76970935, -650.15236712, 359.70108366, 0],
    #                     [222.81186893, 536.56366268, 386.21334066, 0],
    #                     [0, 0, 0, 1]]
    #    frame['RHip'] = np.array([-651.87182756, -493.94862894, 640.38910712])
    #    frame['LHip'] = np.array([624.42435686, 746.90148656, -603.71552902])
    #    accuracyTests.append(frame)

    #    accuracyResults = [
    #        (
    #            [
    #                [245.4757417, 131.1178714, 436.7593954],
    #                [261.0890402, 155.4341163, 500.9176188]
    #            ]
    #        ),
    #        (
    #            [
    #                [207.1182915, 285.604104, 550.9780848],
    #                [1344.7944079, 1237.3558945,  673.3680447]
    #            ]
    #        ),
    #        (
    #            [
    #                [-13.7237354, 126.4764288,  18.3367891],
    #                [627.8602897, 1671.5048695, 1130.4333341]
    #            ]
    #        )
    #    ]
    #    for i in range(len(accuracyTests)):
    #        # Call find_L5_pelvis(frame) with each frame in accuracyTests and round each variable in the 3-element returned list.
    #        result = [np.around(arr, rounding_precision)
    #                  for arr in kinetics.find_L5(accuracyTests[i])]
    #        expected = list(accuracyResults[i])
    #        for j in range(len(result)):
    #            np.testing.assert_almost_equal(result[j], expected[j])

    # def test_get_kinetics(self):
    #    """
    #    This test provides coverage of the get_kinetics function in kinetics.py,
    #    defined as get_kinetics(data, Bodymass), where data is an array of joint centers
    #    and Bodymass is a float or int.

    #    This test uses helper functions to obtain the data variable (aka joint_centers).

    #    Each index in accuracyTests is used as parameters for the function get_kinetics
    #    and the result is then checked to be equal with the same index in
    #    accuracyResults using 8 decimal point precision comparison.
    #    """
    #    # Testing is done by using 5 different bodymasses and the same joint_center obtained from the helper functions.
    #    from pyCGM_Single.pyCGM_Helpers import getfilenames
    #    from pyCGM_Single.pycgmIO import loadData, loadVSK
    #    from pyCGM_Single.pycgmStatic import getStatic
    #    from pyCGM_Single.pycgmCalc import calcAngles

    #    cwd = os.getcwd() + os.sep
    #    # Data is obtained from the sample files.
    #    dynamic_trial, static_trial, vsk_file, _, _ = getfilenames(2)
    #    motionData = loadData(cwd+dynamic_trial)
    #    staticData = loadData(cwd+static_trial)
    #    vsk = loadVSK(cwd+vsk_file, dict=False)

    #    calSM = getStatic(staticData, vsk, flat_foot=False)
    #    _, joint_centers = calcAngles(motionData, start=None, end=None, vsk=calSM,
    #                                  splitAnglesAxis=False, formatData=False, returnjoints=True)

    #    accuracyTests = []
    #    calSM['Bodymass'] = 5.0
    #    # This creates five individual assertions to check, all with the same joint_centers but different bodymasses.
    #    for i in range(5):
    #        accuracyTests.append((joint_centers, calSM['Bodymass']))
    #        # Increment the bodymass by a substantial amount each time.
    #        calSM['Bodymass'] += 35.75

    #    accuracyResults = [
    #        ([246.57466721,  313.55662383, 1026.56323492]),
    #        ([246.59137623,  313.6216639, 1026.56440096]),
    #        ([246.60850798,  313.6856272, 1026.56531282]),
    #        ([246.6260863,  313.74845693, 1026.56594554]),
    #        ([246.64410308,  313.81017167, 1026.5663452]),
    #    ]
    #    for i in range(len(accuracyResults)):
    #        # Call get_kinetics(joint_centers,bodymass) and round each variable in the 3-element returned list to the 8th decimal precision.
    #        result = [np.around(arr, rounding_precision) for arr in kinetics.get_kinetics(
    #            accuracyTests[i][0], accuracyTests[i][1])]

    #        # Compare the result with the values in the expected results, within a rounding precision of 8.
    #        np.testing.assert_almost_equal(
    #            result[i], accuracyResults[i], rounding_precision)
