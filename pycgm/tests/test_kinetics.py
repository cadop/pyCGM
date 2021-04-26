from unittest import TestCase
import pytest
import pycgm.kinetics as kinetics
import numpy as np
import os

rounding_precision = 8


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
        frame['axis'] = [[251.74063624, 392.72694721, 1032.78850073, 0],
                         [250.61711554, 391.87232862, 1032.8741063, 0],
                         [251.60295336, 391.84795134, 1033.88777762, 0],
                         [0, 0, 0, 1]]
        frame['RHip'] = np.array([208.38050472, 122.80342417, 437.98979061])
        frame['LHip'] = np.array([282.57097863, 139.43231855, 435.52900012])
        accuracyTests.append(frame)

        frame = dict()
        frame['axis'] = [[586.81782059, 994.852335, -164.15032491, 0],
                         [367.53692416, -193.11814502, 141.95648112, 0],
                         [814.64795266, 681.51439276, 87.63894117, 0],
                         [0, 0, 0, 1]]
        frame['RHip'] = np.array([-570.727107, 409.48579719, 387.17336605])
        frame['LHip'] = np.array([984.96369008, 161.72241084, 714.78280362])
        accuracyTests.append(frame)

        frame = dict()
        frame['axis'] = [[711.02920886, -701.16459687, 532.55441473, 0],
                         [-229.76970935, -650.15236712, 359.70108366, 0],
                         [222.81186893, 536.56366268, 386.21334066, 0],
                         [0, 0, 0, 1]]
        frame['RHip'] = np.array([-651.87182756, -493.94862894, 640.38910712])
        frame['LHip'] = np.array([624.42435686, 746.90148656, -603.71552902])
        accuracyTests.append(frame)

        accuracyResults = [
            ([[245.4757417, 131.1178714, 436.7593954],
             [261.0890402, 155.4341163, 500.9176188]]),
            ([[207.1182915, 285.604104, 550.9780848], [
             1344.7944079, 1237.3558945,  673.3680447]]),
            ([[-13.7237354, 126.4764288,  18.3367891],
             [627.8602897, 1671.5048695, 1130.4333341]])
        ]
        for i in range(len(accuracyTests)):
            # Call find_L5_pelvis(frame) with each frame in accuracyTests and round each variable in the 3-element returned list.
            result = [np.around(arr, rounding_precision)
                      for arr in kinetics.find_L5(accuracyTests[i])]
            expected = list(accuracyResults[i])
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expected[j])

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
    #    from pycgm.helpers import getfilenames
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
