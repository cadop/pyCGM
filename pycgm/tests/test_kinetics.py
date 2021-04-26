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
        This test provides coverage of the find_L5_thorax function in kinetics.py,
        defined as find_L5(frame), frame contains the markers: C7, RHip, LHip, axis

        Each index in accuracyTests is used as parameters for the function find_L5
        and the result is then checked to be equal with the same index in
        accuracyResults using 8 decimal point precision comparison.
        """
        # Test 3 different frames that contain different markers for C7, RHip, LHip, axis.
        """
        This function tests 3 different frames.
        """
        accuracyTests = []
        frame = dict()
        frame['axis'] = [[[256.3454633226447, 365.7223958512035, 1461.920891187948], [257.26637166499415, 364.69602499862503, 1462.2347234647593], [
            256.1842731803127, 364.4328898435265, 1461.363045336319]], [256.2729542797522, 364.79605748807074, 1462.2905392309394]]
        frame['C7'] = np.array([226.78051758, 311.28042603, 1259.70300293])
        frame['LHip'] = np.array([262.38020472, 242.80342417, 521.98979061])
        frame['RHip'] = np.array([82.53097863, 239.43231855, 835.529000126])
        accuracyTests.append(frame)

        frame = dict()
        frame['axis'] = [[[309.69280961, 700.32003143, 203.66124527], [1111.49874303, 377.00086678, -
                                                                       140.88485905], [917.9480966, 60.89883132, -342.22796426]], [-857.91982333, -869.67870489, 438.51780456]]
        frame['C7'] = np.array([921.981682, 643.5500819, 439.96382993])
        frame['LHip'] = np.array([179.35982654, 815.09778236, 737.19459299])
        frame['RHip'] = np.array([103.01680043, 333.88103831, 823.33260927])
        accuracyTests.append(frame)

        frame = dict()
        frame['axis'] = [[[345.07821036, -746.40495016, -251.18652575], [499.41682335, 40.88439602,
                                                                         507.51025588], [668.3596798, 1476.88140274, 783.47804105]], [1124.81785806, -776.6778811, 999.39015919]]
        frame['C7'] = np.array([537.68019187, 691.49433996, 246.01153709])
        frame['LHip'] = np.array([47.94211912, 338.95742186, 612.52743329])
        frame['RHip'] = np.array([402.57410142, -967.96374463, 575.63618514])
        accuracyTests.append(frame)

        accuracyResults = [
            ([228.5241582, 320.87776246, 998.59374786]),
            ([569.20914046, 602.88531664, 620.68955025]),
            ([690.41775396, 713.36498782, 1139.36061258])
        ]
        for i in range(len(accuracyTests)):
            # Call find_L5_thorax(frame) with each frame in accuracyTests and round each variable in the 3-element returned list.
            result = [np.around(arr, rounding_precision)
                      for arr in kinetics.find_L5(accuracyTests[i])]
            expected = list(accuracyResults[i])
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expected[j])
