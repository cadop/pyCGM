#!/usr/bin/python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import os
import sys
from pycgm.pycgm import StaticCGM, CGM
import pycgm.io as IO

rounding_precision = 6


class TestStaticCGMAxis:
    """
    This class tests the utility functions in pycgm.StaticCGM:
        iad_calculation
        ankle_angle_calc
        static_calculation
    """
    nan_3d = [np.nan, np.nan, np.nan]
    rand_coor = [np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)]

    @pytest.mark.parametrize(["rasi", "lasi", "expected"], [
        (np.array([0, 0, 0]), np.array([0, 0, 0]), 0),
        (np.array([1, 0, 0]), np.array([0, 0, 0]), 1),
        (np.array([0, 0, 0]), np.array([2, 0, 0]), 2),
        (np.array([0, 1, 0]), np.array([0, 1, 0]), 0),
        (np.array([4, 0, 0]), np.array([2, 0, 0]), 2),
        (np.array([4, 0, 0]), np.array([-2, 0, 0]), 6),
        (np.array([0, 2, 1]), np.array([0, 4, 1]), 2),
        (np.array([-5, 3, 0]), np.array([0, 3, 0]), 5),
        (np.array([0, 3, -6]), np.array([0, 2, -5]), 1.4142135623730951),
        (np.array([-6, 4, 0]), np.array([0, 6, -8]), 10.198039027185569),
        (np.array([7, 2, -6]), np.array([3, -7, 2]), 12.68857754044952),
        # Testing that when the markers is composed of lists of ints
        ([7, 2, -6], [3, -7, 2], 12.68857754044952),
        # Testing that when the markers is composed of numpy arrays of ints
        (np.array([7, 2, -6], dtype='int'), np.array([3, -7, 2], dtype='int'), 12.68857754044952),
        # Testing that when the markers is composed of lists of floats
        ([7.0, 2.0, -6.0], [3.0, -7.0, 2.0], 12.68857754044952),
        # Testing that when the markers is composed ofe numpy arrays of floats
        (np.array([7.0, 2.0, -6.0], dtype='float'), np.array([3.0, -7.0, 2.0], dtype='float'), 12.68857754044952)
    ])
    def test_iad_calculation(self, rasi, lasi, expected):
        """
        This test provides coverage of the StaticCGM.iad_calculation,
        defined as iad_calculation(rasi, lasi), where rasi and lasi are the
        arrays representing the positions of the RASI and LASI markers.

        Given the markers RASI and LASI, the Inter ASIS Distance is defined as:
        .. math::
            InterASISDist = \sqrt{(RASI_x-LASI_x)^2 + (RASI_y-LASI_y)^2 + (RASI_z-LASI_z)^2}
        where :math:`RASI_x` is the x-coordinate of the RASI marker

        This unit test ensures that:
        - the distance is measured correctly when some coordinates are the same, all coordinates are the same, and all
        coordinates are different
        - the distance is measured correctly given positive, negative and zero values
        - the resulting output is correct when the markers are composed of lists of ints, numpy arrays of ints, lists of
        floats, and numpy arrays of floats.
        """
        result = StaticCGM.iad_calculation(rasi, lasi)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    @pytest.mark.parametrize(["x_rot", "y_rot", "z_rot", "expected_results"], [
        (0, 0, 0, [0, 0, 0]),
        # X rotations
        (90, 0, 0, [-1.570796, 0, 0]), (30, 0, 0, [-0.523599, 0, 0]), (-30, 0, 0, [0.523599, 0, 0]),
        (120, 0, 0, [-1.047198, 0, 0]), (-120, 0, 0, [1.047198, 0, 0]), (180, 0, 0, [0, 0, 0]),
        # Y rotations
        (0, 90, 0, [0, -1.570796, 0]), (0, 30, 0, [0, -0.523599, 0]), (0, -30, 0, [0, 0.523599, 0]),
        (0, 120, 0, [0, 1.047198, 0]), (0, -120, 0, [0, -1.047198, 0]), (0, 180, 0, [0, 0, 0]),
        # Z rotations
        (0, 0, 90, [0, 0, -1.570796]), (0, 0, 30, [0, 0, -0.523599]), (0, 0, -30, [0, 0, 0.523599]),
        (0, 0, 120, [0, 0, 1.047198]), (0, 0, -120, [0, 0, -1.047198]), (0, 0, 180, [0, 0, 0]),
        # Multiple Rotations
        (150, 30, 0, [-0.447832, 0.588003, 0.281035]), (45, 0, 60, [-0.785398, -0, -1.047198]),
        (0, 90, 120, [0, -1.570796, 1.047198]), (135, 45, 90, [-0.523599, 0.955317, -0.955317])])
    def test_ankle_angle_calc(self, x_rot, y_rot, z_rot, expected_results):
        """
        This test provides coverage of the ankle_angle_calc method in StaticCGM in the file pycgm.py, it is defined
        as ankle_angle_calc(axis_p, axis_d)

        This test takes 3 parameters:
        axis_p: the unit vector of axis_p, the position of the proximal axis
        axis_d: the unit vector of axis_d, the position of the distal axis
        expected_results: the expected result from calling ankle_angle_calc on axis_p and axis_d.
        This returns thex, y, z angles
        from a XYZ Euler angle calculation from a rotational matrix. This rotational matrix is calculated by matrix
        multiplication of axis_d and the inverse of axis_p. This angle is in radians, not degrees.

        The x, y, and z angles are defined as:
        .. math::
            \[ x = \arctan{\frac{M[2][1]}{\sqrt{M[2][0]^2 + M[2][2]^2}}} \]
            \[ y = \arctan{\frac{-M[2][0]}{M[2][2]}} \]
            \[ z = \arctan{\frac{-M[0][1]}{M[1][1]}} \]
        where M is the rotation matrix produced from multiplying axis_d and :math:`axis_p^{-1}`

        This test ensures that:
        - A rotation in one axis will only effect the resulting angle in the corresponding axes
        - A rotation in multiple axes can effect the angles in other axes due to XYZ order
        """
        # Create axis_p as a rotational matrix using the x, y, and z rotations given
        # The rotational matrix will be created using CGM's rotation_matrix.
        axis_p = CGM.rotation_matrix(x_rot, y_rot, z_rot)
        axis_d = CGM.rotation_matrix(0, 0, 0)
        result = StaticCGM.ankle_angle_calc(axis_p, axis_d)
        np.testing.assert_almost_equal(result, expected_results, rounding_precision)

    @pytest.mark.parametrize(["rtoe", "ltoe", "rhee", "lhee", "ankle_axis", "flat_foot", "measurements", "expected"], [
        ([  # Testing from running sample data
            np.array([433.33508301, 354.97229004, 44.27765274]), np.array([31.77310181, 331.23657227, 42.15322876]),
            np.array([381.88534546, 148.47607422, 49.99120331]), np.array([122.18766785, 138.55477905, 46.29433441]),
            np.array([np.array([397.45738291, 217.50712216, 87.83068433]), np.array(rand_coor),
                      np.array([396.73749179, 218.18875543, 87.69979179]), np.array(rand_coor),
                      np.array([112.28082818, 175.83265027, 80.98477997]), np.array(rand_coor),
                      np.array([111.34886681, 175.49163538, 81.10789314]), np.array(rand_coor)]),
            False, {},
            np.array([[-0.015688860223839234, 0.2703999495115947, -0.15237642705642993],
                      [0.009550866847196991, 0.20242596489042683, -0.019420801722458948]])]),
        ([  # Testing with zeros for all parameters.
            np.array([0, 0, 0]), np.array([0, 0, 0]),
            np.array([0, 0, 0]), np.array([0, 0, 0]),
            np.array([np.array([0, 0, 0]), np.array(rand_coor),
                      np.array([0, 0, 0]), np.array(rand_coor),
                      np.array([0, 0, 0]), np.array(rand_coor),
                      np.array([0, 0, 0]), np.array(rand_coor)]),
            False,
            {},
            np.array([nan_3d, nan_3d])]),
        ([  # Testing marker values only
            np.array([1, -4, -1]), np.array([1, -1, -5]),
            np.array([1, -7, -4]), np.array([-6, -5, 1]),
            np.array([np.array([0, 0, 0]), np.array(rand_coor),
                      np.array([0, 0, 0]), np.array(rand_coor),
                      np.array([0, 0, 0]), np.array(rand_coor),
                      np.array([0, 0, 0]), np.array(rand_coor)]),
            False, {},
            np.array([nan_3d, nan_3d])]),
        ([  # Testing with ankle_axis values only
            np.array([0, 0, 0]), np.array([0, 0, 0]),
            np.array([0, 0, 0]), np.array([0, 0, 0]),
            np.array([np.array([3, 3, -2]), np.array(rand_coor),
                      np.array([6, -9, 9]), np.array(rand_coor),
                      np.array([5, 9, -4]), np.array(rand_coor),
                      np.array([-5, 6, 9]), np.array(rand_coor)]),
            False,
            {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
            np.array([nan_3d, nan_3d])]),
        ([  # Testing with measurement values only
            np.array([0, 0, 0]), np.array([0, 0, 0]),
            np.array([0, 0, 0]), np.array([0, 0, 0]),
            np.array([np.array([0, 0, 0]), np.array(rand_coor),
                      np.array([0, 0, 0]), np.array(rand_coor),
                      np.array([0, 0, 0]), np.array(rand_coor),
                      np.array([0, 0, 0]), np.array(rand_coor)]),
            False,
            {},
            np.array([nan_3d, nan_3d])]),
        ([  # Testing with marker and ankle_axis values
            np.array([1, -4, -1]), np.array([1, -1, -5]),
            np.array([1, -7, -4]), np.array([-6, -5, 1]),
            np.array([np.array([3, 3, -2]), np.array(rand_coor),
                      np.array([6, -9, 9]), np.array(rand_coor),
                      np.array([5, 9, -4]), np.array(rand_coor),
                      np.array([-5, 6, 9]), np.array(rand_coor)]),
            False,
            {},
            np.array([[-0.590828, -0.802097, -0.554384], [0.955226, 0.156745, 0.166848]])]),
        ([  # Testing with marker, ankle_axis, and measurement values
            np.array([1, -4, -1]), np.array([1, -1, -5]),
            np.array([1, -7, -4]), np.array([-6, -5, 1]),
            np.array([np.array([3, 3, -2]), np.array(rand_coor),
                      np.array([6, -9, 9]), np.array(rand_coor),
                      np.array([5, 9, -4]), np.array(rand_coor),
                      np.array([-5, 6, 9]), np.array(rand_coor)]),
            False,
            {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
            np.array([[-0.590828, -0.802097, -0.554384], [0.955226, 0.156745, 0.166848]])]),
        ([  # Testing with marker, ankle_axis, and measurement values and flat_foot = True
            np.array([1, -4, -1]), np.array([1, -1, -5]),
            np.array([1, -7, -4]), np.array([-6, -5, 1]),
            np.array([np.array([3, 3, -2]), np.array(rand_coor),
                      np.array([6, -9, 9]), np.array(rand_coor),
                      np.array([5, 9, -4]), np.array(rand_coor),
                      np.array([-5, 6, 9]), np.array(rand_coor)]),
            True,
            {'RightSoleDelta': 0.64, 'LeftSoleDelta': 0.19},
            np.array([[0.041042018208567545, -0.3065439019577841, -0.3106927663413161],
                      [0.39326377295256626, 0.5657243847333632, 0.2128595189127902]])]),
        ([  # Testing with all parameters as numpy arrays of floats, with flat_foot = True
            np.array([1.0, -4.0, -1.0], dtype='float'), np.array([1.0, -1.0, -5.0], dtype='float'),
            np.array([1.0, -7.0, -4.0], dtype='float'), np.array([-6.0, -5.0, 1.0], dtype='float'),
            np.array([np.array([3.0, 3.0, -2.0], dtype='float'), np.array(rand_coor, dtype='float'),
                      np.array([6.0, -9.0, 9.0], dtype='float'), np.array(rand_coor, dtype='float'),
                      np.array([5.0, 9.0, -4.0], dtype='float'), np.array(rand_coor, dtype='float'),
                      np.array([-5.0, 6.0, 9.0], dtype='float'), np.array(rand_coor, dtype='float')]),
            True,
            {'RightSoleDelta': 1.0, 'LeftSoleDelta': -1.0},
            np.array([[0.041042018208567545, -0.30654390195778408, -0.30110158620693045],
                      [00.3932637729525662, 0.56572438473336295, 0.22802611517428609]])])
    ])
    def test_static_calculation(self, rtoe, ltoe, rhee, lhee, ankle_axis, flat_foot, measurements, expected):
        """
        This test provides coverage of the static_calculation function in pycgm.py,
        defined as static_calculation(rtoe, ltoe, rhee, lhee, ankle_axis, flat_foot, measurements)

        This test takes 9 parameters:
        rtoe, ltoe, rhee, lhee : dict
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_axis : ndarray
            An 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.
        flat_foot : boolean
            A boolean indicating if the feet are flat or not.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        expected : ndarray
            The result of static_calculation, which will be a ndarray representing the angle.

        This function first calculates the anatomically incorrect foot axis by calling uncorrect_footaxis. It then
        calculates the anatomically correct foot joint center and axis using either rotaxis_footflat or
        rotaxis_nonfootflat depending on if foot_flat is True or False. It then does some array manipulation and calls
        getankleangle with the anatomically correct and anatomically incorrect axes, once for the left and once for the
        right, to calculate the offset angle between the two axes.

        This test ensures that:
        - Different offset angles are returned depending on whether flat_foot is True or not
        - The resulting output is correct when the parameters are composed of lists of ints, numpy arrays of ints,
        lists of floats, and numpy arrays of floats and when the vsk values are ints and floats.
        """
        result = StaticCGM.static_calculation(rtoe, ltoe, rhee, lhee, ankle_axis, flat_foot, measurements)
        np.testing.assert_almost_equal(result, expected, rounding_precision)


class TestStaticCGMGetStatic:
    """
    This class tests the get_static method in pycgm.py's StaticCGM class.
    """

    @classmethod
    def setup_class(self):
        """
        Called once for all tests. Loads the measurements and motion_data to be used for testing get_static()
        from SampleData/ROM/.
        """
        cwd = os.getcwd()
        if cwd.split(os.sep)[-1] == "refactor":
            parent = os.path.dirname(cwd)
            os.chdir(parent)

        static_trial = 'SampleData/ROM/Sample_Static.c3d'
        measurements_path = 'SampleData/ROM/Sample_SM.vsk'
        self.static = StaticCGM(static_trial, measurements_path)
        self.measurements_copy = self.static.measurements.copy()

    def setup_method(self):
        """
        Called once before all tests in test_get_static_required_markers.
        Resets the measurements dictionary to its original state
        as returned from IO.py.
        """
        self.measurements = self.measurements_copy.copy()

    @pytest.mark.parametrize("key", [
        'LeftLegLength',
        'RightLegLength',
        'Bodymass',
        'LeftAsisTrocanterDistance',
        'InterAsisDistance',
        'LeftTibialTorsion',
        'RightTibialTorsion',
        'LeftShoulderOffset',
        'RightShoulderOffset',
        'LeftElbowWidth',
        'RightElbowWidth',
        'LeftWristWidth',
        'RightWristWidth',
        'LeftHandThickness',
        'RightHandThickness'])
    def test_get_static_required_markers(self, key):
        """
        This function tests that an exception is raised when removing given keys from the measurements dictionary. All
        of the tested markers are required to exist in the measurements dictionary to run StaticCGM.get_static(), so
        deleting those keys should raise an exception.
        """
        for flat_foot in [True, False]:
            del self.static.measurements[key]
            with pytest.raises(Exception):
                self.static.get_static(flat_foot)
            self.static.measurements = self.measurements_copy.copy()

    @pytest.mark.parametrize("key", [
        'RightKneeWidth',
        'LeftKneeWidth',
        'RightAnkleWidth',
        'LeftAnkleWidth'])
    def test_get_static_zero_markers(self, key):
        """
        This function tests that when deleting given keys from the measurements dictionary, the value in the resulting
        calSM is 0. All of the tested markers are set to 0 if they don't exist in StaticCGM.get_static(), so
        deleting these keys should not raise an exception.
        """
        del self.static.measurements[key]
        result = self.static.get_static()
        np.testing.assert_almost_equal(result[key], 0, rounding_precision)

    @pytest.mark.parametrize(["key", "val"], [
        ('Bodymass', 95),
        ('InterAsisDistance', 28),
        ('RightKneeWidth', -11),
        ('LeftKneeWidth', 30),
        ('RightAnkleWidth', -41),
        ('LeftAnkleWidth', -5),
        ('LeftTibialTorsion', 28),
        ('RightTibialTorsion', -37),
        ('LeftShoulderOffset', 48),
        ('RightShoulderOffset', 93),
        ('LeftElbowWidth', -10),
        ('RightElbowWidth', -4),
        ('LeftWristWidth', 71),
        ('RightWristWidth', 9),
        ('LeftHandThickness', 15),
        ('RightHandThickness', -21),
        ('InterAsisDistance', -10)])
    def test_get_static_marker_assignment(self, key, val):
        """
        This function tests that when assigning a value to the given keys from the measurements dictionary, the value in
        the resulting calSM corresponds. All of the tested markers are assigned to calSM in StaticCGM.get_static()
        """
        self.static.measurements = self.measurements_copy.copy()
        self.static.measurements[key] = val
        result = self.static.get_static()
        np.testing.assert_almost_equal(result[key], val, rounding_precision)

    @pytest.mark.parametrize(["left_leg_length", "right_leg_length", "mean_leg_length_expected"], [
        (0, 0, 0),
        (0, 40, 20),
        (-34, 0, -17),
        (-15, 15, 0),
        (5, 46, 25.5)])
    def test_get_static_MeanLegLength(self, left_leg_length, right_leg_length, mean_leg_length_expected):
        """
        This function tests that StaticCGM.get_static() properly calculates calSM['MeanLegLength'] by averaging the
        values in measurements['LeftLegLength'] and measurements['RightLegLength']
        """
        self.static.measurements = self.measurements_copy.copy()
        self.static.measurements['LeftLegLength'] = left_leg_length
        self.static.measurements['RightLegLength'] = right_leg_length
        result = self.static.get_static()
        np.testing.assert_almost_equal(result['MeanLegLength'], mean_leg_length_expected, rounding_precision)

    @pytest.mark.parametrize(["left_val", "right_val", "left_expected", "right_expected"], [
        # Test where left and right are 0
        (0, 0, 72.512, 72.512),
        # Test where left is 0
        (85, 0, 72.512, 72.512),
        # Test where right is 0
        (0, 61, 72.512, 72.512),
        # Test where left and right aren't 0
        (85, 61, 85, 61)])
    def test_get_static_AsisToTrocanterMeasure(self, left_val, right_val, left_expected, right_expected):
        """
        Tests that if LeftAsisTrocanterDistance or RightAsisTrocanterDistance are 0
        in the input measurements dictionary, their corresponding values in calSM
        will be calculated from LeftLegLength and RightLegLength, but if they both have
        values than they will be assigned from the vsk dictionary
        """
        self.static.measurements = self.measurements_copy.copy()
        self.static.measurements['LeftAsisTrocanterDistance'] = left_val
        self.static.measurements['RightAsisTrocanterDistance'] = right_val
        result = self.static.get_static()
        np.testing.assert_almost_equal(result['L_AsisToTrocanterMeasure'], left_expected, rounding_precision)
        np.testing.assert_almost_equal(result['R_AsisToTrocanterMeasure'], right_expected, rounding_precision)

    def test_get_static_iad(self):
        """
        This function tests that when StaticCGM.get_static() is called with measurements['InterAsisDistance'] is 0, that
        the value for calSM['InterAsisDistance'] is calculated from motion_data.
        """
        self.static.measurements = self.measurements_copy.copy()
        self.static.measurements['InterAsisDistance'] = 0
        result = self.static.get_static()
        np.testing.assert_almost_equal(result['InterAsisDistance'], 215.9094195515741, rounding_precision)

    def test_gen(self):
        """
        This function tests that the correct values are returned in calSM['RightStaticRotOff'],
        calSM['RightStaticPlantFlex'], calSM['LeftStaticRotOff'], calSM['LeftStaticPlantFlex'], and calSM['HeadOffset'].
        All of these values are calculated from calls to staticCalculation and staticCalculationHead for every frame
        in motionData
        """
        self.static.measurements = self.measurements_copy.copy()
        result = self.static.get_static()
        np.testing.assert_almost_equal(result['RightStaticRotOff'], 0.015683497632642041, rounding_precision)
        np.testing.assert_almost_equal(result['RightStaticPlantFlex'], 0.27024179070027576, rounding_precision)
        np.testing.assert_almost_equal(result['LeftStaticRotOff'], 0.0094029102924030206, rounding_precision)
        np.testing.assert_almost_equal(result['LeftStaticPlantFlex'], 0.20251085737834015, rounding_precision)
        np.testing.assert_almost_equal(result['HeadOffset'], 0.25719904693106527, rounding_precision)
