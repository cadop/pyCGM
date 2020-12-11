#!/usr/bin/python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import os
import sys
import tempfile
from shutil import rmtree
from pycgm.io import IO
from mock import patch

class TestIO:
    @classmethod
    def setup_class(self):
        """
        Called once for all tests in IO.
        Sets rounding_precision, loads filenames to be used
        for testing load functions, and sets the python version
        being used. Also sets an subject measurement dictionary
        that is an expected return value from load_sm.

        We also run the CGM code to get a frame of output data to get
        IO.write_result.
        """
        self.rounding_precision = 8
        self.cwd = os.getcwd()
        self.pyver = sys.version_info.major

        self.expected_subject_measurements = {
            'ASISx': 0.0, 'ASISy': 140.533599853516, 'ASISz': 0.0,
            'BHDy': 65.2139663696289, 'Beta': 0.314000427722931,
            'Bodymass': 72.0, 'C': 0.0, 'C7x': -160.832626342773,
            'C7z': 35.1444931030273, 'HEADy': 63.3674736022949, 'HJCy': 0.0,
            'HeadOffset': 0.290628731250763, 'HeadOx': 0.167465895414352,
            'HeadOy': 0.0252241939306259, 'HeadOz': 700.027526855469,
            'Height': 1730.0, 'InterAsisDistance': 281.118011474609,
            'LANKy': 0.0, 'LASIx': 0.0, 'LASIz': 0.0,
            'LBHDx': -171.985321044922, 'LELBy': 0.0,
            'LFINy': 0.0, 'LFOOy': -3.58954524993896,
            'LFOOz': -15.6271200180054, 'LHEEx': -71.2924499511719,
            'LKNEy': 0.0, 'LPSIx': -134.874649047852,
            'LPSIy': 66.1733016967773, 'LTHIy': 91.1664733886719,
            'LTHIz': -246.724578857422, 'LTIBy': 86.1824417114258,
            'LTIBz': -241.08772277832, 'LTOEx': 112.053565979004,
            'LWRx': 37.3928489685059, 'LWRy': 0.0,
            'LeftAnkleAbAdd': 0.0, 'LeftAnkleWidth': 90.0,
            'LeftAsisTrocanterDistance': 0.0, 'LeftClavicleLength': 169.407562255859,
            'LeftElbowWidth': 80.0, 'LeftFemurLength': 401.595642089844,
            'LeftFootLength': 176.388000488281, 'LeftHandLength': 86.4382247924805,
            'LeftHandThickness': 17.0, 'LeftHumerusLength': 311.366516113281,
            'LeftKneeWidth': 120.0, 'LeftLegLength': 1000.0,
            'LeftRadiusLength': 264.315307617188, 'LeftShankRotation': 0.0,
            'LeftShoulderOffset': 40.0, 'LeftSoleDelta': 0.0,
            'LeftStaticPlantFlex': 0.137504011392593, 'LeftStaticRotOff': 0.0358467921614647,
            'LeftThighRotation': 0.0, 'LeftTibiaLength': 432.649719238281,
            'LeftTibialTorsion': 0.0, 'LeftWristWidth': 60.0,
            'MeanLegLength': 0.0, 'PelvisLength': 0.0, 'RANKy': 0.0,
            'RASIx': 0.0, 'RASIz': 0.0, 'RBAKx': -217.971115112305,
            'RBAKy': -97.8660354614258, 'RBAKz': -101.454940795898,
            'RBHDx': -159.32258605957, 'RELBy': 0.0,
            'RFINy': 0.0, 'RFOOy': 2.1107816696167,
            'RFOOz': -23.4012489318848, 'RHEEx': -52.7204742431641,
            'RKNEy': 0.0, 'RPSIx': -128.248474121094, 'RPSIy': -77.4204406738281,
            'RTHIy': -93.3007659912109, 'RTHIz': -134.734924316406,
            'RTIBy': -77.5902252197266, 'RTIBz': -201.939865112305,
            'RTOEx': 129.999603271484, 'RWRx': 35.0017356872559,
            'RWRy': 0.0, 'RightAnkleAbAdd': 0.0, 'RightAnkleWidth': 90.0,
            'RightAsisTrocanterDistance': 0.0, 'RightClavicleLength': 172.908142089844,
            'RightElbowWidth': 80.0, 'RightFemurLength': 397.317260742188,
            'RightFootLength': 175.794006347656, 'RightHandLength': 87.4593048095703,
            'RightHandThickness': 17.0, 'RightHumerusLength': 290.563262939453,
            'RightKneeWidth': 120.0, 'RightLegLength': 1000.0,
            'RightRadiusLength': 264.853607177734, 'RightShankRotation': 0.0,
            'RightShoulderOffset': 40.0, 'RightSoleDelta': 0.0,
            'RightStaticPlantFlex': 0.17637075483799, 'RightStaticRotOff': 0.03440235927701,
            'RightThighRotation': 0.0, 'RightTibiaLength': 443.718109130859,
            'RightTibialTorsion': 0.0, 'RightWristWidth': 60.0,
            'STRNz': -207.033920288086, 'T10x': -205.628646850586,
            'T10y': -7.51900339126587, 'T10z': -261.275146484375,
            'Theta': 0.500000178813934, 'ThorOx': 1.28787481784821,
            'ThorOy': 0.0719171389937401, 'ThorOz': 499.705780029297
        }

        self.filename_59993_Frame = 'SampleData/59993_Frame/59993_Frame_Static.c3d'
        self.filename_Sample_Static = 'SampleData/ROM/Sample_Static.csv'
        self.filename_RoboSM_vsk = 'SampleData/Sample_2/RoboSM.vsk'
        self.filename_RoboSM_csv = 'SampleData/Sample_2/RoboSM.csv'

        dynamic_trial = 'SampleData/ROM/Sample_Dynamic.c3d'
        static_trial = 'SampleData/ROM/Sample_Static.c3d'
        measurements = 'SampleData/ROM/Sample_SM.vsk'
        self.subject = CGM(static_trial, dynamic_trial, measurements, start = 0, end = 1)
        self.subject.run()
    
    def setup_method(self):
        """
        Called once before every test method runs.
        Creates a temporary directory to be used for testing 
        functions that write to disk.
        """
        if (self.pyver == 2):
            self.tmp_dir_name = tempfile.mkdtemp()
        else:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.tmp_dir_name = self.tmp_dir.name
    
    def teardown_method(self):
        """
        Called once after every test method is finished running.
        If using Python 2, perform cleanup of previously created
        temporary directory in setup_method(). Cleanup is done
        automatically in Python 3.
        """
        if (self.pyver == 2):
            rmtree(self.tmp_dir_name)

    @pytest.mark.parametrize("frame, data_key, expected_data", [
        (0, 'LFHD',
         np.array([60.1229744, 132.4755249, 1485.8293457])),
        (0, '*113',
         np.array([-173.22341919, 166.87660217, 1273.29980469])),
        (125, 'RASI',
         np.array([144.1030426, -0.36732361, 856.89855957])),
        (125, '*114',
         np.array([169.75387573, -230.69139099, 1264.89257812])),
        (2, 'LPSI',
         np.array([-94.89163208, 49.82866287, 922.64483643])),
        (2, '*113',
         np.array([-172.94085693, 167.04344177, 1273.51000977])),
        (12, 'LKNE',
         np.array([-100.0297699, 126.43037415, 414.15945435])),
        (22, 'C7',
         np.array([-27.38780975, -8.35509396, 1301.37145996])),
        (302, 'RANK',
         np.array([52.61815643, -126.93238068, 58.56194305]))
    ])
    def test_load_c3d_data_accuracy(self, frame, data_key, expected_data):
        """
        This function tests IO.load_c3d(filename), where filename is the string
        giving the path of a c3d file to load.

        This function tests for several markers from different frames to ensure
        that load_c3d works properly.
        """
        data, mappings = IO.load_c3d(self.filename_59993_Frame)
        result_marker_data = data[frame][mappings[data_key]]
        np.testing.assert_almost_equal(result_marker_data, expected_data, self.rounding_precision)

    @pytest.mark.parametrize("data_key, expected_index", [
        ('LFHD', 0),
        ('RFEO', 81),
        ('RTOA', 106),
        ('HEDO', 37)
    ])
    def test_load_c3d_mapping(self, data_key, expected_index):
        """
        This function tests that IO.load_c3d(filename) loads marker mappings
        properly, where the mappings indicate which index corresponds to which
        marker in the loaded data.
        """
        _, mappings = IO.load_c3d(self.filename_59993_Frame)
        result_index = mappings[data_key]
        assert result_index == expected_index

    def test_loadC3D_exceptions(self):
        """
        We test that an exception is raised when loading a non-existent
        file name.
        """
        with pytest.raises(Exception):
            IO.load_c3d("NonExistentFile")

    @pytest.mark.parametrize("frame, data_key, expected_data", [
        (0, 'LFHD',
         np.array([174.5749207, 324.513031, 1728.94397])),
        (0, '*113',
         np.array([-82.65164185, 232.3781891, 1361.853638])),
        (125, 'RASI',
         np.array([353.3344727, 345.1920471, 1033.201172])),
        (125, '*114',
         np.array([567.946106, 261.444458, 1361.898071])),
        (2, 'LPSI',
         np.array([191.5829468, 175.4567261, 1050.240356])),
        (2, '*113',
         np.array([-82.66962433, 232.2470093, 1361.734741])),
        (12, 'LKNE',
         np.array([88.88719177, 242.1836853, 529.8156128])),
        (22, 'C7',
         np.array([251.1347656, 164.8985748, 1527.874634])),
        (270, 'RANK',
         np.array([427.6356201, 188.9467773, 93.36354828]))
    ])
    def test_load_csv_data_accuracy(self, frame, data_key, expected_data):
        """
        This function tests IO.load_csv(filename), where filename is the string
        giving the path of a csv file with marker data to load.

        This function tests for several markers from different frames to ensure
        that load_csv works properly.
        """
        data, mappings = IO.load_csv(self.filename_Sample_Static)
        result_marker_data = data[frame][mappings[data_key]]
        np.testing.assert_almost_equal(result_marker_data, expected_data, self.rounding_precision)

    @pytest.mark.parametrize("data_key, expected_index", [
        ('LFHD', 0),
        ('RFEO', 79),
        ('RTOA', 104),
        ('*113', 113)
    ])
    def test_load_csv_mapping(self, data_key, expected_index):
        """
        This function tests that IO.load_csv(filename) loads marker mappings
        properly, where the mappings indicate which index corresponds to which
        marker in the loaded data.
        """
        _, mappings = IO.load_csv(self.filename_Sample_Static)
        result_index = mappings[data_key]
        assert result_index == expected_index

    def test_load_csv_exceptions(self):
        """
        We test that an exception is raised when loading a non-existent
        file name.
        """
        with pytest.raises(Exception):
            IO.load_csv("NonExistentFile")

    @pytest.mark.parametrize("frame, data_key, expected_data", [
        (0, 'LFHD', np.array([174.5749207, 324.513031, 1728.94397])),
        (16, 'LWRA', np.array([-233.2779846, 485.1967163, 1128.858276])),
        (25, 'C7', np.array([251.1916809, 164.7823639, 1527.859253])),
        (100, 'RANK', np.array([427.6116943, 188.8884583, 93.36972809])),
        (12, 'RKNE', np.array([417.5567017, 241.5111389, 523.7767334]))
    ])
    def test_load_marker_data_csv(self, frame, data_key, expected_data):
        """
        This function tests IO.load_marker_data(filename), where filename is
        the string giving the path of a marker data filename to load.

        This function tests that load_marker_data works correctly when loading
        a csv file.
        """
        data, mappings = IO.load_marker_data(self.filename_Sample_Static)
        result_data = data[frame][mappings[data_key]]
        np.testing.assert_almost_equal(result_data, expected_data, self.rounding_precision)

    @pytest.mark.parametrize("frame, data_key, expected_data", [
        (0, 'LFHD', np.array([60.1229744, 132.4755249, 1485.8293457])),
        (16, 'LWRA', np.array([-422.2036438, 432.76647949, 1199.96057129])),
        (25, 'C7', np.array([-27.17804909, -8.29536247, 1301.43286133])),
        (100, 'RANK', np.array([52.61398697, -127.04923248, 58.46214676])),
        (12, 'RKNE', np.array([96.54218292, -111.24856567, 412.34362793]))
    ])
    def test_load_marker_data_c3d(self, frame, data_key, expected_data):
        """
        This function tests IO.load_marker_data(filename), where filename is
        the string giving the path of a marker data filename to load.

        This function tests that load_marker_data works correctly when loading
        a c3d file.
        """
        data, mappings = IO.load_marker_data(self.filename_59993_Frame)
        result_data = data[frame][mappings[data_key]]
        np.testing.assert_almost_equal(result_data, expected_data, self.rounding_precision)

    def test_load_marker_data_exceptions(self):
        """
        We test that loading a filename without a csv or c3d
        extension returns none.
        """
        assert IO.load_marker_data("NonExistentFile") is None

    def test_load_sm_vsk(self):
        """
        This function tests IO.load_sm_vsk(filename), where filename is
        the string giving the path to a VSK file to load.
        """
        subject_measurements = IO.load_sm_vsk(self.filename_RoboSM_vsk)
        assert isinstance(subject_measurements, dict)
        assert subject_measurements == self.expected_subject_measurements

    def test_load_sm_vsk_exceptions(self):
        """
        Test that loading a non-existent file raises an
        exception.
        """
        with pytest.raises(Exception):
            IO.load_sm_vsk("NonExistentFilename")

    def test_load_sm_csv(self):
        """
        This function tests IO.load_sm_csv(filename), where filename is
        the string giving the path to a csv file with subject measurements to load.
        """
        subject_measurements = IO.load_sm_csv(self.filename_RoboSM_csv)
        assert isinstance(subject_measurements, dict)
        assert subject_measurements == self.expected_subject_measurements

    def test_load_sm_vsk_exceptions(self):
        """
        Test that loading a non-existent file raises an
        exception.
        """
        with pytest.raises(Exception):
            IO.load_sm_csv("NonExistentFilename")

    def test_load_sm_calls_vsk(self):
        """
        This function tests that IO.load_sm(filename) properly calls
        IO.load_sm_vsk when given a filename with a .vsk extension.
        """
        mock_return_value = {'Bodymass': 72.0}
        with patch.object(IO, 'load_sm_vsk', return_value=mock_return_value) as mock_load_sm_vsk:
            subject_measurements = IO.load_sm(self.filename_RoboSM_vsk)
            assert isinstance(subject_measurements, dict)
            assert subject_measurements == {'Bodymass': 72.0}
            mock_load_sm_vsk.assert_called()

    def test_load_sm_calls_csv(self):
        """
        This function tests that IO.load_sm(filename) properly calls
        IO.load_sm_csv when given a filename with a .csv extension.
        """
        mock_return_value = {'Bodymass': 72.0}
        with patch.object(IO, 'load_sm_csv', return_value=mock_return_value) as mock_load_sm_csv:
            subject_measurements = IO.load_sm(self.filename_RoboSM_csv)
            assert isinstance(subject_measurements, dict)
            assert subject_measurements == {'Bodymass': 72.0}
            mock_load_sm_csv.assert_called()

    def test_load_sm_exceptions(self):
        """
        We test that loading a filename without a csv or vsk
        extension returns none.
        """
        assert IO.load_sm("NonExistentFile") is None

    def test_load_scaling_table(self):
        """
        This function tests IO.load_scaling_table(), which loads segment
        scaling factors from the segments.csv file. This test ensures that
        the scaling factors are loaded properly.
        """
        result = IO.load_scaling_table()
        expected = {
            'Humerus': {'y': 0.322, 'mass': 0.028, 'z': 0, 'com': 0.564, 'x': 0.322}, 
            'Head': {'y': 0.495, 'mass': 0.081, 'z': 0.495, 'com': 0.506, 'x': 0.495}, 
            'Hand': {'y': 0.223, 'mass': 0.006, 'z': 0, 'com': 0.5, 'x': 0.223}, 
            'Femur': {'y': 0.329, 'mass': 0.1, 'z': 0.149, 'com': 0.567, 'x': 0.329}, 
            'Radius': {'y': 0.303, 'mass': 0.016, 'z': 0, 'com': 0.57, 'x': 0.303}, 
            'Thorax': {'y': 0.249, 'mass': 0.355, 'z': 0.124, 'com': 0.37, 'x': 0.31}, 
            'Foot': {'y': 0.475, 'mass': 0.0145, 'z': 0, 'com': 0.5, 'x': 0.475}, 
            'Pelvis': {'y': 0.31, 'mass': 0.142, 'z': 0, 'com': 0.5, 'x': 0.31}, 
            'Tibia': {'y': 0.249, 'mass': 0.0465, 'z': 0.124, 'com': 0.567, 'x': 0.255}
        }
        np.testing.assert_equal(result, expected)
    
    @pytest.mark.parametrize("angles, axis, center_of_mass, len_written, truncated_result", [
        (True, True, True, 277, 
         [0, 246.574667211621,	313.556623830123, 1026.56323492199,
          -0.308494914509454, -6.12129279337001, 7.57143110215171]),
        (True, False, False, 58, 
         [0, -0.308494914509454, -6.12129279337001,	7.57143110215171,
          2.91422292971666,	-6.86706898044634,	-18.8210007096431]),
        (False, True, False, 217, 
         [0, 251.608306884766, 391.741317749023, 1032.89349365234,
          251.740636241119,	392.726947206848, 1032.78850073204]),
        (False, False, True, 4, 
         [0, 246.574667211621, 313.556623830123, 1026.56323492199]),
        (['R Hip', 'Head'], False, False, 7, 
         [0, 2.91422292971666, -6.86706898044634, -18.8210007096431,
          0.0211967292758, 5.46225283664947, -91.4960853439643]),
        (False, ['PELO', 'L RADZ'], False, 7, 
         [0, 251.608306884766, 391.741317749023, 1032.89349365234,
          -271.942564463838, 485.19216662335, 1091.96791187486]),
        (['NonExistentKey'], False, False, 1, [0])
    ])
    def test_write_result(self, angles, axis, center_of_mass, len_written, truncated_result):
        """
        This function tests IO.write_result() with several different 
        arguments giving different ways to write outputs.

        This function uses previously computed kinematics and center of mass
        data in setup_method, and writes to a temporary directory for testing.

        We test for a truncated output, and the number of output values written.
        We test writing all angles, axes, and center of mass, only angles, 
        only axes, only center of mass, a list of angles, a list of axes, and
        non-existent keys.
        """
        output_filename = os.path.join(self.tmp_dir_name, 'new_pycgm_output.csv')
        angle_output = self.subject.angle_results
        angle_mapping = self.subject.angle_idx
        axis_output = self.subject.axis_results
        axis_mapping = self.subject.axis_idx
        center_of_mass_output = self.subject.com_results

        IO.write_result(output_filename, angle_output, angle_mapping, axis_output,
                        axis_mapping, center_of_mass_output, angles, axis, 
                        center_of_mass)
        with open(output_filename, 'r') as f:
            lines = f.readlines()
            #Skip the first 6 lines of output since they are headers
            result = lines[7].strip().split(',')
            array_result = np.asarray(result, dtype=np.float64)
            len_result = len(array_result)
            #Test that the truncated results are equal
            np.testing.assert_almost_equal(truncated_result, array_result[:7], self.rounding_precision)
            #Test we have written the correct number of results
            np.testing.assert_equal(len_result, len_written)