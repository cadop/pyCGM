import pytest
import numpy as np
import os
import sys
import tempfile
from shutil import rmtree
import refactor.io as io

class TestIO:
    @classmethod
    def setup_class(self):
        """
        Called once for all tests in IO.
        Sets rounding_precision, loads filenames to be used
        for testing load functions, and sets the python version
        being used.
        """
        self.rounding_precision = 8
        cwd = os.getcwd()
        if(cwd.split(os.sep)[-1]=="pyCGM_Single"):
            parent = os.path.dirname(cwd)
            os.chdir(parent)
        self.cwd = os.getcwd()
        self.pyver = sys.version_info.major

        filename_59993_Frame = 'SampleData' + os.sep + '59993_Frame' + os.sep + '59993_Frame_Static.c3d'
        self.filename_59993_Frame = os.path.join(self.cwd, filename_59993_Frame)
        filename_Sample_Static = 'SampleData' + os.sep + 'ROM' + os.sep + 'Sample_Static.csv'
        self.filename_Sample_Static = os.path.join(self.cwd, filename_Sample_Static)
    
    @pytest.mark.parametrize("frame, data_key, expected_data", [
        (0, 'LFHD',
         np.array([60.1229744, 132.4755249, 1485.8293457])),
        (0, '*113',
         np.array([-173.22341919,  166.87660217, 1273.29980469])),
        (125, 'RASI',
         np.array([144.1030426, -0.36732361, 856.89855957])),
        (125, '*114',
         np.array([ 169.75387573, -230.69139099, 1264.89257812])),
        (2, 'LPSI',
         np.array([-94.89163208, 49.82866287, 922.64483643])),
        (2, '*113',
         np.array([-172.94085693,  167.04344177, 1273.51000977])),
        (12, 'LKNE',
         np.array([-100.0297699, 126.43037415, 414.15945435])),
        (22, 'C7',
         np.array([-27.38780975, -8.35509396, 1301.37145996])),
        (302, 'RANK',
         np.array([52.61815643, -126.93238068, 58.56194305]))
    ])
    def test_load_c3d_data_accuracy(self, frame, data_key, expected_data):
        data,mappings = io.IO.load_c3d(self.filename_59993_Frame)
        result_marker_data = data[frame][mappings[data_key]]
        np.testing.assert_almost_equal(result_marker_data, expected_data, self.rounding_precision)
    
    @pytest.mark.parametrize("data_key, expected_index", [
        ('LFHD', 0),
        ('RFEO', 81),
        ('RTOA', 106),
        ('HEDO', 37)
    ])
    def test_load_c3d_mapping(self, data_key, expected_index):
        _,mappings = io.IO.load_c3d(self.filename_59993_Frame)
        result_index = mappings[data_key]
        assert result_index == expected_index
    
    def test_loadC3D_exceptions(self):
        """
        We test that an exception is raised when loading a non-existent
        file name.
        """
        with pytest.raises(Exception):
            io.IO.load_c3d("NonExistentFile")
    
    @pytest.mark.parametrize("frame, data_key, expected_data", [
        (0, 'LFHD',
         np.array([ 174.5749207,  324.513031 , 1728.94397  ])),
        (0, '*113',
         np.array([ -82.65164185,  232.3781891 , 1361.853638  ])),
        (125, 'RASI',
         np.array([ 353.3344727,  345.1920471, 1033.201172 ])),
        (125, '*114',
         np.array([ 567.946106,  261.444458, 1361.898071])),
        (2, 'LPSI',
         np.array([ 191.5829468,  175.4567261, 1050.240356 ])),
        (2, '*113',
         np.array([ -82.66962433,  232.2470093 , 1361.734741  ])),
        (12, 'LKNE',
         np.array([ 88.88719177, 242.1836853 , 529.8156128 ])),
        (22, 'C7',
         np.array([ 251.1347656,  164.8985748, 1527.874634 ])),
        (270, 'RANK',
         np.array([427.6356201 , 188.9467773 ,  93.36354828]))
    ])
    def test_load_csv_data_accuracy(self, frame, data_key, expected_data):
        data,mappings = io.IO.load_csv(self.filename_Sample_Static)
        result_marker_data = data[frame][mappings[data_key]]
        np.testing.assert_almost_equal(result_marker_data, expected_data, self.rounding_precision)
    
    @pytest.mark.parametrize("data_key, expected_index", [
        ('LFHD', 0),
        ('RFEO', 79),
        ('RTOA', 104),
        ('*113', 113)
    ])
    def test_load_csv_mapping(self, data_key, expected_index):
        _,mappings = io.IO.load_csv(self.filename_Sample_Static)
        result_index = mappings[data_key]
        assert result_index == expected_index
    
    def test_load_csv_exceptions(self):
        """
        We test that an exception is raised when loading a non-existent
        file name.
        """
        with pytest.raises(Exception):
            io.IO.load_csv("NonExistentFile")

    @pytest.mark.parametrize("frame, data_key, expected_data", [
        (0, 'LFHD', np.array([174.5749207, 324.513031, 1728.94397])),
        (16, 'LWRA', np.array([-233.2779846, 485.1967163, 1128.858276])),
        (25, 'C7', np.array([251.1916809, 164.7823639, 1527.859253])),
        (100, 'RANK', np.array([427.6116943, 188.8884583, 93.36972809])),
        (12, 'RKNE', np.array([417.5567017, 241.5111389, 523.7767334]))
    ])
    def test_load_marker_data_csv(self, frame, data_key, expected_data):
        data,mappings = io.IO.load_marker_data(self.filename_Sample_Static)
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
        data, mappings = io.IO.load_marker_data(self.filename_59993_Frame)
        result_data = data[frame][mappings[data_key]]
        np.testing.assert_almost_equal(result_data, expected_data, self.rounding_precision)
    
    def test_load_marker_data_exceptions(self):
        """
        We test that loading a filename without a csv or c3d
        extension returns none.
        """
        assert io.IO.load_marker_data("NonExistentFile") is None