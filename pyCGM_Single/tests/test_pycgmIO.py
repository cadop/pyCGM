import pytest
import numpy as np
import os
import sys
import tempfile
from shutil import rmtree
import pyCGM_Single.pyCGM_Helpers as pyCGM_Helpers
import pyCGM_Single.pycgmStatic as pycgmStatic
import pyCGM_Single.pycgmIO as pycgmIO
import pyCGM_Single.pycgmCalc as pycgmCalc

class TestPycgmIO:
    @classmethod
    def setup_class(cls):
        """
        Called once for all tests for pycgmIO.
        Sets rounding_precision, loads filenames to
        be used for testing load functions, and sets
        the python version being used.

        We also run the pyCGM code to get a frame of
        output data to test pycgmIO.writeResult().
        """
        cls.rounding_precision = 8
        cwd = os.getcwd()
        if(cwd.split(os.sep)[-1]=="pyCGM_Single"):
            parent = os.path.dirname(cwd)
            os.chdir(parent)
        cls.cwd = os.getcwd()
        cls.pyver = sys.version_info.major

        cls.filename_59993_Frame = os.path.join(cls.cwd, pyCGM_Helpers.getfilenames(1)[1])
        cls.filename_Sample_Static = os.path.join(cls.cwd, 'SampleData/ROM/Sample_Static.csv')
        cls.filename_RoboSM_vsk = os.path.join(cls.cwd, pyCGM_Helpers.getfilenames(3)[2])

        dynamic_trial,static_trial,vsk_file,_,_ = pyCGM_Helpers.getfilenames(x=2)
        motion_data = pycgmIO.loadData(os.path.join(cls.cwd, dynamic_trial))
        static_data = pycgmIO.loadData(os.path.join(cls.cwd, static_trial))
        vsk_data = pycgmIO.loadVSK(os.path.join(cls.cwd, vsk_file), dict=False)
        cal_SM = pycgmStatic.getStatic(static_data,vsk_data,flat_foot=False)
        cls.kinematics = pycgmCalc.calcAngles(motion_data,start=0,end=1,\
                         vsk=cal_SM,splitAnglesAxis=False,formatData=False)
        
    def setup_method(self):
        """
        Called once before every test method runs.
        Creates up a temporary directory to be used for testing
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

    @pytest.mark.parametrize("labels, data, expected_result", [
        #Tests lists
        (['A', 'B', 'C'],
         [[[1,2,3],[4,5,6],[7,8,9]],
          [[2,3,4],[5,6,7],[8,9,10]]],
         [{'A':[1,2,3], 'B':[4,5,6], 'C':[7,8,9]},
          {'A':[2,3,4], 'B':[5,6,7], 'C':[8,9,10]}]),
        #Tests numpy arrays
        (['A', 'B', 'C'],
         [[np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9])],
          [np.array([2,3,4]),np.array([5,6,7]),np.array([8,9,10])]],
         [{'A':np.array([1,2,3]), 'B':np.array([4,5,6]), 'C':np.array([7,8,9])},
          {'A':np.array([2,3,4]), 'B':np.array([5,6,7]), 'C':np.array([8,9,10])}]),
        (['A'],
         [[[1,2,3],[4,5,6],[7,8,9]],
          [[2,3,4],[5,6,7],[8,9,10]]],
         [{'A': [1, 2, 3]}, 
          {'A': [2, 3, 4]}]),
        (['A', 'B', 'C'], 
         [[[1,2,3],[4,5,6]],
          [[2,3,4]]],
         [{'A': [1, 2, 3], 'B': [4, 5, 6]}, 
          {'A': [2, 3, 4]}])
    ])            
    def test_createMotionDataDict_accuracy(self, labels, data, expected_result):
        """
        This function tests pycgmIO.createMotionDataDict(labels,data),
        where labels is a list of label names and data is a 2d list 
        or numpy array of coordinates corresponding to labels.

        We test cases where the values in data are lists and numpy arrays.
        We test cases where there are more labels than data coordinates.
        We test cases where there are more data coordinates than labels.
        """
        result = pycgmIO.createMotionDataDict(labels, data)
        np.testing.assert_equal(result, expected_result)

    @pytest.mark.parametrize("motiondata, expected_labels, expected_data", [
        ([{'A': [1, 2, 3], 'B': [4, 5, 6]},
          {'A': [2, 3, 4], 'B': [5, 6, 7]}],
         ['A', 'B'],
         np.array([[[1, 2, 3],[4, 5, 6]],[[2, 3, 4],[5, 6, 7]]])),
        ([{'A': np.array([1, 2, 3]), 'B': np.array([4, 5, 6])},
          {'A': np.array([2, 3, 4]), 'B': np.array([5, 6, 7])}],
         ['A', 'B'],
         np.array([[[1, 2, 3],[4, 5, 6]],[[2, 3, 4],[5, 6, 7]]])),
        ([{'A': np.array([1, 2, 3]), 'B': np.array([4, 5, 6])},
          {'A': np.array([2, 3, 4])}],
         ['A', 'B'],
         np.array([[[1, 2, 3],[4, 5, 6]],[[2, 3, 4],[2, 3, 4]]])),
        ([{'B': np.array([4, 5, 6])},
          {'A': np.array([2, 3, 4])}],
         ['B'],
         np.array([[[4, 5, 6]],[[2, 3, 4]]]))
    ])
    def test_splitMotionDataDict_accuracy(self, motiondata, expected_labels, expected_data):
        """
        This function tests pycgmIO.splitMotionDataDict(motiondata),
        where motiondata is a list of dictionaries of motion capture data.
        This function splits the motiondata into a tuple of labels, data.

        We tests cases where values are lists or numpy arrays.
        We demonstrate unexpected behavior that the function produces when
        keys are not present in every dictionary of motiondata.
        """
        result_labels, result_data = pycgmIO.splitMotionDataDict(motiondata)
        np.testing.assert_equal(result_labels, expected_labels)
        np.testing.assert_equal(result_data, expected_data)
    
    @pytest.mark.parametrize("motiondata", [
        ([{'A': [1, 2]}]),
        ([{'A': []}]),
        ([{'A': [1, 2, 3, 4, 5]}]),
        ([{'A': [[1, 2, 3],
               [4, 5, 6]]}])
    ])
    def test_splitMotionDataDict_exceptions(self, motiondata):
        """
        We test that if the dictionary values in motiondata are not
        1d arrays of 3 elements, an exception is raised.
        """
        with pytest.raises(Exception):
            pycgmIO.splitMotionDataDict(motiondata)

    @pytest.mark.parametrize("labels, data, expected_result", [
        (['MeanLegLength', 'LeftKneeWidth', 'RightAnkleWidth'], [940.0, 105.0, 70.0],
         {'MeanLegLength':940.0, 'LeftKneeWidth':105.0,'RightAnkleWidth':70.0}),
        (['A', 'B', 'C', 'D'], [1, 2, 3, 4],
         {'A': 1, 'B': 2, 'C': 3, 'D': 4}),
        (['A', 'B', 'C', 'D'], np.array([1, 2, 3, 4]),
         {'A': 1, 'B': 2, 'C': 3, 'D': 4}),
        (['A', 'B'], [1, 2, 3, 4, 5, 6],
         {'A': 1, 'B': 2}),
        (['A', 'B', 'C', 'D', 'E'], [1, 2],
         {'A': 1, 'B': 2}),
        ([], [0, 1], {}),
        (['A', 'B', 'C'], [], {}),
        ([], [], {})
    ])
    def test_createVskDataDict_accuracy(self, labels, data, expected_result):
        """
        This function tests pycgmIO.createVskDataDict(labels, data), 
        which creates a dictionary of VSK file values given an array
        of labels and data.

        We test cases where labels and data are the same length,
        labels and data are different lengths,
        labels is empty, data is empty, both are empty, and 
        data is a numpy array.
        """
        result = pycgmIO.createVskDataDict(labels, data)
        np.testing.assert_equal(result, expected_result)

    @pytest.mark.parametrize("vsk, expected_labels, expected_data", [
        ({'MeanLegLength':940.0, 'LeftKneeWidth':105.0,'RightAnkleWidth':70.0},
         ['MeanLegLength', 'LeftKneeWidth', 'RightAnkleWidth'],
         np.array([940., 105.,  70.])),
        ({'A': 1, 'B': 2, 'C': 3, 'D': 4},
         ['A', 'B', 'C', 'D'], 
         np.array([1, 2, 3, 4])),
        ({}, [], np.array([]))
    ])
    def test_splitVskDataDict_accuracy(self, vsk, expected_labels, expected_data):
        """
        This function tests pycgmIO.splitVskDataDict(vsk),
        where vsk is a dictionary of vsk file values. This function
        splits the values into two arrays of labels and data.

        We test the case where the vsk is an empty dictionary.
        """
        result_labels, result_data = pycgmIO.splitVskDataDict(vsk)
        #Convert results to sets so order of the results is ignored
        np.testing.assert_equal(set(result_labels), set(expected_labels))
        np.testing.assert_equal(set(result_data), set(expected_data))            

    def test_markerKeys_accuracy(self):
        """
        This function tests pycgmIO.markerKeys, which returns
        a constant array of marker names.
        """
        result = pycgmIO.markerKeys()
        expected_result = ['RASI', 'LASI', 'RPSI', 'LPSI', 'RTHI', 'LTHI', 'RKNE', 'LKNE', 'RTIB',
            'LTIB', 'RANK', 'LANK', 'RTOE', 'LTOE', 'LFHD', 'RFHD', 'LBHD', 'RBHD', 
            'RHEE', 'LHEE', 'CLAV', 'C7', 'STRN', 'T10', 'RSHO', 'LSHO', 'RELB', 'LELB', 
            'RWRA', 'RWRB', 'LWRA', 'LWRB', 'RFIN', 'LFIN']
        assert result == expected_result

    @pytest.mark.parametrize("frame, data_key, unlabeled_data_key, expected_data, expected_unlabeled_data", [
        (0, 'LFHD', '*113',
         np.array([60.1229744, 132.4755249, 1485.8293457]),
         np.array([-173.22341919,  166.87660217, 1273.29980469])),
        (125, 'RASI', '*114',
         np.array([144.1030426, -0.36732361, 856.89855957]),
         np.array([ 169.75387573, -230.69139099, 1264.89257812])),
        (2, 'LPSI', '*113',
         np.array([-94.89163208, 49.82866287, 922.64483643]),
         np.array([-172.94085693,  167.04344177, 1273.51000977])),
        (12, 'LKNE', '*114',
         np.array([-100.0297699, 126.43037415, 414.15945435]),
         np.array([ 169.80422974, -226.73210144, 1264.15673828])),
        (22, 'C7', '*113',
         np.array([-27.38780975, -8.35509396, 1301.37145996]),
         np.array([-170.55563354,  168.37162781, 1275.37451172])),
        (302, 'RANK', '*114',
         np.array([52.61815643, -126.93238068, 58.56194305]),
         np.array([ 174.65007019, -225.9836731 , 1262.32373047]))
    ])
    def test_loadC3D_data_accuracy(self, frame, data_key, unlabeled_data_key,\
                                                    expected_data, expected_unlabeled_data):
        """
        This function tests pycgmIO.loadC3D(filename), where filename
        is a string indicating the file path of a c3d file to load. This
        function returns an array of [data, dataunlabeled, markers].

        This function uses the file 59993_Frame_Static.c3d in 
        SampleData for testing.

        This function tests for several frames and keys from data
        and dataunlabeled.
        """
        result_59993 = pycgmIO.loadC3D(self.filename_59993_Frame)
        data = result_59993[0]
        dataunlabeled = result_59993[1]
        result_data = data[frame][data_key]
        result_unlabeled_data = dataunlabeled[frame][unlabeled_data_key]
        np.testing.assert_almost_equal(result_data, expected_data, self.rounding_precision)
        np.testing.assert_almost_equal(result_unlabeled_data, expected_unlabeled_data, self.rounding_precision)

    def test_loadC3D_markers(self):
        """
        This function tests that loadC3D loads marker names
        correctly.
        """
        result_59993 = pycgmIO.loadC3D(self.filename_59993_Frame)
        markers_results = result_59993[2][0:50]
        expected_markers_results = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 
        'CLAV', 'STRN', 'RBAK', 'LSHO', 'LELB', 'LWRA', 'LWRB', 'LFIN', 'RSHO', 
        'RELB', 'RWRA', 'RWRB', 'RFIN', 'LASI', 'RASI', 'LPSI', 'RPSI', 'LTHI', 
        'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB', 'RANK', 
        'RHEE', 'RTOE', 'CentreOfMass', 'CentreOfMassFloor', 'HEDO', 'HEDA', 
        'HEDL', 'HEDP', 'LCLO', 'LCLA', 'LCLL', 'LCLP', 'LFEO', 'LFEA', 'LFEL', 'LFEP', 'LFOO']
        assert markers_results == expected_markers_results
    
    def test_loadC3D_exceptions(self):
        """
        We test that an exception is raised when loading a non-existent
        file name.
        """
        with pytest.raises(Exception):
            pycgmIO.loadC3D("NonExistentFile")

    @pytest.mark.parametrize("frame, data_key, unlabeled_data_key, expected_data, expected_unlabeled_data", [
        (0, 'LFHD', '*111',
         np.array([ 174.5749207,  324.513031 , 1728.94397  ]),
         np.array([ 692.8970947,  423.9462585, 1240.289063 ])),
        (125, 'RASI', '*112',
         np.array([ 353.3344727,  345.1920471, 1033.201172 ]),
         np.array([-225.5984955,  403.15448  , 1209.803467 ])),
        (2, 'LPSI', '*113',
         np.array([ 191.5829468,  175.4567261, 1050.240356 ]),
         np.array([ -82.66962433,  232.2470093 , 1361.734741  ])),
        (12, 'LKNE', '*114',
         np.array([ 88.88719177, 242.1836853 , 529.8156128 ]),
         np.array([ 568.6048584,  261.1444092, 1362.141968 ])),
        (22, 'C7', '*112',
         np.array([ 251.1347656,  164.8985748, 1527.874634 ]),
         np.array([-225.2479401,  404.37146  , 1214.369141 ])),
        (273, 'RANK', '*111',
         np.array([427.6519165 , 188.9484558 ,  93.37301636]),
         np.array([ 695.2038574,  421.2562866, 1239.632446 ]))
    ])
    def test_loadCSV_data_accuracy(self, frame, data_key, unlabeled_data_key, \
                                                    expected_data, expected_unlabeled_data):
        """
        This function tests pycgmIO.loadCSV(filename), where filename
        is a string indicating the file path of a CSV file to load. This
        function returns an array of [motionData, unlabeledMotionData, labels].

        This function uses the file Sample_Static.csv in 
        SampleData for testing.

        This function tests for several frames and keys from data 
        and unlabeled.
        """
        result_Sample_Static = pycgmIO.loadCSV(self.filename_Sample_Static)
        data = result_Sample_Static[0]
        dataunlabeled = result_Sample_Static[1]
        result_data = data[frame][data_key]
        result_unlabeled_data = dataunlabeled[frame][unlabeled_data_key]
        np.testing.assert_almost_equal(result_data, expected_data, self.rounding_precision)
        np.testing.assert_almost_equal(result_unlabeled_data, expected_unlabeled_data, self.rounding_precision)

    def test_loadCSV_labels(self):
        """
        This function tests that loadCSV loads label names
        correctly.
        """
        result_Sample_Static = pycgmIO.loadCSV(self.filename_Sample_Static)
        labels_results = result_Sample_Static[2][0:50]
        expected_labels_results = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10',
        'CLAV', 'STRN', 'RBAK', 'LSHO', 'LELB', 'LWRA', 'LWRB', 
        'LFIN', 'RSHO', 'RELB', 'RWRA', 'RWRB', 'RFIN', 'LASI', 
        'RASI', 'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 
        'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 
        'RTOE', 'HEDO', 'HEDA', 'HEDL', 'HEDP', 'LCLO', 'LCLA', 
        'LCLL', 'LCLP', 'LFEO', 'LFEA', 'LFEL', 'LFEP', 'LFOO', 
        'LFOA', 'LFOL']
        assert labels_results == expected_labels_results
    
    def test_loadCSV_exceptions(self):
        """
        We test that an exception is raised when loading a non-existent
        file name.
        """
        with pytest.raises(Exception):
            pycgmIO.loadCSV("NonExistentFile")

    @pytest.mark.parametrize("frame, data_key, expected_data", [
        (0, 'LFHD', np.array([174.5749207, 324.513031, 1728.94397])),
        (16, 'LWRA', np.array([-233.2779846, 485.1967163, 1128.858276])),
        (25, 'C7', np.array([251.1916809, 164.7823639, 1527.859253])),
        (100, 'RANK', np.array([427.6116943, 188.8884583, 93.36972809])),
        (12, 'RKNE', np.array([417.5567017, 241.5111389, 523.7767334]))
    ])
    def test_loadData_csv(self, frame, data_key, expected_data):
        """
        This function tests pycgmIO.loadData(filename), where filename
        is a string indicating the file path of a CSV or C3D file to load.

        This function uses Sample_Static.csv in SampleData for testing.
        """
        csv_results = pycgmIO.loadData(self.filename_Sample_Static)
        result_data = csv_results[frame][data_key]
        np.testing.assert_almost_equal(result_data, expected_data, self.rounding_precision)

    @pytest.mark.parametrize("frame, data_key, expected_data", [
        (0, 'LFHD', np.array([60.1229744, 132.4755249, 1485.8293457])),
        (16, 'LWRA', np.array([-422.2036438, 432.76647949, 1199.96057129])),
        (25, 'C7', np.array([-27.17804909, -8.29536247, 1301.43286133])),
        (100, 'RANK', np.array([52.61398697, -127.04923248, 58.46214676])),
        (12, 'RKNE', np.array([96.54218292, -111.24856567, 412.34362793]))
    ])
    def test_loadData_c3d(self, frame, data_key, expected_data):
        """
        This function tests pycgmIO.loadData(filename), where filename
        is a string indicating the file path of a CSV or C3D file to load.

        This function use 59993_Frame_Static.c3d in SampleData for testing.
        """
        c3d_results = pycgmIO.loadData(self.filename_59993_Frame)
        result_data = c3d_results[frame][data_key]
        np.testing.assert_almost_equal(result_data, expected_data, self.rounding_precision)

    def test_loadData_invalid_filename(self):
        #Test that loading a non-existent filename returns None.
        assert pycgmIO.loadData("NonExistentFile") is None

    @pytest.mark.parametrize("data, expected_result", [
        ({'A': [[1, 2], [4, 5], [7, 8]], 'B': [[4, 5], [7, 8], [10, 11]]},
         [{'A': np.array([1, 4, 7]), 'B': np.array([ 4,  7, 10])}, 
          {'A': np.array([2, 5, 8]), 'B': np.array([ 5,  8, 11])}]),
        ({'A': [np.array([1, 2]), np.array([4, 5]), np.array([7, 8])],
          'B': [np.array([4, 5]), np.array([7, 8]), np.array([10, 11])]},
         [{'A': np.array([1, 4, 7]), 'B': np.array([ 4,  7, 10])}, 
          {'A': np.array([2, 5, 8]), 'B': np.array([ 5,  8, 11])}]),
        ({'A': [[1, 2], [4, 5], [7]]},
         [{'A': np.array([1, 4, 7])}]),
        ({'A': [[2], [4], [6], [8]]},
         [{'A': np.array([2, 4, 6])}]),
        ({'A': [[], [4, 5], [7, 8, 9]]},[])
    ])
    def test_dataAsArray_accuracy(self, data, expected_result):
        """
        This function tests pycgmIO.dataAsArray(data), where
        data is a dictionary of marker data. The function returns
        the dictionary data as an array of dictionaries.

        We test cases where the input is lists or numpy arrays. We test
        cases where the arrays are not all the same shape, when there are more
        than 3 arrays per dictionary key, and when dictionary keys are empty.
        """
        result = pycgmIO.dataAsArray(data)
        np.testing.assert_equal(result, expected_result)

    @pytest.mark.parametrize("data", [
        ({}),
        ({'A': []}),
        ({'A': [[1], [2]]}),
        ({'A': [[1, 2], [2, 3], [3, 4]],
            'B': [[4, 5], [5, 6], [6]]})
    ])
    def test_dataAsArray_exception(self, data):
        """
        We test exceptions from keys with less than 3 arrays of data, 
        empty arrays of data, or inconsistent shapes of arrays across keys.
        """
        with pytest.raises(Exception):
            pycgmIO.dataAsArray(data)

    @pytest.mark.parametrize("data, expected_result", [
        ([{'A': [1, 2, 3], 'B': [4, 5, 6]},
          {'A': [2, 3, 4], 'B': [5, 6, 7]}],
         {'A': [[1, 2, 3], [2, 3, 4]], 'B': [[4, 5, 6], [5, 6, 7]]}),
        ([{'A': [1, 2], 'B': [4]},
          {'A': [4], 'B': []}],
         {'A': [[1, 2], [4]], 'B': [[4], []]}),
        ([{'A': [1, 2]},
          {'A': [4, 5], 'B': [6, 7]}],
         {'A': [[1, 2], [4, 5]], 'B': [[6, 7]]}),
        ([{'A': 2} , {'B': [6, 7]}],
         {'A': [2], 'B': [[6, 7]]}),
        ([], {})
    ])
    def test_dataAsDict_accuracy(self, data, expected_result):
        """
        This function tests pycgmIO.dataAsDict(data, npArray=False), where
        data is a list of dictionaries of marker data. This function returns
        a data as a dictionary.

        We test cases with multiple markers with the same length of data,
        empty arrays, non-array dictionary values, and inconsistent keys
        across the indices of data.
        """
        result = pycgmIO.dataAsDict(data)
        np.testing.assert_equal(result, expected_result)
    
    def test_dataAsDict_numpy_array(self):
        #Test that data is returned as a numpy array if npArray = True
        data = [{'A': [1, 2, 3]}]
        result = pycgmIO.dataAsDict(data, npArray=True)
        result_data = result['A']
        assert isinstance(result_data, np.ndarray)
        assert not isinstance(result_data, list)
    
    @pytest.mark.parametrize("kinetics", [
        ([[1.1, 2.2, 3.3],
          [4.4, 5.5, 6.6],
          [7.7, 8.8, 9.9]]),
        (np.array([[1.1, 2.2, 3.3],
                   [4.4, 5.5, 6.6],
                   [7.7, 8.8, 9.9]]))
    ])
    def test_writeKinetics_accuracy(self, kinetics):
        """
        This function tests pycgmIO.writeKinetics(CoM_output, kinetics),
        where CoM_output is the filename to save output to,
        and kinetics is the array_like output to be saved.

        pycgmIO.writeKinetics() saves array data as .npy files.

        This function tests saving lists and numpy arrays. 
        """
        CoM_output = os.path.join(self.tmp_dir_name, 'CoM')
        pycgmIO.writeKinetics(CoM_output, kinetics)
        write_result = np.load(CoM_output + '.npy')
        np.testing.assert_equal(write_result, kinetics)

    @pytest.mark.parametrize("kwargs, len_written, truncated_result", [
        ({}, 274, 
         [0, -0.308494914509454,-6.121292793370006,7.571431102151712,
          2.914222929716658,-6.867068980446340,-18.821000709643130]),
        ({'angles': False}, 217, 
         [0, 251.608306884765625,391.741317749023438,1032.893493652343750,
          251.740636241118779,392.726947206848479,1032.788500732036255]),
        ({'axis': False}, 58, 
         [0, -0.308494914509454,-6.121292793370006,7.571431102151712,
          2.914222929716658,-6.867068980446340,-18.821000709643130]),
        ({'angles': ['R Hip', 'Head'],'axis': False}, 7, 
         [0, 2.914222929716658,-6.867068980446340,-18.821000709643130,
          0.021196729275744,5.462252836649474,-91.496085343964339]),
        ({'axis': ['PELO', 'L RADZ'], 'angles': False}, 7, 
         [0, 251.608306884765625,391.741317749023438,1032.893493652343750,
          -271.942564463838380,485.192166623350204,1091.967911874857009]),
        ({'axis': ['NonExistentKey'], 'angles': False}, 1, [0])
    ])
    def test_writeResult(self, kwargs, len_written, truncated_result):
        """
        This function tests pycgmIO.writeResult(data, filename, **kwargs),
        where data is the pcygm output data to write, filename is the filename
        to write to, and **kwargs is a dictionary of keyword arguments
        specifying writing options.

        We test for a truncated output, and the number of output values written.
        We test writing all angles and axes, only angles, only axis,
        a list of angles, a list of axis, and non-existent keys.

        This function uses the previously computed kinematics data 
        in setup_method, and writes to a temporary directory for testing.
        """
        data = self.kinematics
        output_filename = os.path.join(self.tmp_dir_name, 'output')
        pycgmIO.writeResult(data, output_filename, **kwargs)
        with open(output_filename + '.csv', 'r') as f:
            lines = f.readlines()
            #Skip the first 6 lines of output since they are headers
            result = lines[7].strip().split(',')
            array_result = np.asarray(result, dtype=np.float64)
            len_result = len(array_result)
            #Test that the truncated results are equal
            np.testing.assert_almost_equal(truncated_result, array_result[:7], 11)
            #Test we have written the correct number of results
            np.testing.assert_equal(len_result, len_written)
        
    def test_smKeys(self):
        """
        This function tests pycgmIO.smKeys(), which returns
        a list of subject measurement keys.
        """
        result = pycgmIO.smKeys()
        expected_result = ['Bodymass', 'Height', 'HeadOffset', 'InterAsisDistance', 'LeftAnkleWidth', 
        'LeftAsisTrocanterDistance', 'LeftClavicleLength', 'LeftElbowWidth', 
        'LeftFemurLength', 'LeftFootLength', 'LeftHandLength', 'LeftHandThickness', 
        'LeftHumerusLength', 'LeftKneeWidth', 'LeftLegLength', 'LeftRadiusLength', 
        'LeftShoulderOffset', 'LeftTibiaLength', 'LeftWristWidth', 'RightAnkleWidth', 
        'RightClavicleLength', 'RightElbowWidth', 'RightFemurLength', 'RightFootLength', 
        'RightHandLength', 'RightHandThickness', 'RightHumerusLength', 'RightKneeWidth', 
        'RightLegLength', 'RightRadiusLength', 'RightShoulderOffset', 'RightTibiaLength', 
        'RightWristWidth']
        assert result == expected_result
    
    def test_loadVSK_list(self):
        """
        This function tests pycgmIO.loadVSK(filename, dict=True),
        where filename is the vsk file to be loaded and dict is a 
        bool indicating whether to return the vsk as a dictionary or list of
        [keys, values].

        RoboSM.vsk in SampleData is used to test the output.

        We test returning as a list.
        """
        result_vsk = pycgmIO.loadVSK(self.filename_RoboSM_vsk, dict=True)
        result_keys = result_vsk[0]
        result_values = result_vsk[1]
        expected_keys = ['Bodymass', 'Height', 'InterAsisDistance', 'LeftLegLength', 'LeftAsisTrocanterDistance', 
        'LeftKneeWidth', 'LeftAnkleWidth', 'LeftTibialTorsion', 'LeftSoleDelta', 'LeftThighRotation', 
        'LeftShankRotation', 'LeftStaticPlantFlex', 'LeftStaticRotOff', 'LeftAnkleAbAdd', 
        'LeftShoulderOffset', 'LeftElbowWidth', 'LeftWristWidth', 'LeftHandThickness', 'RightLegLength', 
        'RightAsisTrocanterDistance', 'RightKneeWidth', 'RightAnkleWidth', 'RightTibialTorsion', 
        'RightSoleDelta', 'RightThighRotation', 'RightShankRotation', 'RightStaticPlantFlex', 
        'RightStaticRotOff', 'RightAnkleAbAdd', 'RightShoulderOffset', 'RightElbowWidth', 
        'RightWristWidth', 'RightHandThickness', 'MeanLegLength', 'C', 'Theta', 'Beta', 'HJCy', 
        'PelvisLength', 'LASIx', 'LASIz', 'RASIx', 'RASIz', 'ASISx', 'ASISz', 'LKNEy', 'LANKy', 
        'RKNEy', 'RANKy', 'LELBy', 'LWRy', 'LFINy', 'RELBy', 'RWRy', 'RFINy', 'HeadOffset', 'HEADy', 
        'LBHDx', 'BHDy', 'RBHDx', 'HeadOx', 'HeadOy', 'HeadOz', 'C7x', 'C7z', 'T10x', 'T10y', 'T10z', 
        'STRNz', 'RBAKx', 'RBAKy', 'RBAKz', 'ThorOx', 'ThorOy', 'ThorOz', 'LeftClavicleLength', 'LeftHumerusLength', 
        'LeftRadiusLength', 'LeftHandLength', 'LWRx', 'RightClavicleLength', 'RightHumerusLength', 'RightRadiusLength', 
        'RightHandLength', 'RWRx', 'ASISy', 'LPSIx', 'LPSIy', 'RPSIx', 'RPSIy', 'LeftFemurLength', 'LeftTibiaLength', 
        'LeftFootLength', 'LTHIy', 'LTHIz', 'LTIBy', 'LTIBz', 'LFOOy', 'LFOOz', 'LHEEx', 'LTOEx', 'RightFemurLength', 
        'RightTibiaLength', 'RightFootLength', 'RTHIy', 'RTHIz', 'RTIBy', 'RTIBz', 'RFOOy', 'RFOOz', 'RHEEx', 'RTOEx']

        expected_values = [72.0, 1730.0, 281.118011474609, 1000.0, 0.0, 120.0, 90.0,
        0.0, 0.0, 0.0, 0.0, 0.137504011392593, 0.0358467921614647, 0.0, 40.0, 80.0,
        60.0, 17.0, 1000.0, 0.0, 120.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.17637075483799,
        0.03440235927701, 0.0, 40.0, 80.0, 60.0, 17.0, 0.0, 0.0, 0.500000178813934, 
        0.314000427722931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.290628731250763, 63.3674736022949, 
        -171.985321044922, 65.2139663696289, -159.32258605957, 0.167465895414352, 
        0.0252241939306259, 700.027526855469, -160.832626342773, 35.1444931030273, 
        -205.628646850586, -7.51900339126587, -261.275146484375, -207.033920288086, 
        -217.971115112305, -97.8660354614258, -101.454940795898, 1.28787481784821, 
        0.0719171389937401, 499.705780029297, 169.407562255859, 311.366516113281, 
        264.315307617188, 86.4382247924805, 37.3928489685059, 172.908142089844, 
        290.563262939453, 264.853607177734, 87.4593048095703, 35.0017356872559, 
        140.533599853516, -134.874649047852, 66.1733016967773, -128.248474121094, 
        -77.4204406738281, 401.595642089844, 432.649719238281, 176.388000488281, 
        91.1664733886719, -246.724578857422, 86.1824417114258, -241.08772277832, 
        -3.58954524993896, -15.6271200180054, -71.2924499511719, 112.053565979004, 
        397.317260742188, 443.718109130859, 175.794006347656, -93.3007659912109, 
        -134.734924316406, -77.5902252197266, -201.939865112305, 2.1107816696167, 
        -23.4012489318848, -52.7204742431641, 129.999603271484]
        
        #Test that loadVSK correctly returned as a list
        assert isinstance(result_vsk, list)
        #Test that len(keys) is the same as len(values)
        assert len(result_keys) == len(result_values)
        #Test accurate loading
        assert expected_keys == result_keys
        assert expected_values == result_values

    def test_loadVSK_dict(self):
        """
        We test pycgmIO.loadVSK returning as a dictionary.
        """
        result_vsk = pycgmIO.loadVSK(self.filename_RoboSM_vsk, dict=False)
        expected_result = {
            'ASISx': 0.0,'ASISy': 140.533599853516,'ASISz': 0.0,
            'BHDy': 65.2139663696289,'Beta': 0.314000427722931,
            'Bodymass': 72.0,'C': 0.0,'C7x': -160.832626342773,
            'C7z': 35.1444931030273,'HEADy': 63.3674736022949,'HJCy': 0.0,
            'HeadOffset': 0.290628731250763,'HeadOx': 0.167465895414352,
            'HeadOy': 0.0252241939306259,'HeadOz': 700.027526855469,
            'Height': 1730.0,'InterAsisDistance': 281.118011474609,
            'LANKy': 0.0,'LASIx': 0.0,'LASIz': 0.0,
            'LBHDx': -171.985321044922,'LELBy': 0.0,
            'LFINy': 0.0,'LFOOy': -3.58954524993896,
            'LFOOz': -15.6271200180054,'LHEEx': -71.2924499511719,
            'LKNEy': 0.0,'LPSIx': -134.874649047852,
            'LPSIy': 66.1733016967773,'LTHIy': 91.1664733886719,
            'LTHIz': -246.724578857422,'LTIBy': 86.1824417114258,
            'LTIBz': -241.08772277832,'LTOEx': 112.053565979004,
            'LWRx': 37.3928489685059,'LWRy': 0.0,
            'LeftAnkleAbAdd': 0.0,'LeftAnkleWidth': 90.0,
            'LeftAsisTrocanterDistance': 0.0,'LeftClavicleLength': 169.407562255859,
            'LeftElbowWidth': 80.0,'LeftFemurLength': 401.595642089844,
            'LeftFootLength': 176.388000488281,'LeftHandLength': 86.4382247924805,
            'LeftHandThickness': 17.0,'LeftHumerusLength': 311.366516113281,
            'LeftKneeWidth': 120.0,'LeftLegLength': 1000.0,
            'LeftRadiusLength': 264.315307617188,'LeftShankRotation': 0.0,
            'LeftShoulderOffset': 40.0,'LeftSoleDelta': 0.0,
            'LeftStaticPlantFlex': 0.137504011392593,'LeftStaticRotOff': 0.0358467921614647,
            'LeftThighRotation': 0.0,'LeftTibiaLength': 432.649719238281,
            'LeftTibialTorsion': 0.0,'LeftWristWidth': 60.0,
            'MeanLegLength': 0.0,'PelvisLength': 0.0,'RANKy': 0.0,
            'RASIx': 0.0,'RASIz': 0.0,'RBAKx': -217.971115112305,
            'RBAKy': -97.8660354614258,'RBAKz': -101.454940795898,
            'RBHDx': -159.32258605957,'RELBy': 0.0,
            'RFINy': 0.0,'RFOOy': 2.1107816696167,
            'RFOOz': -23.4012489318848,'RHEEx': -52.7204742431641,
            'RKNEy': 0.0,'RPSIx': -128.248474121094,'RPSIy': -77.4204406738281,
            'RTHIy': -93.3007659912109,'RTHIz': -134.734924316406,
            'RTIBy': -77.5902252197266,'RTIBz': -201.939865112305,
            'RTOEx': 129.999603271484,'RWRx': 35.0017356872559,
            'RWRy': 0.0,'RightAnkleAbAdd': 0.0,'RightAnkleWidth': 90.0,
            'RightAsisTrocanterDistance': 0.0,'RightClavicleLength': 172.908142089844,
            'RightElbowWidth': 80.0,'RightFemurLength': 397.317260742188,
            'RightFootLength': 175.794006347656,'RightHandLength': 87.4593048095703,
            'RightHandThickness': 17.0,'RightHumerusLength': 290.563262939453,
            'RightKneeWidth': 120.0,'RightLegLength': 1000.0,
            'RightRadiusLength': 264.853607177734,'RightShankRotation': 0.0,
            'RightShoulderOffset': 40.0,'RightSoleDelta': 0.0,
            'RightStaticPlantFlex': 0.17637075483799,'RightStaticRotOff': 0.03440235927701,
            'RightThighRotation': 0.0,'RightTibiaLength': 443.718109130859,
            'RightTibialTorsion': 0.0,'RightWristWidth': 60.0,
            'STRNz': -207.033920288086,'T10x': -205.628646850586,
            'T10y': -7.51900339126587,'T10z': -261.275146484375,
            'Theta': 0.500000178813934,'ThorOx': 1.28787481784821,
            'ThorOy': 0.0719171389937401,'ThorOz': 499.705780029297
        }

        #Test that loadVSK correctly returned as a dictionary
        assert isinstance(result_vsk, dict)
        #Test accurate loading
        assert result_vsk == expected_result

    def test_loadVSK_exceptions(self):
        """
        Test that loading a non-existent file raises an
        exception.
        """
        with pytest.raises(Exception):
            pycgmIO.loadVSK("NonExistentFilename")

    @pytest.mark.parametrize("motionData, expected_labels, expected_values", [
        ([{'A': [1, 2, 3], 'B': [4, 5, 6]},
          {'A': [2, 3, 4], 'B': [5, 6, 7]}],
         ['A', 'B'],
         [np.array([[1, 2, 3],[4, 5, 6]]), np.array([[2, 3, 4],[5, 6, 7]])]),
        ([{'A': np.array([1, 2, 3]), 'B': np.array([4, 5, 6])},
          {'A': np.array([2, 3, 4]), 'B': np.array([5, 6, 7])}],
         ['A', 'B'],
         [np.array([[1, 2, 3],[4, 5, 6]]), np.array([[2, 3, 4],[5, 6, 7]])]),
        ([{'A': np.array([1, 2, 3]), 'B': np.array([4, 5, 6])},
          {'A': np.array([2, 3, 4])}],
         ['A', 'B'],
         [np.array([[1, 2, 3],[4, 5, 6]]), np.array([[2, 3, 4]])]),
        ([{'B': np.array([4, 5, 6])},
          {'A': np.array([2, 3, 4])}],
         ['B'],
         [np.array([[4, 5, 6]]), np.array([[2, 3, 4]])])
    ])
    def test_splitDataDict_accuracy(self, motionData, expected_labels, expected_values):
        """
        This function tests pycgmIO.splitDataDict(motionData),
        where motionData is a list of dictionaries of motion capture data.
        This function splits the motionData into a tuple of values, labels.

        We tests cases where values are lists or numpy arrays.
        We demonstrate unexpected behavior that the function produces when
        keys are not present in every dictionary of motiondata.
        """
        result_values, result_labels = pycgmIO.splitDataDict(motionData)
        np.testing.assert_equal(result_labels, expected_labels)
        np.testing.assert_equal(result_values, expected_values)
    
    @pytest.mark.parametrize("values, labels, expected_result", [
        #Tests lists
        ([[[1,2,3],[4,5,6],[7,8,9]],
          [[2,3,4],[5,6,7],[8,9,10]]],
         ['A', 'B', 'C'],
         [{'A':[1,2,3], 'B':[4,5,6], 'C':[7,8,9]},
          {'A':[2,3,4], 'B':[5,6,7], 'C':[8,9,10]}]),
        #Tests numpy arrays
        ([[np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9])],
          [np.array([2,3,4]),np.array([5,6,7]),np.array([8,9,10])]],
         ['A', 'B', 'C'],
         [{'A':np.array([1,2,3]), 'B':np.array([4,5,6]), 'C':np.array([7,8,9])},
          {'A':np.array([2,3,4]), 'B':np.array([5,6,7]), 'C':np.array([8,9,10])}]),
        ([[[1,2,3],[4,5,6]],
          [[2,3,4]]],
         ['A', 'B', 'C'],
         [{'A': [1, 2, 3], 'B': [4, 5, 6]},
          {'A': [2, 3, 4]}])
    ]) 
    def test_combineDataDict_accuracy(self, values, labels, expected_result):
        """
        This function tests pycgmIO.combineDataDict(values, labels), where
        values is an array of motion data values and labels is a list of 
        marker names.

        We test cases where the arrays in values are lists and numpy arrays.
        We test the case where there are more labels than values.
        """
        result = pycgmIO.combineDataDict(values, labels)
        np.testing.assert_equal(result, expected_result)
    
    @pytest.mark.parametrize("values, labels", [
        ([[[1,2,3],[4,5,6],[7,8,9]],
          [[2,3,4],[5,6,7],[8,9,10]]],
         ['A', 'B'])
    ])
    def test_combineDataDict_exceptions(self, values, labels):
        """
        We test that an exception is raised when there are more
        values than labels.
        """
        with pytest.raises(Exception):
            pycgmIO.combineDataDict(values, labels)
        
    def test_make_sure_path_exists(self):
        """
        This function tests pycgmIO.make_sure_path_exists(path),
        where path is the path to create.

        This function creates a file path. We use a temporary
        directory for testing.
        """
        new_directory = os.path.join(self.tmp_dir_name, 'new_directory')
        pycgmIO.make_sure_path_exists(new_directory)
        assert os.path.isdir(new_directory)


        

    
