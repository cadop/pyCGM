import pytest
from pyCGM_Single.pyCGM_Helpers import getfilenames
import pyCGM_Single.pycgmIO as pycgmIO
import numpy as np
import os

class TestPycgmIO:
    @classmethod
    def setup_class(cls):
        """
        Called once for all tests for pycgmIO.
        Sets rounding_precision, and loads filenames to
        be used for testing load functions.
        """
        cls.rounding_precision = 8
        cwd = os.getcwd()
        if(cwd.split(os.sep)[-1]=="pyCGM_Single"):
            parent = os.path.dirname(cwd)
            os.chdir(parent)
        cls.cwd = os.getcwd()

        cls.filename_59993_Frame = os.path.join(cls.cwd, getfilenames(1)[1])
        cls.filename_Sample_Static = os.path.join(cls.cwd, 'SampleData/ROM/Sample_Static.csv')

    @pytest.mark.parametrize("labels, data, expected_result", [
        (
         #Tests lists
         ['A', 'B', 'C'],
         [[[1,2,3],[4,5,6],[7,8,9]],
          [[2,3,4],[5,6,7],[8,9,10]]],
         [{'A':[1,2,3], 'B':[4,5,6], 'C':[7,8,9]},
          {'A':[2,3,4], 'B':[5,6,7], 'C':[8,9,10]}]
        ),

        (
         #Tests numpy arrays
         ['A', 'B', 'C'],
         [[np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9])],
          [np.array([2,3,4]),np.array([5,6,7]),np.array([8,9,10])]],
         [{'A':np.array([1,2,3]), 'B':np.array([4,5,6]), 'C':np.array([7,8,9])},
          {'A':np.array([2,3,4]), 'B':np.array([5,6,7]), 'C':np.array([8,9,10])}]
        ),

        (
         ['A'],
         [[[1,2,3],[4,5,6],[7,8,9]],
          [[2,3,4],[5,6,7],[8,9,10]]],
         [{'A': [1, 2, 3]}, 
          {'A': [2, 3, 4]}]
        ),

        (
         ['A', 'B', 'C'], 
         [[[1,2,3],[4,5,6]],
          [[2,3,4]]],
         [{'A': [1, 2, 3], 'B': [4, 5, 6]}, 
          {'A': [2, 3, 4]}]
        )
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
        (
         [{'A': [1, 2, 3], 'B': [4, 5, 6]},
          {'A': [2, 3, 4], 'B': [5, 6, 7]}],
         ['A', 'B'],
         np.array([[[1, 2, 3],[4, 5, 6]],[[2, 3, 4],[5, 6, 7]]])
        ),
        (
         [{'A': np.array([1, 2, 3]), 'B': np.array([4, 5, 6])},
          {'A': np.array([2, 3, 4]), 'B': np.array([5, 6, 7])}],
         ['A', 'B'],
         np.array([[[1, 2, 3],[4, 5, 6]],[[2, 3, 4],[5, 6, 7]]])
        ),
        (
         [{'A': np.array([1, 2, 3]), 'B': np.array([4, 5, 6])},
          {'A': np.array([2, 3, 4])}],
         ['A', 'B'],
         np.array([[[1, 2, 3],[4, 5, 6]],[[2, 3, 4],[2, 3, 4]]])
        ),
        (
         [{'B': np.array([4, 5, 6])},
          {'A': np.array([2, 3, 4])}],
         ['B'],
         np.array([[[4, 5, 6]],[[2, 3, 4]]])
        )
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
        assert result_labels == expected_labels
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
        (
         ['MeanLegLength', 'LeftKneeWidth', 'RightAnkleWidth'], [940.0, 105.0, 70.0],
         {'MeanLegLength':940.0, 'LeftKneeWidth':105.0,'RightAnkleWidth':70.0}
        ),
        (
         ['A', 'B', 'C', 'D'], [1, 2, 3, 4],
         {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        ),
        (
         ['A', 'B', 'C', 'D'], np.array([1, 2, 3, 4]),
         {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        ),
        (
         ['A', 'B'], [1, 2, 3, 4, 5, 6],
         {'A': 1, 'B': 2}
        ),
        (
         ['A', 'B', 'C', 'D', 'E'], [1, 2],
         {'A': 1, 'B': 2}
        ),
        (
         [], [0, 1], {}
        ),
        (
         ['A', 'B', 'C'], [], {}
        ),
        (
         [], [], {}
        )
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
        (
         {'MeanLegLength':940.0, 'LeftKneeWidth':105.0,'RightAnkleWidth':70.0},
         ['MeanLegLength', 'LeftKneeWidth', 'RightAnkleWidth'],
         np.array([940., 105.,  70.])
        ),
        (
         {'A': 1, 'B': 2, 'C': 3, 'D': 4},
         ['A', 'B', 'C', 'D'], np.array([1, 2, 3, 4])
        ),
        (
         {}, [], np.array([])
        )
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
        (
         0, 'LFHD', '*113',
         np.array([60.1229744, 132.4755249, 1485.8293457]),
         np.array([-173.22341919,  166.87660217, 1273.29980469])
        ),
        (
         125, 'RASI', '*114',
         np.array([144.1030426, -0.36732361, 856.89855957]),
         np.array([ 169.75387573, -230.69139099, 1264.89257812])
        ),
        (
         2, 'LPSI', '*113',
         np.array([-94.89163208, 49.82866287, 922.64483643]),
         np.array([-172.94085693,  167.04344177, 1273.51000977])
        ),
        (
         12, 'LKNE', '*114',
         np.array([-100.0297699, 126.43037415, 414.15945435]),
         np.array([ 169.80422974, -226.73210144, 1264.15673828])
        ),
        (
         22, 'C7', '*113',
         np.array([-27.38780975, -8.35509396, 1301.37145996]),
         np.array([-170.55563354,  168.37162781, 1275.37451172])
        ),
        (
         302, 'RANK', '*114',
         np.array([52.61815643, -126.93238068, 58.56194305]),
         np.array([ 174.65007019, -225.9836731 , 1262.32373047])
        )
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
        (
         0, 'LFHD', '*111',
         np.array([ 174.5749207,  324.513031 , 1728.94397  ]),
         np.array([ 692.8970947,  423.9462585, 1240.289063 ])
        ),
        (
         125, 'RASI', '*112',
         np.array([ 353.3344727,  345.1920471, 1033.201172 ]),
         np.array([-225.5984955,  403.15448  , 1209.803467 ])
        ),
        (
         2, 'LPSI', '*113',
         np.array([ 191.5829468,  175.4567261, 1050.240356 ]),
         np.array([ -82.66962433,  232.2470093 , 1361.734741  ])
        ),
        (
         12, 'LKNE', '*114',
         np.array([ 88.88719177, 242.1836853 , 529.8156128 ]),
         np.array([ 568.6048584,  261.1444092, 1362.141968 ])
        ),
        (
         22, 'C7', '*112',
         np.array([ 251.1347656,  164.8985748, 1527.874634 ]),
         np.array([-225.2479401,  404.37146  , 1214.369141 ])
        ),
        (
         273, 'RANK', '*111',
         np.array([427.6519165 , 188.9484558 ,  93.37301636]),
         np.array([ 695.2038574,  421.2562866, 1239.632446 ])
        )
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
        (
         {'A': [[1, 2], [4, 5], [7, 8]], 'B': [[4, 5], [7, 8], [10, 11]]},
         [{'A': np.array([1, 4, 7]), 'B': np.array([ 4,  7, 10])}, 
          {'A': np.array([2, 5, 8]), 'B': np.array([ 5,  8, 11])}]
        ),
        (
         {'A': [np.array([1, 2]), np.array([4, 5]), np.array([7, 8])],
          'B': [np.array([4, 5]), np.array([7, 8]), np.array([10, 11])]},
         [{'A': np.array([1, 4, 7]), 'B': np.array([ 4,  7, 10])}, 
          {'A': np.array([2, 5, 8]), 'B': np.array([ 5,  8, 11])}]
        ),
        (
         {'A': [[1, 2], [4, 5], [7]]},
         [{'A': np.array([1, 4, 7])}]
        ),
        (
         {'A': [[2], [4], [6], [8]]},
         [{'A': np.array([2, 4, 6])}]
        ),
        (
         {'A': [[], [4, 5], [7, 8, 9]]},
         []
        )
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
        (
         [{'A': [1, 2, 3], 'B': [4, 5, 6]},
          {'A': [2, 3, 4], 'B': [5, 6, 7]}],
         {'A': [[1, 2, 3], [2, 3, 4]], 'B': [[4, 5, 6], [5, 6, 7]]}
        ),
        (
         [{'A': [1, 2], 'B': [4]},
          {'A': [4], 'B': []}],
         {'A': [[1, 2], [4]], 'B': [[4], []]}
        ),
        (
         [{'A': [1, 2]},
          {'A': [4, 5], 'B': [6, 7]}],
         {'A': [[1, 2], [4, 5]], 'B': [[6, 7]]}
        ),
        (
         [{'A': 2} , {'B': [6, 7]}],
         {'A': [2], 'B': [[6, 7]]}
        ),
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