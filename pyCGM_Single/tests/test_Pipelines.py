import pytest
import numpy as np
from pyCGM_Single.pyCGM_Helpers import getfilenames
from pyCGM_Single.pycgmIO import loadData, dataAsDict
from pyCGM_Single.clusterCalc import target_dict, segment_dict
from pyCGM_Single.pyCGM import pelvisJointCenter
import pyCGM_Single.Pipelines as Pipelines
import os

class TestPipelines:
    @classmethod
    def setup_class(cls):
        """
        Called once for all tests for Pipelines.
        Sets rounding_precision, and does a one-time load
        of dynamic and static trial data as dictionaries from
        SampleData/Sample_2/.
        """
        cls.rounding_precision = 8
        cwd = os.getcwd()
        if(cwd.split(os.sep)[-1]=="pyCGM_Single"):
            parent = os.path.dirname(cwd)
            os.chdir(parent)
        cls.cwd = os.getcwd()

        cur_dir = cls.cwd
        dynamic_trial, static_trial,_,_,_ = getfilenames(3)
        dynamic_trial = os.path.join(cur_dir, dynamic_trial)
        static_trial = os.path.join(cur_dir, static_trial)
        motionData = loadData(dynamic_trial)
        staticData = loadData(static_trial)
        cls.data_original = dataAsDict(motionData, npArray=True)
        cls.static_original = dataAsDict(staticData, npArray=True)
        for frame in motionData:
            frame['SACR'] = pelvisJointCenter(frame)[2]
        cls.data_with_sacrum_original = dataAsDict(motionData, npArray=True)

    def setup_method(self):
        """
        Runs before every test, resetting data and static
        to their original states.
        """
        self.data = self.data_original.copy()
        self.static = self.static_original.copy()
        self.data_with_sacrum = self.data_with_sacrum_original.copy()

    #Setup some inputs that are reused across tests
    valid_filter_data = np.array([[0,1], [1,2], [2,3], [3,4], 
                                  [4,5], [5,6], [6,7], [7,8], 
                                  [8,9],[9,10],[10,11],[11,12], 
                                  [12,13],[13,14],[14,15],[15,16]])

    invalid_filter_data = [[0], [1], [2], [3], [4], [5], [6], [7], [8], 
                           [9], [10], [11], [12], [13], [14], [15], [16]]

    #Test positive and negative floats and ints
    list_tests = [
            [-1003.58361816, -1003.50396729, -1003.42358398, -1003.3425293,
            -1003.26068115, -1003.17810059, -1003.09484863, -1003.01080322,
            -1002.92608643, -1002.84063721, -1002.75445557, -1002.6675415,
            -1002.57995605, -1002.49157715, -1002.40252686, -1002.31274414],
            [70, 76, 87, 52, 28, 36, 65, 95, 69, 62, 60, 17, 12, 97, 11, 48],
            [-70, 76, 87, -52, 28, 36, 65, 95, -69, 62, 60, 17, 12, -97, 11, 48]
    ]
    #Expected results are the same for both lists and numpy arrays
    expected_results = [
        np.array([-1003.58359202, -1003.50390817, -1003.42363665, -1003.34252947,
                -1003.26066539, -1003.17811173, -1003.09483484, -1003.01081474,
                -1002.92608178, -1002.8406421 , -1002.7544567 , -1002.66753023,
                -1002.57993488, -1002.49164743, -1002.40251014, -1002.31267097]),

        np.array([69.98542399, 84.63525626, 78.90545928, 52.35480234, 
                30.49005341, 37.66166803, 65.18623797, 84.35193788, 
                82.07291528, 64.17505539, 41.24155139, 28.10878834,
                33.98739195, 47.41539095, 52.2509855,  48.00079742]),

        np.array([-69.96417558, 62.8459697, 68.01436426, -1.36719047,
                -12.10645143, 48.64339139,  81.14625417,  41.74781614,
                1.18057508, 21.16904888, 57.9718148, 39.85271513, 
                -21.05573085, -53.81198321, -23.50929002, 47.94081346])
    ]
    @pytest.mark.parametrize("data, cutoff, Fs, expected_result", [
        (list_tests[0], 20, 120, expected_results[0]),
        (list_tests[1], 20, 120, expected_results[1]),
        (list_tests[2], 20, 120, expected_results[2]),
        (np.array(list_tests[0]), 20, 120, expected_results[0]),
        (np.array(list_tests[1]), 20, 120, expected_results[1]),
        (np.array(list_tests[2]), 20, 120, expected_results[2])
    ])
    def test_butterFilter_accuracy(self, data, cutoff, Fs, expected_result):
        """
        This function tests Pipelines.butterFilter(data, cutoff, Fs),
        where data is an array of numbers to filter, cutoff is the cutoff
        frequency to filter, and Fs is the sampling frequency of the data.

        We test cases where inputs are positive and negative floats and 
        integers. We test cases where inputs are lists and numpy arrays.
        """
        #Call butterFilter() with parametrized values
        result = Pipelines.butterFilter(data, cutoff, Fs)
        np.testing.assert_almost_equal(result, expected_result, self.rounding_precision)

    @pytest.mark.parametrize("data, cutoff, Fs", [
        #Test that if len(data) < 15, an exception is raised
        ([], 20, 120),
        (range(3), 20, 120),
        (range(15), 20, 120),
        (np.arange(0, 0.4, 0.1), 20, 120),
        #Test invalid values for cutoff frequency will raise exceptions
        (range(20), -10, 120),
        (range(20), 0, 120),
        (range(20), 40, 120),
        #Test invalid values for Fs, the sampling frequency will raise exceptions
        (range(20), 20, -10),
        (range(20), 20, 0),
        (range(20), 20, 20),
        (range(20), 20, 60)
    ])
    def test_butterFilter_exceptions(self, data, cutoff, Fs):
        """
        We test exceptions raised when the length of the input data is too short, 
        when the cutoff frequency value is negative, zero, or too large, 
        and when the sampling frequency value is negative, zero, or too small.
        """
        with pytest.raises(Exception):
            Pipelines.butterFilter(data, cutoff, Fs)
    
    @pytest.mark.parametrize("data, cutoff, Fs, expected_result", [
        (
            np.array([[-1003.58361816, 81.00761414, 1522.23693848], [-1003.50396729, 81.02921295, 1522.18493652],
                [-1003.42358398, 81.05059814, 1522.13598633], [-1003.3425293,  81.07178497, 1522.09020996],
                [-1003.26068115, 81.09275818, 1522.04760742], [-1003.17810059, 81.11352539, 1522.00805664],
                [-1003.09484863, 81.13407898, 1521.97167969], [-1003.01080322, 81.1544342,  1521.93835449],
                [-1002.92608643, 81.17457581, 1521.90820312], [-1002.84063721, 81.19451141, 1521.88122559],
                [-1002.75445557, 81.21424103, 1521.8572998 ], [-1002.6675415,  81.23376465, 1521.83654785],
                [-1002.57995605, 81.25308228, 1521.81884766], [-1002.49157715, 81.27219391, 1521.80419922],
                [-1002.40252686, 81.29109955, 1521.79284668], [-1002.31274414, 81.30980682, 1521.78442383]], dtype=np.float32),
            20, 120, 
            np.array([[-1003.58359202, 81.00761957, 1522.2369362 ], [-1003.50390817, 81.02919582, 1522.18514327],
                [-1003.42363665, 81.05060565, 1522.135889  ], [-1003.34252947, 81.07178881, 1522.0901444 ],
                [-1003.26066539, 81.09275628, 1522.04763945], [-1003.17811173, 81.11352192, 1522.00810375],
                [-1003.09483484, 81.13408185, 1521.97164458], [-1003.01081474, 81.15443324, 1521.93835939],
                [-1002.92608178, 81.17457741, 1521.90820438], [-1002.8406421 , 81.19451144, 1521.88119076],
                [-1002.7544567 , 81.21423685, 1521.85735721], [-1002.66753023, 81.23376495, 1521.83657557],
                [-1002.57993488, 81.25309204, 1521.81874533], [-1002.49164743, 81.27219203, 1521.80416229],
                [-1002.40251014, 81.29107514, 1521.79304586], [-1002.31267097, 81.30982637, 1521.78437862]]),
        ),
        (
            np.array([[0, 1], [1, 2], [2, -3], [3, 4],
                    [4, 5], [-5, 6], [6, 7], [-7, 8],
                    [8, 9], [9, 10], [10, 11], [11, 12],
                    [12, -13], [-13, 14], [14, 15], [-15, 16]], dtype=np.int32),
            20, 120,
            np.array([[-0.000935737255,  0.998712399], [ 1.01481885, -0.253654830],[ 2.33933854,  0.00948734868], [ 2.95259703,  2.13587233],
                        [ 1.99482481,  4.97199325], [ 0.123360866,  6.87747083], [-1.18427846,  7.06168539], [-0.287575584,  6.80213161],
                        [ 3.88087521,  8.89186494], [ 9.65108926,  12.49212252], [ 12.33700072,  11.15786714] ,[ 9.51590015,  3.47236703],
                        [ 5.42470382, -0.198189462], [ 3.62729076,  6.87305425], [-1.60886230,  15.2201909], [-14.9976228,  15.99966724]])
        )
    ])
    def test_filt_accuracy(self, data, cutoff, Fs, expected_result):
        """
        This function tests Pipelines.filt(data, cutoff, Fs),
        where data is a 2darray of numbers to filter, cutoff is the cutoff
        frequency to filter, and Fs is the sampling frequency of the data.

        We test cases where inputs are lists of positive and negative floats 
        and ints.
        """
        result = Pipelines.filt(data, cutoff, Fs)
        np.testing.assert_almost_equal(result, expected_result, self.rounding_precision)

    @pytest.mark.parametrize("data, cutoff, Fs", [
        #Test that if the shape of each array in data is not the same, an exception is raised
        (np.array([[0,1],[1,2],[2,3],[3],[4],[5],[6],[7], 
        [8],[9],[10],[11],[12],[13],[14],[15]]), 20, 120),
        #Test that data must be a numpy array
        (invalid_filter_data, 20, 120),
        #Test that len(data) must be > 15
        (np.array([range(15)]), 20, 120),
        (np.arange(0, 0.4, 0.1), 20, 120),
        #Test invalid values for the cutoff frequency
        (valid_filter_data, -10, 120),
        (valid_filter_data, 0, 120),
        (valid_filter_data, 40, 120),
        #Test invalid values for the sampling frequency
        (valid_filter_data, 20, -10),
        (valid_filter_data, 20, 0),
        (valid_filter_data, 20, 20),
        (valid_filter_data, 20, 60)
    ])
    def test_filt_exceptions(self, data, cutoff, Fs):
        """
        We test exceptions raised when the arrays in data are not of the same
        shape, when the length of data is too short, when the cutoff frequency 
        value is negative, zero, or too large, when the sampling frequency 
        value is negative, zero, or too small, and when data is not a numpy array.
        """
        with pytest.raises(Exception):
            Pipelines.filt(data, cutoff, Fs)
    
    @pytest.mark.parametrize("trajs, expected_result", [
        ({'trajOne': np.array([[217.19961548, -82.35484314, 332.2684021 ],
                            [257.19961548, -32.35484314, 382.2684021 ]])},
        [{'trajOne': np.array([217.19961548, -82.35484314, 332.2684021 ])}, 
        {'trajOne': np.array([257.19961548, -32.35484314, 382.2684021 ])}]),
        ({'a': np.array([[1, 2, 3], [4, 5, 6]]), 
        'b': np.array([[7, 8, 9], [10, 11, 12]])},
        [{'a': np.array([1, 2, 3]), 'b': np.array([7, 8, 9])}, 
        {'a': np.array([4, 5, 6]), 'b': np.array([10, 11, 12])}])
    ])
    def test_prep_accuracy(self, trajs, expected_result):
        """
        This function tests Pipelines.prep(trajs), where trajs
        is a dictionary containing numpy arrays.
        """
        result = Pipelines.prep(trajs)
        np.testing.assert_equal(result, expected_result)

    @pytest.mark.parametrize("trajs", [
        #Test that if data[key] is not a numpy array, an exception is raised
        ({'trajOne': [[217.19961548, -82.35484314, 332.2684021 ],
                    [257.19961548, -32.35484314, 382.2684021 ]]})
    ])
    def test_prep_exceptions(self, trajs):
        with pytest.raises(Exception):
            Pipelines.prep(trajs)
    
    @pytest.mark.parametrize("data, name, expected_result", [
        (
        [{'LTIL': np.array([-268.1545105, 327.53512573,  30.17036057]),
        'RFOP': np.array([-38.4509964, -148.6839447 ,  59.21961594])},
        {'LTIL': np.array([-273.1545105, 324.53512573,  36.17036057]),
        'RFOP': np.array([-38.4509964, -148.6839447 ,  59.21961594])}],

        'LTIL',

        [{'LTIL': np.array([np.nan, np.nan, np.nan]), 
        'RFOP': np.array([-38.4509964, -148.6839447, 59.21961594])}, 
        {'LTIL': np.array([np.nan, np.nan, np.nan]), 
        'RFOP': np.array([-38.4509964, -148.6839447, 59.21961594])}]
        ),
        (
        #Test that data[key] can be a list
        [{'LTIL': [-268.1545105, 327.53512573, 30.17036057],
        'RFOP': [-38.4509964, -148.6839447 , 59.21961594]},
        {'LTIL': [-273.1545105, 324.53512573, 36.17036057],
        'RFOP': [-38.4509964, -148.6839447 , 59.21961594]}],

        'LTIL',

        [{'LTIL': np.array([np.nan, np.nan, np.nan]), 
        'RFOP': np.array([-38.4509964, -148.6839447, 59.21961594])}, 
        {'LTIL': np.array([np.nan, np.nan, np.nan]), 
        'RFOP': np.array([-38.4509964, -148.6839447, 59.21961594])}]
        ),
        (
        #Test removing a non-existent marker
        [{'LTIL': np.array([-268.1545105, 327.53512573, 30.17036057])}],

        'non_existent_marker',

        #removing a non-existent marker adds it as a key
        [{'LTIL': np.array([-268.1545105, 327.53512573, 30.17036057]),
        'non_existent_marker': np.array([np.nan, np.nan, np.nan])}]
        )
    ])
    def test_clearMarker_accuracy(self, data, name, expected_result):
        """
        This function tests Pipelines.clearMarker(data, name), where
        data is an array of dictionary of markers lists and name is the 
        name of a marker to clear.

        We test cases where the values in data are lists and numpy arrays.
        We test the case where the marker name is not a key in data.
        """
        result = Pipelines.clearMarker(data, name)
        np.testing.assert_equal(result, expected_result)

    @pytest.mark.parametrize("Data, expected_result", [
        (
        #Test numpy arrays of positive and negative floats and ints
        {'LFHD': np.array([[-1003.58361816, 81.00761414, 1522.23693848], [-1003.50396729, 81.02921295, 1522.18493652],
                        [-1003.42358398, 81.05059814, 1522.13598633], [-1003.3425293,  81.07178497, 1522.09020996],
                        [-1003.26068115, 81.09275818, 1522.04760742], [-1003.17810059, 81.11352539, 1522.00805664],
                        [-1003.09484863, 81.13407898, 1521.97167969], [-1003.01080322, 81.1544342,  1521.93835449],
                        [-1002.92608643, 81.17457581, 1521.90820312], [-1002.84063721, 81.19451141, 1521.88122559],
                        [-1002.75445557, 81.21424103, 1521.8572998 ], [-1002.6675415,  81.23376465, 1521.83654785],
                        [-1002.57995605, 81.25308228, 1521.81884766], [-1002.49157715, 81.27219391, 1521.80419922],
                        [-1002.40252686, 81.29109955, 1521.79284668], [-1002.31274414, 81.30980682, 1521.78442383]], dtype=np.float32)
        },
        {
        'LFHD': np.array([[-1003.58359202, 81.00761957, 1522.2369362 ], [-1003.50390817, 81.02919582, 1522.18514327],
                            [-1003.42363665, 81.05060565, 1522.135889  ], [-1003.34252947, 81.07178881, 1522.0901444 ],
                            [-1003.26066539, 81.09275628, 1522.04763945], [-1003.17811173, 81.11352192, 1522.00810375],
                            [-1003.09483484, 81.13408185, 1521.97164458], [-1003.01081474, 81.15443324, 1521.93835939],
                            [-1002.92608178, 81.17457741, 1521.90820438], [-1002.8406421 , 81.19451144, 1521.88119076],
                            [-1002.7544567 , 81.21423685, 1521.85735721], [-1002.66753023, 81.23376495, 1521.83657557],
                            [-1002.57993488, 81.25309204, 1521.81874533], [-1002.49164743, 81.27219203, 1521.80416229],
                            [-1002.40251014, 81.29107514, 1521.79304586], [-1002.31267097, 81.30982637, 1521.78437862]])
        }    
        ),
        (
        {'LFHD': valid_filter_data}, #[[0,1], [1,2], ... [15, 16]]
        {
        'LFHD': np.array([[0.000290878541, 1.00029088], [1.00001659, 2.00001659],
                        [1.99986579, 2.99986579], [3.00000047, 4.00000047],
                        [4.00006844, 5.00006844], [4.99998738, 5.99998738],
                        [5.99995125, 6.99995125], [7.00003188, 8.00003188],
                        [8.00006104, 9.00006104], [8.99992598, 9.99992598],
                        [9.99988808, 10.99988808], [11.0001711, 12.0001711],
                        [12.00023202, 13.00023202], [12.99960463, 13.99960463],
                        [13.99950627, 14.99950627], [15.00091227, 16.00091227]])
        }
        )
    ])
    def test_filtering_accuracy(self, Data, expected_result):
        """
        The following functions test Pipelines.filtering(Data), where
        Data is a dictionary of marker lists. This function calls
        Pipelines.filt().

        We test cases where inputs are numpy arrays of positive and 
        negative floats and ints.
        """
        result = Pipelines.filtering(Data)
        for key in Data:
            np.testing.assert_almost_equal(result[key], expected_result[key], self.rounding_precision)

    @pytest.mark.parametrize("Data", [
        #Test that if data is not a numpy array, an exception is raised
        ({'LFHD':invalid_filter_data})
    ])
    def test_filtering_exceptions(self, Data):
        with pytest.raises(Exception):
            Pipelines.filtering(Data)

    @pytest.mark.parametrize("key, useables, s, expected_result", [
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 0, 
        [-1007.8145678233541, 71.28465078977477, 1522.6626006179151]),
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 1, 
        [-1007.7357797476452, 71.30567599088612, 1522.6056345492811]),
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 2, 
        [-1007.6561772477821, 71.32644261551039, 1522.5516787767372]),
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], -1, 
        [710.8111428914814, -18.282265916438064, 1549.7284035675332]),
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 10, 
        [-1006.9916284913861, 71.48482387826286, 1522.2367625952083]),
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 100, 
        [-995.8183141045178, 73.11905329024174, 1526.9072499889455]),
    ])
    def test_transform_from_static_accuracy(self, key, useables, s, expected_result):
        """
        This function tests Pipelines.transform_from_static(data,static,key,useables,s),
        where data is an array of dictionaries of marker data,
        static is an array of static marker data, key is the name of the
        missing marker to perform gap filling for,
        useables is a list of markers in the same cluster as key,
        and s is the frame number that the marker data is missing for.

        This function performs gap filling based on static data.

        We use files from SampleData/Sample_2/ for testing.
        We test for 6 missing frames from the loaded data.
        """
        data = self.data
        static = self.static
        #Clear given key and frame to test gap filling
        data[key][s] = np.array([np.nan, np.nan, np.nan])
        result = Pipelines.transform_from_static(data,static,key,useables,s)
        np.testing.assert_almost_equal(result, expected_result, self.rounding_precision)

    @pytest.mark.parametrize("key, useables, s", [
        #Test that if useables is not at least 3 unique markers, an exception is raised
        ('LFHD', ['RFHD', 'RFHD', 'RFHD'], 0),
        ('LFHD', ['RFHD'], 0),
        ('LFHD', [], 0),
        #Test that if s, the frame number, is out of range, an exception is raised
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 6100),
        #Test that if the marker name does not exist, an exception is raised
        ('InvalidKey', ['RFHD', 'RBHD', 'LBHD'], 0)
    ])
    def test_transform_from_static_exceptions(self, key, useables, s):
        """
        We test exceptions raised when there are not enough usable markers,
        the frame number is out of range, and the marker name does not exist.
        """
        data = self.data 
        static = self.static
        with pytest.raises(Exception):
            Pipelines.transform_from_static(data,static,key,useables,s)
    
    @pytest.mark.parametrize("key,clust,last_time,i,expected_result", [
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 3, 11,
        [-1002.66354241,    81.22543097,  1521.82434027]),
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 3, 12,
        [-1002.57542092,    81.24378237,  1521.80517086]),
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 3, -1,
        [714.4191660275219, -8.268045936969543, 1550.088229312965]),
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 3, 15,
        [-1002.30681304,    81.29768863,  1521.76708531]),
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 3, 100,
        [-991.7315609567293, 82.91868701883672, 1526.597213251877]),
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 15, 16,
        [np.nan, np.nan, np.nan])
    ])
    def test_transform_from_mov_accuracy(self,key,clust,last_time,i,expected_result):
        """
        This function tests Pipelines.transform_from_mov(data,key,clust,last_time,i),
        where data is an array of dictionaries of marker data,
        key is the name of the missing marker to perform gap filling for,
        clust is a list of markers in the same cluster as key,
        last_time is the last frame the the missing marker was visible, and
        i is the frame number the marker is missing for.

        This function performs gap filling based on surrounding data.

        We use files from SampleData/Sample_2/ for testing.
        We test for 6 missing frames from the loaded data.
        """
        data = self.data
        data[key][i] = np.array([np.nan, np.nan, np.nan])
        result = Pipelines.transform_from_mov(data,key,clust,last_time,i)
        np.testing.assert_almost_equal(result, expected_result, self.rounding_precision)

    @pytest.mark.parametrize("key,clust,last_time,i", [
        #Test that if clust is not at least 3 unique markers, an exception is raised
        ('LFHD', ['RFHD', 'RFHD', 'RFHD'], 0, 1),
        ('LFHD', ['RFHD'], 0, 1),
        ('LFHD', [], 0, 1),
        #Test that if i, the frame number, it out of range, an exception is raised
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 0, 6100),
        #Test that if the marker name does not exist, an exception is raised
        ('InvalidKey', ['RFHD', 'RBHD', 'LBHD'], 0, 1)
    ])
    def test_transform_from_mov_exceptions(self,key,clust,last_time,i):
        """
        We test exceptions raised when there are not enough usable markers,
        the frame number is out of range, and the marker name does not exist.
        """
        data = self.data
        with pytest.raises(Exception):
            Pipelines.transform_from_mov(data,key,clust,last_time,i)
    
    target_dict = target_dict()
    segment_dict = segment_dict()
    @pytest.mark.parametrize("key,target_dict,segment_dict,j,missings,expected_result", [
        ('LFHD',target_dict,segment_dict,10,{},
            ['RFHD', 'RBHD', 'LBHD']),
        ('LFHD',target_dict,segment_dict,10,{'LFHD':[]},
            ['RFHD', 'RBHD', 'LBHD']),
        ('LFHD',target_dict,segment_dict,10,{'RFHD':[10]},
            ['RBHD', 'LBHD']),
        ('LFHD',target_dict,segment_dict,10,{'LBHD':[10], 'RFHD':[10]},
            ['RBHD']),
        ('LFHD',target_dict,segment_dict,10,{'LBHD':[10], 'RFHD':[10], 'RBHD':[10]},
            []),
        ('C7',target_dict,segment_dict,10,{},
            ['STRN', 'CLAV', 'T10', 'RBAK', 'RSHO', 'LSHO']),
        ('RPSI',target_dict,segment_dict,10,{},
            ['LPSI', 'LASI', 'RASI']),
        ('LKNE',target_dict,segment_dict,10,{},
            ['LTHI'])
    ])
    def test_segmentFinder_accuracy(self,key,target_dict,segment_dict,j,missings,expected_result):
        """
        This function tests Pipelines.segmentFinder(key,data,target_dict,segment_dict,j,missings),
        where data is an array of dictionaries of marker data,
        key is the name of the missing marker to find the segment for,
        target_dict is a dictionary of marker to segment,
        segment_dict is a dictionary of segments to marker names,
        j is the frame number the marker is missing for, 
        and missings is a dictionary indicating other missing markers.

        We test to ensure that missing markers are not used to reconstruct
        other missing markers. 
        """
        data = self.data
        result = Pipelines.segmentFinder(key,data,target_dict,segment_dict,j,missings)
        assert result == expected_result

    @pytest.mark.parametrize("key,target_dict,segment_dict,j,missings", [
        #Test that the marker name must exist
        ('InvalidKey',target_dict,segment_dict,10,{})
    ])
    def test_segmentFinder_exceptions(self,key,target_dict,segment_dict,j,missings):
        """
        We test the exception that is raised when the marker name does not
        exist.
        """
        data = self.data
        with pytest.raises(Exception):
            Pipelines.segmentFinder(key,data,target_dict,segment_dict,j,missings)
    
    @pytest.mark.parametrize("key, frameNumber, expected_result", [
        ('LFHD', 1, [-1007.73577975, 71.30567599, 1522.60563455]),
        ('LFHD', 10, [-1002.75400789, 81.21320267, 1521.8559697 ]),
        ('LFHD', 12, [-1002.57942155, 81.2521139, 1521.81737938]),
        ('LFHD', 15, [-1002.31227389, 81.30886597, 1521.78327027]),
        ('LFHD', 100, [-991.70150174, 82.79915329, 1526.67335699]),
        ('LFHD', -1, [ 714.9015343, -9.04757279, 1550.23346378]),
    ])
    def test_rigid_fill_accuracy(self, key, frameNumber, expected_result):
        """
        This function tests Pipelines.rigid_fill(Data, static),
        where Data is an array of dictionaries of marker data,
        and static is an array of dictionaries of static trial data.

        This function fills gaps for frames with missing data.

        We simulate missing data by clearing data of the given key at
        the given frame number and then test the gap filling result.
        """
        #Sacrum marker is required to use rigid_fill
        data = self.data_with_sacrum
        static = self.static
        #Clear the given frame number to test gap filling
        data[key][frameNumber] = np.array([np.nan, np.nan, np.nan])
        #Call Pipelines.rigid_fill()
        rigid_fill_data = Pipelines.rigid_fill(data, static)
        #Test that the missing frame was filled in accurately
        result = rigid_fill_data[key][frameNumber]
        np.testing.assert_almost_equal(result, expected_result, self.rounding_precision)

    def test_rigid_fill_exceptions(self):
        """
        We test the exception raised when the marker SACR does not
        exist in Data.
        """
        #Test that the marker SACR must exist:
        with pytest.raises(Exception):
            data = self.data
            static = self.static
            Pipelines.rigid_fill(data, static)
    