import pytest
import numpy as np

from refactor.pycgm import CGM
from refactor.io import IO
import refactor.prep as prep
import os
import mock


class TestPrepFiltering:
    @classmethod
    def setup_class(self):
        """
        Called once for all tests in TestPrepFiltering.
        Sets rounding precision to 8 decimal places.
        """
        self.rounding_precision = 8

    #Setup some inputs that are reused across tests
    valid_filter_data = np.array([[0,1], [1,2], [2,3], [3,4], 
                                  [4,5], [5,6], [6,7], [7,8], 
                                  [8,9],[9,10],[10,11],[11,12], 
                                  [12,13],[13,14],[14,15],[15,16]])

    invalid_filter_data = [[0], [1], [2], [3], [4], [5], [6], [7], [8], 
                           [9], [10], [11], [12], [13], [14], [15], [16]]
    
    #Test positive and negative floats and ints in butter_filter
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
    @pytest.mark.parametrize("data, cutoff_frequency, sampling_frequency, expected", [
        (list_tests[0], 20, 120, expected_results[0]),
        (list_tests[1], 20, 120, expected_results[1]),
        (list_tests[2], 20, 120, expected_results[2]),
        (np.array(list_tests[0]), 20, 120, expected_results[0]),
        (np.array(list_tests[1]), 20, 120, expected_results[1]),
        (np.array(list_tests[2]), 20, 120, expected_results[2])
    ])
    def test_butter_filter_accuracy(self, data, cutoff_frequency, sampling_frequency, expected):
        """
        This function tests prep.butter_filter(data, cutoff_frequency, sampling_frequency),
        where data is an array of numbers to filter, cutoff_frequency is the cutoff
        frequency to filter, and sampling_frequency is the sampling frequency of the data.

        prep.butter_filter applies a fourth-order low-pass Butterworth filter
        that is constructed using the scipy functions butter() and filtfilt().

        For each case, we test that the Butterworth filter is applied correctly
        by testing its output is accurate in cases where inputs are positive and negative 
        floats and integers. We test cases where inputs are lists and numpy arrays.
        """
        result = prep.butter_filter(data, cutoff_frequency, sampling_frequency)
        np.testing.assert_almost_equal(result, expected, self.rounding_precision)
    
    @pytest.mark.parametrize("data, cutoff_frequency, sampling_frequency", [
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
    def test_butter_filter_exceptions(self, data, cutoff_frequency, sampling_frequency):
        """
        We test exceptions raised when the length of the input data is too short, 
        when the cutoff frequency value is negative, zero, or too large, 
        and when the sampling frequency value is negative, zero, or too small.
        """
        with pytest.raises(Exception):
            prep.butter_filter(data, cutoff_frequency, sampling_frequency)
    
    @pytest.mark.parametrize("data, cutoff_frequency, sampling_frequency, expected", [
        (np.array([[-1003.58361816, 81.00761414, 1522.23693848], [-1003.50396729, 81.02921295, 1522.18493652],
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
                [-1002.40251014, 81.29107514, 1521.79304586], [-1002.31267097, 81.30982637, 1521.78437862]])),
        (np.array([[0, 1], [1, 2], [2, -3], [3, 4],
                    [4, 5], [-5, 6], [6, 7], [-7, 8],
                    [8, 9], [9, 10], [10, 11], [11, 12],
                    [12, -13], [-13, 14], [14, 15], [-15, 16]], dtype=np.int32),
         20, 120,
        np.array([[-0.000935737255,  0.998712399], [ 1.01481885, -0.253654830],[ 2.33933854,  0.00948734868], [ 2.95259703,  2.13587233],
                        [ 1.99482481,  4.97199325], [ 0.123360866,  6.87747083], [-1.18427846,  7.06168539], [-0.287575584,  6.80213161],
                        [ 3.88087521,  8.89186494], [ 9.65108926,  12.49212252], [ 12.33700072,  11.15786714] ,[ 9.51590015,  3.47236703],
                        [ 5.42470382, -0.198189462], [ 3.62729076,  6.87305425], [-1.60886230,  15.2201909], [-14.9976228,  15.99966724]]))
    ])
    def test_filt_accuracy(self, data, cutoff_frequency, sampling_frequency, expected):
        """
        This function tests prep.filt(data, cutoff_frequency, sampling_frequency),
        where data is a 2darray of numbers to filter, cutoff_frequency is the cutoff
        frequency to filter, and sampling_frequency is the sampling frequency of the data.

        prep.filt() uses the same fourth-order Butterworth filter constructed
        in Pipelines.butterFilter().

        We test cases where inputs are lists of positive and negative floats 
        and ints.
        """
        result = prep.filt(data, cutoff_frequency, sampling_frequency)
        np.testing.assert_almost_equal(result, expected, self.rounding_precision)
    
    @pytest.mark.parametrize("data, cutoff_frequency, sampling_frequency", [
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
    def test_filt_exceptions(self, data, cutoff_frequency, sampling_frequency):
        """
        We test exceptions raised when the arrays in data are not of the same
        shape, when the length of data is too short, when the cutoff frequency 
        value is negative, zero, or too large, when the sampling frequency 
        value is negative, zero, or too small, and when data is not a numpy array.
        """
        with pytest.raises(Exception):
            prep.filt(data, cutoff_frequency, sampling_frequency)
    
    @pytest.mark.parametrize("data, cutoff_frequency, sampling_frequency, expected", [
        (np.array([[[-1003.58361816, 81.00761414, 1522.23693848]], [[-1003.50396729, 81.02921295, 1522.18493652]],
                   [[-1003.42358398, 81.05059814, 1522.13598633]], [[-1003.3425293,  81.07178497, 1522.09020996]],
                   [[-1003.26068115, 81.09275818, 1522.04760742]], [[-1003.17810059, 81.11352539, 1522.00805664]],
                   [[-1003.09484863, 81.13407898, 1521.97167969]], [[-1003.01080322, 81.1544342,  1521.93835449]],
                   [[-1002.92608643, 81.17457581, 1521.90820312]], [[-1002.84063721, 81.19451141, 1521.88122559]],
                   [[-1002.75445557, 81.21424103, 1521.8572998 ]], [[-1002.6675415,  81.23376465, 1521.83654785]],
                   [[-1002.57995605, 81.25308228, 1521.81884766]], [[-1002.49157715, 81.27219391, 1521.80419922]],
                   [[-1002.40252686, 81.29109955, 1521.79284668]], [[-1002.31274414, 81.30980682, 1521.78442383]]], dtype=np.float32),
         20, 120,
         np.array([[[-1003.58359202, 81.00761957, 1522.2369362 ]], [[-1003.50390817, 81.02919582, 1522.18514327]],
                    [[-1003.42363665, 81.05060565, 1522.135889  ]], [[-1003.34252947, 81.07178881, 1522.0901444 ]],
                    [[-1003.26066539, 81.09275628, 1522.04763945]], [[-1003.17811173, 81.11352192, 1522.00810375]],
                    [[-1003.09483484, 81.13408185, 1521.97164458]], [[-1003.01081474, 81.15443324, 1521.93835939]],
                    [[-1002.92608178, 81.17457741, 1521.90820438]], [[-1002.8406421 , 81.19451144, 1521.88119076]],
                    [[-1002.7544567 , 81.21423685, 1521.85735721]], [[-1002.66753023, 81.23376495, 1521.83657557]],
                    [[-1002.57993488, 81.25309204, 1521.81874533]], [[-1002.49164743, 81.27219203, 1521.80416229]],
                    [[-1002.40251014, 81.29107514, 1521.79304586]], [[-1002.31267097, 81.30982637, 1521.78437862]]])),
        (np.array([[[0,1]],   [[1,2]],   [[2,3]],   [[3,4]], 
                   [[4,5]],   [[5,6]],   [[6,7]],   [[7,8]], 
                   [[8,9]],   [[9,10]],  [[10,11]], [[11,12]], 
                   [[12,13]], [[13,14]], [[14,15]], [[15,16]]]),
         20, 120,
         np.array([[[0.000290878541, 1.00029088]], [[1.00001659, 2.00001659]],
                   [[1.99986579, 2.99986579]], [[3.00000047, 4.00000047]],
                   [[4.00006844, 5.00006844]], [[4.99998738, 5.99998738]],
                   [[5.99995125, 6.99995125]], [[7.00003188, 8.00003188]],
                   [[8.00006104, 9.00006104]], [[8.99992598, 9.99992598]],
                   [[9.99988808, 10.99988808]], [[11.0001711, 12.0001711]],
                   [[12.00023202, 13.00023202]], [[12.99960463, 13.99960463]],
                   [[13.99950627, 14.99950627]], [[15.00091227, 16.00091227]]]))
    ])
    def test_filtering_accuracy(self, data, cutoff_frequency, sampling_frequency, expected):
        """
        This function tests prep.filtering(data, cutoff_frequency, sampling_frequency), where
        data is a 3d numpy array of data to filter, cutoff_frequency is the desired cutoff
        frequency, and sampling_frequency is the sampling frequency of the data.

        We test cases where inputs are numpy arrays of positive and negative floats
        and ints.
        """
        result = prep.filtering(data, cutoff_frequency, sampling_frequency)
        np.testing.assert_almost_equal(result, expected, self.rounding_precision)
    
    @pytest.mark.parametrize("data, cutoff_frequency, sampling_frequency", [
        #Test that if data is not a numpy array, an exception is raised
        ([[[0,1]],   [[1,2]],   [[2,3]],   [[3,4]], 
         [[4,5]],    [[5,6]],   [[6,7]],   [[7,8]], 
         [[8,9]],    [[9,10]],  [[10,11]], [[11,12]], 
         [[12,13]],  [[13,14]], [[14,15]], [[15,16]]],20,120)
    ])
    def test_filtering_exceptions(self, data, cutoff_frequency, sampling_frequency):
        with pytest.raises(Exception):
            prep.filtering(data, cutoff_frequency, sampling_frequency)

class TestPrepGapFilling:
    @classmethod
    def setup_class(self):
        """
        Called once for all tests in TestPrepGapFilling.
        Sets rounding precision, and does a one-time load of
        dynamic and static trial data from
        SampleData/Sample_2/.
        """
        self.rounding_precision = 8
        self.cwd = os.getcwd()

        dynamic_trial = 'SampleData/Sample_2/RoboWalk.c3d'
        static_trial = 'SampleData/Sample_2/RoboStatic.c3d'
        self.data_original, self.data_mapping_original = IO.load_marker_data(dynamic_trial)
        self.static_original, self.static_mapping_original = IO.load_marker_data(static_trial)
        self.segment_dict = prep.default_segment_dict()
    
    def setup_method(self):
        """
        Runs before every test, resetting dynamid and static
        trial data to their orignal states.
        """
        self.data = self.data_original.copy()
        self.data_mapping = self.data_mapping_original.copy()
        self.static = self.static_original.copy()
        self.static_mapping = self.static_mapping_original.copy()
    
    @pytest.mark.parametrize("pm, c, expected", [
        ([420.53975659, 30.76040902, 555.49711768],
         [np.array([-343.59864907, 238.25329134, -755.16883877]), 
          np.array([8.1286508, 495.13257337, 885.7371809]), 
          np.array([384.38897987, 741.88310889, 289.56653492])],
          [-550.7935827 ,  773.66338285, -330.72688418]),
        ([290.67647141, -887.27170397, -928.18965884],
         [np.array([975.77145761, 169.07228161, 714.73898307]), 
          np.array([34.79840373, 437.98858319, 342.44994367]), 
          np.array([386.38290967, 714.0373601, -254.71890944])],
          [1198.80575306,  683.68559959, 1716.24804919]),
        ([451.73055418, 25.29186874, 212.82059603],
         [np.array([750.37513208, 777.29644972, 814.25477338]), 
          np.array([785.58183092, 45.27606372, 228.32835519]), 
          np.array([-251.24340957, 71.99704479, -70.78517678])],
         [800.20530287, 543.8441708 , 356.96134002]),
        (np.array([198.67934839, 617.12145922, -942.60245177]),
         [np.array([-888.79518579, 677.00555294, 580.34056878]), 
          np.array([-746.7046053, 365.85692077, 964.74398363]), 
          np.array([488.51200254, 242.19485233, -875.4405979])],
          [-986.86909809, -247.71320943, -44.9278578]),
        (np.array([151.45958988, 228.60024976, 571.69254842]),
         [np.array([329.65958402, 338.27760766, 893.98328401]), 
          np.array([185.96009811, 933.21745694, 203.23381269]), 
          np.array([370.92610191, 763.93031647, -624.83623717])],
         [-213.42223861,  284.87178829,  486.68789429])
    ])
    def test_get_marker_location(self, pm, c, expected):
        """
        This function tests prep.get_marker_location(pm, c), where
        pm is a 1x3 list or array of the location of the missing marker
        in the cluster frame, and c is 2d array or list indicating 
        locations of other markers in the same cluster and same frame 
        as the missing marker.

        This test ensures get_marker_location is accurate with lists, 
        numpy arrays, negative numbers and floats.
        """
        result = prep.get_marker_location(pm, c)
        np.testing.assert_almost_equal(result, expected, self.rounding_precision)

    @pytest.mark.parametrize("p, c, expected", [
        ([61.25716038, 819.60483461, 871.28563964],
         [np.array([-109.75683574, -703.39208609, 23.40503888]), 
          np.array([8.1286508, 495.13257337, 885.7371809]), 
          np.array([384.38897987, 741.88310889, 289.56653492])],
         [1391.11399871,  396.16136313,   77.71905097]),
        ([-108.77877024, 164.72037283, -487.34574257], 
         [np.array([-840.15274045, -477.4003232, 989.63441835]), 
          np.array([34.79840373, 437.98858319, 342.44994367]), 
          np.array([386.38290967, 714.0373601, -254.71890944])],
         [-3890.74300865,  5496.20861951,    81.25476604]),
        ([172.60504672, 189.51963254, 714.76733718],
         [np.array([100.28243274, -308.77342489, 823.62871217]), 
          np.array([785.58183092, 45.27606372, 228.32835519]), 
          np.array([-251.24340957, 71.99704479, -70.78517678])],
          [233.03998819, 154.49967147, 395.87517602]),
        (np.array([606.02393735, 905.3131133, -759.04662559]),
         [np.array([144.30930144, -843.7618657, 105.12793356]), 
          np.array([-746.7046053, 365.85692077, 964.74398363]), 
          np.array([488.51200254, 242.19485233, -875.4405979])],
         [ 435.34624558, 1905.72175012, -305.53267563]),
        (np.array([-973.61392617, 246.51405629, 558.66195333]),
         [np.array([763.69368715, 709.90434444, 650.91067694]), 
          np.array([185.96009811, 933.21745694, 203.23381269]), 
          np.array([370.92610191, 763.93031647, -624.83623717])],
         [ 2083.22592722, -1072.83708766,  1139.02963846])
    ])
    def test_get_static_transform(self, p, c, expected):
        """
        This function tests prep.get_static_transform(p, c), where
        p is a 1x3 list or array of the location of the mssing marker
        at a previously visible frame, and c is a 2d array or list indicating
        locations of other markers in the same cluster as the missing marker.

        This test ensures test_get_static_transform is accurate with lists, 
        numpy arrays, negative numbers and floats.
        """
        result = prep.get_static_transform(p, c)
        np.testing.assert_almost_equal(result, expected, self.rounding_precision)

    @pytest.mark.parametrize("key, useables, s, expected", [
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
    def test_transform_from_static_sample_data(self, key, useables, s, expected):
        """
        This function tests prep.transform_from_static(), which takes in dynamic and
        static trial data and mappings, the name of the missing marker (key), useables, which
        is the other markers in the same cluster as key, and s, which is the frame number the
        marker data is missing for.

        We use files from SampleData/Sample_2/ for testing.
        We test for 6 missing frames from the loaded data.
        """
        data = self.data
        data_mapping = self.data_mapping
        static = self.static
        static_mapping = self.static_mapping
        #Clear given key and frame to test gap filling
        data[s][data_mapping[key]] = np.array([np.nan, np.nan, np.nan])
        result = prep.transform_from_static(data, data_mapping, static, static_mapping, key, useables, s)
        np.testing.assert_almost_equal(result, expected, self.rounding_precision)
    
    def test_transform_from_static(self):
        """
        This function tests prep.transform_from_static().

        prep.transform_from_static() uses a transformation that is stored between a 4 marker cluster.  
        It requires an inverse transformation matrix to be stored between the 
        combination of 3 marker groupings, with the 4th marker stored in relation to that frame. 
        
        prep.transform_from_static() takes in as input the missing marker, and uses static data to 
        create an inverse transformation matrix, multiplying the new frame by the stored 
        inverse transform to get the missing marker position.

        We test that prep.transform_from_mov() is accurate by creating four markers
        in a square in static data, and adding one to all but one of their x-coordinates
        for motion data.

        We ensure that the correct position is reconstructed from the static 
        data.
        """
        #Define the four markers, A, B, C, D, as the four corners of a square.
        static = np.array([[[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]]])

        #Define static mapping
        static_mapping = {
            'A':0, 'B':1, 'C':2, 'D':3
        }

        #Use the three markers B, C, D to reconstruct A
        useables = ['B', 'C', 'D']

        #Add one to all of the x-coordinates of B, C, D
        #Set A to nan
        data = np.array([[[np.nan, np.nan, np.nan], 
                          [0, 1, 0], [2, -1, 0], [0, -1, 0]]])
        
        #Define data mapping
        data_mapping =  {
            'A':0, 'B':1, 'C':2, 'D':3
        }

        #We expect that A is at [2, 1, 0]
        expected = np.array([2, 1, 0])
        result = prep.transform_from_static(data, data_mapping, static, static_mapping, 'A', useables, 0)
        np.testing.assert_equal(result, expected)
    
    @pytest.mark.parametrize("key, useables, s", [
        #Test that if useables is not at least 3 unique markers, an exception is raised
        ('LFHD', ['RFHD', 'RFHD'], 0),
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
        the frame number is out of range, or the marker name does not exist.
        """
        data = self.data
        data_mapping = self.data_mapping
        static = self.static
        static_mapping = self.static_mapping
        with pytest.raises(Exception):
            prep.transform_from_static(data, data_mapping, static, static_mapping, key, useables, s)
    
    @pytest.mark.parametrize("key,clust,last_time,i,expected", [
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
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 15, 15,
        [np.nan, np.nan, np.nan])
    ])
    def test_transform_from_mov_sample_data(self, key, clust, last_time, i, expected):
        """
        This function tests prep.transform_from_mov(), which takes in dynamic and
        static trial data and mappings, the name of the missing marker (key), clust, which
        is the other markers in the same cluster as key, and i, which is the frame number the
        marker data is missing for.

        We use files from SampleData/Sample_2/ for testing.
        We test for 6 missing frames from the loaded data.
        """
        data = self.data
        data_mapping = self.data_mapping
        #Clear given key and frame to test gap filling
        data[i][data_mapping[key]] = np.array([np.nan, np.nan, np.nan])
        result = prep.transform_from_mov(data, data_mapping, key, clust, last_time, i)
        np.testing.assert_almost_equal(result, expected, self.rounding_precision)

    def test_transform_from_mov(self):
        """
        This function tests prep.transform_from_mov().

        prep.transform_from_mov() uses a transformation that is stored between a 4 marker cluster.  
        It requires an inverse transformation matrix to be stored between the 
        combination of 3 marker groupings, with the 4th marker stored in relation to that frame. 
        
        prep.transform_from_mov() takes in as input the missing marker, and uses previous frames of motion data to 
        create an inverse transformation matrix, multiplying the new frame by the stored 
        inverse transform to get the missing marker position.

        We test that prep.transform_from_mov() is accurate by creating four markers
        in a square at frame zero, and adding one to all but one of their x-coordinates
        in frame 1.

        We ensure that the correct position is reconstructed from the previous frame 
        data, and that the fourth marker is in the correct position after
        the transform.
        """
        #Define the four markers, A, B, C, D, as the four corners of a square in frame 0.
        #Add one to the x-coordinate of all markers but A in frame 1.
        data = np.array([[[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]],
                         [[np.nan, np.nan, np.nan], [0, 1, 0], [2, -1, 0], [0, -1, 0]]])
        #Define data mapping
        data_mapping = {
            'A':0, 'B':1, 'C':2, 'D':3
        }

        #Use the three markers, B, C, D to reconstruct the position of A
        clust = ['B', 'C', 'D']
        #Last frame in which all four markers were visible
        last_time = 0
        #Frame for which a marker is missing data
        i = 1

        #We expect that A is at [2, 1, 0]
        expected = np.array([2, 1, 0])

        result = prep.transform_from_mov(data, data_mapping, 'A', clust, last_time, i)
        np.testing.assert_equal(result, expected)
    
    @pytest.mark.parametrize("key,clust,last_time,i", [
        #Test that if clust is not at least 3 unique markers, an exception is raised
        ('LFHD', ['RFHD', 'RFHD'], 0, 1),
        ('LFHD', ['RFHD'], 0, 1),
        ('LFHD', [], 0, 1),
        #Test that if i, the frame number, it out of range, an exception is raised
        ('LFHD', ['RFHD', 'RBHD', 'LBHD'], 0, 6100),
        #Test that if the marker name does not exist, an exception is raised
        ('InvalidKey', ['RFHD', 'RBHD', 'LBHD'], 0, 1)
    ])
    def test_transform_from_mov_exceptions(self, key, clust, last_time, i):
        """
        We test exceptions raised when there are not enough usable markers,
        the frame number is out of range, and the marker name does not exist.
        """
        data = self.data
        data_mapping = self.data_mapping
        with pytest.raises(Exception):
            prep.transform_from_mov(data, data_mapping, key, clust, last_time, i)

    @pytest.mark.parametrize("key, j, missings, expected", [
        ('LFHD', 10, {}, ['RFHD', 'RBHD', 'LBHD']),
        ('LFHD', 10, {'LFHD':[]}, ['RFHD', 'RBHD', 'LBHD']),
        ('LFHD', 10, {'RFHD':[10]},['RBHD', 'LBHD']),
        ('LFHD', 10, {'LBHD':[10], 'RFHD':[10]},['RBHD']),
        ('LFHD', 10, {'LBHD':[10], 'RFHD':[10], 'RBHD':[10]},[]),
        ('C7',   10, {}, ['STRN', 'CLAV', 'T10', 'RBAK', 'RSHO', 'LSHO']),
        ('RPSI', 10, {},['LPSI', 'LASI', 'RASI']),
    ])
    def test_segment_finder_accuracy(self, key, j, missings, expected):
        """
        This function tests prep.segment_finder(), which takes in dynamic trial
        data and mappings, key, the name of the missing marker to find the segment for,
        segment_dict, which gives a mapping from segment to marker, j, the frame number
        the marker is missing for, and missings, a dictionary indicating other missing
        markers.

        We use the default segment_dict returned by prep.default_segment_dict().

        We test to ensure that missing markers are not used to reconstruct other
        missing markers.
        """
        data = self.data
        data_mapping = self.data_mapping
        segment_dict = prep.default_segment_dict()
        result = prep.segment_finder(key, data, data_mapping, segment_dict, j, missings)
        assert result == expected
    
    @pytest.mark.parametrize("key,j,missings", [
        #Test that the marker name must exist
        ('InvalidKey',10,{})
    ])
    def test_segment_finder_exceptions(self, key, j, missings):
        """
        We test the exception that is raised when the marker name does not
        exist.
        """
        data = self.data
        mapping = self.data_mapping
        segment_dict = prep.default_segment_dict()
        with pytest.raises(Exception):
            prep.segment_finder(key, data, data_mapping, segment_dict, j, missings) 
    
    def test_rigid_fill_transform_from_static(self):
        """
        This function tests prep.rigid_fill(data, data_mapping, static, static_mapping, segment_dict),
        where data and static are 3d numpy arrays of marker data, 
        data_mapping, and static_mapping, are mapping dictionaries for 
        data and static, and segment_dict is a dictionary that
        gives the segments that each marker is a part of.

        prep.rigid_fill() fills gaps for frames with missing data using
        the transform functions prep.transform_from_static and
        prep.transform_from_mov.

        prep.rigid_fill() determines whether to call transform_from_static
        or transform_from_mov by determining if there is a previous frame where
        all markers in the cluster of the missing marker exist, and can be used
        to create an inverse transformation matrix to determine
        the position of the missing marker in the current frame. If there is,
        transform_from_mov is used. Otherwise, transform_from_static, which uses
        static data to reconstruct missing markers, is used.
        
        This function tests that rigid_fill will properly call transform_from_static
        by clearing the value of a given key at all frames.
        Since there are no other frames to reconstruct the inverse transform
        off of, we expect the function to use transform_from_static.
        """

        data = self.data
        data_mapping = self.data_mapping
        static = self.static
        static_mapping = self.static_mapping
        segment_dict = prep.default_segment_dict()

        for i in range(len(data)):
            data[i][data_mapping['LFHD']] = np.array([np.nan, np.nan, np.nan])
        
        mock_return_val = np.array([1, 2, 3])
        with mock.patch.object(prep, 'transform_from_static', return_value=mock_return_val) as mock_transform_from_static:
            rigid_fill_data = prep.rigid_fill(data, data_mapping, static, static_mapping, segment_dict)
            mock_transform_from_static.assert_called()
        
        result = rigid_fill_data[10][data_mapping['LFHD']]
        np.testing.assert_equal(result, np.array([1, 2, 3]))
    
    def test_rigid_fill_transform_from_mov(self):
        """        
        This function tests that prep.rigid_fill() will properly call transform_from_mov
        by clearing the value of a marker at only one frame. We expect transform_from_mov
        to be called in this case since there are previous markers to create the
        inverse transform matrix from.
        """
        data = self.data
        data_mapping = self.data_mapping
        static = self.static
        static_mapping = self.static_mapping
        segment_dict = prep.default_segment_dict()

        data[10][data_mapping['LFHD']] = np.array([np.nan, np.nan, np.nan])

        mock_return_val = np.array([1, 2, 3])
        with mock.patch.object(prep, 'transform_from_mov', return_value=mock_return_val) as mock_transform_from_mov:
            rigid_fill_data = prep.rigid_fill(data, data_mapping, static, static_mapping, segment_dict)
            mock_transform_from_mov.assert_called()
        
        result = rigid_fill_data[10][data_mapping['LFHD']]
        np.testing.assert_equal(result, np.array([1, 2, 3]))
