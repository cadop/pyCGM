import unittest
import numpy as np
from pyCGM_Single.pyCGM_Helpers import getfilenames
from pyCGM_Single.pycgmIO import loadData, dataAsDict
from pyCGM_Single.clusterCalc import target_dict, segment_dict
from pyCGM_Single.pyCGM import pelvisJointCenter
import pyCGM_Single.Pipelines as Pipelines
import os

class TestPipelines(unittest.TestCase):
    rounding_precision = 8
    cwd = os.getcwd()
    if(cwd.split(os.sep)[-1]=="pyCGM_Single"):
        parent = os.path.dirname(cwd)
        os.chdir(parent)
    cwd = os.getcwd()

    def load_data_from_file(self, x, calcSacrum = False):
        """
        Loads motion capture data from dynamic and static trials.
        Returns as data dictionaries. Optionally calculate the 
        sacrum marker before creating the data dictionaries.
        """
        cur_dir = self.cwd
        dynamic_trial, static_trial,_,_,_ = getfilenames(x)
        dynamic_trial = os.path.join(cur_dir, dynamic_trial)
        static_trial = os.path.join(cur_dir, static_trial)
        motionData = loadData(dynamic_trial)
        staticData = loadData(static_trial)
        if (calcSacrum):
            for frame in motionData:
                frame['SACR'] = pelvisJointCenter(frame)[2]
        data = dataAsDict(motionData, npArray=True)
        static = dataAsDict(staticData, npArray=True)
        return data, static

    def test_butterFilter(self):
        """
        This function tests Pipelines.butterFilter(data, cutoff, Fs),
        where data is an array of numbers to filter, cutoff is the cutoff
        frequency to filter, and Fs is the sampling frequency of the data.

        We test cases where inputs are positive and negative floats and 
        integers. We test cases where inputs are lists and numpy arrays.

        We test exceptions raised when the length of the input data is too short, 
        when the cutoff frequency value is negative, zero, or too large, 
        and when the sampling frequency value is negative, zero, or too small.
        """
        #Test positive and negative floats and ints
        listTests = [
                [-1003.58361816, -1003.50396729, -1003.42358398, -1003.3425293,
                -1003.26068115, -1003.17810059, -1003.09484863, -1003.01080322,
                -1002.92608643, -1002.84063721, -1002.75445557, -1002.6675415,
                -1002.57995605, -1002.49157715, -1002.40252686, -1002.31274414],
                [70, 76, 87, 52, 28, 36, 65, 95, 69, 62, 60, 17, 12, 97, 11, 48],
                [-70, 76, 87, -52, 28, 36, 65, 95, -69, 62, 60, 17, 12, -97, 11, 48]
        ]

        #Same input as listTests, but as numpy arrays
        numpyTests = [np.array(arr) for arr in listTests]

        #Expected results are the same for both lists and numpy arrays
        expectedResults = [
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

        for i in range(len(listTests)):
            cutoff = 20
            Fs = 120
            #Call butterFilter() with the values from listTests and numpyTests
            result = Pipelines.butterFilter(listTests[i], cutoff, Fs)
            numpyResult = Pipelines.butterFilter(numpyTests[i], cutoff, Fs)
            expectedResult = expectedResults[i]

            np.testing.assert_almost_equal(result, expectedResult, self.rounding_precision)
            np.testing.assert_almost_equal(numpyResult, expectedResult, self.rounding_precision)

        #Test inputs that will cause exceptions
        exceptionTests = [
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
        ]
        
        for test in exceptionTests:
            with self.assertRaises(Exception):
                Pipelines.butterFilter(test[0], test[1], test[2])

    def test_filt(self):
        """
        This function tests Pipelines.filt(data, cutoff, Fs),
        where data is a 2darray of numbers to filter, cutoff is the cutoff
        frequency to filter, and Fs is the sampling frequency of the data.

        We test cases where inputs are lists of positive and negative floats 
        and ints.

        We test exceptions raised when the arrays in data are not of the same
        shape, when the length of data is too short, when the cutoff frequency 
        value is negative, zero, or too large, when the sampling frequency 
        value is negative, zero, or too small, and when data is not a numpy array.
        """
        #Test positive and negative floats and ints of numpy arrays
        tests = [
            np.array([[-1003.58361816, 81.00761414, 1522.23693848], [-1003.50396729, 81.02921295, 1522.18493652],
                      [-1003.42358398, 81.05059814, 1522.13598633], [-1003.3425293,  81.07178497, 1522.09020996],
                      [-1003.26068115, 81.09275818, 1522.04760742], [-1003.17810059, 81.11352539, 1522.00805664],
                      [-1003.09484863, 81.13407898, 1521.97167969], [-1003.01080322, 81.1544342,  1521.93835449],
                      [-1002.92608643, 81.17457581, 1521.90820312], [-1002.84063721, 81.19451141, 1521.88122559],
                      [-1002.75445557, 81.21424103, 1521.8572998 ], [-1002.6675415,  81.23376465, 1521.83654785],
                      [-1002.57995605, 81.25308228, 1521.81884766], [-1002.49157715, 81.27219391, 1521.80419922],
                      [-1002.40252686, 81.29109955, 1521.79284668], [-1002.31274414, 81.30980682, 1521.78442383]], dtype=np.float32),
            np.array([[0, 1], [1, 2], [2, -3], [3, 4],
                      [4, 5], [-5, 6], [6, 7], [-7, 8],
                      [8, 9], [9, 10], [10, 11], [11, 12],
                      [12, -13], [-13, 14], [14, 15], [-15, 16]], dtype=np.int32)
        ]
        expectedResults = [
                np.array([[-1003.58359202, 81.00761957, 1522.2369362 ], [-1003.50390817, 81.02919582, 1522.18514327],
                          [-1003.42363665, 81.05060565, 1522.135889  ], [-1003.34252947, 81.07178881, 1522.0901444 ],
                          [-1003.26066539, 81.09275628, 1522.04763945], [-1003.17811173, 81.11352192, 1522.00810375],
                          [-1003.09483484, 81.13408185, 1521.97164458], [-1003.01081474, 81.15443324, 1521.93835939],
                          [-1002.92608178, 81.17457741, 1521.90820438], [-1002.8406421 , 81.19451144, 1521.88119076],
                          [-1002.7544567 , 81.21423685, 1521.85735721], [-1002.66753023, 81.23376495, 1521.83657557],
                          [-1002.57993488, 81.25309204, 1521.81874533], [-1002.49164743, 81.27219203, 1521.80416229],
                          [-1002.40251014, 81.29107514, 1521.79304586], [-1002.31267097, 81.30982637, 1521.78437862]]),
                np.array([[-0.000935737255,  0.998712399], [ 1.01481885, -0.253654830],[ 2.33933854,  0.00948734868], [ 2.95259703,  2.13587233],
                          [ 1.99482481,  4.97199325], [ 0.123360866,  6.87747083], [-1.18427846,  7.06168539], [-0.287575584,  6.80213161],
                          [ 3.88087521,  8.89186494], [ 9.65108926,  12.49212252], [ 12.33700072,  11.15786714] ,[ 9.51590015,  3.47236703],
                          [ 5.42470382, -0.198189462], [ 3.62729076,  6.87305425], [-1.60886230,  15.2201909], [-14.9976228,  15.99966724]])
        ]

        for i in range(len(tests)):
            #Call filt() with the values from tests
            cutoff = 20
            Fs = 120
            result = Pipelines.filt(tests[i], cutoff, Fs)
            expectedResult = expectedResults[i]
            np.testing.assert_almost_equal(result, expectedResult, self.rounding_precision)
        
        #Test that if the shape of each array in data is not the same, an IndexError is raised
        invalidData = [[i] for i in range(16)] #Generate 16 arrays of length 1
        invalidData[0].append(1) #Change shape of first array to length 2
        invalidData = np.array(invalidData) #convert to numpy array
        with self.assertRaises(IndexError):
            Pipelines.filt(invalidData, 20, 120)

        validData = np.array([[i, i+1] for i in range(16)])
        exceptionTests = [
            #Test that data must be a numpy array
            ([[i] for i in range(20)], 20, 120),
            #Test that len(data) must be > 15
            (np.array([range(15)]), 20, 120),
            (np.arange(0, 0.4, 0.1), 20, 120),
            #Test invalid values for the cutoff frequency
            (validData, -10, 120),
            (validData, 0, 120),
            (validData, 40, 120),
            #Test invalid values for the sampling frequency
            (validData, 20, -10),
            (validData, 20, 0),
            (validData, 20, 20),
            (validData, 20, 60)
        ]

        for test in exceptionTests:
            with self.assertRaises(Exception):
                Pipelines.filt(test[0], test[1], test[2])

    def test_prep(self):
        """
        This function tests Pipelines.prep(trajs), where trajs
        is a dictionary containing numpy arrays.
        """
        tests = [
                #Tests one marker and two frames
                {'trajOne': np.array([[217.19961548, -82.35484314, 332.2684021 ],
                                      [257.19961548, -32.35484314, 382.2684021 ]])}, 
                #Tests two markers and two frames
                {'a': np.array([[1, 2, 3], [4, 5, 6]]), 
                 'b': np.array([[7, 8, 9], [10, 11, 12]])}
        ]
        expectedResults = [
                [
                 {'trajOne': np.array([217.19961548, -82.35484314, 332.2684021 ])}, 
                 {'trajOne': np.array([257.19961548, -32.35484314, 382.2684021 ])}
                ],
                [
                 {'a': np.array([1, 2, 3]), 'b': np.array([7, 8, 9])}, 
                 {'a': np.array([4, 5, 6]), 'b': np.array([10, 11, 12])}
                ]
        ]
        for i in range(len(tests)):
            #Call prep with the values from tests
            result = Pipelines.prep(tests[i])
            expectedResult = expectedResults[i]
            for j in range(len(result)):
                np.testing.assert_equal(result[j], expectedResult[j])

        #Test that if data[key] is not a numpy array, an exception is raised
        data = {'trajOne': [[217.19961548, -82.35484314, 332.2684021 ],
                            [257.19961548, -32.35484314, 382.2684021 ]]}
        with self.assertRaises(TypeError):
            Pipelines.prep(data)

    def test_clearMarker(self):
        """
        This function tests Pipelines.clearMarker(data, name), where
        data is an array of dictionary of markers lists and name is the 
        name of a marker to clear.

        We test cases where the values in data are lists and numpy arrays.
        We test the case where the marker name is not a key in data.
        """
        tests = [
            ([{'LTIL': np.array([-268.1545105, 327.53512573,  30.17036057]),
               'RFOP': np.array([-38.4509964, -148.6839447 ,  59.21961594])},
              {'LTIL': np.array([-273.1545105, 324.53512573,  36.17036057]),
               'RFOP': np.array([-38.4509964, -148.6839447 ,  59.21961594])}], 'LTIL'),
            #Test that data[key] can be a list
            ([{'LTIL': [-268.1545105, 327.53512573, 30.17036057],
               'RFOP': [-38.4509964, -148.6839447 , 59.21961594]},
              {'LTIL': [-273.1545105, 324.53512573, 36.17036057],
               'RFOP': [-38.4509964, -148.6839447 , 59.21961594]}], 'LTIL'),
            #Test removing a non-existent marker
            ([{'LTIL': np.array([-268.1545105, 327.53512573, 30.17036057])}], 'non_existent_marker')
        ]
        expectedResults = [
            [{'LTIL': np.array([np.nan, np.nan, np.nan]), 
              'RFOP': np.array([-38.4509964, -148.6839447, 59.21961594])}, 
             {'LTIL': np.array([np.nan, np.nan, np.nan]), 
              'RFOP': np.array([-38.4509964, -148.6839447, 59.21961594])}],
            [{'LTIL': np.array([np.nan, np.nan, np.nan]), 
              'RFOP': np.array([-38.4509964, -148.6839447, 59.21961594])}, 
             {'LTIL': np.array([np.nan, np.nan, np.nan]), 
              'RFOP': np.array([-38.4509964, -148.6839447, 59.21961594])}],
            #removing a non-existent marker adds it as a key
            [{'LTIL': np.array([-268.1545105, 327.53512573, 30.17036057]),
              'non_existent_marker': np.array([np.nan, np.nan, np.nan])}], 
        ]
        for i in range(len(tests)):
            #Call clearMarker() with the values from tests
            result = Pipelines.clearMarker(tests[i][0], tests[i][1])
            expectedResult = expectedResults[i]
            np.testing.assert_equal(result, expectedResult)

    def test_filtering(self):
        """
        This function tests Pipelines.filtering(Data), where
        Data is a dictionary of marker lists. This function calls
        Pipelines.filt().

        We test cases where inputs are numpy arrays of positive and 
        negative floats and ints.

        We test the exception raised when the values in Data are 
        not numpy arrays.
        """
        #Test numpy arrays of positive and negative floats and ints
        tests = []

        Data = {}
        Data['LFHD'] = np.array([[-1003.58361816, 81.00761414, 1522.23693848], [-1003.50396729, 81.02921295, 1522.18493652],
                                  [-1003.42358398, 81.05059814, 1522.13598633], [-1003.3425293,  81.07178497, 1522.09020996],
                                  [-1003.26068115, 81.09275818, 1522.04760742], [-1003.17810059, 81.11352539, 1522.00805664],
                                  [-1003.09484863, 81.13407898, 1521.97167969], [-1003.01080322, 81.1544342,  1521.93835449],
                                  [-1002.92608643, 81.17457581, 1521.90820312], [-1002.84063721, 81.19451141, 1521.88122559],
                                  [-1002.75445557, 81.21424103, 1521.8572998 ], [-1002.6675415,  81.23376465, 1521.83654785],
                                  [-1002.57995605, 81.25308228, 1521.81884766], [-1002.49157715, 81.27219391, 1521.80419922],
                                  [-1002.40252686, 81.29109955, 1521.79284668], [-1002.31274414, 81.30980682, 1521.78442383]], dtype=np.float32)
        tests.append(Data)

        Data = {}
        Data['LFHD'] = np.array([[i,i+1] for i in range(16)], dtype=np.int32) #[[0,1], [1,2], ... [15, 16]]
        tests.append(Data)

        expectedResults = [
            {
                'LFHD': np.array([[-1003.58359202, 81.00761957, 1522.2369362 ], [-1003.50390817, 81.02919582, 1522.18514327],
                                  [-1003.42363665, 81.05060565, 1522.135889  ], [-1003.34252947, 81.07178881, 1522.0901444 ],
                                  [-1003.26066539, 81.09275628, 1522.04763945], [-1003.17811173, 81.11352192, 1522.00810375],
                                  [-1003.09483484, 81.13408185, 1521.97164458], [-1003.01081474, 81.15443324, 1521.93835939],
                                  [-1002.92608178, 81.17457741, 1521.90820438], [-1002.8406421 , 81.19451144, 1521.88119076],
                                  [-1002.7544567 , 81.21423685, 1521.85735721], [-1002.66753023, 81.23376495, 1521.83657557],
                                  [-1002.57993488, 81.25309204, 1521.81874533], [-1002.49164743, 81.27219203, 1521.80416229],
                                  [-1002.40251014, 81.29107514, 1521.79304586], [-1002.31267097, 81.30982637, 1521.78437862]])
            },
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
        ]

        for i in range(len(tests)):
            #Call filtering() with the values from tests
            result = Pipelines.filtering(tests[i])
            expectedResult = expectedResults[i]
            for key in result:
                np.testing.assert_almost_equal(result[key], expectedResult[key], self.rounding_precision)
        
        #Test that if data is not a numpy array, an exception is raised
        invalidData = [[i] for i in range(20)] 
        with self.assertRaises(Exception):
            Pipelines.filtering(invalidData)

    def test_transform_from_static(self):
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
        
        We test exceptions raised when there are not enough usable markers,
        the frame number is out of range, and the marker name does not exist.
        """

        data,static = self.load_data_from_file(3) #Indicates that we use files from SampleData/Sample_2/ for testing

        #Tests for the marker 'LFHD' missing in 6 different frames
        key = 'LFHD'
        useables = ['RFHD', 'RBHD', 'LBHD']
        frameNumTests = [0, 1, 2, -1, 10, 100] #Frame numbers to be cleared to test gap filling.
        for frame in frameNumTests:
            data[key][frame] = np.array([np.nan, np.nan, np.nan]) 
        expectedResults = [
            [-1007.8145678233541, 71.28465078977477, 1522.6626006179151],
            [-1007.7357797476452, 71.30567599088612, 1522.6056345492811],
            [-1007.6561772477821, 71.32644261551039, 1522.5516787767372],
            [710.8111428914814, -18.282265916438064, 1549.7284035675332],
            [-1006.9916284913861, 71.48482387826286, 1522.2367625952083],
            [-995.8183141045178, 73.11905329024174, 1526.9072499889455]
        ]

        for i in range(len(frameNumTests)):
            #Call transform_from_static() with the values from frameNumTests
            result = Pipelines.transform_from_static(data,static,key,useables,frameNumTests[i])
            expectedResult = expectedResults[i]
            np.testing.assert_almost_equal(result, expectedResult, self.rounding_precision)
        
        exceptionTests = [
            #Test that if useables is not at least 3 unique markers, an exception is raised
            (data, static, 'LFHD', ['RFHD', 'RFHD', 'RFHD'], 0),
            (data, static, 'LFHD', ['RFHD'], 0),
            (data, static, 'LFHD', [], 0),
            #Test that if s, the frame number, is out of range, an exception is raised
            (data, static, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 6100),
            #Test that if the marker name does not exist, an exception is raised
            (data, static, 'InvalidKey', ['RFHD', 'RBHD', 'LBHD'], 0)
        ]

        for test in exceptionTests:
            with self.assertRaises(Exception):
                Pipelines.transform_from_static(test[0],test[1],test[2],test[3],test[4])

    def test_transform_from_mov(self):
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
        
        We test exceptions raised when there are not enough usable markers,
        the frame number is out of range, and the marker name does not exist.
        """
        data,_ = self.load_data_from_file(3) #Indicates that we use files from SampleData/Sample_2/ for testing
        key = 'LFHD'
        clust = ['RFHD', 'RBHD', 'LBHD'] #Markers in the same cluster as 'LFHD'
        last_time = 3
        frameNumTests = [0, 1, 2, -1, 10, 100] #Frame numbers to be cleared to test gap filling
        for frame in frameNumTests:
            data[key][frame] = np.array([np.nan, np.nan, np.nan])
        expectedFrameNumResults = [
            [-1003.5853191302489, 81.01088363383448, 1522.2423219324983],
            [-1003.5051421962658, 81.03139890186682, 1522.1885615624824],
            [-1003.4242027260506, 81.05167282762505, 1522.1377603754775],
            [714.4191660275219, -8.268045936969543, 1550.088229312965],
            [-1002.750909308156, 81.20689050873727, 1521.8463616672468],
            [-991.7315609567293, 82.91868701883672, 1526.597213251877]
        ]
        #Test that gap filling is accurate for the missing frame numbers in frameNumTests
        for i in range(len(frameNumTests)):
            result = Pipelines.transform_from_mov(data,key,clust,3,frameNumTests[i])
            expectedResult = expectedFrameNumResults[i]
            np.testing.assert_almost_equal(result, expectedResult, self.rounding_precision)


        #This tests that if data[last_time] is not visible, gap filling is not performed.
        key = 'LFHD'
        clust = ['RFHD', 'RBHD', 'LBHD'] 
        last_time = 20
        data[key][last_time] = np.array([np.nan, np.nan, np.nan])
        i = 21

        result = Pipelines.transform_from_mov(data,key,clust,last_time,i)
        expectedResult = [np.nan, np.nan, np.nan] #The expected result is nan since data[last_time] is already cleared
        np.testing.assert_equal(result, expectedResult)

        exceptionTests = [
            #Test that if clust is not at least 3 unique markers, an exception is raised
            (data, 'LFHD', ['RFHD', 'RFHD', 'RFHD'], 0, 1),
            (data, 'LFHD', ['RFHD'], 0, 1),
            (data, 'LFHD', [], 0, 1),
            #Test that if i, the frame number, it out of range, an exception is raised
            (data, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 0, 6100),
            #Test that if the marker name does not exist, an exception is raised
            (data, 'InvalidKey', ['RFHD', 'RBHD', 'LBHD'], 0, 1),
        ]

        for test in exceptionTests:
            with self.assertRaises(Exception):
                Pipelines.transform_from_mov(test[0],test[1],test[2],test[3],test[4])


    def test_segmentFinder(self):
        """
        This function tests Pipelines.segmentFinder(key,data,targetDict,segmentDict,j,missings),
        where data is an array of dictionaries of marker data,
        key is the name of the missing marker to find the segment for,
        targetDict is a dictionary of marker to segment,
        segmentDict is a dictionary of segments to marker names,
        j is the frame number the marker is missing for, 
        and missings is a dictionary indicating other missing markers.

        We test to ensure that missing markers are not used to reconstruct
        other missing markers. 

        We test the exception that is raised when the marker name does not
        exist.
        """
        data,_ = self.load_data_from_file(3) #Indicates that we use files from SampleData/Sample_2/ for testing
        targetDict = target_dict()
        segmentDict = segment_dict()

        tests = [
            ('LFHD',data,targetDict,segmentDict,10,{}),
            ('LFHD',data,targetDict,segmentDict,10,{'LFHD':[]}),
            ('LFHD',data,targetDict,segmentDict,10,{'RFHD':[10]}),
            ('LFHD',data,targetDict,segmentDict,10,{'LBHD':[10], 'RFHD':[10]}),
            ('LFHD',data,targetDict,segmentDict,10,{'LBHD':[10], 'RFHD':[10], 'RBHD':[10]}),
            ('C7',data,targetDict,segmentDict,10,{}),
            ('RPSI',data,targetDict,segmentDict,10,{}),
            ('LKNE',data,targetDict,segmentDict,10,{})
        ]

        expectedResults = [
            ['RFHD', 'RBHD', 'LBHD'],
            ['RFHD', 'RBHD', 'LBHD'],
            ['RBHD', 'LBHD'],
            ['RBHD'],
            [],
            ['STRN', 'CLAV', 'T10', 'RBAK', 'RSHO', 'LSHO'],
            ['LPSI', 'LASI', 'RASI'],
            ['LTHI']
        ]

        for i in range(len(tests)):
            test = tests[i]
            result = Pipelines.segmentFinder(test[0],test[1],test[2],test[3],test[4],test[5])
            expectedResult = expectedResults[i]
            self.assertEqual(result, expectedResult)

        #Test that the marker name must exist:
        with self.assertRaises(KeyError):
            Pipelines.segmentFinder('InvalidKey',data,targetDict,segmentDict,10,{})

    def test_rigid_fill(self):
        """
        This function tests Pipelines.rigid_fill(Data, static),
        where Data is an array of dictionaries of marker data,
        and static is an array of dictionaries of static trial data.

        This function fills gaps for frames with missing data.

        We test simulate missing data by clearing six different frames
        and testing the gap filling result.
        
        We test the exception raised when the marker SACR does not
        exist in Data.
        """
        #Test that the marker SACR must exist:
        with self.assertRaises(KeyError):
            data,static = self.load_data_from_file(3) #Indicates that we use files from SampleData/Sample_2/ for testing
            Pipelines.rigid_fill(data, static)

        data,static = self.load_data_from_file(3, calcSacrum=True) #True indicates we calculate SACR before testing
        key = 'LFHD' #clear from LFHD to test gap filling
        framesToClear = [1, 10, 12, 15, 100, -1]
        for frameNum in framesToClear:
            data[key][frameNum] = np.array([np.nan, np.nan, np.nan])
        
        result = Pipelines.rigid_fill(data, static)
        expectedResults = [
            [-1007.73577975, 71.30567599, 1522.60563455],
            [-1002.75400789, 81.21320267, 1521.8559697 ],
            [-1002.57942155, 81.2521139, 1521.81737938],
            [-1002.31227389, 81.30886597, 1521.78327027],
            [-991.70150174, 82.79915329, 1526.67335699],
            [ 714.9015343, -9.04757279, 1550.23346378]
        ]
        index = 0
        for frameNum in framesToClear:
            np.testing.assert_almost_equal(result[key][frameNum], expectedResults[index], self.rounding_precision)
            index += 1