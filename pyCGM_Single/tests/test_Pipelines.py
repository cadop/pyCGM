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

    def loadDataDicts(self, x, calcSacrum = False):
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

        #Define valid inputs for data, cutoff, and Fs
        data = [70, 76, 87, 52, 28, 36, 65, 95, 69, 62, 60, 17, 12, 97, 11, 48]
        cutoff = 20
        Fs = 120
        
        #Test inputs that will cause exceptions
        #Test that if len(data) < 15, an exception is raised
        dataLenTests = [[],range(3),np.arange(0, 0.4, 0.1), range(15)]
        for test in dataLenTests:
            with self.assertRaises(Exception):
                Pipelines.butterFilter(test, cutoff, Fs)
        
        #Test invalid values for cutoff frequency will raise exceptions
        cutoffExceptionTests = [-10, 0, Fs/3]
        for test in cutoffExceptionTests:
            with self.assertRaises(Exception):
                Pipelines.butterFilter(data, test, Fs)

        #Test invalid values for Fs, the sampling frequency will raise exceptions
        FsExceptionTests = [-10, 0, 20, cutoff*3]
        for test in FsExceptionTests:
            with self.assertRaises(Exception):
                Pipelines.butterFilter(data, cutoff, test)

    def test_filt(self):
        """
        This function tests Pipelines.filt(data, cutoff, Fs),
        where data is a 2darray of numbers to filter, cutoff is the cutoff
        frequency to filter, and Fs is the sampling frequency of the data.
        """
        #Test positive and negative floats and ints
        tests = [
            np.array([[-1003.58361816, 81.00761414, 1522.23693848], [-1003.50396729, 81.02921295, 1522.18493652],
                      [-1003.42358398, 81.05059814, 1522.13598633], [-1003.3425293,  81.07178497, 1522.09020996],
                      [-1003.26068115, 81.09275818, 1522.04760742], [-1003.17810059, 81.11352539, 1522.00805664],
                      [-1003.09484863, 81.13407898, 1521.97167969], [-1003.01080322, 81.1544342,  1521.93835449],
                      [-1002.92608643, 81.17457581, 1521.90820312], [-1002.84063721, 81.19451141, 1521.88122559],
                      [-1002.75445557, 81.21424103, 1521.8572998 ], [-1002.6675415,  81.23376465, 1521.83654785],
                      [-1002.57995605, 81.25308228, 1521.81884766], [-1002.49157715, 81.27219391, 1521.80419922],
                      [-1002.40252686, 81.29109955, 1521.79284668], [-1002.31274414, 81.30980682, 1521.78442383]]),
            np.array([[0, 1], [1, 2], [2, -3], [3, 4],
                      [4, 5], [-5, 6], [6, 7], [-7, 8],
                      [8, 9], [9, 10], [10, 11], [11, 12],
                      [12, -13], [-13, 14], [14, 15], [-15, 16]])
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
        
        #Test inputs that will cause exceptions

        #Test that if data is not a numpy array, a TypeError is raised
        data = [[[i,i+1] for i in range(16)]]
        with self.assertRaises(TypeError):
            Pipelines.filt(data, 20, 120)

        #Define valid inputs for data, cutoff, and Fs
        data = np.array([[[i,i+1] for i in range(16)]])
        cutoff = 20
        Fs = 120

        #Test that if the shape of each array in data is not the same, an IndexError is raised
        invalidData = [[i] for i in range(16)] #Generate 16 arrays of length 1
        invalidData[0].append(1) #Change shape of first array to length 2
        invalidData = np.array(invalidData) #convert to numpy array
        with self.assertRaises(IndexError):
            Pipelines.filt(invalidData, cutoff, Fs)

        #Test that if len(data) < 15, a TypeError is raised
        dataLenTests = [[range(3)], [np.arange(0, 0.4, 0.1)], [range(15)]]
        for test in dataLenTests:
            with self.assertRaises(TypeError):
                Pipelines.filt(test, cutoff, Fs)

        #Test invalid values for cutoff frequency will raise exceptions
        cutoffExceptionTests = [-10, 0, Fs/3]
        for test in cutoffExceptionTests:
            with self.assertRaises(Exception):
                Pipelines.butterFilter(data, test, Fs)

        #Test invalid values for Fs, the sampling frequency will raise exceptions
        FsExceptionTests = [-10, 0, 20, cutoff*3]
        for test in FsExceptionTests:
            with self.assertRaises(Exception):
                Pipelines.butterFilter(data, cutoff, test)

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
                #Test removing non-existent marker
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
            for j in range(len(result)):
                np.testing.assert_equal(result[j], expectedResult[j])

    def test_filtering(self):
        """
        This function tests Pipelines.filtering(Data), where
        Data is a dictionary of marker lists. This function calls
        Pipelines.filt().
        """
        tests = [
            {
                'LFHD': np.array([[-1003.58361816, 81.00761414, 1522.23693848], [-1003.50396729, 81.02921295, 1522.18493652],
                                  [-1003.42358398, 81.05059814, 1522.13598633], [-1003.3425293,  81.07178497, 1522.09020996],
                                  [-1003.26068115, 81.09275818, 1522.04760742], [-1003.17810059, 81.11352539, 1522.00805664],
                                  [-1003.09484863, 81.13407898, 1521.97167969], [-1003.01080322, 81.1544342,  1521.93835449],
                                  [-1002.92608643, 81.17457581, 1521.90820312], [-1002.84063721, 81.19451141, 1521.88122559],
                                  [-1002.75445557, 81.21424103, 1521.8572998 ], [-1002.6675415,  81.23376465, 1521.83654785],
                                  [-1002.57995605, 81.25308228, 1521.81884766], [-1002.49157715, 81.27219391, 1521.80419922],
                                  [-1002.40252686, 81.29109955, 1521.79284668], [-1002.31274414, 81.30980682, 1521.78442383]])
            },
            {
                'LFHD': np.array([[i,i+1] for i in range(16)]) #[[0,1], [1,2], ... [15, 16]]
            }
        ]
        
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
        data,static = self.loadDataDicts(3) #Indicates that we use files from SampleData/Sample_2/ for testing
        key = 'LFHD'
        useables = ['RFHD', 'RBHD', 'LBHD']
        frameNumTests = [0, 1, 2, -1, 10, 100]
        expectedResults = [[-1007.8145678233541, 71.28465078977477, 1522.6626006179151],
                    [-1007.7357797476452, 71.30567599088612, 1522.6056345492811],
                    [-1007.6561772477821, 71.32644261551039, 1522.5516787767372],
                    [710.8111428914814, -18.282265916438064, 1549.7284035675332],
                    [-1006.9916284913861, 71.48482387826286, 1522.2367625952083],
                    [-995.8183141045178, 73.11905329024174, 1526.9072499889455]]

        for i in range(len(frameNumTests)):
            result = Pipelines.transform_from_static(data,static,key,useables,frameNumTests[i])
            expectedResult = expectedResults[i]
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expectedResult[j], self.rounding_precision)
        
        #Test that if useables is not at least 3 unique markers, an exception is raised
        useablesExceptionTests = [
              ['RFHD', 'RFHD', 'RFHD'],
              [],
              ['RFHD'],
        ]
        with self.assertRaises(Exception):
            for test in useablesExceptionTests:
                Pipelines.transform_from_static(data,static,key,test,frameNumTests[0])

        #Test that frameNum must be in range:
        with self.assertRaises(IndexError):
                Pipelines.transform_from_static(data,static,key,useables,6000) #6000 is out of range

        #Test that the marker name must exist:
        with self.assertRaises(KeyError):
                Pipelines.transform_from_static(data,static,'InvalidKey',useables,frameNumTests[0])

    def test_transform_from_mov(self):
        data,_ = self.loadDataDicts(3) #Indicates that we use files from SampleData/Sample_2/ for testing
        key = 'LFHD'
        clust = ['RFHD', 'RBHD', 'LBHD'] #Markers in the same cluster as 'LFHD'
        frameNumTests = [0, 1, 2, -1, 10, 100] #Frame numbers to be cleared to test gap filling
        for frame in frameNumTests:
            data[key][frame] = np.array([np.nan, np.nan, np.nan])
        lastTimeTests = [0, 20] #Frame numbers where the marker was last visible
        expectedFrameNumResults = [[-1003.5853191302489, 81.01088363383448, 1522.2423219324983],
                           [-1003.5051421962658, 81.03139890186682, 1522.1885615624824],
                           [-1003.4242027260506, 81.05167282762505, 1522.1377603754775],
                           [714.4191660275219, -8.268045936969543, 1550.088229312965],
                           [-1002.750909308156, 81.20689050873727, 1521.8463616672468],
                           [-991.7315609567293, 82.91868701883672, 1526.597213251877]]
        expectedLastTimeResults = [[np.nan, np.nan, np.nan], #nan since data[last_time] is already cleared
                                   [-991.7393505887787, 82.93448317847992, 1526.6200219105137]]

        #Testing for several missing frames
        for i in range(len(frameNumTests)):
            result = Pipelines.transform_from_mov(data,key,clust,3,frameNumTests[i])
            expectedResult = expectedFrameNumResults[i]
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expectedResult[j], self.rounding_precision)

        #Testing for different last visible times
        for i in range(len(lastTimeTests)):
            result = Pipelines.transform_from_mov(data,key,clust,lastTimeTests[i],100)
            expectedResult = expectedLastTimeResults[i]
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expectedResult[j], self.rounding_precision)

        #Test that if clust is not at least 3 unique markers, an exception is raised
        clustExceptionTests = [
              ['RFHD', 'RFHD', 'RFHD'],
              [],
              ['RFHD'],
        ]
        with self.assertRaises(Exception):
            for test in clustExceptionTests:
                Pipelines.transform_from_mov(data,key,test,0,10)

        #Test that frameNum must be in range:
        with self.assertRaises(Exception):
                Pipelines.transform_from_mov(data,key,clust,0,6100)

        #Test that the marker name must exist:
        with self.assertRaises(KeyError):
                Pipelines.transform_from_mov(data,'InvalidKey',clust,0,10)

    def test_segmentFinder(self):
        data,_ = self.loadDataDicts(3) #Indicates that we use files from SampleData/Sample_2/ for testing
        key = 'LFHD'
        targetDict = target_dict()
        segmentDict = segment_dict()
        j = 10

        missingsTests = [
            #Normal cases, no other missing markers
            {}, 
            {'LFHD':[]},
            #tests to ensure we dont reconstruct based on missing markers 
            {'RFHD':[j]}, 
            {'LBHD':[j], 'RFHD':[j]},
            {'LBHD':[j], 'RFHD':[j], 'RBHD':[j]}
        ]
        expectedMissingsResults = [
            ['RFHD', 'RBHD', 'LBHD'],
            ['RFHD', 'RBHD', 'LBHD'],
            ['RBHD', 'LBHD'],
            ['RBHD'],
            []
        ]

        for i in range(len(missingsTests)):
            result = Pipelines.segmentFinder(key,data,targetDict,segmentDict,j,missingsTests[i])
            self.assertEqual(result, expectedMissingsResults[i])

        #Tests for markers in different clusters
        keyTests = ['LFHD', 'C7', 'RPSI', 'LKNE']
        expectedKeyResults = [
            ['RFHD', 'RBHD', 'LBHD'],
            ['STRN', 'CLAV', 'T10', 'RBAK', 'RSHO', 'LSHO'],
            ['LPSI', 'LASI', 'RASI'],
            ['LTHI']
        ]

        for i in range(len(keyTests)):
            result = Pipelines.segmentFinder(keyTests[i],data,targetDict,segmentDict,j,missingsTests[0])
            self.assertEqual(result, expectedKeyResults[i])

        #Test that the marker name must exist:
        with self.assertRaises(KeyError):
            Pipelines.segmentFinder('InvalidKey',data,targetDict,segmentDict,j,missingsTests[0])

    def test_rigid_fill(self):
        #Test that the marker SACR must exist:
        with self.assertRaises(KeyError):
            data,static = self.loadDataDicts(3) #Indicates that we use files from SampleData/Sample_2/ for testing
            Pipelines.rigid_fill(data, static)

        data,static = self.loadDataDicts(3, calcSacrum=True) #True indicates we calculate SACR before testing
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


        



       

