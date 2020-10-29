import unittest
import numpy as np
from pyCGM_Single.pyCGM_Helpers import getfilenames
from pyCGM_Single.pycgmIO import loadData, dataAsDict
import pyCGM_Single.Pipelines as Pipelines
import os

class TestPipelines(unittest.TestCase):
    rounding_precision = 8
    cwd = os.getcwd()
    if(cwd.split(os.sep)[-1]=="pyCGM_Single"):
        parent = os.path.dirname(cwd)
        os.chdir(parent)
    cwd = os.getcwd()

    def loadDataDicts(self, x):
        cur_dir = self.cwd
        dynamic_trial, static_trial,_,_,_ = getfilenames(x)
        dynamic_trial = os.path.join(cur_dir, dynamic_trial)
        static_trial = os.path.join(cur_dir, static_trial)
        motionData = loadData(dynamic_trial)
        staticData = loadData(static_trial)
        data = dataAsDict(motionData, npArray=True)
        static = dataAsDict(staticData, npArray=True)
        return data, static

    def test_butterFilter(self):
        tests = [[-1003.58361816, -1003.50396729, -1003.42358398, -1003.3425293,
                  -1003.26068115, -1003.17810059, -1003.09484863, -1003.01080322,
                  -1002.92608643, -1002.84063721, -1002.75445557, -1002.6675415,
                  -1002.57995605, -1002.49157715, -1002.40252686, -1002.31274414], #normal case
                 [70, 76, 87, 52, 28, 36, 65, 95, 69, 62, 60, 17, 12, 97, 11, 48], #test ints
                 [-70, 76, 87, -52, 28, 36, 65, 95, -69, 62, 60, 17, 12, -97, 11, 48]] #test ints, mixed positive and negative
        cutoff = 20
        Fs = 120
        expectedResults = [np.array([-1003.58359202, -1003.50390817, -1003.42363665, -1003.34252947,
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
                                   -21.05573085, -53.81198321, -23.50929002, 47.94081346])]
        for i in range(len(tests)):
            data = tests[i]
            numpyData = np.array(tests[i])
            result = Pipelines.butterFilter(data, cutoff, Fs)
            numpyResult = Pipelines.butterFilter(numpyData, cutoff, Fs)
            expectedResult = expectedResults[i]
            for i in range(len(result)):
                np.testing.assert_almost_equal(result[i], expectedResult[i], self.rounding_precision)
                np.testing.assert_almost_equal(numpyResult[i], expectedResult[i], self.rounding_precision)

        #Exception Tests
        #Test that len(data) > 15
        dataLenTests = [[],
                        [0, 1, 2, 3], 
                        [0.1, 0.2, 0.3], 
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
        for test in dataLenTests:
            with self.assertRaises(Exception):
                Pipelines.butterFilter(test, cutoff, Fs)
        
        cutoffExceptionTests = [-10, 0, Fs/3]
        for test in cutoffExceptionTests:
            with self.assertRaises(Exception):
                Pipelines.butterFilter(tests[0], test, Fs)

        FsExceptionTests = [-10, 0, 20, cutoff*3]
        for test in FsExceptionTests:
            with self.assertRaises(Exception):
                Pipelines.butterFilter(tests[0], cutoff, test)

    def test_filt(self):
        cutoff = 20
        Fs = 120
        tests = [
                [[-1003.58361816, 81.00761414, 1522.23693848], [-1003.50396729, 81.02921295, 1522.18493652],
                 [-1003.42358398, 81.05059814, 1522.13598633], [-1003.3425293,  81.07178497, 1522.09020996],
                 [-1003.26068115, 81.09275818, 1522.04760742], [-1003.17810059, 81.11352539, 1522.00805664],
                 [-1003.09484863, 81.13407898, 1521.97167969], [-1003.01080322, 81.1544342,  1521.93835449],
                 [-1002.92608643, 81.17457581, 1521.90820312], [-1002.84063721, 81.19451141, 1521.88122559],
                 [-1002.75445557, 81.21424103, 1521.8572998 ], [-1002.6675415,  81.23376465, 1521.83654785],
                 [-1002.57995605, 81.25308228, 1521.81884766], [-1002.49157715, 81.27219391, 1521.80419922],
                 [-1002.40252686, 81.29109955, 1521.79284668], [-1002.31274414, 81.30980682, 1521.78442383]],
                [[0, 1], [1, 2], [2, 3], [3, 4],
                 [4, 5], [5, 6], [6, 7], [7, 8],
                 [8, 9], [9, 10], [10, 11], [11, 12],
                 [12, 13], [13, 14], [14, 15], [15, 16]]
        ]
        expectedResults = [
                [[-1003.58359202, 81.00761957, 1522.2369362 ], [-1003.50390817, 81.02919582, 1522.18514327],
                 [-1003.42363665, 81.05060565, 1522.135889  ], [-1003.34252947, 81.07178881, 1522.0901444 ],
                 [-1003.26066539, 81.09275628, 1522.04763945], [-1003.17811173, 81.11352192, 1522.00810375],
                 [-1003.09483484, 81.13408185, 1521.97164458], [-1003.01081474, 81.15443324, 1521.93835939],
                 [-1002.92608178, 81.17457741, 1521.90820438], [-1002.8406421 , 81.19451144, 1521.88119076],
                 [-1002.7544567 , 81.21423685, 1521.85735721], [-1002.66753023, 81.23376495, 1521.83657557],
                 [-1002.57993488, 81.25309204, 1521.81874533], [-1002.49164743, 81.27219203, 1521.80416229],
                 [-1002.40251014, 81.29107514, 1521.79304586], [-1002.31267097, 81.30982637, 1521.78437862]],
                [[0.000290878541, 1.00029088], [1.00001659, 2.00001659],
                 [1.99986579, 2.99986579], [3.00000047, 4.00000047],
                 [4.00006844, 5.00006844], [4.99998738, 5.99998738],
                 [5.99995125, 6.99995125], [7.00003188, 8.00003188],
                 [8.00006104, 9.00006104], [8.99992598, 9.99992598],
                 [9.99988808, 10.99988808], [11.0001711, 12.0001711],
                 [12.00023202, 13.00023202], [12.99960463, 13.99960463],
                 [13.99950627, 14.99950627], [15.00091227, 16.00091227]]
        ]

        for i in range(len(tests)):
            data = np.array(tests[i])
            result = Pipelines.filt(data, cutoff, Fs)
            expectedResult = expectedResults[i]
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expectedResult[j], self.rounding_precision)

        #Test that data must be a numpy array
        with self.assertRaises(Exception):
            Pipelines.filt(tests[0], cutoff, Fs)

        dataLenTests = [[[0, 1, 2, 3]], 
                        [[0.1], [0.2], [0.3]], 
                        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]]

        #Test that len(data) > 15
        for test in dataLenTests:
            with self.assertRaises(TypeError):
                Pipelines.filt(test, cutoff, Fs)

        cutoffExceptionTests = [-10, 0, Fs/3]
        for test in cutoffExceptionTests:
            with self.assertRaises(Exception):
                Pipelines.butterFilter(tests[0], test, Fs)

        FsExceptionTests = [-10, 0, 20, cutoff*3]
        for test in FsExceptionTests:
            with self.assertRaises(Exception):
                Pipelines.butterFilter(tests[0], cutoff, test)

    def test_prep(self):
        tests = [
                {'trajOne': np.array([[217.19961548, -82.35484314, 332.2684021 ],
                                      [257.19961548, -32.35484314, 382.2684021 ]])},
                 {'a': np.array([[1, 2, 3], [4, 5, 6]]),
                  'b': np.array([[7, 8, 9], [10, 11, 12]])}]
        expectedResults = [
                [{'trajOne': np.array([217.19961548, -82.35484314, 332.2684021 ])}, 
                 {'trajOne': np.array([257.19961548, -32.35484314, 382.2684021 ])}],
                [{'a': np.array([1, 2, 3]),
                  'b': np.array([7, 8, 9])}, 
                 {'a': np.array([4, 5, 6]),
                  'b': np.array([10, 11, 12])}]
        ]
        for i in range(len(tests)):
            result = Pipelines.prep(tests[i])
            expectedResult = expectedResults[i]
            for j in range(len(result)):
                for key in result[j]:
                    np.testing.assert_equal(result[j][key], expectedResult[j][key])

        #Test that data[key] must be a numpy array
        data = {'trajOne': [[217.19961548, -82.35484314, 332.2684021 ],
                            [257.19961548, -32.35484314, 382.2684021 ]]}
        with self.assertRaises(TypeError):
            Pipelines.prep(data)

    def test_clearMarker(self):
        tests = [
                ([{'LTIL': np.array([-268.1545105, 327.53512573,  30.17036057]),
                   'RFOP': np.array([-38.4509964, -148.6839447 ,  59.21961594])},
                  {'LTIL': np.array([-273.1545105, 324.53512573,  36.17036057]),
                   'RFOP': np.array([-38.4509964, -148.6839447 ,  59.21961594])}], 'LTIL'),
                ([{'LTIL': np.array([-268.1545105, 327.53512573, 30.17036057])}], 'RFOP'), #removing non-existent name
                ([{'LTIL': [-268.1545105, 327.53512573, 30.17036057],
                   'RFOP': [-38.4509964, -148.6839447 , 59.21961594]},
                  {'LTIL': [-273.1545105, 324.53512573, 36.17036057],
                   'RFOP': [-38.4509964, -148.6839447 , 59.21961594]}], 'LTIL') #data[key] can be a list
        ]
        expectedResults = [
                [{'LTIL': np.array([np.nan, np.nan, np.nan]), 
                  'RFOP': np.array([-38.4509964, -148.6839447, 59.21961594])}, 
                 {'LTIL': np.array([np.nan, np.nan, np.nan]), 
                  'RFOP': np.array([-38.4509964, -148.6839447, 59.21961594])}],
                [{'LTIL': np.array([-268.1545105, 327.53512573, 30.17036057]),
                  'RFOP': np.array([np.nan, np.nan, np.nan])}],
                [{'LTIL': np.array([np.nan, np.nan, np.nan]), 
                  'RFOP': np.array([-38.4509964, -148.6839447, 59.21961594])}, 
                 {'LTIL': np.array([np.nan, np.nan, np.nan]), 
                  'RFOP': np.array([-38.4509964, -148.6839447, 59.21961594])}]
        ]
        for i in range(len(tests)):
            result = Pipelines.clearMarker(tests[i][0], tests[i][1])
            expectedResult = expectedResults[i]
            for j in range(len(result)):
                for key in result[j]:
                    np.testing.assert_equal(result[j][key], expectedResult[j][key])

    def test_filtering(self):
        tests = [
                {'LFHD': [[-1003.58361816, 81.00761414, 1522.23693848], [-1003.50396729, 81.02921295, 1522.18493652],
                 [-1003.42358398, 81.05059814, 1522.13598633], [-1003.3425293,  81.07178497, 1522.09020996],
                 [-1003.26068115, 81.09275818, 1522.04760742], [-1003.17810059, 81.11352539, 1522.00805664],
                 [-1003.09484863, 81.13407898, 1521.97167969], [-1003.01080322, 81.1544342,  1521.93835449],
                 [-1002.92608643, 81.17457581, 1521.90820312], [-1002.84063721, 81.19451141, 1521.88122559],
                 [-1002.75445557, 81.21424103, 1521.8572998 ], [-1002.6675415,  81.23376465, 1521.83654785],
                 [-1002.57995605, 81.25308228, 1521.81884766], [-1002.49157715, 81.27219391, 1521.80419922],
                 [-1002.40252686, 81.29109955, 1521.79284668], [-1002.31274414, 81.30980682, 1521.78442383]]},
                {'LFHD': [[0, 1], [1, 2], [2, 3], [3, 4],
                 [4, 5], [5, 6], [6, 7], [7, 8],
                 [8, 9], [9, 10], [10, 11], [11, 12],
                 [12, 13], [13, 14], [14, 15], [15, 16]]}
        ]
        expectedResults = [
                {'LFHD': [[-1003.58359202, 81.00761957, 1522.2369362 ], [-1003.50390817, 81.02919582, 1522.18514327],
                 [-1003.42363665, 81.05060565, 1522.135889  ], [-1003.34252947, 81.07178881, 1522.0901444 ],
                 [-1003.26066539, 81.09275628, 1522.04763945], [-1003.17811173, 81.11352192, 1522.00810375],
                 [-1003.09483484, 81.13408185, 1521.97164458], [-1003.01081474, 81.15443324, 1521.93835939],
                 [-1002.92608178, 81.17457741, 1521.90820438], [-1002.8406421 , 81.19451144, 1521.88119076],
                 [-1002.7544567 , 81.21423685, 1521.85735721], [-1002.66753023, 81.23376495, 1521.83657557],
                 [-1002.57993488, 81.25309204, 1521.81874533], [-1002.49164743, 81.27219203, 1521.80416229],
                 [-1002.40251014, 81.29107514, 1521.79304586], [-1002.31267097, 81.30982637, 1521.78437862]]},
                {'LFHD': [[0.000290878541, 1.00029088], [1.00001659, 2.00001659],
                 [1.99986579, 2.99986579], [3.00000047, 4.00000047],
                 [4.00006844, 5.00006844], [4.99998738, 5.99998738],
                 [5.99995125, 6.99995125], [7.00003188, 8.00003188],
                 [8.00006104, 9.00006104], [8.99992598, 9.99992598],
                 [9.99988808, 10.99988808], [11.0001711, 12.0001711],
                 [12.00023202, 13.00023202], [12.99960463, 13.99960463],
                 [13.99950627, 14.99950627], [15.00091227, 16.00091227]]}
        ]

        #Test that data must be a numpy array
        with self.assertRaises(TypeError):
            Pipelines.filtering(tests[0])

        for i in range(len(tests)):
            data = tests[i]
            #data[key] must be a numpy array
            for key in data:
                data[key] = np.array(data[key])
            result = Pipelines.filtering(data)
            expectedResult = expectedResults[i]
            for key in result:
                for j in range(len(result[key])):
                    np.testing.assert_almost_equal(result[key][j], expectedResult[key][j], self.rounding_precision)


    def test_transform_from_static(self):
        data,static = self.loadDataDicts(3)
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
        
        #useables must be at least of length 3:
        useablesExceptionTests = [
              ['RFHD', 'RFHD', 'RFHD'],
              [],
              ['RFHD'],
        ]
        with self.assertRaises(Exception):
            for test in useablesExceptionTests:
                Pipelines.transform_from_static(data,static,key,test,frameNumTests[0])

        #test that frameNum must be in range:
        with self.assertRaises(IndexError):
                Pipelines.transform_from_static(data,static,key,useables,6000)

        #test that the key must exist:
        with self.assertRaises(KeyError):
                Pipelines.transform_from_static(data,static,'InvalidKey',useables,frameNumTests[0])

    def test_transform_from_mov(self):
        data,_ = self.loadDataDicts(3)
        key = 'LFHD'
        clust = ['RFHD', 'RBHD', 'LBHD']
        frameNumTests = [0, 1, 2, -1, 10, 100]
        for frame in frameNumTests:
            data[key][frame] = np.array([np.nan, np.nan, np.nan])
        lastTimeTests = [0, 20]
        expectedFrameNumResults = [[-1003.5853191302489, 81.01088363383448, 1522.2423219324983],
                           [-1003.5051421962658, 81.03139890186682, 1522.1885615624824],
                           [-1003.4242027260506, 81.05167282762505, 1522.1377603754775],
                           [714.4191660275219, -8.268045936969543, 1550.088229312965],
                           [-1002.750909308156, 81.20689050873727, 1521.8463616672468],
                           [-991.7315609567293, 82.91868701883672, 1526.597213251877]]
        expectedLastTimeResults = [[np.nan, np.nan, np.nan],[-991.7393505887787, 82.93448317847992, 1526.6200219105137]]

        #Standard case
        for i in range(len(frameNumTests)):
            result = Pipelines.transform_from_mov(data,key,clust,3,frameNumTests[i])
            expectedResult = expectedFrameNumResults[i]
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expectedResult[j], self.rounding_precision)

        #Testing last_time
        for i in range(len(lastTimeTests)):
            result = Pipelines.transform_from_mov(data,key,clust,lastTimeTests[i],100)
            expectedResult = expectedLastTimeResults[i]
            for j in range(len(result)):
                np.testing.assert_almost_equal(result[j], expectedResult[j], self.rounding_precision)

        #clust must be at least of length 3:
        clustExceptionTests = [
              ['RFHD', 'RFHD', 'RFHD'],
              [],
              ['RFHD'],
        ]
        with self.assertRaises(Exception):
            for test in clustExceptionTests:
                Pipelines.transform_from_mov(data,key,test,0,10)

        #test that frameNum must be in range:
        with self.assertRaises(Exception):
                Pipelines.transform_from_mov(data,key,clust,0,6100)

        #test that the key must exist:
        with self.assertRaises(KeyError):
                Pipelines.transform_from_mov(data,'InvalidKey',clust,0,10)

        



       

