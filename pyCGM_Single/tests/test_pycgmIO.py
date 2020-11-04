import unittest
from pyCGM_Single.pyCGM_Helpers import getfilenames
import pyCGM_Single.pycgmIO as pycgmIO
import numpy as np
import os

class Test_pycgmIO(unittest.TestCase):
    rounding_precision = 8
    cwd = os.getcwd()
    if (cwd.split(os.sep)[-1]=="pyCGM_Single"):
        parent = os.path.dirname(cwd)
        os.chdir(parent)
    cwd = os.getcwd()

    def test_createMotionDataDict(self):
        """
        This function tests pycgmIO.createMotionDataDict(labels,data),
        where labels is a list of label names and data is a 2d list 
        or numpy array of coordinates corresponding to labels.

        We test cases where the values in data are lists and numpy arrays.
        We test cases where there are more labels than data coordinates.
        We test cases where there are more data coordinates than labels.
        """
        #Tests lists and numpy arrays
        tests = [
            (['A', 'B', 'C'],
            [[[1,2,3],[4,5,6],[7,8,9]],
             [[2,3,4],[5,6,7],[8,9,10]]]),

            (['A', 'B', 'C'],
             [[np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9])],
              [np.array([2,3,4]),np.array([5,6,7]),np.array([8,9,10])]]),

            (['A'],
            [[[1,2,3],[4,5,6],[7,8,9]],
             [[2,3,4],[5,6,7],[8,9,10]]]),

            (['A', 'B', 'C'], 
            [[[1,2,3],[4,5,6]],
             [[2,3,4]]])
        ]
        expectedResults = [
            [{'A':[1,2,3], 'B':[4,5,6], 'C':[7,8,9]},
             {'A':[2,3,4], 'B':[5,6,7], 'C':[8,9,10]}],

            [{'A':np.array([1,2,3]), 'B':np.array([4,5,6]), 'C':np.array([7,8,9])},
             {'A':np.array([2,3,4]), 'B':np.array([5,6,7]), 'C':np.array([8,9,10])}],

            [{'A': [1, 2, 3]}, 
             {'A': [2, 3, 4]}],

            [{'A': [1, 2, 3], 'B': [4, 5, 6]}, 
             {'A': [2, 3, 4]}]
        ]

        for i in range(len(tests)):
            test = tests[i]
            #Call createMotionDataDict() with the values from tests
            result = pycgmIO.createMotionDataDict(test[0], test[1])
            expectedResult = expectedResults[i]
            np.testing.assert_equal(result, expectedResult)

    def test_splitMotionDataDict(self):
        """
        This function tests pycgmIO.splitMotionDataDict(motiondata),
        where motiondata is a list of dictionaries of motion capture data.
        This function splits the motiondata into a tuple of labels, data.

        We tests cases where values are lists or numpy arrays.
        We demonstrate unexpected behavior that the function produces when
        keys are not present in every dictionary of motiondata.

        We test that if the dictionary values in motiondata are not
        1d arrays of 3 elements, an exception is raised.
        """
        tests = []
        motiondata = [
            {'A': [1, 2, 3], 'B': [4, 5, 6]},
            {'A': [2, 3, 4], 'B': [5, 6, 7]}
        ]
        tests.append(motiondata)

        motiondata = [
            {'A': np.array([1, 2, 3]), 'B': np.array([4, 5, 6])},
            {'A': np.array([2, 3, 4]), 'B': np.array([5, 6, 7])}
        ]
        tests.append(motiondata)

        motiondata = [
            {'A': np.array([1, 2, 3]), 'B': np.array([4, 5, 6])},
            {'A': np.array([2, 3, 4])}
        ]
        tests.append(motiondata)
        
        motiondata = [
            {'B': np.array([4, 5, 6])},
            {'A': np.array([2, 3, 4])}
        ]
        tests.append(motiondata)

        expectedResults = [
            (['A', 'B'], np.array([[[1, 2, 3],[4, 5, 6]],[[2, 3, 4],[5, 6, 7]]])),
            (['A', 'B'], np.array([[[1, 2, 3],[4, 5, 6]],[[2, 3, 4],[5, 6, 7]]])),
            (['A', 'B'], np.array([[[1, 2, 3],[4, 5, 6]],[[2, 3, 4],[2, 3, 4]]])),
            (['B'], np.array([[[4, 5, 6]],[[2, 3, 4]]]))
        ]

        for i in range(len(tests)):
            test = tests[i]
            #Call splitMotionDataDict() with the values from tests
            resultLabels, resultData = pycgmIO.splitMotionDataDict(test)
            expectedResult = expectedResults[i]
            expectedLabels = expectedResult[0]
            expectedData = expectedResult[1]
            self.assertEqual(resultLabels, expectedLabels)
            np.testing.assert_equal(resultData, expectedData)
        
        #Test that if the dictionary values are not 
        #1d arrays of 3 elements, an exception is raised.
        exceptionTests = [
            {'A': [1, 2]},
            {'A': []},
            {'A': [1, 2, 3, 4, 5]},
            {'A': [[1, 2, 3],
                   [4, 5, 6]]}
        ]

        for test in exceptionTests:
            with self.assertRaises(Exception):
                pycgmIO.splitMotionDataDict(test)

    def test_createVskDataDict(self):
        """
        This function tests pycgmIO.createVskDataDict(labels, data), 
        which creates a dictionary of VSK file values given an array
        of labels and data.

        We test cases where labels and data are the same length,
        labels and data are different lengths,
        labels is empty, data is empty, both are empty, and 
        data is a numpy array.
        """
        tests = [
            (['MeanLegLength', 'LeftKneeWidth', 'RightAnkleWidth'],
             [940.0, 105.0, 70.0]),
            (['A', 'B', 'C', 'D'],
             [1, 2, 3, 4]),
            (['A', 'B', 'C', 'D'],
             np.array([1, 2, 3, 4])),
            (['A', 'B'],
             [1, 2, 3, 4, 5, 6]),
            (['A', 'B', 'C', 'D', 'E'],
             [1, 2]),
            ([], [0, 1]),
            (['A', 'B', 'C'], []),
            ([], [])
        ]

        expectedResults = [
            {'MeanLegLength':940.0, 'LeftKneeWidth':105.0,'RightAnkleWidth':70.0},
            {'A': 1, 'B': 2, 'C': 3, 'D': 4},
            {'A': 1, 'B': 2, 'C': 3, 'D': 4},
            {'A': 1, 'B': 2},
            {'A': 1, 'B': 2},
            {},
            {},
            {}
        ]

        for i in range(len(tests)):
            test = tests[i]
            #Call createVskDataDict() with the values from tests
            result = pycgmIO.createVskDataDict(test[0], test[1])
            expectedResult = expectedResults[i]
            np.testing.assert_equal(result, expectedResult)

    def test_splitVskDataDict(self):
        """
        This function tests pycgmIO.splitVskDataDict(vsk),
        where vsk is a dictionary of vsk file values. This function
        splits the values into two arrays of labels and data.

        We test the case where the vsk is an empty dictionary.
        """
        tests = [
            {'MeanLegLength':940.0, 'LeftKneeWidth':105.0,'RightAnkleWidth':70.0},
            {'A': 1, 'B': 2, 'C': 3, 'D': 4},
            {'A': 1, 'B': 2},
            {}
        ]

        expectedResults = [
            (['MeanLegLength', 'LeftKneeWidth', 'RightAnkleWidth'], np.array([940., 105.,  70.])),
            (['A', 'B', 'C', 'D'], np.array([1, 2, 3, 4])),
            (['A', 'B'], np.array([1, 2])),
            ([], np.array([]))
        ]

        for i in range(len(tests)):
            test = tests[i]
            #Call splitVskDataDict with the values from tests
            resultLabels, resultData = pycgmIO.splitVskDataDict(test)
            expectedResult = expectedResults[i]
            expectedLabels = expectedResult[0]
            expectedData = expectedResult[1]
            #Convert results to sets so order of the results is ignored
            np.testing.assert_equal(set(resultLabels), set(expectedLabels))
            np.testing.assert_equal(set(resultData), set(expectedData))

    def test_markerKeys(self):
        """
        This function tests pycgmIO.markerKeys, which returns
        a constant array of marker names.
        """
        result = pycgmIO.markerKeys()
        expectedResult = ['RASI', 'LASI', 'RPSI', 'LPSI', 'RTHI', 'LTHI', 'RKNE', 'LKNE', 'RTIB',
            'LTIB', 'RANK', 'LANK', 'RTOE', 'LTOE', 'LFHD', 'RFHD', 'LBHD', 'RBHD', 
            'RHEE', 'LHEE', 'CLAV', 'C7', 'STRN', 'T10', 'RSHO', 'LSHO', 'RELB', 'LELB', 
            'RWRA', 'RWRB', 'LWRA', 'LWRB', 'RFIN', 'LFIN']
        self.assertEqual(result, expectedResult)

    def test_loadC3D(self):
        """
        This function test pycgmIO.loadC3D(filename), where filename
        is a string indicating the file path of a c3d file to load. This
        function returns an array of [data, dataunlabeled, markers].

        This function uses the file 59993_Frame_Static.c3d in 
        SampleData for testing.

        We test that an exception is raised when loading a non-existent
        file name.
        """
        #Use pyCGM_Helpers.getfilenames() to retrieve filenames
        filename_59993_Frame = os.path.join(self.cwd, getfilenames(1)[1])

        #Call loadC3D with the filename
        result_59993 = pycgmIO.loadC3D(filename_59993_Frame)

        #Test for some frames from data
        data = result_59993[0]
        dataResults = [
            data[0]['LFHD'],
            data[125]['RASI'],
            data[2]['LPSI'],
            data[12]['LKNE'],
            data[22]['C7'],
            data[302]['RANK']
        ]
        expectedDataResults = [
            np.array([60.1229744, 132.4755249, 1485.8293457]),
            np.array([144.1030426, -0.36732361, 856.89855957]),
            np.array([-94.89163208, 49.82866287, 922.64483643]),
            np.array([-100.0297699, 126.43037415, 414.15945435]),
            np.array([-27.38780975, -8.35509396, 1301.37145996]),
            np.array([52.61815643, -126.93238068, 58.56194305])
        ]
        for i in range(len(dataResults)):
            result = dataResults[i]
            expectedResult = expectedDataResults[i]
            np.testing.assert_almost_equal(result, expectedResult, self.rounding_precision)
        
        #Test for some frames from dataunlabeled
        dataunlabeled = result_59993[1]
        dataunlabeledResults = [
            dataunlabeled[0]['*113'],
            dataunlabeled[10]['*114'],
            dataunlabeled[302]['*113'],
            dataunlabeled[1]['*114'],
            dataunlabeled[12]['*113'],
            dataunlabeled[94]['*114']
        ]
        expectedDataunlabeledResults = [
            np.array([-173.22341919,  166.87660217, 1273.29980469]), 
            np.array([ 169.66015625, -226.81838989, 1264.20507812]), 
            np.array([-166.02009583,  170.07366943, 1278.88745117]), 
            np.array([ 168.91772461, -227.32530212, 1264.42041016]), 
            np.array([-171.64820862,  167.7848053 , 1274.49621582]), 
            np.array([ 170.23286438, -229.47463989, 1264.04638672])
        ]
        for i in range(len(dataunlabeledResults)):
            result = dataunlabeledResults[i]
            expectedResult = expectedDataunlabeledResults[i]
            np.testing.assert_almost_equal(result, expectedResult, self.rounding_precision)
        
        #Test for first 50 marker names
        markersResults = result_59993[2][0:50]
        expectedMarkersResults = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 
        'CLAV', 'STRN', 'RBAK', 'LSHO', 'LELB', 'LWRA', 'LWRB', 'LFIN', 'RSHO', 
        'RELB', 'RWRA', 'RWRB', 'RFIN', 'LASI', 'RASI', 'LPSI', 'RPSI', 'LTHI', 
        'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB', 'RANK', 
        'RHEE', 'RTOE', 'CentreOfMass', 'CentreOfMassFloor', 'HEDO', 'HEDA', 
        'HEDL', 'HEDP', 'LCLO', 'LCLA', 'LCLL', 'LCLP', 'LFEO', 'LFEA', 'LFEL', 'LFEP', 'LFOO']
        self.assertEqual(markersResults, expectedMarkersResults)

        #Test that loading a non-existent filename will raise an exception.
        with self.assertRaises(Exception):
            pycgmIO.loadC3D("NonExistentFile")

    def test_loadCSV(self):
        """
        This function tests pycgmIO.loadCSV(filename), where filename
        is a string indicating the file path of a CSV file to load. This
        function returns an array of [motionData, unlabeledMotionData, labels].

        This function uses the file Sample_Static.csv in 
        SampleData for testing.

        We test that an exception is raised when loading a non-existent
        file name.
        """
        filename = 'SampleData/ROM/Sample_Static.csv'
        Sample_Static_filename = os.path.join(self.cwd, filename)

        #Call loadCSV with the filename from SampleData
        Sample_Static_result = pycgmIO.loadCSV(Sample_Static_filename)

        #Test for some frames from motionData
        motionData = Sample_Static_result[0]
        motionDataResults = [
            motionData[0]['LFHD'],
            motionData[125]['RASI'],
            motionData[2]['LPSI'],
            motionData[12]['LKNE'],
            motionData[22]['C7'],
            motionData[273]['RANK']
        ]
        expectedMotionDataResults = [
            np.array([ 174.5749207,  324.513031 , 1728.94397  ]),
            np.array([ 353.3344727,  345.1920471, 1033.201172 ]),
            np.array([ 191.5829468,  175.4567261, 1050.240356 ]),
            np.array([ 88.88719177, 242.1836853 , 529.8156128 ]),
            np.array([ 251.1347656,  164.8985748, 1527.874634 ]),
            np.array([427.6519165 , 188.9484558 ,  93.37301636])
        ]
        for i in range(len(motionDataResults)):
            result = motionDataResults[i]
            expectedResult = expectedMotionDataResults[i]
            np.testing.assert_almost_equal(result, expectedResult, self.rounding_precision)

        #Test for some frames from unlabeledMotionData
        unlabeledMotionData = Sample_Static_result[1]
        unlabeledMotionDataResults = [
            unlabeledMotionData[0]['*111'],
            unlabeledMotionData[10]['*112'],
            unlabeledMotionData[272]['*113'],
            unlabeledMotionData[1]['*114'],
            unlabeledMotionData[12]['*112'],
            unlabeledMotionData[94]['*111']
        ]
        expectedUnlabeledMotionDataResults = [
            np.array([692.8970947, 423.9462585, 1240.289063 ]),
            np.array([-226.0103607, 404.8680725, 1214.214111]),
            np.array([-82.37133789, 230.3377533, 1359.47583]),
            np.array([568.5709229, 260.5100708, 1361.798462]),
            np.array([-225.9597473, 404.8319092, 1214.285522]),
            np.array([692.8276978, 422.5860901, 1239.788574])
        ]
        for i in range(len(unlabeledMotionDataResults)):
            result = unlabeledMotionDataResults[i]
            expectedResult = expectedUnlabeledMotionDataResults[i]
            np.testing.assert_almost_equal(result, expectedResult, self.rounding_precision)

        #Test for first 50 label names
        labelsResults = Sample_Static_result[2][0:50]
        expectedLabelsResults = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10',
         'CLAV', 'STRN', 'RBAK', 'LSHO', 'LELB', 'LWRA', 'LWRB', 
         'LFIN', 'RSHO', 'RELB', 'RWRA', 'RWRB', 'RFIN', 'LASI', 
         'RASI', 'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 
         'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 
         'RTOE', 'HEDO', 'HEDA', 'HEDL', 'HEDP', 'LCLO', 'LCLA', 
         'LCLL', 'LCLP', 'LFEO', 'LFEA', 'LFEL', 'LFEP', 'LFOO', 
         'LFOA', 'LFOL']
        self.assertEqual(labelsResults, expectedLabelsResults)

        #Test that loading a non-existent filename will raise an exception.
        with self.assertRaises(Exception):
            pycgmIO.loadCSV("NonExistentFile")

    def test_loadData(self):
        """
        This function tests pycgmIO.loadData(filename), where filename
        is a string indicating the file path of a CSV or C3D file to load.

        This function uses the files 59993_Frame_Static.c3d and 
        Sample_Static.csv in SampleData for testing.

        This function calls loadC3D and loadCSV, so we only test loadData
        to ensure that those functions are called properly.

        We also test loading a non-existent filename.
        """
        filename = 'SampleData/ROM/Sample_Static.csv'
        csv_filename = os.path.join(self.cwd, filename)
        c3d_filename = os.path.join(self.cwd, getfilenames(1)[1])

        #Test loading a csv file
        csvData = pycgmIO.loadData(csv_filename)
        
        #Test for some frames in csvResultData
        csvResults = [
            csvData[0]['LFHD'],
            csvData[16]['LWRA'],
            csvData[25]['C7'],
            csvData[100]['RANK'],
            csvData[12]['RKNE']
        ]
        expectedCsvResults = [
            np.array([174.5749207, 324.513031, 1728.94397]),
            np.array([-233.2779846, 485.1967163, 1128.858276]),
            np.array([251.1916809, 164.7823639, 1527.859253]),
            np.array([427.6116943, 188.8884583, 93.36972809]),
            np.array([417.5567017, 241.5111389, 523.7767334])
        ]
        for i in range(len(csvResults)):
            result = csvResults[i]
            expectedResult = expectedCsvResults[i]
            np.testing.assert_almost_equal(result, expectedResult, self.rounding_precision)
        
        #Test loading a c3d file
        c3dData = pycgmIO.loadData(c3d_filename)

        #Test for some frames in c3dResultData
        c3dResults = [
            c3dData[0]['LFHD'],
            c3dData[16]['LWRA'],
            c3dData[25]['C7'],
            c3dData[100]['RANK'],
            c3dData[12]['RKNE']
        ]
        expectedC3dResults = [
            np.array([60.1229744, 132.4755249, 1485.8293457]),
            np.array([-422.2036438, 432.76647949, 1199.96057129]),
            np.array([-27.17804909, -8.29536247, 1301.43286133]),
            np.array([52.61398697, -127.04923248, 58.46214676]),
            np.array([96.54218292, -111.24856567, 412.34362793])
        ]
        for i in range(len(c3dResults)):
            result = c3dResults[i]
            expectedResult = expectedC3dResults[i]
            np.testing.assert_almost_equal(result, expectedResult, self.rounding_precision)

        #Test that loading a non-existent filename returns None.
        self.assertIsNone(pycgmIO.loadData("NonExistentFile"))

    def test_dataAsArray(self):
        """
        This function tests pycgmIO.dataAsArray(data), where
        data is a dictionary of marker data. The function returns
        the dictionary data as an array of dictionaries.

        We test cases where the input is lists or numpy arrays. We test
        cases where the arrays are not all the same shape, when there are more
        than 3 arrays per dictionary key, and when dictionary keys are empty.

        We test exceptions from keys with less than 3 arrays of data, 
        empty arrays of data, or inconsistent shapes of arrays across keys.
        """

        tests = [
            {'A': [[1, 2], [4, 5], [7, 8]], 'B': [[4, 5], [7, 8], [10, 11]]},

            {'A': [np.array([1, 2]), np.array([4, 5]), np.array([7, 8])],
             'B': [np.array([4, 5]), np.array([7, 8]), np.array([10, 11])]},

            {'A': [[1, 2], [4, 5], [7]]},

            {'A': [[2], [4], [6], [8]]},

            {'A': [[], [4, 5], [7, 8, 9]]},
        ]
        expectedResults = [
            [{'A': np.array([1, 4, 7]), 'B': np.array([ 4,  7, 10])}, 
             {'A': np.array([2, 5, 8]), 'B': np.array([ 5,  8, 11])}],

            [{'A': np.array([1, 4, 7]), 'B': np.array([ 4,  7, 10])}, 
             {'A': np.array([2, 5, 8]), 'B': np.array([ 5,  8, 11])}],

            [{'A': np.array([1, 4, 7])}],

            [{'A': np.array([2, 4, 6])}],

            []
        ]

        for i in range(len(tests)):
            #Call dataAsArray() with the values from tests
            result = pycgmIO.dataAsArray(tests[i])
            expectedResult = expectedResults[i]
            np.testing.assert_equal(result, expectedResult)

        exceptionTests = [
            {},
            {'A': []},
            {'A': [[1], [2]]},
            {'A': [[1, 2], [2, 3], [3, 4]],
             'B': [[4, 5], [5, 6], [6]]}
        ]
        for test in exceptionTests:
            with self.assertRaises(Exception):
                pycgmIO.dataAsArray(test)

    def test_dataAsDict(self):
        """
        This function tests pycgmIO.dataAsDict(data, npArray=False), where
        data is a list of dictionaries of marker data. This function returns
        a data as a dictionary.

        We test cases with multiple markers with the same length of data,
        empty arrays, non-array dictionary values, and inconsistent keys
        across the indices of data.
        """

        tests = [
            [{'A': [1, 2, 3], 'B': [4, 5, 6]},
            {'A': [2, 3, 4], 'B': [5, 6, 7]}],

            [{'A': [1, 2], 'B': [4]},
             {'A': [4], 'B': []}],

            [{'A': [1, 2]},
             {'A': [4, 5], 'B': [6, 7]}],

            [{'A': 2} , {'B': [6, 7]}],

            []
        ]
        expectedResults = [
            {'A': [[1, 2, 3], [2, 3, 4]], 'B': [[4, 5, 6], [5, 6, 7]]},

            {'A': [[1, 2], [4]], 'B': [[4], []]},
            
            {'A': [[1, 2], [4, 5]], 'B': [[6, 7]]},

            {'A': [2], 'B': [[6, 7]]},

            {}
        ]

        for i in range(len(tests)):
            result = pycgmIO.dataAsDict(tests[i])
            expectedResult = expectedResults[i]
            np.testing.assert_equal(result, expectedResult)

        #Test that data is returned as a numpy array if npArray = True
        data = [{'A': [1, 2, 3]}]
        result = pycgmIO.dataAsDict(data, npArray=True)

        for key in result:
            resultData = result[key]
            self.assertIsInstance(resultData, np.ndarray)
            self.assertNotIsInstance(resultData, list)