import unittest
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
        labels = ['A', 'B', 'C']
        dataTests = [
            [[[1,2,3],[4,5,6],[7,8,9]],
            [[2,3,4],[5,6,7],[8,9,10]]],
            [[np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9])],
            [np.array([2,3,4]),np.array([5,6,7]),np.array([8,9,10])]]
        ]
        expectedResults = [
            [{'A':[1,2,3], 'B':[4,5,6], 'C':[7,8,9]},
             {'A':[2,3,4], 'B':[5,6,7], 'C':[8,9,10]}],
            [{'A':[1,2,3], 'B':[4,5,6], 'C':[7,8,9]},
             {'A':[2,3,4], 'B':[5,6,7], 'C':[8,9,10]}],
        ]

        for testNum in range(len(dataTests)):
            result = pycgmIO.createMotionDataDict(labels, dataTests[testNum])
            expected = expectedResults[testNum]
            for i in range(len(expected)):
                np.testing.assert_equal(result[i], expected[i])
                
        #Test that if 