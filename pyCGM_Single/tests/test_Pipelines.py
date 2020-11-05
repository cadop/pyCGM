import pytest
import numpy as np
from pyCGM_Single.pyCGM_Helpers import getfilenames
from pyCGM_Single.pycgmIO import loadData, dataAsDict
from pyCGM_Single.clusterCalc import target_dict, segment_dict
from pyCGM_Single.pyCGM import pelvisJointCenter
import pyCGM_Single.Pipelines as Pipelines
import os

rounding_precision = 8
cwd = os.getcwd()
if(cwd.split(os.sep)[-1]=="pyCGM_Single"):
    parent = os.path.dirname(cwd)
    os.chdir(parent)
cwd = os.getcwd()

def load_data_from_file(x, calcSacrum = False):
    """
    Loads motion capture data from dynamic and static trials.
    Returns as data dictionaries. Optionally calculate the 
    sacrum marker before creating the data dictionaries.
    """
    cur_dir = cwd
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

"""
The following functions test Pipelines.butterFilter(data, cutoff, Fs),
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
@pytest.mark.parametrize("data, cutoff, Fs, expectedResult", [
    (listTests[0], 20, 120, expectedResults[0]),
    (listTests[1], 20, 120, expectedResults[1]),
    (listTests[2], 20, 120, expectedResults[2]),
    (np.array(listTests[0]), 20, 120, expectedResults[0]),
    (np.array(listTests[1]), 20, 120, expectedResults[1]),
    (np.array(listTests[2]), 20, 120, expectedResults[2])
])
def test_butterFilterAccuracy(data, cutoff, Fs, expectedResult):
    #Call butterFilter() with the values from listTests and numpyTests
    result = Pipelines.butterFilter(data, cutoff, Fs)
    np.testing.assert_almost_equal(result, expectedResult, rounding_precision)

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
def test_butterFilterExceptions(data, cutoff, Fs):
    with pytest.raises(Exception):
        Pipelines.butterFilter(data, cutoff, Fs)

"""
The following functions test Pipelines.filt(data, cutoff, Fs),
where data is a 2darray of numbers to filter, cutoff is the cutoff
frequency to filter, and Fs is the sampling frequency of the data.

We test cases where inputs are lists of positive and negative floats 
and ints.

We test exceptions raised when the arrays in data are not of the same
shape, when the length of data is too short, when the cutoff frequency 
value is negative, zero, or too large, when the sampling frequency 
value is negative, zero, or too small, and when data is not a numpy array.
"""
@pytest.mark.parametrize("data, cutoff, Fs, expectedResult", [
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
def test_filtAccuracy(data, cutoff, Fs, expectedResult):
    result = Pipelines.filt(data, cutoff, Fs)
    np.testing.assert_almost_equal(result, expectedResult, rounding_precision)

validData = np.array([[i, i+1] for i in range(16)])
@pytest.mark.parametrize("data, cutoff, Fs", [
    #Test that if the shape of each array in data is not the same, an exception is raised
    (np.array([[0,1],[1,2],[2,3],[3],[4],[5],[6],[7], 
      [8],[9],[10],[11],[12],[13],[14],[15]]), 20, 120),
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
])
def test_filtExceptions(data, cutoff, Fs):
    with pytest.raises(Exception):
        Pipelines.filt(data, cutoff, Fs)

"""
The following functions test Pipelines.prep(trajs), where trajs
is a dictionary containing numpy arrays.
"""
@pytest.mark.parametrize("trajs, expectedResult", [
    ({'trajOne': np.array([[217.19961548, -82.35484314, 332.2684021 ],
                           [257.19961548, -32.35484314, 382.2684021 ]])},
    [{'trajOne': np.array([217.19961548, -82.35484314, 332.2684021 ])}, 
     {'trajOne': np.array([257.19961548, -32.35484314, 382.2684021 ])}]),
    ({'a': np.array([[1, 2, 3], [4, 5, 6]]), 
      'b': np.array([[7, 8, 9], [10, 11, 12]])},
    [{'a': np.array([1, 2, 3]), 'b': np.array([7, 8, 9])}, 
     {'a': np.array([4, 5, 6]), 'b': np.array([10, 11, 12])}])
])
def test_prepAccuracy(trajs, expectedResult):
    result = Pipelines.prep(trajs)
    np.testing.assert_equal(result, expectedResult)

@pytest.mark.parametrize("trajs", [
    #Test that if data[key] is not a numpy array, an exception is raised
    ({'trajOne': [[217.19961548, -82.35484314, 332.2684021 ],
                [257.19961548, -32.35484314, 382.2684021 ]]})
])

def test_prepExceptions(trajs):
    with pytest.raises(Exception):
        Pipelines.prep(trajs)

"""
The following functions test Pipelines.clearMarker(data, name), where
data is an array of dictionary of markers lists and name is the 
name of a marker to clear.

We test cases where the values in data are lists and numpy arrays.
We test the case where the marker name is not a key in data.
"""

@pytest.mark.parametrize("data, name, expectedResult", [
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
def test_clearMarkerAccuracy(data, name, expectedResult):
    result = Pipelines.clearMarker(data, name)
    np.testing.assert_equal(result, expectedResult)

"""
The following functions test Pipelines.filtering(Data), where
Data is a dictionary of marker lists. This function calls
Pipelines.filt().

We test cases where inputs are numpy arrays of positive and 
negative floats and ints.

We test the exception raised when the values in Data are 
not numpy arrays.
"""
@pytest.mark.parametrize("Data, expectedResult", [
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
    {'LFHD': np.array([[i,i+1] for i in range(16)], dtype=np.int32)}, #[[0,1], [1,2], ... [15, 16]]
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
def test_filteringAccuracy(Data, expectedResult):
    result = Pipelines.filtering(Data)
    for key in Data:
        np.testing.assert_almost_equal(result[key], expectedResult[key], rounding_precision)

@pytest.mark.parametrize("Data", [
    #Test that if data is not a numpy array, an exception is raised
    ({'LFHD':[[i] for i in range(20)]})
])
def test_filteringExceptions(Data):
    with pytest.raises(Exception):
        Pipelines.filtering(Data)

"""
The following functions test Pipelines.transform_from_static(data,static,key,useables,s),
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
data,static = load_data_from_file(3) #Indicates that we use files from SampleData/Sample_2/ for testing
frameNumTests = [0, 1, 2, -1, 10, 100] #Frame numbers to be cleared for LFHD to test gap filling.
for frame in frameNumTests:
    data['LFHD'][frame] = np.array([np.nan, np.nan, np.nan])

@pytest.mark.parametrize("data, static, key, useables, s, expectedResult", [
    (data, static, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 0, 
    [-1007.8145678233541, 71.28465078977477, 1522.6626006179151]),
    (data, static, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 1, 
    [-1007.7357797476452, 71.30567599088612, 1522.6056345492811]),
    (data, static, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 2, 
    [-1007.6561772477821, 71.32644261551039, 1522.5516787767372]),
    (data, static, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], -1, 
    [710.8111428914814, -18.282265916438064, 1549.7284035675332]),
    (data, static, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 10, 
    [-1006.9916284913861, 71.48482387826286, 1522.2367625952083]),
    (data, static, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 100, 
    [-995.8183141045178, 73.11905329024174, 1526.9072499889455]),
])
def test_transform_from_staticAccuracy(data, static, key, useables, s, expectedResult):
    result = Pipelines.transform_from_static(data,static,key,useables,s)
    np.testing.assert_almost_equal(result, expectedResult, rounding_precision)

@pytest.mark.parametrize("data, static, key, useables, s", [
    #Test that if useables is not at least 3 unique markers, an exception is raised
    (data, static, 'LFHD', ['RFHD', 'RFHD', 'RFHD'], 0),
    (data, static, 'LFHD', ['RFHD'], 0),
    (data, static, 'LFHD', [], 0),
    #Test that if s, the frame number, is out of range, an exception is raised
    (data, static, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 6100),
    #Test that if the marker name does not exist, an exception is raised
    (data, static, 'InvalidKey', ['RFHD', 'RBHD', 'LBHD'], 0)
])

def test_transform_from_staticExceptions(data, static, key, useables, s):
    with pytest.raises(Exception):
        Pipelines.transform_from_static(data,static,key,useables,s)

"""
The following functions test Pipelines.transform_from_mov(data,key,clust,last_time,i),
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
data,_ = load_data_from_file(3) #Indicates that we use files from SampleData/Sample_2/ for testing
frameNumTests = [11, 12, -1, 15, 100] #Frame numbers to be cleared to test gap filling
for frame in frameNumTests:
    data['LFHD'][frame] = np.array([np.nan, np.nan, np.nan])

@pytest.mark.parametrize("data,key,clust,last_time,i,expectedResult", [
    (data, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 3, 11,
    [-1002.66354241,    81.22543097,  1521.82434027]),
    (data, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 3, 12,
    [-1002.57542092,    81.24378237,  1521.80517086]),
    (data, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 3, -1,
    [714.4191660275219, -8.268045936969543, 1550.088229312965]),
    (data, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 3, 15,
    [-1002.30681304,    81.29768863,  1521.76708531]),
    (data, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 3, 100,
    [-991.7315609567293, 82.91868701883672, 1526.597213251877]),
    (data, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 15, 16,
    [np.nan, np.nan, np.nan])
])
def test_transform_from_movAccuracy(data,key,clust,last_time,i,expectedResult):
    result = Pipelines.transform_from_mov(data,key,clust,last_time,i)
    np.testing.assert_almost_equal(result, expectedResult, rounding_precision)

@pytest.mark.parametrize("data,key,clust,last_time,i", [
    #Test that if clust is not at least 3 unique markers, an exception is raised
    (data, 'LFHD', ['RFHD', 'RFHD', 'RFHD'], 0, 1),
    (data, 'LFHD', ['RFHD'], 0, 1),
    (data, 'LFHD', [], 0, 1),
    #Test that if i, the frame number, it out of range, an exception is raised
    (data, 'LFHD', ['RFHD', 'RBHD', 'LBHD'], 0, 6100),
    #Test that if the marker name does not exist, an exception is raised
    (data, 'InvalidKey', ['RFHD', 'RBHD', 'LBHD'], 0, 1)
])
def test_transform_from_movExceptions(data,key,clust,last_time,i):
    with pytest.raises(Exception):
        Pipelines.transform_from_mov(data,key,clust,last_time,i)

"""
The following functions test Pipelines.segmentFinder(key,data,targetDict,segmentDict,j,missings),
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
data,_ = load_data_from_file(3) #Indicates that we use files from SampleData/Sample_2/ for testing
targetDict = target_dict()
segmentDict = segment_dict()
@pytest.mark.parametrize("key,data,targetDict,segmentDict,j,missings,expectedResult", [
    ('LFHD',data,targetDict,segmentDict,10,{},
        ['RFHD', 'RBHD', 'LBHD']),
    ('LFHD',data,targetDict,segmentDict,10,{'LFHD':[]},
        ['RFHD', 'RBHD', 'LBHD']),
    ('LFHD',data,targetDict,segmentDict,10,{'RFHD':[10]},
        ['RBHD', 'LBHD']),
    ('LFHD',data,targetDict,segmentDict,10,{'LBHD':[10], 'RFHD':[10]},
        ['RBHD']),
    ('LFHD',data,targetDict,segmentDict,10,{'LBHD':[10], 'RFHD':[10], 'RBHD':[10]},
        []),
    ('C7',data,targetDict,segmentDict,10,{},
        ['STRN', 'CLAV', 'T10', 'RBAK', 'RSHO', 'LSHO']),
    ('RPSI',data,targetDict,segmentDict,10,{},
        ['LPSI', 'LASI', 'RASI']),
    ('LKNE',data,targetDict,segmentDict,10,{},
        ['LTHI'])
])
def test_segmentFinderAccuracy(key,data,targetDict,segmentDict,j,missings,expectedResult):
    result = Pipelines.segmentFinder(key,data,targetDict,segmentDict,j,missings)
    assert result == expectedResult

@pytest.mark.parametrize("key,data,targetDict,segmentDict,j,missings", [
    #Test that the marker name must exist
    ('InvalidKey',data,targetDict,segmentDict,10,{})
])
def test_segmentFinderExceptions(key,data,targetDict,segmentDict,j,missings):
    with pytest.raises(Exception):
        Pipelines.segmentFinder(key,data,targetDict,segmentDict,j,missings)

"""
The following functions test Pipelines.rigid_fill(Data, static),
where Data is an array of dictionaries of marker data,
and static is an array of dictionaries of static trial data.

This function fills gaps for frames with missing data.

We test simulate missing data by clearing six different frames
and testing the gap filling result.

We test the exception raised when the marker SACR does not
exist in Data.
"""
data,static = load_data_from_file(3, calcSacrum=True) #True indicates we calculate SACR before testing
framesToClear = [1, 10, 12, 15, 100, -1]
for frameNum in framesToClear:
    data['LFHD'][frameNum] = np.array([np.nan, np.nan, np.nan])

result = Pipelines.rigid_fill(data, static)

@pytest.mark.parametrize("result, expectedResult", [
    (result['LFHD'][1], [-1007.73577975, 71.30567599, 1522.60563455]),
    (result['LFHD'][10], [-1002.75400789, 81.21320267, 1521.8559697 ]),
    (result['LFHD'][12], [-1002.57942155, 81.2521139, 1521.81737938]),
    (result['LFHD'][15], [-1002.31227389, 81.30886597, 1521.78327027]),
    (result['LFHD'][100], [-991.70150174, 82.79915329, 1526.67335699]),
    (result['LFHD'][-1], [ 714.9015343, -9.04757279, 1550.23346378]),
])
def test_rigid_fill_accuracy(result, expectedResult):
    np.testing.assert_almost_equal(result, expectedResult, rounding_precision)

@pytest.mark.parametrize("data, static", [
    (load_data_from_file(3)) #Loading without calculating sacrum causes an Exception
])
def test_rigid_fill_exceptions(data, static):
    #Test that the marker SACR must exist:
    with pytest.raises(Exception):
        Pipelines.rigid_fill(data, static)