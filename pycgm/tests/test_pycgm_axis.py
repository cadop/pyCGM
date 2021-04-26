import pycgm.axis as axis
import pytest
import numpy as np

rounding_precision = 5


class TestLowerBodyAxis:
    """
    This class tests the lower body axis functions in pyCGM.py:
    pelvisJointCenter
    hipJointCenter
    """
    nan_3d = [np.nan, np.nan, np.nan]
    rand_int = np.random.randint(0, 10)
    rand_coor = [np.random.randint(0, 10), np.random.randint(
        0, 10), np.random.randint(0, 10)]

    @pytest.mark.parametrize(["frame", "expected"], [
        # Test from running sample data
        ({'C7': np.array([251.22619629, 229.75683594, 1533.77624512]), 'T10': np.array([228.64323425, 192.32041931, 1279.6418457]), 'CLAV': np.array([256.78051758, 371.28042603, 1459.70300293]), 'STRN': np.array([251.67492676, 414.10391235, 1292.08508301])},
         [[[256.23991128535846, 365.30496976939753, 1459.662169500559], [257.1435863244796, 364.21960599061947, 1459.588978712983], [256.0843053658035, 364.32180498523223, 1458.6575930699294]], [256.149810236564, 364.3090603933987, 1459.6553639290375]]),
        # Basic test with a variance of 1 in the x and y dimensions of the markers
        ({'C7': np.array([1, 1, 0]), 'T10': np.array([0, 1, 0]), 'CLAV': np.array([1, 0, 0]), 'STRN': np.array([0, 0, 0])},
         [[[1, 6, 0], [1, 7, 1], [0, 7, 0]], [1, 7, 0]]),
        # Setting the markers so there's no variance in the x-dimension
        ({'C7': np.array([0, 1, 0]), 'T10': np.array([0, 1, 0]), 'CLAV': np.array([0, 0, 0]), 'STRN': np.array([0, 0, 0])},
         [[nan_3d, nan_3d, nan_3d], nan_3d]),
        # Setting the markers so there's no variance in the y-dimension
        ({'C7': np.array([1, 0, 0]), 'T10': np.array([0, 0, 0]), 'CLAV': np.array([1, 0, 0]), 'STRN': np.array([0, 0, 0])},
         [[nan_3d, nan_3d, nan_3d], nan_3d]),
        # Setting each marker in a different xy quadrant
        ({'C7': np.array([-1, 1, 0]), 'T10': np.array([1, 1, 0]), 'CLAV': np.array([-1, -1, 0]), 'STRN': np.array([1, -1, 0])},
         [[[-1, 5, 0], [-1, 6, -1], [0, 6, 0]], [-1, 6, 0]]),
        # Setting values of the markers so that midpoints will be on diagonals
        ({'C7': np.array([-2, 1, 0]), 'T10': np.array([1, 2, 0]), 'CLAV': np.array([-1, -2, 0]), 'STRN': np.array([2, -1, 0])},
         [[[-2.8973666, 3.69209979, 0], [-3.21359436, 4.64078309, -1], [-2.26491106, 4.95701085, 0]], [-3.21359436, 4.64078309, 0]]),
        # Adding the value of 1 in the z dimension for all 4 markers
        ({'C7': np.array([1, 1, 1]), 'T10': np.array([0, 1, 1]), 'CLAV': np.array([1, 0, 1]), 'STRN': np.array([0, 0, 1])},
         [[[1, 6, 1], [1, 7, 2], [0, 7, 1]], [1, 7, 1]]),
        # Setting the z dimension value higher for C7 and CLAV
        ({'C7': np.array([1, 1, 2]), 'T10': np.array([0, 1, 1]), 'CLAV': np.array([1, 0, 2]), 'STRN': np.array([0, 0, 1])},
         [[[1, 6, 2], [0.29289322, 7, 2.70710678], [0.29289322, 7, 1.29289322]], [1, 7, 2]]),
        # Setting the z dimension value higher for C7 and T10
        ({'C7': np.array([1, 1, 2]), 'T10': np.array([0, 1, 2]), 'CLAV': np.array([1, 0, 1]), 'STRN': np.array([0, 0, 1])},
         [[[1, 4.24264069, 5.24264069], [1, 4.24264069, 6.65685425], [0, 4.94974747, 5.94974747]], [1, 4.94974747, 5.94974747]]),
        # Testing that when frame is a list of ints
        ({'C7': [1, 1, 2], 'T10': [0, 1, 2], 'CLAV': [1, 0, 1], 'STRN': [0, 0, 1]},
         [[[1, 4.24264069, 5.24264069], [1, 4.24264069, 6.65685425], [0, 4.94974747, 5.94974747]],
          [1, 4.94974747, 5.94974747]]),
        # Testing that when frame is a numpy array of ints
        ({'C7': np.array([1, 1, 2], dtype='int'), 'T10': np.array([0, 1, 2], dtype='int'),
          'CLAV': np.array([1, 0, 1], dtype='int'), 'STRN': np.array([0, 0, 1], dtype='int')},
         [[[1, 4.24264069, 5.24264069], [1, 4.24264069, 6.65685425], [0, 4.94974747, 5.94974747]],
          [1, 4.94974747, 5.94974747]]),
        # Testing that when frame is a list of floats
        ({'C7': [1.0, 1.0, 2.0], 'T10': [0.0, 1.0, 2.0], 'CLAV': [1.0, 0.0, 1.0], 'STRN': [0.0, 0.0, 1.0]},
         [[[1, 4.24264069, 5.24264069], [1, 4.24264069, 6.65685425], [0, 4.94974747, 5.94974747]],
          [1, 4.94974747, 5.94974747]]),
        # Testing that when frame is a numpy array of floats
        ({'C7': np.array([1.0, 1.0, 2.0], dtype='float'), 'T10': np.array([0.0, 1.0, 2.0], dtype='float'),
          'CLAV': np.array([1.0, 0.0, 1.0], dtype='float'), 'STRN': np.array([0.0, 0.0, 1.0], dtype='float')},
         [[[1, 4.24264069, 5.24264069], [1, 4.24264069, 6.65685425], [0, 4.94974747, 5.94974747]],
          [1, 4.94974747, 5.94974747]])])
    def test_thorax_axis(_, frame, expected):
        """
        This test provides coverage of the thoraxJC function in pyCGM.py, defined as thoraxJC(frame)

        This test takes 2 parameters:
        frame: dictionary of marker lists
        expected: the expected result from calling thoraxJC on frame

        The function uses the CLAV, C7, STRN, and T10 markers from the frame to calculate the midpoints of the front, back, left, and right center positions of the thorax.
        The thorax axis vector components are then calculated using subtracting the pairs (left to right, back to front) of the aforementioned midpoints.
        Afterwords, the axes are made orthogonal by calculating the cross product of each individual axis.
        Finally, the head axis is then rotated around the x axis based off the thorax offset angle in the VSK.

        This test is checking to make sure the thorax joint center and thorax joint axis are calculated correctly given
        the 4 coordinates given in frame. This includes testing when there is no variance in the coordinates,
        when the coordinates are in different quadrants, when the midpoints will be on diagonals, and when the z
        dimension is variable. Lastly, it checks that the resulting output is correct when frame is a list of ints, a
        numpy array of ints, a list of floats, and a numpy array of floats.
        """
        c7 = frame["C7"] if "C7" in frame else None
        t10 = frame["T10"] if "T10" in frame else None
        clav = frame["CLAV"] if "CLAV" in frame else None
        strn = frame["STRN"] if "STRN" in frame else None

        expected_axis = np.zeros((4, 4))
        expected_axis[3, 3] = 1.0
        expected_axis[0, :3] = np.subtract(expected[0][0], expected[1])
        expected_axis[1, :3] = np.subtract(expected[0][1], expected[1])
        expected_axis[2, :3] = np.subtract(expected[0][2], expected[1])
        expected_axis[:3, 3] = expected[1]

        result = axis.thorax_axis(clav, c7, strn, t10)

        np.testing.assert_almost_equal(
            result, expected_axis, rounding_precision)
