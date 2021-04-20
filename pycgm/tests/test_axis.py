from mock import patch
import pycgm.axis as axis
import pytest
import numpy as np

rounding_precision = 6


class TestUpperBodyAxis:
    """
    This class tests the upper body axis functions in axis.py:
        hand_axis
    """
    rand_num = np.random.randint(0, 10)
    nan = np.nan

    @pytest.mark.parametrize(["rwra", "rwrb", "lwra", "lwrb", "rfin", "lfin", "wrist_jc", "vsk", "mock_return_val", "expected_mock_args", "expected",],
        [
            # Test from running sample data
            (
                np.array([776.51898193, 495.68103027, 1108.38464355]), np.array([830.9072876, 436.75341797, 1119.11901855]),
                np.array([-249.28146362, 525.32977295, 1117.09057617]), np.array([-311.77532959, 477.22512817, 1125.1619873]),
                np.array([863.71374512, 524.4475708, 1074.54248047]), np.array([-326.65890503, 558.34338379, 1091.04284668]),
                [
                    [
                        [rand_num, rand_num, rand_num, 793.3281430325068],
                        [rand_num, rand_num, rand_num, 451.2913478825204],
                        [rand_num, rand_num, rand_num, 1084.4325513020426],
                        [0, 0, 0, 1],
                    ],
                    [
                        [rand_num, rand_num, rand_num, -272.4594189740742],
                        [rand_num, rand_num, rand_num, 485.801522109477],
                        [rand_num, rand_num, rand_num, 1091.3666238350822],
                        [0, 0, 0, 1],
                    ],
                ],
                {"RightHandThickness": 34.0, "LeftHandThickness": 34.0},
                [ [-324.53477798, 551.88744289, 1068.02526837], [859.80614366, 517.28239823, 1051.97278945] ],
                [
                    [
                        [-280.528396605, 501.27745056000003, 1121.126281735],
                        [-272.4594189740742, 485.801522109477, 1091.3666238350822],
                        [-326.65890503, 558.34338379, 1091.04284668],
                        24.0,
                    ],
                    [
                        [803.713134765, 466.21722411999997, 1113.75183105],
                        [793.3281430325068, 451.2913478825204, 1084.4325513020426],
                        [863.71374512, 524.4475708, 1074.54248047],
                        24.0,
                    ],
                ],
                [
                    np.array(
                        [
                            [0.15061613, 0.31001409, 0.93872576, 859.80614366],
                            [-0.72638693, 0.67880636, -0.10762882, 517.28239823],
                            [-0.67057946, -0.66566748, 0.32742937, 1051.97278944],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [-0.08516279, 0.27149019, 0.95866593, -324.53477798],
                            [-0.79815387, -0.59451803, 0.09746127, 551.88744289],
                            [0.59640397, -0.75686285, 0.26732176, 1068.02526837],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
            ),
            # Testing when values are added to wrist_jc
            (
                np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
                [
                    [
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, 4],
                        [rand_num, rand_num, rand_num, 3],
                        [0, 0, 0, 1],
                    ],
                    [
                        [rand_num, rand_num, rand_num, 9],
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, -6],
                        [0, 0, 0, 1],
                    ],
                ],
                {"RightHandThickness": 0.0, "LeftHandThickness": 0.0},
                [[0, 0, 0], [0, 0, 0]],
                [ [[0, 0, 0], [9, 0, -6], [0, 0, 0], 7.0], [[0, 0, 0], [0, 4, 3], [0, 0, 0], 7.0], ],
                [
                    np.array(
                        [
                            [nan, nan, nan, 0.0],
                            [nan, nan, nan, 0.0],
                            [0.0, 0.8, 0.6, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [nan, nan, nan, 0.0],
                            [nan, nan, nan, 0.0],
                            [0.83205029, 0.0, -0.5547002, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
            ),
            # Testing when values are added to wrist_jc, frame['RFIN'] and frame['LFIN']
            (
                np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([1, -9, 6]), np.array([-6, 3, 8]),
                [
                    [
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, 4],
                        [rand_num, rand_num, rand_num, 3],
                        [0, 0, 0, 1],
                    ],
                    [
                        [rand_num, rand_num, rand_num, 9],
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, -6],
                        [0, 0, 0, 1],
                    ],
                ],
                {"RightHandThickness": 0.0, "LeftHandThickness": 0.0},
                [[0, 0, 0], [0, 0, 0]],
                [
                    [[0, 0, 0], [9, 0, -6], [-6, 3, 8], 7.0],
                    [[0, 0, 0], [0, 4, 3], [1, -9, 6], 7.0],
                ],
                [
                    np.array(
                        [
                            [nan, nan, nan, 0.0],
                            [nan, nan, nan, 0.0],
                            [0.0, 0.8, 0.6, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [nan, nan, nan, 0.0],
                            [nan, nan, nan, 0.0],
                            [0.83205029, 0.0, -0.5547002, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
            ),
            # Testing when values are added to wrist_jc, frame['RFIN'], frame['LFIN'], frame['RWRA'], and frame['LWRA']
            (
                np.array([4, 7, 6]), np.array([0, 0, 0]), np.array([-4, 5, 3]), np.array([0, 0, 0]), np.array([1, -9, 6]), np.array([-6, 3, 8]),
                [
                    [
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, 4],
                        [rand_num, rand_num, rand_num, 3],
                        [0, 0, 0, 1],
                    ],
                    [
                        [rand_num, rand_num, rand_num, 9],
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, -6],
                        [0, 0, 0, 1],
                    ],
                ],
                {"RightHandThickness": 0.0, "LeftHandThickness": 0.0},
                [[0, 0, 0], [0, 0, 0]],
                [ [[-2.0, 2.5, 1.5], [9, 0, -6], [-6, 3, 8], 7.0], [[2.0, 3.5, 3.0], [0, 4, 3], [1, -9, 6], 7.0], ],
                [
                    np.array(
                        [
                            [-0.14834045, -0.59336181, 0.79114908, 0.0],
                            [0.98893635, -0.08900427, 0.11867236, 0.0],
                            [0.0, 0.8, 0.6, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [0.55384878, -0.05538488, 0.83077316, 0.0],
                            [-0.030722, -0.99846508, -0.046083, 0.0],
                            [0.83205029, 0.0, -0.5547002, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
            ),
            # Testing when values are added to frame and wrist_jc
            (
                np.array([4, 7, 6]), np.array([0, -5, 4]), np.array([-4, 5, 3]), np.array([-3, 2, -7]), np.array([1, -9, 6]), np.array([-6, 3, 8]),
                [
                    [
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, 4],
                        [rand_num, rand_num, rand_num, 3],
                        [0, 0, 0, 1],
                    ],
                    [
                        [rand_num, rand_num, rand_num, 9],
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, -6],
                        [0, 0, 0, 1],
                    ],
                ],
                {"RightHandThickness": 0.0, "LeftHandThickness": 0.0},
                [[0, 0, 0], [0, 0, 0]],
                [
                    [[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 7.0],
                    [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 7.0],
                ],
                [
                    np.array(
                        [
                            [0.81373347, -0.34874292, 0.46499055, 0.0],
                            [0.58123819, 0.48824008, -0.65098678, 0.0],
                            [0.0, 0.8, 0.6, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [0.19988898, -0.93281525, 0.29983347, 0.0],
                            [-0.5174328, -0.36035499, -0.7761492, 0.0],
                            [0.83205029, 0.0, -0.5547002, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
            ),
            # Testing when values are added to frame, wrist_jc, and vsk
            (
                np.array([4, 7, 6]), np.array([0, -5, 4]), np.array([-4, 5, 3]), np.array([-3, 2, -7]), np.array([1, -9, 6]), np.array([-6, 3, 8]),
                [
                    [
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, 4],
                        [rand_num, rand_num, rand_num, 3],
                        [0, 0, 0, 1],
                    ],
                    [
                        [rand_num, rand_num, rand_num, 9],
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, -6],
                        [0, 0, 0, 1],
                    ],
                ],
                {"RightHandThickness": 36.0, "LeftHandThickness": -9.0},
                [[0, 0, 0], [0, 0, 0]],
                [
                    [[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 2.5],
                    [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 25.0],
                ],
                [
                    np.array(
                        [
                            [0.81373347, -0.34874292, 0.46499055, 0.0],
                            [0.58123819, 0.48824008, -0.65098678, 0.0],
                            [0.0, 0.8, 0.6, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [0.19988898, -0.93281525, 0.29983347, 0.0],
                            [-0.5174328, -0.36035499, -0.7761492, 0.0],
                            [0.83205029, 0.0, -0.5547002, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
            ),
            # Testing when values are added to frame, wrist_jc, vsk and mock_return_val
            (
                np.array([4, 7, 6]), np.array([0, -5, 4]), np.array([-4, 5, 3]), np.array([-3, 2, -7]), np.array([1, -9, 6]), np.array([-6, 3, 8]),
                [
                    [
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, 4],
                        [rand_num, rand_num, rand_num, 3],
                        [0, 0, 0, 1],
                    ],
                    [
                        [rand_num, rand_num, rand_num, 9],
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, -6],
                        [0, 0, 0, 1],
                    ],
                ],
                {"RightHandThickness": 36.0, "LeftHandThickness": -9.0},
                [[-6, 4, -4], [2, 8, 1]],
                [
                    [[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 2.5],
                    [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 25.0],
                ],
                [
                    np.array(
                        [
                            [0.91168461, -0.34188173, 0.22792115, 2.0],
                            [-0.04652421, 0.46524211, 0.88396, 8.0],
                            [-0.40824829, -0.81649658, 0.40824829, 1.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [-0.21615749, -0.94092085, 0.2606605, -6.0],
                            [-0.18683841, -0.22217524, -0.9569376, 4.0],
                            [0.95831485, -0.25555063, -0.12777531, -4.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
            ),
            # Testing that when frame and wrist_jc are composed of lists of ints and vsk values are ints
            (
                [4, 7, 6], [0, -5, 4], [-4, 5, 3], [-3, 2, -7], [1, -9, 6], [-6, 3, 8],
                [
                    [
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, 4],
                        [rand_num, rand_num, rand_num, 3],
                        [0, 0, 0, 1],
                    ],
                    [
                        [rand_num, rand_num, rand_num, 9],
                        [rand_num, rand_num, rand_num, 0],
                        [rand_num, rand_num, rand_num, -6],
                        [0, 0, 0, 1],
                    ],
                ],
                {"RightHandThickness": 36, "LeftHandThickness": -9},
                [[-6, 4, -4], [2, 8, 1]],
                [
                    [[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 2.5],
                    [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 25.0],
                ],
                [
                    np.array(
                        [
                            [0.91168461, -0.34188173, 0.22792115, 2.0],
                            [-0.04652421, 0.46524211, 0.88396, 8.0],
                            [-0.40824829, -0.81649658, 0.40824829, 1.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [-0.21615749, -0.94092085, 0.2606605, -6.0],
                            [-0.18683841, -0.22217524, -0.9569376, 4.0],
                            [0.95831485, -0.25555063, -0.12777531, -4.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
            ),
            # Testing that when frame and wrist_jc are composed of numpy arrays of ints and vsk values are ints
            (
                np.array([4, 7, 6]), np.array([0, -5, 4]), np.array([-4, 5, 3]), np.array([-3, 2, -7]), np.array([1, -9, 6]), np.array([-6, 3, 8]),
                np.array(
                    [
                        [
                            [rand_num, rand_num, rand_num, 0],
                            [rand_num, rand_num, rand_num, 4],
                            [rand_num, rand_num, rand_num, 3],
                            [0, 0, 0, 1],
                        ],
                        [
                            [rand_num, rand_num, rand_num, 9],
                            [rand_num, rand_num, rand_num, 0],
                            [rand_num, rand_num, rand_num, -6],
                            [0, 0, 0, 1],
                        ],
                    ],
                    dtype="int",
                ),
                {"RightHandThickness": 36, "LeftHandThickness": -9},
                [[-6, 4, -4], [2, 8, 1]],
                [
                    [[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 2.5],
                    [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 25.0],
                ],
                [
                    np.array(
                        [
                            [0.91168461, -0.34188173, 0.22792115, 2.0],
                            [-0.04652421, 0.46524211, 0.88396, 8.0],
                            [-0.40824829, -0.81649658, 0.40824829, 1.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [-0.21615749, -0.94092085, 0.2606605, -6.0],
                            [-0.18683841, -0.22217524, -0.9569376, 4.0],
                            [0.95831485, -0.25555063, -0.12777531, -4.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
            ),
            # Testing that when frame and wrist_jc are composed of lists of floats and vsk values are floats
            (
                [4.0, 7.0, 6.0], [0.0, -5.0, 4.0], [-4.0, 5.0, 3.0], [-3.0, 2.0, -7.0], [1.0, -9.0, 6.0], [-6.0, 3.0, 8.0],
                [
                    [
                        [rand_num, rand_num, rand_num, 0.0],
                        [rand_num, rand_num, rand_num, 4.0],
                        [rand_num, rand_num, rand_num, 3.0],
                        [0, 0, 0, 1],
                    ],
                    [
                        [rand_num, rand_num, rand_num, 9.0],
                        [rand_num, rand_num, rand_num, 0.0],
                        [rand_num, rand_num, rand_num, -6.0],
                        [0, 0, 0, 1],
                    ],
                ],
                {"RightHandThickness": 36.0, "LeftHandThickness": -9.0},
                [[-6, 4, -4], [2, 8, 1]],
                [
                    [[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 2.5],
                    [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 25.0],
                ],
                [
                    np.array(
                        [
                            [0.91168461, -0.34188173, 0.22792115, 2.0],
                            [-0.04652421, 0.46524211, 0.88396, 8.0],
                            [-0.40824829, -0.81649658, 0.40824829, 1.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [-0.21615749, -0.94092085, 0.2606605, -6.0],
                            [-0.18683841, -0.22217524, -0.9569376, 4.0],
                            [0.95831485, -0.25555063, -0.12777531, -4.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
            ),
            # Testing that when frame and wrist_jc are composed of numpy arrays of floats and vsk values are floats
            # "rwra", "rwrb", "lwra", "lwrb", "rfin", "lfin", "wrist_jc", "vsk", "mock_return_val", "expected_mock_args", "expected"
            (
                np.array([4.0, 7.0, 6.0], dtype="float"), np.array([0.0, -5.0, 4.0], dtype="float"),
                np.array([-4.0, 5.0, 3.0], dtype="float"), np.array([-3.0, 2.0, -7.0], dtype="float"),
                np.array([1.0, -9.0, 6.0], dtype="float"), np.array([-6.0, 3.0, 8.0], dtype="float"),
                np.array(
                    [
                        [
                            [rand_num, rand_num, rand_num, 0.0],
                            [rand_num, rand_num, rand_num, 4.0],
                            [rand_num, rand_num, rand_num, 3.0],
                            [0, 0, 0, 1],
                        ],
                        [
                            [rand_num, rand_num, rand_num, 9.0],
                            [rand_num, rand_num, rand_num, 0.0],
                            [rand_num, rand_num, rand_num, -6.0],
                            [0, 0, 0, 1],
                        ],
                    ],
                    dtype="float",
                ),
                {"RightHandThickness": 36.0, "LeftHandThickness": -9.0},
                [[-6, 4, -4], [2, 8, 1]],
                [
                    [[-3.5, 3.5, -2.0], [9, 0, -6], [-6, 3, 8], 2.5],
                    [[2.0, 1.0, 5.0], [0, 4, 3], [1, -9, 6], 25.0],
                ],
                [
                    np.array(
                        [
                            [0.91168461, -0.34188173, 0.22792115, 2.0],
                            [-0.04652421, 0.46524211, 0.88396, 8.0],
                            [-0.40824829, -0.81649658, 0.40824829, 1.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [-0.21615749, -0.94092085, 0.2606605, -6.0],
                            [-0.18683841, -0.22217524, -0.9569376, 4.0],
                            [0.95831485, -0.25555063, -0.12777531, -4.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
            ),
        ],
    )
    def test_hand_joint_center(
        self, rwra, rwrb, lwra, lwrb, rfin, lfin, wrist_jc, vsk, mock_return_val, expected_mock_args, expected,
    ):
        """
        This test provides coverage of the hand_joint_center function in axis.py, defined as
        hand_joint_center(rwra, rwrb, lwra, lwrb, rfin, lfin, wrist_jc, vsk)

        This test takes 11 parameters:
        rwra: 1x3 RWRA marker
        rwrb: 1x3 RWRB marker
        lwra: 1x3 LWRA marker
        lwrb: 1x3 LWRB marker
        rfin: 1x3 RFIN marker
        lfin: 1x3 LFIN marker
        wrist_jc: array containing the x,y,z position of the wrist joint center
        vsk: dictionary containing subject measurements from a VSK file
        mock_return_val: the value to be returned by the mock for find_joint_center
        expected_mock_args: the expected arguments used to call the mocked function, find_joint_center
        expected: the expected result from calling hand_joint_center on frame, wrist_jc, and vsk

        This test is checking to make sure the hand joint axis is calculated correctly given the input parameters.
        This tests mocks find_joint_center to make sure the correct parameters are being passed into it given the parameters
        passed into hand_joint_center, and to also ensure that hand_joint_center returns the correct value considering
        the return value of find_joint_center, mock_return_val.

        Using RWRA, RWRB, LWRA, and LWRB from the given frame dictionary,
        RWRI = (RWRA+RWRB)/2
        LWRI = (LWRA+LWRB)/2
        aka the midpoints of the markers for each direction.

        LHND is calculated using the Rodriques' rotation formula with the LWRI, LWJC, and LFIN as reference points. The thickness of the left hand is also applied in the calculations.
        The same can be said for the RHND, but with respective markers and measurements (aka RWRI, RWJC, and RFIN).
        z_axis = LWJC - LHND
        y-axis = LWRI - LRWA
        x-axis = y-axis \cross z-axis
        y-axis = z-axis \cross x-axis

        This is for the handJC left axis, and is the same for the right axis but with the respective markers.
        The origin for each direction is calculated by adding each axis to each HND marker.

        Lastly, it checks that the resulting output is correct when frame and wrist_jc are composed of lists of ints,
        numpy arrays of ints, lists of floats, and numpy arrays of floats and vsk values are either an int or a float.
        wrist_jc cannot be a numpy array due to it not being shaped like a multi-dimensional array.
        """
        
        with patch.object(
            axis, "find_joint_center", side_effect=mock_return_val
        ) as mock_find_joint_center:
            result = axis.hand_axis(rwra, rwrb, lwra, lwrb, rfin, lfin, wrist_jc, vsk)

        # Asserting that there were only 2 calls to find_joint_center
        np.testing.assert_equal(mock_find_joint_center.call_count, 2)

        # Asserting that the correct params were sent in the 1st (left) call to find_joint_center
        np.testing.assert_almost_equal(expected_mock_args[0][0], mock_find_joint_center.call_args_list[0][0][0], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][1], mock_find_joint_center.call_args_list[0][0][1], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][2], mock_find_joint_center.call_args_list[0][0][2], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[0][3], mock_find_joint_center.call_args_list[0][0][3], rounding_precision)

        ## Asserting that the correct params were sent in the 2nd (right) call to find_joint_center
        np.testing.assert_almost_equal(expected_mock_args[1][0], mock_find_joint_center.call_args_list[1][0][0], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][1], mock_find_joint_center.call_args_list[1][0][1], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][2], mock_find_joint_center.call_args_list[1][0][2], rounding_precision)
        np.testing.assert_almost_equal(expected_mock_args[1][3], mock_find_joint_center.call_args_list[1][0][3], rounding_precision)

        ## Asserting that findShoulderJC returned the correct result given the return value given by mocked find_joint_center
        np.testing.assert_almost_equal(result[0], expected[0], rounding_precision)
        np.testing.assert_almost_equal(result[1], expected[1], rounding_precision)
