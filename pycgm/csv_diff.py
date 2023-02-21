import numpy as np


def diff_pycgm_csv(model, trial_name, csv_filename):
    """
    Tests the output axes and angles of a trial against 
    a CSV file from the original PyCGM.

    Returns if a mismatch is found.
    """

    # Default names of output axes
    axis_array_fields = ['Pelvis', 'Hip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RFoot', 'LFoot', 'Head',
                         'Thorax', 'RClav', 'LClav', 'RHum', 'LHum', 'RRad', 'LRad', 'RHand', 'LHand']

    # Default names of output angles
    angle_array_fields = ['Pelvis', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RFoot', 'LFoot', 'Head', 'Thorax',
                          'Neck', 'Spine', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist']

    # 'Pelvis' in the structured axis array corresponds to 'PELO', 'PELX', 'PELY', 'PELZ' in the csv (12 values)
    axis_slice_map = { key: slice( index*12, index*12+12, 1) for index, key in enumerate(axis_array_fields) }

    # 'Pelvis' in the structured angle array corresponds to 'X', 'Y', 'Z' in the csv (3 values)
    angle_slice_map = { key: slice( index*3, index*3+3, 1) for index, key in enumerate(angle_array_fields) }

    model_output_axes   = model.data.dynamic[trial_name].axes
    model_output_angles = model.data.dynamic[trial_name].angles

    csv_results = np.genfromtxt(csv_filename, delimiter=',')
    print(f"\nLoaded csv: {csv_filename}")

    accurate = True
    # Compare axes, frame by frame 
    for frame_idx, frame in enumerate(csv_results):
            frame = frame[58:]

            for key, slc in axis_slice_map.items():
                original_o = frame[slc.start    :slc.start + 3]
                original_x = frame[slc.start + 3:slc.start + 6] - original_o
                original_y = frame[slc.start + 6:slc.start + 9] - original_o
                original_z = frame[slc.start + 9:slc.stop]      - original_o

                original_matrix = np.array([original_x, original_y, original_z, original_o]).T
                model_matrix = model_output_axes[key][0][frame_idx]

                if not np.allclose(original_matrix, model_matrix):
                    accurate = False
                    error = abs((original_matrix - model_matrix) / original_matrix) * 100
                    print(f'\nAxis mismatch, Frame {frame_idx}: {key} ({error=})%')
                    print(f"{original_matrix=}\n{model_matrix=}")
                    return

    print(f'{trial_name} axes match:', accurate)

    # Compare angles, frame by frame 
    for frame_idx, frame in enumerate(csv_results):
            frame = frame[1:59]

            for key, slc in angle_slice_map.items():
                original_angle = np.array(frame[slc])
                refactored_angle = model_output_angles[key][0][frame_idx]
                if not np.allclose(original_angle, refactored_angle):
                    accurate = False
                    error = abs((original_angle - refactored_angle) / original_angle) * 100
                    print(f'\nAngle mismatch, Frame {frame_idx}: {key} ({error=})%')
                    print(f"{original_angle=}\n{refactored_angle=}")
                    return
    print(f'{trial_name} angles match:', accurate)
