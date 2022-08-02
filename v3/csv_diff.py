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
                refactored_x = model_output_axes[key][0][frame_idx][:, 0]
                refactored_y = model_output_axes[key][0][frame_idx][:, 1]
                refactored_z = model_output_axes[key][0][frame_idx][:, 2]
                refactored_o = model_output_axes[key][0][frame_idx][:, 3]
                if not np.allclose(original_o, refactored_o):
                    accurate = False
                    error = abs((original_o - refactored_o) / original_o) * 100
                    print(f'\nAxis mismatch, Frame {frame_idx}: {key}, origin ({error})%')
                    print(f"{original_o=}\n{refactored_o=}")
                    return
                if not np.allclose(original_x, refactored_x):
                    accurate = False
                    error = abs((original_x - refactored_x) / original_x) * 100
                    print(f'\nAxis mismatch, Frame {frame_idx}: {key}, x-axis ({error})%')
                    print(f"{original_x=}\n{refactored_x=}")
                    return
                if not np.allclose(original_y, refactored_y):
                    accurate = False
                    error = abs((original_y - refactored_y) / original_y) * 100
                    print(f'\nAxis mismatch, Frame {frame_idx}: {key}, y-axis ({error})%')
                    print(f"{original_y=}\n{refactored_y=}")
                    return
                if not np.allclose(original_z, refactored_z):
                    accurate = False
                    error = abs((original_z - refactored_z) / original_z) * 100
                    print(f'\nAxis mismatch, Frame {frame_idx}: {key}, z-axis ({error})%')
                    print(f"{original_z=}\n{refactored_z=}")
                    return
    print(f'{trial_name} axes match:', accurate)

    # Compare angles, frame by frame 
    for frame_idx, frame in enumerate(csv_results):
            frame = frame[1:59]

            for key, slc in angle_slice_map.items():
                original_x = frame[slc][0]
                original_y = frame[slc][1]
                original_z = frame[slc][2]
                refactored_x = model_output_angles[key][0][frame_idx][0]
                refactored_y = model_output_angles[key][0][frame_idx][1]
                refactored_z = model_output_angles[key][0][frame_idx][2]
                if not np.allclose(original_x, refactored_x):
                    accurate = False
                    error = abs((original_x - refactored_x) / original_x) * 100
                    print(f'\nAngle mismatch, Frame {frame_idx}: {key} X ({error})%')
                    print(f"{original_x=}\n{refactored_x=}")
                    return
                if not np.allclose(original_y, refactored_y):
                    accurate = False
                    error = abs((original_y - refactored_y) / original_y) * 100
                    print(f'\nAngle mismatch, Frame {frame_idx}: {key} Y ({error})%')
                    print(f"{original_y=}\n{refactored_y=}")
                    return
                if not np.allclose(original_z, refactored_z):
                    accurate = False
                    error = abs((original_z - refactored_z) / original_z) * 100
                    print(f'\nAngle mismatch, Frame {frame_idx}: {key} Z ({error})%')
                    print(f"{original_z=}\n{refactored_z=}")
                    return
    print(f'{trial_name} angles match:', accurate)


