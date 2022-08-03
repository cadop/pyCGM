import re

import numpy as np
from numpy.lib import recfunctions as rfn

from .constants import POINT_DTYPE
from ..calc.kinematics import static
from .io import load_c3d, loadVSK


def structure_model(static_trial_filename, dynamic_trials, measurement_filename, axis_result_keys, angle_result_keys):
    '''Create a structured array containing a model's data

    Parameters
    ----------
    static_trial_filename : str
        Filename of the static trial .c3d

    dynamic_trials : str or list of str
        Filename or list of filenames of dynamic trial .c3d(s)

    measurement_filename : str
        Filename of the subject measurement .vsk

    axis_result_keys : list of str
        A list containing the names of the model's returned axes

    angle_result_keys : list of str
        A list containing the names of the model's returned angles

    Returns
    -------
    model : structured array
        Structured array containing the model's measurements, static trial,
        and dynamic trial(s)

    Notes
    -----
    Accessing measurement data:
        model.static.measurements.{measurement name}
        e.g. model.static.measurements.LeftLegLength

    Accessing static trial data:
        model.static.markers.{marker name}.point.{x, y, z}
        e.g. model.static.markers.LASI.point.x

    Accessing dynamic trial data:
        model.dynamic.{filename}.markers.{marker name}.point.{x, y, z}
        e.g. model.RoboWalk.markers.LASI.point.x
        e.g. model.RoboWalk.axes.Pelvis
        e.g. model.RoboWalk.angles.RHip
    '''

    def structure_measurements(measurements):
        sm_names = measurements[0]
        sm_dtype = []
        for key in sm_names:
            if key == "GCS":
                sm_dtype.append((key, 'f8', (3,3)))
            else:
                sm_dtype.append((key, 'f8'))
        measurements_struct = np.array(tuple(measurements[1]), dtype=sm_dtype)
        return measurements_struct

    if isinstance(dynamic_trials, str):
        dynamic_trials = [dynamic_trials]


    # Load measurements
    uncalibrated_measurements = loadVSK(measurement_filename)
    uncalibrated_measurements_dict = dict(zip(uncalibrated_measurements[0], uncalibrated_measurements[1]))

    # Load static
    static_trial = load_c3d(static_trial_filename)

    # Calibrate subject measurements
    calibrated_measurements_dict = static.getStatic(uncalibrated_measurements_dict, static_trial)
    calibrated_measurements_split = [list(calibrated_measurements_dict.keys()), list(calibrated_measurements_dict.values())]

    # Structure calibrated measurements
    measurements_struct = structure_measurements(calibrated_measurements_split)

    dynamic_dtype = []
    marker_structs = []
    parsed_filenames = []

    for trial_name in dynamic_trials:
        dynamic_trial, num_frames = load_c3d(trial_name, return_frame_count=True)

        marker_dtype = dynamic_trial.dtype
        axes_dtype   = np.dtype([(key, 'f8', (num_frames, 3, 4)) for key in axis_result_keys])
        angles_dtype = np.dtype([(key, 'f8', (num_frames, 3)) for key in angle_result_keys])

        # parse just the name of the trial
        filename = re.findall(r'[^\/]+(?=\.)', trial_name)[0]
        parsed_filenames.append(filename)

        marker_structs.append(dynamic_trial)

        trial_dtype = [('markers', marker_dtype),
                       ('axes',    axes_dtype),
                       ('angles',  angles_dtype)]

        dynamic_dtype.append((filename, trial_dtype))


    model_dtype = [('static', [('markers', static_trial.dtype), \
                               ('measurements', measurements_struct.dtype)]), \
                   ('dynamic', dynamic_dtype)]

    model = np.zeros((1), dtype=model_dtype)
    model['static']['markers'] = static_trial
    model['static']['measurements'] = measurements_struct

    for i, trial_name in enumerate(parsed_filenames):
        model['dynamic'][trial_name]['markers'] = marker_structs[i]

    model = model.view(np.recarray)

    return model


def get_markers(arr, names, points_only=True, debug=False):

    if isinstance(names, str):
        names = [names]
    num_frames = arr[0][0].shape[0]

    rec = rfn.repack_fields(arr[names]).view(POINT_DTYPE).reshape(len(names), int(num_frames))


    if points_only:
        rec = rec['point'][['x', 'y', 'z']]

    rec = rfn.structured_to_unstructured(rec)

    return rec

