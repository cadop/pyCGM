import re

import numpy as np
from numpy.lib import recfunctions as rfn

from ..calc.kinematics import static
from .constants import POINT_DTYPE
from .io import load_c3d
from .io import loadVSK


def structure_vsk(measurements):
    sm_names = measurements[0]
    sm_dtype = []
    for key in sm_names:
        if key == "GCS":
            sm_dtype.append((key, 'f8', (3,3)))
        else:
            sm_dtype.append((key, 'f8'))
    sm_dtype.append(("FlatFoot", '?'))
    measurements[1].append(0)
    measurements_struct = np.array(tuple(measurements[1]), dtype=sm_dtype)
    return measurements_struct


def structure_model(static_trial_filename, dynamic_trials, measurement_filename, static_calculations, dynamic_calculations):

    if isinstance(dynamic_trials, str):
        dynamic_trials = [dynamic_trials]

    # Load measurements
    measurements = loadVSK(measurement_filename)

    # Structure uncalibrated measurements
    measurements_struct = structure_vsk(measurements)
    
    # Get intersection of (VSK input + static input/ouput + dynamic input/output) names
    required_static_measurements = static_calculations.required_measurements
    returned_static_measurements = static_calculations.returned_measurements
    required_dynamic_measurements = dynamic_calculations.required_measurements
    
    calibrated_measurement_names = required_static_measurements.union(returned_static_measurements).union(required_dynamic_measurements)

    # Load static
    static_trial, num_frames = load_c3d(static_trial_filename, return_frame_count=True)

    # Structure static dtype
    calibrated_axes_dtype         = np.dtype([(key, 'f8', (num_frames, 3, 4)) for key in static_calculations.returned_axes])
    calibrated_angles_dtype       = np.dtype([(key, 'f8', (num_frames, 3)) for key in static_calculations.returned_angles])

    calibrated_measurements_dtype = []
    for key in calibrated_measurement_names:
        if key == "GCS":
            calibrated_measurements_dtype.append((key, 'f8', (3,3)))
        elif key == "FlatFoot":
            calibrated_measurements_dtype.append((key, '?'))
        else:
            calibrated_measurements_dtype.append((key, 'f8'))
    calibrated_measurements_dtype = np.dtype(calibrated_measurements_dtype)

    static_dtype = [ ('markers', static_trial.dtype), \
                     ('measurements', measurements_struct.dtype), \
                     ('calibrated', [
                                      ('axes', calibrated_axes_dtype), \
                                      ('angles', calibrated_angles_dtype), \
                                      ('measurements', calibrated_measurements_dtype)
                                    ]
                     )
                   ]

    dynamic_dtype = []
    marker_structs = []
    parsed_filenames = []

    for trial_name in dynamic_trials:
        dynamic_trial, num_frames = load_c3d(trial_name, return_frame_count=True)

        marker_dtype = dynamic_trial.dtype
        axes_dtype   = np.dtype([(key, 'f8', (num_frames, 3, 4)) for key in dynamic_calculations.returned_axes])
        angles_dtype = np.dtype([(key, 'f8', (num_frames, 3)) for key in dynamic_calculations.returned_angles])

        # parse just the name of the trial
        filename = re.findall(r'[^\/]+(?=\.)', trial_name)[0]
        parsed_filenames.append(filename)

        marker_structs.append(dynamic_trial)

        trial_dtype = [('markers', marker_dtype),
                       ('axes',    axes_dtype),
                       ('angles',  angles_dtype)]

        dynamic_dtype.append((filename, trial_dtype))

    model_dtype = [('static', static_dtype), \
                   ('dynamic', dynamic_dtype)]

    model = np.zeros((1), dtype=model_dtype)
    
    # Set input data
    model['static']['markers'] = static_trial
    model['static']['measurements'] = measurements_struct

    # Copy required input measurements to calibrated measurements struct
    vsk_names = set(measurements[0])
    required_by_both = list(vsk_names.intersection(calibrated_measurement_names))
    intersecting_values = model['static']['measurements'][required_by_both]
    model['static']['calibrated']['measurements'][required_by_both] = intersecting_values

    for i, trial_name in enumerate(parsed_filenames):
        model['dynamic'][trial_name]['markers'] = marker_structs[i]

    model = model.view(np.recarray)

    return model
