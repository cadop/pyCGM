import numpy as np
from numpy.lib import recfunctions as rfn

from .constants import POINT_DTYPE



# TODO
# load before
# don't pass filenames, pass data
def structure_model(measurement_data, static_data, dynamic_data, static_calculations, dynamic_calculations):
    """

    Note
    ----
    Using different dtypes for measurements|markers|axes|angles of the same name
    will throw an error because the tuples are different
        ("FlatFoot", np.bool_)
        ("FlatFoot", bool)

    """

    static_data, num_frames = static_data.data, static_data.num_frames
    
    # Get intersection of (VSK input + static input/ouput + dynamic input/output) names
    required_static_measurements = static_calculations.required_measurements
    returned_static_measurements = static_calculations.returned_measurements
    required_dynamic_measurements = dynamic_calculations.required_measurements
    calibrated_measurement_tuples = required_static_measurements | returned_static_measurements | required_dynamic_measurements

    # Structure static dtype
    calibrated_measurements_dtype = np.dtype([(name, dtype) for name, dtype in calibrated_measurement_tuples])
    calibrated_axes_dtype         = np.dtype([(key, dtype.base, (num_frames,) + dtype.shape) for key, dtype in static_calculations.returned_axes])
    calibrated_angles_dtype       = np.dtype([(key, dtype.base, (num_frames,) + dtype.shape) for key, dtype in static_calculations.returned_angles])

    static_dtype = [ ('markers', static_data.dtype), \
                     ('measurements', measurement_data.dtype), \
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

    for dynamic_trial in dynamic_data:
        dynamic_trial, num_frames, dynamic_trial_filename = dynamic_trial.data, dynamic_trial.num_frames, dynamic_trial.trial_name

        marker_dtype = dynamic_trial.dtype
        axes_dtype   = np.dtype([(key, dtype.base, (num_frames,) + dtype.shape) for key, dtype in dynamic_calculations.returned_axes])
        angles_dtype = np.dtype([(key, dtype.base, (num_frames,) + dtype.shape) for key, dtype in dynamic_calculations.returned_angles])

        marker_structs.append(dynamic_trial)

        trial_dtype = [('markers', marker_dtype),
                       ('axes',    axes_dtype),
                       ('angles',  angles_dtype)]

        dynamic_dtype.append((dynamic_trial_filename, trial_dtype))

    model_dtype = [('static', static_dtype), \
                   ('dynamic', dynamic_dtype)]

    model = np.zeros((1), dtype=model_dtype)
    
    # Set input data
    model['static']['markers'] = static_data
    model['static']['measurements'] = measurement_data

    # Copy required input measurements to calibrated measurements struct
    vsk_names = set(measurement_data.dtype.names)
    required_by_both = list({x[0] for x in calibrated_measurement_tuples} & vsk_names)
    intersecting_values = model['static']['measurements'][required_by_both]
    model['static']['calibrated']['measurements'][required_by_both] = intersecting_values

    for i, trial_name in enumerate([trial.trial_name for trial in dynamic_data]):
        model['dynamic'][trial_name]['markers'] = marker_structs[i]

    model = model.view(np.recarray)

    return model
