import re
import time

import numpy as np
from numpy.lib import recfunctions as rfn

from ..calc import static
from .new_io import marker_dtype, load_c3d, loadVSK
from .pycgmIO import loadData


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

    start = time.time()


    # HACK
    # load static trial, measurements for use in getStatic (has not been refactored)
    old_static_data = loadData(static_trial_filename)
    uncalibrated_measurements = loadVSK(measurement_filename)
    uncalibrated_measurements_dict = dict(zip(uncalibrated_measurements[0], uncalibrated_measurements[1]))

    # calibrate subject measurements
    calibrated_measurements_dict = static.getStatic(old_static_data, uncalibrated_measurements_dict)
    calibrated_measurements_split = [list(calibrated_measurements_dict.keys()), list(calibrated_measurements_dict.values())]

    measurements_struct = structure_measurements(calibrated_measurements_split)
    static_trial = load_c3d(static_trial_filename)

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

    end = time.time()
    print(f'Total time to load and structure model: {end-start}\n')

    return model


def get_markers(arr, names, points_only=True, debug=False):
    start = time.time()

    if isinstance(names, str):
        names = [names]
    num_frames = arr[0][0].shape[0]

    rec = rfn.repack_fields(arr[names]).view(marker_dtype()).reshape(len(names), int(num_frames))


    if points_only:
        rec = rec['point'][['x', 'y', 'z']]

    rec = rfn.structured_to_unstructured(rec)

    end = time.time()
    if debug:
        print(f'Time to get {len(names)} markers: {end-start}')

    return rec


def add_virtual_marker(dynamic_trial, name, marker_data):
    '''
    Params:

    dynamic_trial: structured array containing dynamic trial data
        e.g. subject.dynamic.filename

    name: name of the virtual marker

    marker_data: structured array containing the data for the virtual marker
        dtype is [('frame', '<f8'), ('point', [('x', '<f8'), ('y', '<f8'), ('z', '<f8')])]


    Returns a new dynamic trial with the virtual marker added
        return dtype is [('markers', dynamic_struct.dtype)]
    '''
    start = time.time()

    if np.ndim(marker_data) == 2:
        marker_data = marker_data[0]

    num_frames = dynamic_trial.markers[0][0].shape[0]
    num_markers = len(dynamic_trial.markers.dtype.names)
    dtype_names = list(dynamic_trial.markers.dtype.names)

    trial_uns = get_markers(dynamic_trial.markers, dtype_names, False).reshape(num_markers, num_frames, 4)

    marker_data.dtype = np.dtype(("4f8"))
    marker_data = marker_data.reshape(num_frames, 4)
    trial_uns = np.insert(trial_uns, 0, marker_data, axis=0)

    dtype_names.insert(0, name)
    num_markers = len(dtype_names)

    marker_positions = np.empty((num_markers, num_frames), dtype=(("4f8")))
    marker_positions[:] = trial_uns
    marker_positions.dtype = marker_dtype()

    marker_xyz = [(key, (marker_dtype(), (num_frames,))) for key in dtype_names]
    dynamic_struct = np.empty((1), dtype=marker_xyz)

    for i, key in enumerate(dtype_names):
        dynamic_struct[key][0][:, np.newaxis] = marker_positions[i]

    trial_dtype = [('markers', dynamic_struct.dtype)]

    dynamic_trial = np.zeros((1), dtype=trial_dtype)
    dynamic_trial['markers'] = dynamic_struct
    dynamic_trial = dynamic_trial.view(np.recarray)

    end = time.time()
    print(f'Time to extend dynamic struct with marker {name}: {end-start}')

    return dynamic_trial


def update_subject_struct(subject, modified_trial_name, with_virtual_markers):
    '''
    Restructures a subject after a trial has been modified
    '''
    start = time.time()

    # Create a new dynamic trial dtype
    # Uses the existing dynamic trial dtypes, unless it's the modified trial
    dynamic_dtype = []
    for trial_name in subject.dynamic.dtype.names:
        if trial_name == modified_trial_name:
            dynamic_dtype.append((trial_name, with_virtual_markers.dtype))
        else:
            dynamic_dtype.append((trial_name, subject.dynamic[trial_name].dtype))


    # Create a new subject
    subject_dtype = [('static', [('markers', subject.static.markers.dtype), \
                                 ('measurements', subject.static.measurements.dtype)]), \
                     ('dynamic', dynamic_dtype)]
    new_subject = np.zeros((1), dtype=subject_dtype)

    # verify that the new dtype is correct
    print(f'\n{new_subject["dynamic"][modified_trial_name].dtype == with_virtual_markers.dtype=}')

    # Copy the static data
    new_subject['static']['markers'] = subject.static.markers
    new_subject['static']['measurements'] = subject.static.measurements

    # Copy the dynamic data, replacing the modified trial
    for trial_name in subject.dynamic.dtype.names:
        if trial_name == modified_trial_name:
            new_subject['dynamic'][trial_name] = with_virtual_markers
        else:
            new_subject['dynamic'][trial_name]['markers'] = subject.dynamic[trial_name].markers

    new_subject = new_subject.view(np.recarray)

    end = time.time()
    print(f'Time to update subject struct: {end-start}\n')

    return new_subject


def add_dynamic_marker(subject, dynamic_trial_name, marker_name, marker_data):
    """ 
    TODO consider whether or not marker_data already has frame numbers, add if not
    """ 
    with_added_marker = add_virtual_marker(subject.dynamic[dynamic_trial_name], marker_name, marker_data)
    new_subject = update_subject_struct(subject, dynamic_trial_name, with_added_marker)

    return new_subject
