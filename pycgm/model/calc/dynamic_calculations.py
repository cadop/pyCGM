import copy
import itertools
import time

import numpy.lib.recfunctions as rfn

from ..utils.constants import POINT_DTYPE
from .kinematics.dynamic import CalcDynamic


class DynamicCalc:
    def __init__(self, function_set=None):
        self.function_set = copy.deepcopy(CalcDynamic().funcs) if function_set is None else function_set

    @property
    def required_measurements(self):
        return set(itertools.chain.from_iterable([function.required_measurements for function in self.function_set]))

    @property
    def returned_axes(self):
        return set(itertools.chain.from_iterable([function.returned_axes for function in self.function_set]))

    @property
    def returned_angles(self):
        return set(itertools.chain.from_iterable([function.returned_angles for function in self.function_set]))


    def index_of_function(self, function_name):
        for idx, function in enumerate(self.function_set):
            if function.name == function_name:
                return idx
        return len(self.function_set)


    def update_trial_names(self, trial_names):
        for trial_name in trial_names:
            for function in self.function_set:
                function.parameter_values[trial_name] =  []

    def expand_parameters_from_data(self, trials):
        """
        Expand each function's parameter names to values to in passed data
        """
        for trial_name in trials.dynamic.dtype.names:
            for function in self.function_set:
                function.parameter_values[trial_name] = []

                for parameter_name in function.required_markers:
                # ============== Markers ============== 
                    if parameter_name in trials.dynamic[trial_name].markers.dtype.names:
                        # Use marker name to retrieve from marker struct
                        expanded_parameter = self.get_markers(trials.dynamic[trial_name].markers, parameter_name, True)[0]
                        function.parameter_values[trial_name].append(expanded_parameter)
                    else:
                        function.parameter_values[trial_name].append(None)


                for parameter_name in function.required_measurements:
                # ============== Measurements ============== 
                    if parameter_name in trials.static.calibrated.measurements.dtype.names:
                        # Use measurement name to retrieve from measurements struct
                        function.parameter_values[trial_name].append(trials.static.calibrated.measurements[parameter_name][0])
                    else:
                        function.parameter_values[trial_name].append(None)

                if 'axes' in trials.dynamic[trial_name].dtype.names:
                    for parameter_name in function.required_axes:
                            
                    # ============== Axes ============== 
                        if parameter_name in trials.dynamic[trial_name].axes.dtype.names:
                            # Add parameter from axes struct
                            function.parameter_values[trial_name].append(trials.dynamic[trial_name].axes[parameter_name][0])
                        else:
                            function.parameter_values[trial_name].append(None)

                if 'angles' in trials.dynamic[trial_name].dtype.names:
                    for parameter_name in function.required_angles:
                    # ============== Angles ============== 
                        if parameter_name in trials.dynamic[trial_name].angles.dtype.names:
                            # Add parameter from angles struct
                            function.parameter_values[trial_name].append(trials.dynamic[trial_name].angles[parameter_name])
                        else:
                            function.parameter_values[trial_name].append(None)


    def get_markers(self, arr, names, points_only=True, debug=False):
        start = time.time()

        if isinstance(names, str):
            names = [names]
        num_frames = arr[0][0].shape[0]

        if any(name not in arr[0].dtype.names for name in names):
            return None

        rec = rfn.repack_fields(arr[names]).view(POINT_DTYPE).reshape(len(names), int(num_frames))


        if points_only:
            rec = rec['point'][['x', 'y', 'z']]

        rec = rfn.structured_to_unstructured(rec)

        end = time.time()
        if debug:
            print(f'Time to get {len(names)} markers: {end-start}')

        return rec
