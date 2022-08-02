import itertools
import time

from .kinematics.dynamic import CalcAxes, CalcAngles
from ..utils.constants import POINT_DTYPE

import numpy.lib.recfunctions as rfn


class Calculations:
    def __init__(self, axis_function_set, angle_function_set):
        self.axis_function_set = axis_function_set
        self.angle_function_set = angle_function_set

    @property
    def axis_functions(self):
        if self.axis_function_set is None:
            return CalcAxes().funcs
        else:
            return self.axis_function_set

    @property
    def angle_functions(self):
        if self.angle_function_set is None:
            return CalcAngles().funcs
        else:
            return self.angle_function_set

    @property
    def returned_axes(self):
        return list(itertools.chain.from_iterable([function.returned_axes for function in self.axis_functions]))

    @property
    def returned_angles(self):
        return list(itertools.chain.from_iterable([function.returned_angles for function in self.angle_functions]))


    def update_trial_names(self, trial_names):
        for trial_name in trial_names:
            for function in self.axis_functions:
                function.parameter_values[trial_name] =  []
            for function in self.angle_functions:
                function.parameter_values[trial_name] =  []

    def expand_parameters_from_data(self, trials):
        """
        Expand each function's parameter names to values to in passed data
        """
        for trial_name in trials.dynamic.dtype.names:
            for function in self.axis_functions + self.angle_functions:

                for parameter_name in function.required_markers:
                # ============== Markers ============== 
                    if parameter_name in trials.dynamic[trial_name].markers.dtype.names:
                        # Use marker name to retrieve from marker struct
                        expanded_parameter = self.get_markers(trials.dynamic[trial_name].markers, parameter_name, True)
                        if expanded_parameter is not None:
                            expanded_parameter = expanded_parameter[0]
                        function.parameter_values[trial_name].append(expanded_parameter)
                    else:
                        function.parameter_values[trial_name].append(None)


                for parameter_name in function.required_measurements:
                # ============== Measurements ============== 
                    if parameter_name in trials.static.measurements.dtype.names:
                        # Use measurement name to retrieve from measurements struct
                        try:
                            expanded_parameter = trials.static.measurements[parameter_name][0]
                        except ValueError:
                            expanded_parameter = None

                        function.parameter_values[trial_name].append(expanded_parameter)
                    else:
                        function.parameter_values[trial_name].append(None)

                for parameter_name in function.required_axes:
                # ============== Axes ============== 
                    if parameter_name in trials.dynamic[trial_name].axes.dtype.names:
                        # Add parameter from axes struct
                        function.parameter_values[trial_name].append(trials.dynamic[trial_name].axes[parameter_name][0])
                    else:
                        function.parameter_values[trial_name].append(None)

                for parameter_name in function.required_angles:
                # ============== Angles ============== 
                    if parameter_name in trials.dynamic[trial_name].angles.dtype.names:
                        # Add parameter from angles struct
                        function.parameter_values[trial_name].append(trials.dynamic[trial_name].angles[parameter_name])
                    else:
                        function.parameter_values[trial_name].append(None)

                for parameter_name in function.required_constants:
                # ============== Constants ============== 
                    function.parameter_values[trial_name].append(parameter_name)


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

