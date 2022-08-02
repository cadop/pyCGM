import itertools
import time

from .function import Function
from .kinematics.dynamic import CalcAxes, CalcAngles
from ..utils.constants import POINT_DTYPE

import numpy.lib.recfunctions as rfn

class Calculations:
    def __init__(self):
        self.axis_functions = self.construct_axis_functions()
        self.angle_functions = self.construct_angle_functions()

    def construct_axis_functions(self):
        return [Function(func) for func in CalcAxes().funcs]

    def construct_angle_functions(self):
        return [Function(func) for func in CalcAngles().funcs]

    def update_trial_names(self, trial_names):
        for trial_name in trial_names:
            for function in self.axis_functions:
                function.parameter_values[trial_name] =  []
            for function in self.angle_functions:
                function.parameter_values[trial_name] =  []

    @property
    def returned_axes(self):
        return list(itertools.chain.from_iterable([function.result_names for function in self.axis_functions]))

    @property
    def returned_angles(self):
        return list(itertools.chain.from_iterable([function.result_names for function in self.angle_functions]))

    def expand_parameters_from_data(self, data):
        """
        Expand each function's parameter names to values to in passed data
        """
        for trial_name in data.dynamic.dtype.names:
            for function in self.axis_functions + self.angle_functions:
                for parameter_name in function.parameter_names:

                    if parameter_name in data.dynamic[trial_name].markers.dtype.names:
                        # Use marker name to retrieve from marker struct
                        new_parameter = self.get_markers(data.dynamic[trial_name].markers, parameter_name, True)
                        if new_parameter is not None:
                            new_parameter = new_parameter[0]
                        function.parameter_values[trial_name].append(new_parameter)

                    elif parameter_name in data.static.measurements.dtype.names:
                        # Use measurement name to retrieve from measurements struct
                        try:
                            new_parameter = data.static.measurements[parameter_name][0]
                        except ValueError:
                            new_parameter = None

                        function.parameter_values[trial_name].append(new_parameter)

                    elif parameter_name in data.dynamic[trial_name].axes.dtype.names:
                        # Add parameter from axes struct
                        function.parameter_values[trial_name].append(data.dynamic[trial_name].axes[parameter_name][0])

                    elif parameter_name in data.dynamic[trial_name].angles.dtype.names:
                        # Add parameter from angles struct
                        function.parameter_values[trial_name].append(data.dynamic[trial_name].angles[parameter_name])

                    else:
                        if not isinstance(parameter_name, str):
                            # Parameter is a constant, append as is
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

