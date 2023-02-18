import itertools
import time

import numpy as np
import numpy.lib.recfunctions as rfn

from ..utils.constants import POINT_DTYPE
from .kinematics.static import CalcStatic


class StaticCalc:
    def __init__(self, function_set=None):
        self.function_set = function_set
        self.update_parameter_values()

    @property
    def static_functions(self):
        if self.function_set is None:
            return CalcStatic().funcs
        return self.function_set

    @property
    def required_measurements(self):
        return set(itertools.chain.from_iterable([function.required_measurements for function in self.static_functions]))

    @property
    def returned_measurements(self):
        return set(itertools.chain.from_iterable([function.returned_measurements for function in self.static_functions]))

    @property
    def returned_axes(self):
        return set(itertools.chain.from_iterable([function.returned_axes for function in self.static_functions]))

    @property
    def returned_angles(self):
        return set(itertools.chain.from_iterable([function.returned_angles for function in self.static_functions]))

    @property
    def returned_measurements(self):
        return set(itertools.chain.from_iterable([function.returned_measurements for function in self.static_functions]))


    def index_of_function(self, function_name):
        for idx, function in enumerate(self.static_functions):
            if function.name == function_name:
                return idx
        return len(self.static_functions)


    def update_parameter_values(self):
        for function in self.static_functions:
            function.parameter_values['static'] =  []

    def expand_parameters_from_data(self, data):
        """
        Expand each function's parameter names to values to in passed data
        """
        for function in self.static_functions:
            function.parameter_values['static'] = []

            for parameter_name in function.required_markers:
            # ============== Markers ============== 
                if parameter_name in data.static.markers.dtype.names:
                    # Use marker name to retrieve from marker struct
                    expanded_parameter = self.get_markers(data.static.markers, parameter_name, True)[0]
                    function.parameter_values['static'].append(expanded_parameter)
                else:
                    function.parameter_values['static'].append(None)

            for parameter_name in function.required_measurements:
            # ============== Measurements ============== 
                if parameter_name in data.static.calibrated.measurements.dtype.names:
                    # Use measurement name to retrieve from calibrated measurements struct
                    function.parameter_values['static'].append(data.static.calibrated.measurements[parameter_name])
                else:
                    function.parameter_values['static'].append(None)

            for parameter_name in function.required_axes:
            # ============== Axes ============== 
                if parameter_name in data.static.calibrated.axes.dtype.names:
                    # Add parameter from axes dict
                    function.parameter_values['static'].append(data.static.calibrated.axes[parameter_name][0])
                else:
                    function.parameter_values['static'].append(None)

            for parameter_name in function.required_angles:
            # ============== Angles ============== 
                if parameter_name in data.static.calibrated.angles:
                    # Add parameter from angles dict
                    function.parameter_values['static'].append(data.static.calibrated.angles[parameter_name][0])
                else:
                    function.parameter_values['static'].append(None)


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
