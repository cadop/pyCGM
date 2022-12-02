import itertools
import time

from ..utils.constants import POINT_DTYPE

import numpy.lib.recfunctions as rfn


class StaticCalc:
    def __init__(self, function_set):
        self.function_set = function_set
        self.update_parameter_values()

    @property
    def functions(self):
        return self.function_set

    @property
    def returned_measurements(self):
        return list(itertools.chain.from_iterable([function.returned_measurements for function in self.functions]))

    @property
    def returned_axes(self):
        return list(itertools.chain.from_iterable([function.returned_axes for function in self.functions]))

    @property
    def returned_angles(self):
        return list(itertools.chain.from_iterable([function.returned_angles for function in self.functions]))

    @property
    def returned_constants(self):
        return list(itertools.chain.from_iterable([function.returned_constants for function in self.functions]))


    def index_of_function(self, function_name):
        for idx, function in enumerate(self.function_set):
            if function.name == function_name:
                return idx
        return len(self.function_set)


    def update_parameter_values(self):
        for function in self.function_set:
            function.parameter_values['static'] =  []

    def expand_parameters_from_data(self, trial, calibrated_dict):
        """
        Expand each function's parameter names to values to in passed data
        """
        for function in self.function_set:

            for parameter_name in function.required_markers:
            # ============== Markers ============== 
                if parameter_name in trial.static.markers.dtype.names:
                    # Use marker name to retrieve from marker struct
                    expanded_parameter = self.get_markers(trial.static.markers, parameter_name, True)
                    if expanded_parameter is not None:
                        expanded_parameter = expanded_parameter[0]
                    function.parameter_values['static'].append(expanded_parameter)
                else:
                    function.parameter_values['static'].append(None)


            for parameter_name in function.required_measurements:
            # ============== Measurements ============== 
                if parameter_name in trial.static.measurements.dtype.names:
                    # Use measurement name to retrieve from measurements struct
                    try:
                        expanded_parameter = trial.static.measurements[parameter_name][0]
                    except ValueError:
                        expanded_parameter = None

                    function.parameter_values['static'].append(expanded_parameter)
                else:
                    function.parameter_values['static'].append(None)

            for parameter_name in function.required_axes:
            # ============== Axes ============== 
                if parameter_name in calibrated_dict['axes']:
                    # Add parameter from axes dict
                    function.parameter_values['static'].append(calibrated_dict['axes'][parameter_name])
                else:
                    function.parameter_values['static'].append(None)

            for parameter_name in function.required_angles:
            # ============== Angles ============== 
                if parameter_name in calibrated_dict['angles']:
                    # Add parameter from angles dict
                    function.parameter_values['static'].append(calibrated_dict['angles'][parameter_name])
                else:
                    function.parameter_values['static'].append(None)

            for parameter_name in function.required_constants:
            # ============== Constants ============== 
                function.parameter_values['static'].append(parameter_name)
            
        print("\nTODO: getStatic. Parameters should be in place at least\n")


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

