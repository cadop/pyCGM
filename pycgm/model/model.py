import time

import numpy as np
import numpy.lib.recfunctions as rfn

from ..defaults.parameters import Angle, Axis, Marker, Measurement
from ..utils import subject_utils
from .model_creator import ModelCreator


class Model(ModelCreator):
    def __init__(self, static_filename, dynamic_filenames, measurement_filename):
        super().__init__(static_filename, dynamic_filenames, measurement_filename)


    def run(self):
        """
        Run each trial in the model and insert output values into 
        the model's data struct.
        """

        for trial_name in self.trial_names:
            for index, func in enumerate(self.axis_functions):

                # Retrieve the names of the axes returned by this function
                # e.g. 'calc_axis_pelvis' -> 'Pelvis'
                returned_axis_names = self.axis_function_to_return[func.__name__]

                start = time.time()

                # Get the parameters for this function, run it
                parameters = self.axis_func_parameters[trial_name][self.axis_execution_order[func.__name__]]
                ret_axes = np.array(func(*parameters))

                # Insert returned axes into the model structured array
                if ret_axes.ndim == 4:
                    # Multiple axes returned by one function
                    for ret_axes_index, axis in enumerate(ret_axes):
                        # Insert each axis into model
                        self.data.dynamic[trial_name].axes[returned_axis_names[ret_axes_index]] = axis

                else:
                    # Insert returned axis into model
                    self.data.dynamic[trial_name].axes[returned_axis_names[0]] = ret_axes

                end = time.time()

                print(f"\t{trial_name:<20}\t{func.__name__:<25}\t{end-start:.5f}s")

            for index, func in enumerate(self.angle_functions):

                # Retrieve the names of the angles returned by this function
                # e.g. 'calc_angle_pelvis' -> 'Pelvis'
                returned_angle_names = self.angle_function_to_return[func.__name__]

                start = time.time()

                # Get the parameters for this function, run it
                parameters = self.angle_func_parameters[trial_name][self.angle_execution_order[func.__name__]]
                ret_angles = np.array(func(*parameters))

                # Insert returned angles into the model structured array
                if ret_angles.ndim == 3:
                    # Multiple angles returned by one function
                    for ret_angles_index, angle in enumerate(ret_angles):
                        # Insert each angle into model
                        self.data.dynamic[trial_name].angles[returned_angle_names[ret_angles_index]] = angle

                else:
                    # Insert returned angle into model
                    self.data.dynamic[trial_name].angles[returned_angle_names[0]] = ret_angles

                end = time.time()

                print(f"\t{trial_name:<20}\t{func.__name__:<25}\t{end-start:.5f}s")


    def get_markers(self, arr, names, points_only=True, debug=False):
        start = time.time()

        if isinstance(names, str):
            names = [names]
        num_frames = arr[0][0].shape[0]

        if any(name not in arr[0].dtype.names for name in names):
            return None

        rec = rfn.repack_fields(arr[names]).view(subject_utils.marker_dtype()).reshape(len(names), int(num_frames))


        if points_only:
            rec = rec['point'][['x', 'y', 'z']]

        rec = rfn.structured_to_unstructured(rec)

        end = time.time()
        if debug:
            print(f'Time to get {len(names)} markers: {end-start}')

        return rec


    def modify_function(self, function, markers=None, measurements=None, axes=None, angles=None, returns_axes=None, returns_angles=None):
        """Modify an existing function's parameters and returned values

        Parameters
        ----------
        function : str
            Name of the function that is to be modified.
        measurements : list of str, optional
            Name(s) of required measurement parameters.
        markers : list of str, optional
            Name(s) of required marker parameters.
        axes : list of str, optional
            Name(s) of required axis parameters.
        angles : list of str, optional
            Name(s) of required angle parameters.
        returns_axes : list of str, optional
            Name(s) of returned axes.
        returns_angles : list of str, optional
            Name(s) of returned angles.

        Raises
        ------
        Exception
            If the function returns both axes and angles
            If function is not of type str
        """

        if returns_axes is not None and returns_angles is not None:
            raise Exception(f'{function} must return either an axis or an angle, not both')

        # Create list of parameter objects from parameter names
        params = []
        for measurement_name in [measurement_name for measurement_name in (measurements or [])]:
            params.append(Measurement(measurement_name))

        for marker_name in [marker_name for marker_name in (markers or [])]:
            params.append(Marker(marker_name))

        for axis_name in [axis_name for axis_name in (axes or [])]:
            params.append(Axis(axis_name))

        for angle_name in [angle_name for angle_name in (angles or [])]:
            params.append(Angle(angle_name))


        if isinstance(function, str):  # make sure a function name is passed
            if function in self.axis_execution_order:
                self.axis_func_parameter_names[self.axis_execution_order[function]] = params

            elif function in self.angle_execution_order:
                self.angle_func_parameter_names[self.angle_execution_order[function]] = params

            else:
                raise KeyError(f"Unable to find function {function} in execution order")
        else:
            raise Exception(f'Pass the name of the function as a string like so: \'{function.__name__}\'')

        if returns_axes is not None:
            # Add returned axes, update related attributes
            self.axis_function_to_return[function] = returns_axes

        if returns_angles is not None:
            # Add returned angles, update related attributes
            self.angle_function_to_return[function] = returns_angles

        # Expand required parameter names to their values in each trial's dataset
        self.axis_func_parameters, self.angle_func_parameters = self.update_trial_parameters()


    def add_function(self, function, order=None, measurements=None, markers=None, axes=None, angles=None, returns_axes=None, returns_angles=None):
        """Add a custom function to the model.

        Parameters
        ----------
        function : str or function
            Name or function object that is to be added.
        order : list or tuple of [str, int], optional
            Index in the execution order the function is to be run, represented as [function_name, offset].
        measurements : list of str, optional
            Name(s) of required measurement parameters.
        markers : list of str, optional
            Name(s) of required marker parameters.
        axes : list of str, optional
            Name(s) of required axis parameters.
        angles : list of str, optional
            Name(s) of required angle parameters.
        returns_axes : list of str, optional
            Name(s) of returned axes.
        returns_angles : list of str, optional
            Name(s) of returned angles.

        Raises
        ------
        Exception
            If the function returns both axes and angles
            If the function does not return a custom axis or angle
        KeyError
            If function is not found in the function execution order.
        
        Notes
        -----
        order is represented by [function_name, offset]
            - function_name is the name of the function that the new function will be run relative to.
            - offset is offset from the target function_name that the new function will be run.
            - an order of ['calc_axis_knee', -1] will run the custom function immediately before calc_axis_knee.
        """

        def insert_axis_function(target_function_name, offset, func):
            """Insert a custom axis function at the desired offset from a target function name.

            Parameters
            ----------
            target_function_name : str
                The name of the function that the axis function will be run relative to.
            offset : int
                The offset from the target function that the new function will be run.

            Raises
            ------
            KeyError
                If target_function_name is not found in the axis function execution order.

            Notes
            -----
                A target name of 'calc_axis_knee' and an offset of -1 will run the custom function
                immediately before calc_axis_knee.
            """

            # Get the index in the execution order where the new function is to be run
            try:
                target_index = self.axis_execution_order[target_function_name] + offset
            except KeyError:
                raise KeyError(f"Unable to find function {target_function_name} in axis execution order")

            # If the target index is out of bounds, append the function to the end or beginning
            if target_index > len(self.axis_execution_order):
                target_index = len(self.axis_execution_order)
            elif target_index < 0:
                target_index = 0

            # Insert at specified index and update execution order
            self.axis_functions.insert(target_index, func)
            self.axis_execution_order, self.angle_execution_order = self.map_function_names_to_index()

            # Extend the returned axes of the function BEFORE the new function
            # e.g. calc_joint_center_hip returns [RHipJC, LHipJC]
            #      We don't want to position the returned axes between RHipJC and LHipJC,
            #      so calc_joint_center_hip's returned axes must be extended
            function_to_extend = self.axis_functions[target_index - 1].__name__
            self.axis_function_to_return[function_to_extend].extend(returns_axes)

            # Update return keys 
            self.axis_keys, self.angle_keys = self.update_return_keys()           

            # Insert the function's parameters into the target index
            self.axis_func_parameter_names.insert(self.axis_execution_order[func_name], params)

            # Add empty space for custom function's parameters in each trial
            for trial_name in self.trial_names:
                self.axis_func_parameters[trial_name].insert(target_index+1, [])


        def insert_angle_function(target_function_name, offset, func):
            """Insert a custom angle function at the desired offset from a target function name.

            Parameters
            ----------
            target_function_name : str
                The name of the function that the angle function will be run relative to.
            offset : int
                The offset from the target function that the new function will be run.

            Raises
            ------
            KeyError
                If target_function_name is not found in the angle function execution order.

            Notes
            -----
                A target name of 'calc_angle_knee' and an offset of -1 will run the custom function
                immediately before calc_angle_knee.
            """

            # Get the index in the execution order where the new function is to be run
            try:
                target_index = self.angle_execution_order[target_function_name] + offset
            except KeyError:
                raise KeyError(f"Unable to find function {target_function_name} in angle execution order")

            # If the target index is out of bounds, append the function to the end
            if target_index > len(self.angle_execution_order):
                target_index = len(self.angle_execution_order)
            elif target_index < 0:
                target_index = 0

            # Insert at specified index and update execution order
            self.angle_functions.insert(target_index, func)
            self.angle_execution_order, self.angle_execution_order = self.map_function_names_to_index()

            # Extend the returned angles of the function BEFORE the new function
            # e.g. calc_joint_center_hip returns [RHipJC, LHipJC]
            #      We don't want to position the returned angles between RHipJC and LHipJC,
            #      so calc_joint_center_hip's returned angles must be extended
            function_to_extend = self.angle_functions[target_index - 1].__name__
            self.angle_function_to_return[function_to_extend].extend(returns_angles)

            # Update return keys 
            self.angle_keys, self.angle_keys = self.update_return_keys()           

            # Insert the function's parameters into the target index
            self.angle_func_parameter_names.insert(self.angle_execution_order[func_name], params)

            # Add empty space for custom function's parameters in each trial
            for trial_name in self.trial_names:
                self.angle_func_parameters[trial_name].insert(target_index+1, [])


        # Get func object and name
        if isinstance(function, str):
            func_name = function
            func      = getattr(self, func_name)
        elif callable(function):
            func_name = function.__name__
            func      = function

        if returns_axes is not None and returns_angles is not None:
            raise Exception(f'{func_name} must return either an axis or an angle, not both')
        if returns_axes is None and returns_angles is None:
            raise Exception(f'{func_name} must return a custom axis or angle. If the axis or angle already exists in the model, use self.modify_function()')

        # Create list of parameter objects from parameter names
        params = []
        for measurement_name in [measurement_name for measurement_name in (measurements or [])]:
            params.append(Measurement(measurement_name))

        for marker_name in [marker_name for marker_name in (markers or [])]:
            params.append(Marker(marker_name))

        for axis_name in [axis_name for axis_name in (axes or [])]:
            params.append(Axis(axis_name))

        for angle_name in [angle_name for angle_name in (angles or [])]:
            params.append(Angle(angle_name))


        if returns_axes is not None:
            # Add returned axes, update related attributes

            if order is not None:
                # Find the position of the function name in the execution order
                # Then apply the offset to get the desired execution index
                # e.g. ['calc_axis_knee', -1] -> run the custom function right before calc_axis_knee

                target_function_name = order[0]
                offset = order[1] + 1

                if offset > 0:
                    offset -= 1

                insert_axis_function(target_function_name, offset, func)

            else:
                # Append to the end
                self.axis_functions.append(func)
                self.axis_keys.extend(returns_axes)

                for trial_name in self.trial_names:
                    self.axis_func_parameters[trial_name].append([])

            # Update parameters and returns
            self.axis_func_parameter_names[self.axis_execution_order[function]] = params
            self.axis_function_to_return[function] = returns_axes

        if returns_angles is not None:
            # Add returned angles, update related attributes

            if order is not None:
                # Find the position of the function name in the execution order
                # Then apply the offset to get the desired execution index
                # e.g. ['calc_angle_knee', -1] -> run the custom function right before calc_angle_knee

                target_function_name = order[0]
                offset = order[1] + 1

                if offset > 0:
                    offset -= 1

                insert_angle_function(target_function_name, offset, func)

            else:
                # Append to the end
                self.angle_functions.append(func)
                self.angle_keys.extend(returns_angles)

                for trial_name in self.trial_names:
                    self.angle_func_parameters[trial_name].append([])

            # Update parameters and returns
            self.angle_func_parameter_names[self.angle_execution_order[function]] = params
            self.angle_function_to_return[function] = returns_angles

        # Remake data struct with new return keys
        self.data = self.make_data_struct()

        # Expand required parameter names to their values in each trial's dataset
        self.axis_func_parameters, self.angle_func_parameters = self.update_trial_parameters()

