from itertools import chain

import numpy as np

from ..calc.dynamic import CalcAngles, CalcAxes
from ..defaults import return_keys
from ..defaults.parameters import (Angle, AngleFunctions, Axis, AxisFunctions,
                                   Marker, Measurement)
from ..utils import subject_utils


class ModelCreator():
    def __init__(self, static_filename, dynamic_filenames, measurement_filename):
        self.static_filename = static_filename
        self.dynamic_filenames = dynamic_filenames
        self.measurement_filename = measurement_filename

        # Add non-overridden default dynamic funcs to funcs list
        self.axis_functions  = self.get_axis_functions()
        self.angle_functions = self.get_angle_functions()

        # Map function names to indices: 'calc_pelvis_axis': 0 ...
        self.axis_execution_order, self.angle_execution_order = self.map_function_names_to_index()

        # Map function names to the names of the data they return: 'calc_pelvis_axis': ['Pelvis'] ...
        self.axis_function_to_return, self.angle_function_to_return = self.map_function_names_to_returns()

        # Update returned axis and angle list to be used in their respective structed
        # array datatypes
        #   self.axis_keys:  ['Pelvis','RHipJC', 'LHipJC', 'Hip',   'RKnee', 'LKnee', ...]
        #   self.angle_keys: ['Pelvis','RHip',   'LHip',   'RKnee', 'LKnee',  ...]
        self.axis_keys, self.angle_keys = self.update_return_keys()           

        # Structure subject data
        self.data = self.make_data_struct()
        self.trial_names = self.data.dynamic.dtype.names

        # Get default parameter objects
        self.axis_func_parameter_names  = AxisFunctions().parameters()
        self.angle_func_parameter_names = AngleFunctions().parameters()

        # Expand required parameter names to their values in each trial's dataset
        self.axis_func_parameters, self.angle_func_parameters = self.update_trial_parameters()

    def make_data_struct(self):
        return subject_utils.structure_model(self.static_filename,
                                             self.dynamic_filenames,
                                             self.measurement_filename,
                                             self.axis_keys,
                                             self.angle_keys)

    def get_axis_functions(self):
        """
        Initialize axis functions from dynamic.CalcAxes if they have not
        already been defined in a custom model.
        """
        axis_functions = []
        for func in CalcAxes().funcs:
            if hasattr(self, func.__name__):
                axis_functions.append(getattr(self, func.__name__))
            else:
                axis_functions.append(func)

        return axis_functions


    def get_angle_functions(self):
        """
        Initialize angle functions from dynamic.CalcAngles if they have not
        already been defined in a custom model.
        """
        angle_functions = []
        for func in CalcAngles().funcs:
            if hasattr(self, func.__name__):
                angle_functions.append(getattr(self, func.__name__))
            else:
                angle_functions.append(func)

        return angle_functions


    def map_function_names_to_index(self):
        """Map function names to indices.

        e.g: Axis functions
        self.axis_execution_order  = { 'calc_pelvis_axis': 0,
                                         'calc_joint_center_hip: 1,
                                         ...
                                       }

        e.g: Angle functions
        self.angle_function_to_index = { 'calc_angle_pelvis': 0,
                                         'calc_angle_hip': 1,
                                         ...
                                       }

        }
        """
        axis_execution_order  = {}
        angle_execution_order = {}

        for index, function in enumerate(self.axis_functions):
            axis_execution_order[function.__name__] = index

        for index, function in enumerate(self.angle_functions):
            angle_execution_order[function.__name__] = index

        return axis_execution_order, angle_execution_order


    def map_function_names_to_returns(self):
        """Map function names to the names of the data they return.

        e.g: Axis functions
        self.axis_function_to_return  = { 'calc_pelvis_axis': ['Pelvis'],
                                          'calc_joint_center_hip: ['RHipJC', 'LHipJC'],
                                          ...
                                        }

        e.g: Angle functions
        self.angle_function_to_return = { 'calc_angle_pelvis': ['Pelvis'],
                                          'calc_angle_hip': ['RHip', 'LHip'],
                                          ...
                                        }

        }
        """
        axis_function_to_return  = return_keys.axes()
        angle_function_to_return = return_keys.angles()
        return axis_function_to_return, angle_function_to_return


    def update_return_keys(self):
        axis_keys = list(chain.from_iterable(self.axis_function_to_return.values()))
        angle_keys = list(chain.from_iterable(self.angle_function_to_return.values()))

        return axis_keys, angle_keys


    def names_to_values(self, function_parameters, trial_name):
        """Convert a list of function parameter objects to their values in each trial's dataset

        Parameters
        ----------
        function_parameters : list of lists of parameter objects
            Required parameter objects for all functions
        trial_name : str
            The name of the trial
        
        Returns
        -------
        updated_parameters_list : list of list of ndarray
            The values of all specified parameters of all functions of the given trial

        Notes
        -----
        function_parameters is a list of lists of parameter objects like so:
            [
                [
                    # knee_axis parameters
                    Marker('RTHI'),
                    Marker('LTHI'),
                    Marker('RKNE'),
                    Marker('LKNE'),
                    Axis('RHipJC'),
                    Axis('LHipJC'),
                    Measurement('RightKneeWidth'),
                    Measurement('LeftKneeWidth')
                ],
                ...other functions
            ]
        """

        updated_parameters_list = [[] for _ in range(len(function_parameters))]

        for function_index, function_parameters in enumerate(function_parameters):
            for parameter in function_parameters:

                if isinstance(parameter, Marker):
                    # Use marker name to retrieve from marker struct
                    new_parameter = self.get_markers(self.data.dynamic[trial_name].markers, parameter.name, True)
                    if new_parameter is not None:
                        new_parameter = new_parameter[0]
                    updated_parameters_list[function_index].append(new_parameter)

                elif isinstance(parameter, Measurement):
                    # Use measurement name to retrieve from measurements struct
                    try:
                        new_parameter = self.data.static.measurements[parameter.name][0]
                    except ValueError:
                        new_parameter = None

                    updated_parameters_list[function_index].append(new_parameter)

                elif isinstance(parameter, Axis):
                    # Add parameter from axes struct
                    updated_parameters_list[function_index].append(self.data.dynamic[trial_name].axes[parameter.name][0])

                elif isinstance(parameter, Angle):
                    # Add parameter from angles struct
                    updated_parameters_list[function_index].append(self.data.dynamic[trial_name].angles[parameter.name])

                else:
                    # Parameter is a constant, append as is
                    updated_parameters_list[function_index].append(parameter)

        return updated_parameters_list


    def update_trial_parameters(self):
        axis_func_parameters  = {}
        angle_func_parameters = {}
        for trial in self.trial_names:
            axis_func_parameters[trial]  = self.names_to_values(self.axis_func_parameter_names,  trial)
            angle_func_parameters[trial] = self.names_to_values(self.angle_func_parameter_names, trial)

        return axis_func_parameters, angle_func_parameters

