import time

from .calc.kinematics import dynamic, static
from .calc.calculations import DynamicCalc
from .calc.static_calculations import StaticCalc
from .utils.structure import structure_model, structure_model_input
from .trial_set import DynamicTrialSet
from .static_trial import StaticTrial


class Model():
    def __init__(self,
                 static_filename,
                 dynamic_filenames,
                 measurement_filename,
                 static_functions=None,
                 dynamic_axis_functions=None,
                 dynamic_angle_functions=None):
        """
        Represents a Model
        """
        self.static_filename = static_filename
        self.dynamic_filenames = dynamic_filenames
        self.measurement_filename = measurement_filename

        if dynamic_axis_functions is not None:
            dynamic_axis_functions  = [dynamic_axis_functions] if not isinstance(dynamic_axis_functions, list) else dynamic_axis_functions
        else:
            dynamic_axis_functions = dynamic.CalcAxes().funcs

        if dynamic_angle_functions is not None:
            dynamic_angle_functions = [dynamic_angle_functions] if not isinstance(dynamic_angle_functions, list) else dynamic_angle_functions
        else:
            dynamic_axis_functions = dynamic.CalcAngles().funcs

        if static_functions is not None:
            static_functions  = [static_functions] if not isinstance(static_functions, list) else static_functions
        else:
            static_functions = static.StaticCalc().funcs

        self.static_calc = StaticCalc(static_functions)
        self.dynamic_calc = DynamicCalc(dynamic_axis_functions, dynamic_angle_functions)
        self.structure()


    def run(self):
        """
        Run each of the Model's trials
        """
        self.static_trial.run(self.static_calc)
        # self.dynamic_trials.run(self.dynamic_calc)

    def structure(self):
        start = time.time()
        # self.data = structure_model(self.static_filename,
        #                             self.dynamic_filenames,
        #                             self.measurement_filename,
        #                             self.calc.returned_axes,
        #                             self.calc.returned_angles)

        # self.trial_set = TrialSet(self.data)

        # self.calc.update_trial_names(self.trial_set.dynamic_trial_names)
        # self.calc.expand_parameters_from_data(self.data)

        # static_data = structure_model_input(self.static_filename,
        #                                     self.dynamic_filenames,
        #                                     self.measurement_filename)

        static_data = structure_model_input(self.static_filename,
                                            self.dynamic_filenames,
                                            self.measurement_filename)
        self.static_trial = StaticTrial()
        self.static_calc.expand_parameters_from_data(static_data, self.static_trial.calibrated_measurements)
        # self.static_trial.run(self.static_calc)

        # self.dynamic_trials = TrialSet(self.data, self.calc)
        end = time.time()
        print(f"Time to structure model: {end - start}s")

    
    def insert_axis_function(self, function, index=None, before=None, after=None):

        if index is not None:
            self.calc.axis_function_set.insert(index, function)

        elif before is not None:
            func_idx = self.calc.index_of_axis_function(before)
            self.calc.axis_function_set.insert(func_idx, function)

        elif after is not None:
            func_idx = self.calc.index_of_axis_function(after)
            self.calc.axis_function_set.insert(func_idx + 1, function)

        else:
            self.calc.axis_function_set.append(function)

        self.structure()

    def insert_angle_function(self, function, index=None, before=None, after=None):

        if index is not None:
            self.calc.angle_function_set.insert(index, function)

        elif before is not None:
            func_idx = self.calc.index_of_angle_function(before)
            self.calc.angle_function_set.insert(func_idx, function)

        elif after is not None:
            func_idx = self.calc.index_of_angle_function(after)
            self.calc.angle_function_set.insert(func_idx + 1, function)

        else:
            self.calc.angle_function_set.append(function)

        self.structure()
                                    
