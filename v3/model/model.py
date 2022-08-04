import time
from .calc.calculations import Calculations
from .utils.structure import structure_model
from .trial_set import TrialSet


class Model():
    def __init__(self,
                 static_filename,
                 dynamic_filenames,
                 measurement_filename,
                 axis_functions=None,
                 angle_functions=None):
        """
        Represents a Model
        """
        self.static_filename = static_filename
        self.dynamic_filenames = dynamic_filenames
        self.measurement_filename = measurement_filename

        if axis_functions is not None:
            axis_functions  = [axis_functions] if not isinstance(axis_functions, list) else axis_functions

        if angle_functions is not None:
            angle_functions = [angle_functions] if not isinstance(angle_functions, list) else angle_functions

        self.calc = Calculations(axis_functions, angle_functions)
        self.structure()


    def run(self):
        """
        Run each of the Model's trials
        """
        self.trial_set.run(self.calc)

    def structure(self):
        start = time.time()
        self.data = structure_model(self.static_filename,
                                    self.dynamic_filenames,
                                    self.measurement_filename,
                                    self.calc.returned_axes,
                                    self.calc.returned_angles)

        self.trial_set = TrialSet(self.data, self.calc)

        self.calc.update_trial_names(self.trial_set.dynamic_trial_names)
        self.calc.expand_parameters_from_data(self.data)

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
                                    
