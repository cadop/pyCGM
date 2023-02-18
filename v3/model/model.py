import time

import numpy as np

from .calc.dynamic_calculations import DynamicCalc
from .calc.kinematics import dynamic
from .calc.kinematics import static
from .calc.static_calculations import StaticCalc
from .dynamic_trial_set import DynamicTrialSet
from .static_trial import StaticTrial
from .utils.structure import structure_model


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

        self.static_calc  = StaticCalc(static_functions)
        self.dynamic_calc = DynamicCalc(dynamic_axis_functions, dynamic_angle_functions)
        self.structure()


    def run(self):
        """
        Run each of the Model's trials
        """
        # Run static trial 
        self.static_trial.run(self.static_calc)

        # Load calibrated parameters
        self.dynamic_trials = DynamicTrialSet(self.data)
        self.dynamic_calc.update_trial_names(self.dynamic_trials.trial_names)
        self.dynamic_calc.expand_parameters_from_data(self.data)

        # Run dynamic trials
        self.dynamic_trials.run(self.dynamic_calc)

    def structure(self):
        start = time.time()

        self.data = structure_model(self.static_filename,
                                    self.dynamic_filenames,
                                    self.measurement_filename,
                                    self.static_calc, 
                                    self.dynamic_calc)
        
        # TODO consider adding flat_foot as a flag to Model init
        self.data.static.calibrated.measurements.FlatFoot = 0;
        
        # TODO calculate GCS
        self.data.static.calibrated.measurements.GCS = np.array([ [1, 0, 0], [0, 1, 0], [0, 0, 1] ])

        # Structure static trial
        self.static_trial = StaticTrial(self.data.static)
        self.static_calc.expand_parameters_from_data(self.data)

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
