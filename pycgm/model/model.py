import time

import numpy as np

from .calc.dynamic_config import DynamicConfig
from .calc.kinematics.dynamic import CalcDynamic
from .calc.kinematics.static import CalcStatic
from .calc.static_config import StaticConfig
from .dynamic_trial import DynamicTrial
from .preprocess.preprocess import Preprocess
from .static_trial import StaticTrial
from .utils.io import IO
from .utils.structure import structure_model


class Model():
    def __init__(self,
                 static_filename,
                 dynamic_filenames,
                 measurement_filename,
                 static_functions=None,
                 dynamic_functions=None,
                 preprocess=None):
        """
        Represents a Model
        """
        self.static_filename = static_filename
        self.dynamic_filenames = dynamic_filenames
        self.measurement_filename = measurement_filename

        if issubclass(type(static_functions), CalcStatic):
            self.static_config = StaticConfig(static_functions.funcs)
        else:
            self.static_config = StaticConfig(static_functions)

        if issubclass(type(dynamic_functions), CalcDynamic):
            self.dynamic_config = DynamicConfig(dynamic_functions.funcs)
        else:
            self.dynamic_config = DynamicConfig(dynamic_functions)

        if issubclass(type(preprocess), Preprocess):
            self.preprocess_config = preprocess
        else:
            self.preprocess_config = Preprocess()

        self.load()
        self.preprocess()
        self.structure()


    def load(self):
        io = IO(self.static_filename, self.dynamic_filenames, self.measurement_filename)
        self.static_data, self.dynamic_data, self.measurement_data = io.load()

    def preprocess(self):
        self.preprocess_config.measurements = self.measurement_data
        self.preprocess_config.static_data = self.static_data
        self.preprocess_config.dynamic_data = self.dynamic_data
        self.preprocess_config.run()

    def structure(self):
        start = time.time()

        self.data = structure_model(self.measurement_data,
                                    self.static_data,
                                    self.dynamic_data,
                                    self.static_config, 
                                    self.dynamic_config)
        
        # TODO consider adding helpers to structure custom data
        self.data.static.calibrated.measurements.FlatFoot = 0;
        
        # TODO calculate GCS
        self.data.static.calibrated.measurements.GCS = np.array([ [1, 0, 0], [0, 1, 0], [0, 0, 1] ])

        # Structure static trial
        self.static_trial = StaticTrial(self.data.static)
        self.static_config.expand_parameters_from_data(self.data)

        # Load calibrated parameters
        self.dynamic_trials = DynamicTrial(self.data)
        self.dynamic_config.update_trial_names(self.dynamic_trials.trial_names)


        end = time.time()
        # print(f"Time to structure model: {end - start}s")

    def run(self):
        """
        Run each of the Model's trials
        """
        self.static_trial.run(self.static_config)

        self.dynamic_config.expand_parameters_from_data(self.data)
        self.dynamic_trials.run(self.dynamic_config)


    
    def insert_static_function(self, function, index=None, before=None, after=None, replaces=None):

        if index is not None:
            self.static_config.function_set.insert(index, function)

        elif before is not None:
            func_idx = self.static_config.index_of_function(before)
            self.static_config.function_set.insert(func_idx, function)

        elif after is not None:
            func_idx = self.static_config.index_of_function(after)
            self.static_config.function_set.insert(func_idx + 1, function)

        elif replaces is not None:
            func_idx = self.static_config.index_of_function(replaces)
            self.static_config.function_set[func_idx] = function

        else:
            self.static_config.function_set.append(function)

        self.structure()

    def insert_dynamic_function(self, function, index=None, before=None, after=None, replaces=None):

        if index is not None:
            self.dynamic_config.function_set.insert(index, function)

        elif before is not None:
            func_idx = self.dynamic_config.index_of_function(before)
            self.dynamic_config.function_set.insert(func_idx, function)

        elif after is not None:
            func_idx = self.dynamic_config.index_of_function(after)
            self.dynamic_config.function_set.insert(func_idx + 1, function)

        elif replaces is not None:
            func_idx = self.dynamic_config.index_of_function(replaces)
            self.dynamic_config.function_set[func_idx] = function

        else:
            self.dynamic_config.function_set.append(function)

        self.structure()
