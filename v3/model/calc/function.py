from . import defaults
import numpy as np

class Function:
    def __init__(self, func, parameter_names=None, result_names=None):
        self.func = func
        self.name = func.__name__
        self.parameter_values = {}

        if parameter_names is None:
            self.parameter_names = defaults.parameters()[self.name]
        else:
            self.parameter_names = parameter_names

        if result_names is None:
            self.result_names = defaults.returns()[self.name]
        else:
            self.result_names = result_names


    def run(self, trial_name):
        return np.asarray(self.func(*self.parameter_values[trial_name]))

