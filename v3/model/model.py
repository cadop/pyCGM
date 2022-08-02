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

        self.calc = Calculations(axis_functions, angle_functions)
        self.data = structure_model(self.static_filename,
                                    self.dynamic_filenames,
                                    self.measurement_filename,
                                    self.calc.returned_axes,
                                    self.calc.returned_angles)

        self.trial_set = TrialSet(self.data, self.calc)


    def run(self):
        """
        Run each of the Model's trials
        """
        self.trial_set.run(self.calc)
                                    
