import time

from .calc.calculations import Calculations
from .utils.structure import structure_model

class Model():
    def __init__(self, static_filename, dynamic_filenames, measurement_filename):
        self.static_filename = static_filename
        self.dynamic_filenames = dynamic_filenames
        self.measurement_filename = measurement_filename
        self.calc = Calculations()
        self.data = structure_model(self.static_filename,
                                    self.dynamic_filenames,
                                    self.measurement_filename,
                                    self.calc.returned_axes,
                                    self.calc.returned_angles)
        self.calc.update_trial_names(self.data.dynamic.dtype.names)

        self.calc.expand_parameters_from_data(self.data)


    def run(self):
        for trial_name in self.data.dynamic.dtype.names:
            for function in self.calc.axis_functions:
                start = time.time()
                ret_axes = function.run(trial_name)

                # Insert returned axes into the model structured array
                if ret_axes.ndim == 4:
                    # Multiple axes returned by one function
                    for ret_axes_index, axis in enumerate(ret_axes):
                        # Insert each axis into model
                        self.data.dynamic[trial_name].axes[function.result_names[ret_axes_index]] = axis

                else:
                    # Insert returned axis into model
                    self.data.dynamic[trial_name].axes[function.result_names[0]] = ret_axes

                end = time.time()

                print(f"\t{trial_name:<20}\t{function.name:<25}\t{end-start:.5f}s")

            for function in self.calc.angle_functions:
                start = time.time()

                ret_angles = function.run(trial_name)

                # Insert returned angles into the model structured array
                if ret_angles.ndim == 3:
                    # Multiple angles returned by one function
                    for ret_angles_index, angle in enumerate(ret_angles):
                        # Insert each angle into model
                        self.data.dynamic[trial_name].angles[function.result_names[ret_angles_index]] = angle

                else:
                    # Insert returned angle into model
                    self.data.dynamic[trial_name].angles[function.result_names[0]] = ret_angles

                end = time.time()

                print(f"\t{trial_name:<20}\t{function.name:<25}\t{end-start:.5f}s")
                                    
