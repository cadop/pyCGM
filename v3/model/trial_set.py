import time


class DynamicTrialSet():
    def __init__(self, data):
        self.trials = data.dynamic
        self.trial_names = data.dynamic.dtype.names


    def run(self, calc):
        for trial_name in self.trial_names:
            for function in calc.axis_functions:
                start = time.time()
                ret_axes = function.run(trial_name)

                # Insert returned axes into the model structured array
                if ret_axes.ndim == 4:
                    # Multiple axes returned by one function
                    for ret_axes_index, axis in enumerate(ret_axes):
                        # Insert each axis into model
                        self.trials[trial_name].axes[function.returned_axes[ret_axes_index]] = axis

                else:
                    # Insert returned axis into model
                    self.trials[trial_name].axes[function.returned_axes[0]] = ret_axes

                end = time.time()

                print(f"\t{trial_name:<20}\t{function.name:<25}\t{end-start:.5f}s")

            for function in calc.angle_functions:
                start = time.time()

                ret_angles = function.run(trial_name)

                # Insert returned angles into the model structured array
                if ret_angles.ndim == 3:
                    # Multiple angles returned by one function
                    for ret_angles_index, angle in enumerate(ret_angles):
                        # Insert each angle into model
                        self.trials[trial_name].angles[function.returned_angles[ret_angles_index]] = angle

                else:
                    # Insert returned angle into model
                    self.trials[trial_name].angles[function.returned_angles[0]] = ret_angles

                end = time.time()

                print(f"\t{trial_name:<20}\t{function.name:<25}\t{end-start:.5f}s")

