import os
import time


class DynamicTrial():
    def __init__(self, data):
        self.trials = data.dynamic
        self.trial_names = data.dynamic.dtype.names


    def run(self, calc):
        for trial_name in self.trial_names:
            for function in calc.function_set:
                start = time.time()
                returned = function.run(trial_name)

                if function.returns['axes'] != []:
                    if returned.ndim == 4:
                        for idx, axis in enumerate(function.returns['axes']):
                            axis_name = axis[0]
                            self.trials[trial_name].axes[axis_name] = returned[idx]
                    else:
                        axis = function.returns['axes'][0][0]
                        self.trials[trial_name].axes[axis] = returned

                elif function.returns['angles'] != []:
                    if returned.ndim == 3:
                        for idx, angle in enumerate(function.returns['angles']):
                            angle_name = angle[0]
                            self.trials[trial_name].angles[angle_name] = returned[idx]
                    else:
                        angle = function.returns['angles'][0][0]
                        self.trials[trial_name].angles[angle] = returned

                elif function.returns['measurements'] != []:
                    if returned.ndim == 1:
                        for idx, measurement in enumerate(function.returns['measurements']):
                            self.trials[trial_name].measurements[measurement] = returned[idx]
                    else:
                        for measurement in function.returns['measurements']:
                            self.trials[trial_name].measurements[measurement] = returned


                end = time.time()
                print(f"\t{trial_name:<20}\t{function.name:<25}\t{end-start:.5f}s")
