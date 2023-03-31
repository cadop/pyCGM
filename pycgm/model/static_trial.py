import itertools
import time


class StaticTrial():
    def __init__(self, static_trial_struct):
        self.struct = static_trial_struct


    def run(self, calc):
        for function in calc.function_set:
            start = time.time()
            returned = function.run('static')

            if function.returns['axes'] != []:
                if returned.ndim == 4:
                    for idx, axis in enumerate(function.returns['axes']):
                        axis_name = axis[0]
                        self.struct.calibrated.axes[axis_name][0] = returned[idx]
                else:
                    axis = function.returns['axes'][0][0]
                    self.struct.calibrated.axes[axis] = returned

            elif function.returns['angles'] != []:
                for angle in function.returns['angles']:
                    self.struct.calibrated.angles[angle] = returned

            elif function.returns['measurements'] != []:
                if returned.ndim == 1:
                    for idx, measurement in enumerate(function.returns['measurements']):
                        measurement_name = measurement[0]
                        self.struct.calibrated.measurements[measurement_name] = returned[idx]
                else:
                    for measurement in function.returns['measurements']:
                        measurement_name = measurement[0]
                        self.struct.calibrated.measurements[measurement_name] = returned


            end = time.time()
            print(f"\t{'static':<20}\t{function.name:<25}\t{end-start:.5f}s")
