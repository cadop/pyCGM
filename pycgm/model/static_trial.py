import itertools
import time


class StaticTrial():
    def __init__(self, static_trial_struct):
        self.struct = static_trial_struct


    def run(self, calc):
        start = time.time()
        for function in calc.static_functions:
            returned = function.run('static')

            if function.returns['axes'] != [None]:
                if returned.ndim == 4:
                    for idx, axis in enumerate(function.returns['axes']):
                        self.struct.calibrated.axes[axis] = returned[idx]
                else:
                    axis = function.returns['axes'][0]
                    self.struct.calibrated.axes[axis] = returned

            elif function.returns['angles'] != [None]:
                for angle in function.returns['angles']:
                    self.struct.calibrated.angles[angle] = returned

            elif function.returns['measurements'] != [None]:
                if returned.ndim == 1:
                    for idx, measurement in enumerate(function.returns['measurements']):
                        self.struct.calibrated.measurements[measurement] = returned[idx]
                else:
                    for measurement in function.returns['measurements']:
                        self.struct.calibrated.measurements[measurement] = returned


            end = time.time()
            print(f"\t{'static':<20}\t{function.name:<25}\t{end-start:.5f}s")
