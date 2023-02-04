import time
import itertools


class StaticTrial():
    def __init__(self, static_trial_struct):
        self.struct = static_trial_struct


    def run(self, calc):
        for function in calc.functions:
            start = time.time()
            returned = function.run('static')

            if function.returns['axes'] != [None]:
                for axis in function.returns['axes']:
                    self.struct.calibrated.axes[axis] = returned[0]
            elif function.returns['angles'] != [None]:
                for angle in function.returns['angles']:
                    self.struct.calibrated.angles[angle] = returned
            elif function.returns['constants'] != [None]:
                if returned.ndim == 1:
                    for idx, constant in enumerate(function.returns['constants']):
                        self.struct.calibrated.measurements[constant] = returned[idx]

                else:
                    for constant in function.returns['constants']:
                        self.struct.calibrated.measurements[constant] = returned


            # # Insert returned axes into the model structured array
            # if returned.ndim == 4:
            #     # Multiple axes returned by one function
            #     for axis in returned:
            #         # Insert each axis into model
            #         self.calibrated_measurements[function.returns] = axis

            # else:
            #     # Insert returned axis into model
            #     self.calibrated_measurements[function.returns[0]] = returned

            end = time.time()

            print(f"\t{'static':<20}\t{function.name:<25}\t{end-start:.5f}s")

