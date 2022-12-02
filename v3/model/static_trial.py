import time
import itertools


class StaticTrial():
    def __init__(self):
        self.calibrated_measurements = {}


    def run(self, calc):
        for function in calc.functions:
            start = time.time()
            returned = function.run('static')

            print(function.returns)

            if function.returns['axes'] != [None]:
                self.calibrated_measurements['axes'][function.returns] = returned
            elif function.returns['angles'] != [None]:
                self.calibrated_measurements['angles'][function.returns] = returned
            elif function.returns['other'] != [None]:
                self.calibrated_measurements['other'][function.returns] = returned


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

