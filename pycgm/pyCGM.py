import time

from .model.model import Model


class PyCGM():
    def __init__(self, models):
        if isinstance(models, Model):
            models = [models]

        self.models = models


    def run_all(self):
        for i, model in enumerate(self.models):
            print(f"Running model {i+1} of {len(self.models)}")

            start = time.time()
            model.run()
            end = time.time()
            print(f'\tModel {i+1} runtime:\t\t\t\t\t{end-start:.5f}s\n')


    def __getitem__(self, index):
        return self.models[index]

