import numpy.lib.recfunctions as rfn

from .constants import POINT_DTYPE

class Dataset():
    def __init__(self, data, num_frames, trial_name=None):
        self.data = data
        self.num_frames = num_frames
        self.trial_name = trial_name
