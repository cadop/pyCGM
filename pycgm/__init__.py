import os
from .model.model import Model
from pycgm.pyCGM import PyCGM

def get_data_dir():
    """
    Returns the directory of the package.
    """
    return os.path.join(os.path.dirname(__file__), 'SampleData')