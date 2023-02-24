import os

from .model.calc.function import Function
from .model.model import Model
from .model.model_set import ModelSet

def get_data_dir():
    """
    Returns the directory of the package.
    """
    return os.path.join(os.path.dirname(__file__), 'SampleData')
