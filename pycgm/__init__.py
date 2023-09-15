import os

from .model.calc.function import Function
from .model.calc.kinematics.dynamic import CalcDynamic
from .model.calc.kinematics.static import CalcStatic
from .model.model import Model
from .model.model_set import ModelSet
from .model.preprocess.preprocess import Preprocess
from .model.utils.constants import POINT_DTYPE

def get_data_dir():
    """
    Returns the directory of the package.
    """
    return os.path.join(os.path.dirname(__file__), 'SampleData')
