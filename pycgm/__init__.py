from . _about import __version__
import os

def get_data_dir():
    """Returns the directory of the package.
    """
    return os.path.join(os.path.dirname(__file__), 'examples', 'SampleData')

def get_rom_data():
    dir = get_data_dir() + "/ROM"
    static = dir + "/Sample_Static.c3d"
    dynamic = dir + "/Sample_Dynamic.c3d"
    measurements = dir + "/Sample_SM.vsk"
    return static, dynamic, measurements
