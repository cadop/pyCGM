from ._about import __version__
import pycgm.pycgm
import pycgm.io
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


def get_rom_csv():
    dir = get_data_dir() + "/ROM"
    return dir + "/Sample_Static.csv"


def get_59993_data():
    dir = get_data_dir() + "/59993_Frame"
    static = dir + "/59993_Frame_Static.c3d"
    dynamic = dir + "/59993_Frame_Dynamic.c3d"
    measurements = dir + "/59993_Frame_SM.vsk"
    return static, dynamic, measurements


def get_robo_data():
    dir = get_data_dir() + "/Sample_2"
    static = dir + "/RoboStatic.c3d"
    dynamic = dir + "/RoboWalk.c3d"
    measurements = dir + "/RoboSM.vsk"
    return static, dynamic, measurements


def get_robo_results():
    dir = get_data_dir() + "/Sample_2"
    return dir + "/RoboResults.csv"


def get_robo_measurements():
    dir = get_data_dir() + "/Sample_2"
    vsk = dir + "/RoboSM.vsk"
    csv = dir + "/RoboSM.csv"
    return vsk, csv
