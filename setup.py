import sys
sys.path.append('./pyCGM_Single') # TODO update to pycgm when fixed
from _about import __version__
from io import open
import setuptools

with open("README.md", "r",encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pycgm", 
    version= __version__,
    author="", # Many
    author_email="cadop@umich.edu",
    description="A Python Implementation of the Conventional Gait Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cadop/pycgm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7, !=3.0, !=3.1, !=3.2, !=3.3, !=3.4, !=3.5',
    install_requires=['numpy>=1.15','scipy'],
    package_data={
        "": [
            "../SampleData/*/*.c3d",
            "../SampleData/*/*.csv",
            "../SampleData/*/*.vsk",
            "segments.csv"
        ], # TODO Need to fix
    },
    include_package_data=True,   
)
