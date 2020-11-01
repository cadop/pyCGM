import sys
sys.path.append('./pyCGM_Single') # TODO update to pycgm when fixed
from _about import __version__
import setuptools

pyver = sys.version_info.major
if pyver == 2:
    with open("README.md", "r") as fh:
        long_description = fh.read()
else:
    with open("README.md", "r",encoding="utf8") as fh:
        long_description = fh.read()

requires = ">=2, <3" if pyver == 2 else ">=3"

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
    python_requires=requires,
    install_requires=['numpy>=1.15','scipy'],
    package_data={
        "": ["SampleData/ROM/*.c3d","SampleData/ROM/*.vsk","segments.csv"], # TODO Need to fix
    },
    include_package_data=True,   
)
