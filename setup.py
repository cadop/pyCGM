import sys
sys.path.append('./pyCGM_Single')
from _about import __version__
import setuptools

with open("README.md", "r",encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyCGM_Single", # TODO change this when folder is properly named  
    version= __version__,
    author="Mathew Schwartz, et al.",
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
    python_requires='>=3',
    install_requires=['numpy>=1.15'],
    # package_data={
    #     "": ["SampleData/ROM/*.c3d","SampleData/ROM/*.vsk"],
    # },
    include_package_data=True,   
)
