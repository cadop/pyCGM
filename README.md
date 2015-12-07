# pyCGM

This is a work in progress document.  Full documentation of the module will be available soon.  This is part of ongoing research that will be published in the upcoming months.

Python Module for the Conventional Gait Model, calculates kinematics.

This is a python module for calculating conventional gait model.  It can read a .c3d file (thanks to https://pypi.python.org/pypi/c3d/0.2.1)  or .csv file generated from Vicon Nexus under the Export ASCII settings.

The goal of this project is to release an easy to understand conventional gait model that users can implement in their own projects via a single python file.  We have included a runpycgm.py file to demonstrate how the pycgm file can be called. 

Soon, this will develop into a few options, one in which the file size is drastically reduced by relying much more on the numpy functions, and another which we will try to reduce all dependencies to make the file as portable as possible.

Our aim is to provide researchers and students a tool that can aid in understanding and developing modifications to the conventional gait model through python without much more. 

Requirements:
Python 2.7, Numpy, C3D


Uses:

See runpycgm.py

Credits:

Developed in the Digital Human Research Center http://dhrc.snu.ac.kr at the 

Advanced Institutes of Convergence Technology (AICT) http://aict.snu.ac.kr

Project Lead: Mathew Schwartz (umcadop at gmail) For issues, use github or email me directly

Contributors: Seungeun Yeon (연승은),Filipe Alves Caixeta, Robert Van-Wesep
