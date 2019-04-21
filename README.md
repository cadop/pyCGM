# pyCGM

**Python Module for the Conventional Gait Model**
*calculates Kinematics and Center of Mass*

The goal of this project is to release an easy to understand conventional gait model that users can implement in their own projects via a single python file.  While the project is multiple files, the kinematics code is contained in a single file pyCGM.py.  With the update, another kinetics file is also used for the center of mass calculation.   

Our aim is to provide researchers and students a tool that can aid in understanding and developing modifications to the conventional gait model through python without much more. 

The kinematics are validated against Nexus 1.8, and file types are known from Nexus 1.8.  Newer C3D files may not work (but files re-exported from Mokka usually work). 

## How to use?
For getting started, please check the [WIKI on github] (https://github.com/cadop/pyCGM/wiki/Overview)

### Requirements:
* Python 2.7 or Python 3
* Numpy
* Scipy (only if using the pipeline operations)

### Requirements for HPC:
* Python 2.7
* Numpy
* MPI Preferably Linux (MPI is not as simple to setup on windows)

Uses a modified version of the c3d.py loader from github. https://pypi.python.org/pypi/c3d/0.2.1

## Credits:

Originally developed in the Digital Human Research Center at the Advanced Institutes of Convergence Technology (AICT), Seoul National University http://aict.snu.ac.kr

Project Lead: Mathew Schwartz (umcadop at gmail) For issues, use github or email me directly

Contributors: Neil M. Thomas, Philippe C. Dixon,  Seungeun Yeon (연승은),Filipe Alves Caixeta, Robert Van-Wesep

## Reference
Read about this code and if you find it useful in your work please cite:

Schwartz, Mathew, and Philippe C. Dixon. "*The effect of subject measurement error on joint kinematics in the conventional gait model: Insights from the open-source pyCGM tool using high performance computing methods.*" PloS one 13.1 (2018): e0189984.
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0189984

### Bibtex:
@article{schwartz2018effect,
  title={The effect of subject measurement error on joint kinematics in the conventional gait model: Insights from the open-source pyCGM tool using high performance computing methods},
  author={Schwartz, Mathew and Dixon, Philippe C},
  journal={PloS one},
  volume={13},
  number={1},
  pages={e0189984},
  year={2018},
  publisher={Public Library of Science}
}
