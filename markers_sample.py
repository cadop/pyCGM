#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycgm.pycgm import CGM
from pycgm.customizations import CustomCGM3
import pycgm

if __name__ == "__main__":
    static_trial, dynamic_trial, measurements = pycgm.get_rom_data()
    subject1 = CGM(static_trial, dynamic_trial, measurements, end=5)
    subject1.run()
    print("head jc standard")
    print(subject1.head_joint_centers[0])

    static_trial_renamed = static_trial[:-4] + "_Renamed.csv"
    dynamic_trial_renamed = dynamic_trial[:-4] + "_Renamed.c3d"
    # Static and Dynamic trial contain the same set of renamed markers
    # Doing a renaming to the dynamic trial applies it to the static by default
    subject2 = CGM(static_trial_renamed, dynamic_trial_renamed, measurements, end=5)
    subject2.rename_marker('RFHD', 'RFHead')  # Fix one marker with remap(), or multiple with bulk_remap()
    subject2.rename_markers({'LFHD': 'LFHead', 'RBHD': 'RBHead', 'LBHD': 'LBHead'})
    subject2.run()
    print("head jc with renamed head markers")
    print(subject2.head_joint_centers[0])

    # Static trial, but not Dynamic, contains renamed markers
    # Override must be set to False in this case to ensure that they are different
    subject3 = CGM(static_trial_renamed, dynamic_trial, measurements, override_static=False, end=5)
    subject3.rename_static_markers({'RFHD': 'RFHead', 'LFHD': 'LFHead',
                                    'RBHD': 'RBHead', 'LBHD': 'LBHead'})
    subject3.run()
    print("head jc with renamed static only")
    print(subject2.head_joint_centers[0])

    # Dynamic trial, but not Static, contains renamed markers
    # Override must be set to False in this case to ensure that they are different
    subject4 = CGM(static_trial, dynamic_trial_renamed, measurements, override_static=False, end=5)
    subject4.rename_markers({'RFHD': 'RFHead', 'LFHD': 'LFHead',
                             'RBHD': 'RBHead', 'LBHD': 'LBHead'})
    subject4.run()
    print("head jc with renamed dynamic only")
    print(subject4.head_joint_centers[0])

    dynamic_trial_extra = dynamic_trial[:-4] + "_Extra.c3d"
    static_trial_extra = static_trial[:-4] + "_Extra.c3d"
    # Dynamic trial now contains fake markers RWRI and LWRI which replace their respective calculations
    subject5 = CustomCGM3(static_trial_extra, dynamic_trial_extra, measurements, end=5)
    subject5.rename_markers({"RWRI": "RWRI", "LWRI": "LWRI"})  # Must define new markers
    subject5.run()
    print("wrist jc standard")
    print(subject1.wrist_joint_centers[0])
    print("wrist jc extra markers")
    print(subject5.wrist_joint_centers[0])
