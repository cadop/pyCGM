"""
An example of more advanced settings and customizations that pycgm offers.

    >>> from pycgm.pycgm import CGM
    >>> from pycgm.customizations import CustomCGM1, CustomCGM2

    Load in or define the custom subclass(es), and locate input data as usual.

    >>> sample_dir = "SampleData/ROM/"
    >>> static_trial = sample_dir + "Sample_Static.c3d"
    >>> dynamic_trial = sample_dir + "Sample_Dynamic.c3d"
    >>> measurements = sample_dir + "Sample_SM.vsk"

    Basic CGM object created and run in the normal way, except only the first 5 frames are calculated.

    >>> subject1 = CGM(static_trial, dynamic_trial, measurements, start=0, end=5)
    >>> subject1.run()

    Various types of output data in arrays, indexable by frame.

    Properties labeled "axes" return the origin and unit vectors for a total of four vectors per frame.

    Properties labeled "joint center" return only the vector containing the origin.

    Properties labeled "angle" return three vectors containing the flexion, abduction, and rotation angles.

    Properties labeled with one of the aforementioned angles return only that angle.

    Joints which have both a right and left counterpart return a two-element array containing the
    right and left portions of the property in question.

    >>> print(subject1.pelvis_axes, subject1.pelvis_joint_centers,
    ...       subject1.pelvis_angles, subject1.pelvis_flexions,
    ...       subject1.pelvis_abductions, subject1.pelvis_rotations)  # doctest: +SKIP

    Custom CGM object with a different pelvis calculation is created in the same manner as the default.

    >>> subject2 = CustomCGM1(static_trial, dynamic_trial, measurements, start=0, end=5)
    >>> subject2.run()

    Compare the outputs of only the affected values, in this case the pelvis joint center.

    >>> print("Pelvis JC default")  # doctest: +SKIP
    >>> print(subject1.pelvis_joint_centers)  # doctest: +SKIP
    >>> print("Pelvis JC custom")  # doctest: +SKIP
    >>> print(subject2.pelvis_joint_centers)  # doctest: +SKIP

    Custom CGM object that also comes with a custom static calculation, with the result modifying the CGM's output.

    >>> subject3 = CustomCGM2(static_trial, dynamic_trial, measurements, start=0, end=5)
    >>> subject3.run()

    Compare the outputs of only the affected values, in this case the entire hip axis.

    >>> print("Hip Axes default")  # doctest: +SKIP
    >>> print(subject1.hip_axes)  # doctest: +SKIP
    >>> print("Hip Axis custom")  # doctest: +SKIP
    >>> print(subject3.hip_axes)  # doctest: +SKIP

    # TODO: Need examples with: renaming markers, adding/removing markers, adding additional joint outputs,
    # TODO: running and comparing multiple subjects

"""

if __name__ == "__main__":
    from pycgm.pycgm import CGM
    from pycgm.customizations import CustomCGM1, CustomCGM2

    sample_dir = "SampleData/ROM/"
    static_trial = sample_dir + "Sample_Static.c3d"
    dynamic_trial = sample_dir + "Sample_Dynamic.c3d"
    measurements = sample_dir + "Sample_SM.vsk"
    subject1 = CGM(static_trial, dynamic_trial, measurements, start=0, end=5)
    subject1.run()
    print("Pelvis JC default")
    print(subject1.pelvis_joint_centers)

    subject2 = CustomCGM1(static_trial, dynamic_trial, measurements, start=0, end=5)
    subject2.run()
    print("Pelvis JC custom")
    print(subject2.pelvis_joint_centers)

    subject3 = CustomCGM2(static_trial, dynamic_trial, measurements, start=0, end=5)
    subject3.run()
    print("Hip Axes default")
    print(subject1.hip_axes)

    print("Hip Axis custom")
    print(subject3.hip_axes)
