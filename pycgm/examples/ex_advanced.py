"""
An example of more advanced settings and customizations that pycgm offers.

.. plot::
    :context: close-figs

    >>> from pycgm.pycgm import CGM
    >>> from pycgm.customizations import CustomCGM1, CustomCGM2
    >>> import pycgm
    >>> import matplotlib.pyplot as plt
    >>> static_trial, dynamic_trial, measurements = pycgm.get_rom_data()

    Basic CGM object created and run, but using optional parameters to specify a range of frames.

    >>> subject1 = CGM(static_trial, dynamic_trial, measurements, start=0, end=5)  # doctest: +ELLIPSIS
    >>> subject1.run()

    Demonstration of the various properties used for accessing useful subsets of the results.
    Joints which have both a right and left counterpart return a two-element array containing the
    appropriate right and left portions.

    >>> print(subject1.pelvis_axes, subject1.pelvis_joint_centers,
    ...       subject1.pelvis_angles, subject1.pelvis_flexions,
    ...       subject1.pelvis_abductions, subject1.pelvis_rotations)  # doctest: +SKIP

    Custom CGM object with a different pelvis calculation is created in the same manner as the default.

    >>> subject2 = CustomCGM1(static_trial, dynamic_trial, measurements, start=0, end=5)  # doctest: +ELLIPSIS
    >>> subject2.run()

    Visually compare the outputs of only the affected values, in this case the pelvis joint center.

    >>> plt.plot(subject1.pelvis_joint_centers[:, 0])  # doctest: +SKIP
    >>> plt.plot(subject2.pelvis_joint_centers[:, 0])  # doctest: +SKIP
    >>> plt.plot(subject1.pelvis_joint_centers[:, 1])  # doctest: +SKIP
    >>> plt.plot(subject2.pelvis_joint_centers[:, 1])  # doctest: +SKIP
    >>> plt.plot(subject1.pelvis_joint_centers[:, 2])  # doctest: +SKIP
    >>> plt.plot(subject2.pelvis_joint_centers[:, 2])  # doctest: +SKIP
    >>> plt.xlabel('Frame')  # doctest: +SKIP
    >>> plt.ylabel('Pelvis Joint Centers')  # doctest: +SKIP
    >>> plt.tight_layout()
    >>> plt.show()  # doctest: +SKIP

    Custom CGM object that also comes with a custom static calculation, which stores an extra value as a measurement.
    The value is accessed and used to modify the CGM's output.

    >>> subject3 = CustomCGM2(static_trial, dynamic_trial, measurements, start=0, end=5)  # doctest: +ELLIPSIS
    >>> subject3.run()

.. plot::
    :context: close-figs

    Visually compare the outputs of only the affected values, in this case the entire hip axis.

    >>> plt.plot(subject1.hip_axes[0][:, 0])  # doctest: +SKIP
    >>> plt.plot(subject3.hip_axes[0][:, 0])  # doctest: +SKIP
    >>> plt.xlabel('Frame')  # doctest: +SKIP
    >>> plt.ylabel('Hip Axes')  # doctest: +SKIP
    >>> plt.tight_layout()
    >>> plt.show()  # doctest: +SKIP

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
