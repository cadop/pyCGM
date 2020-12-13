"""
An example of the most basic use case of pycgm to perform joint angle calculations.

.. plot::
    :context: close-figs

    >>> from pycgm.pycgm import CGM
    >>> import pycgm
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP

    After importing, locate the input files (Static trial, Dynamic trial and Subject Measurements).

    >>> static_trial, dynamic_trial, measurements = pycgm.get_rom_data()

    Create the `CGM` object by passing in the input file paths.

    >>> subject1 = CGM(static_trial, dynamic_trial, measurements, end=10)  # doctest: +ELLIPSIS

    Explicitly tell the object when to perform the calculations with `run()`.

    >>> subject1.run()

    Plot results of head flexion angle

    >>> head_flex = subject1.head_flexions
    >>> plt.plot(head_flex)  # doctest: +SKIP
    >>> plt.xlabel('Frame')  # doctest: +SKIP
    >>> plt.ylabel('Head Flexion Angle')  # doctest: +SKIP
    >>> plt.tight_layout()  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

    # TODO: Show accessing output

"""

if __name__ == "__main__":
    from pycgm.pycgm import CGM

    sample_dir = "SampleData/ROM/"
    static_trial = sample_dir + "Sample_Static.c3d"
    dynamic_trial = sample_dir + "Sample_Dynamic.c3d"
    measurements = sample_dir + "Sample_SM.vsk"
    subject1 = CGM(static_trial, dynamic_trial, measurements, start=0, end=5)
    subject1.run()
