import numpy as np


class Function:
    def __init__(self, func, parameters=None, returns=None):
        self.func = func
        self.name = func.__name__

        self.parameter_values = {}
        self.parameters = parameters
        self.returns = returns

    @property
    def required_markers(self):
        markers = self.parameters.get('markers')
        if markers[0] is not None:
            return markers
        return []

    @property
    def required_measurements(self):
        measurements = self.parameters.get('measurements')
        if measurements[0] is not None:
            return measurements
        return []

    @property
    def required_axes(self):
        axes = self.parameters.get('axes')
        if axes[0] is not None:
            return axes
        return []

    @property
    def required_angles(self):
        angles = self.parameters.get('angles')
        if angles[0] is not None:
            return angles
        return []

    @property
    def required_constants(self):
        constants = self.parameters.get('constants')
        if constants[0] is not None:
            return constants
        return []

    @property
    def returned_axes(self):
        axes = self.returns.get('axes')
        if axes[0] is not None:
            return axes
        return []

    @property
    def returned_angles(self):
        angles = self.returns.get('angles')
        if angles[0] is not None:
            return angles
        return []

    def run(self, trial_name):
        return np.asarray(self.func(*self.parameter_values[trial_name]))


    def info(*args, **kwargs):

        def structure(func):
            to_list = lambda value : [value] if not isinstance(value, list) else value

            required_markers      = to_list(kwargs.get('markers'))
            required_measurements = to_list(kwargs.get('measurements'))
            required_axes         = to_list(kwargs.get('axes'))
            required_angles       = to_list(kwargs.get('angles'))
            required_constants    = to_list(kwargs.get('constants'))

            returns_axes   = to_list(kwargs.get('returns_axes'))
            returns_angles = to_list(kwargs.get('returns_angles'))


            # if returns_axes is not [None] and returns_angles is not [None]:
            #     raise Exception(func.__name__ + " returns both axes and angles")

            parameters = { "markers":      required_markers,
                           "measurements": required_measurements,
                           "axes":         required_axes,
                           "angles":       required_angles,
                           "constants":    required_constants }

            returns = { "axes":   returns_axes,
                        "angles": returns_angles }
             
            return Function(func, parameters, returns)
             
        return structure

