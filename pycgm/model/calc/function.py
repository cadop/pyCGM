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
    def returned_measurements(self):
        measurements = self.returns.get('measurements')
        if measurements[0] is not None:
            return measurements
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
        params = []
        for value in self.parameter_values[trial_name]:
            if type(value) == np.ndarray:
                if value.shape == (1,) and value.dtype in [np.float64, np.bool_]:
                    params.append(value[0])
                else:
                    params.append(value)
            else:
                params.append(value)
        return np.asarray(self.func(*params))


    def info(*args, **kwargs):
        def structure(func):
            to_list = lambda value : [value] if not isinstance(value, list) else value

            required_markers      = to_list(kwargs.get('markers'))
            required_measurements = to_list(kwargs.get('measurements'))
            required_axes         = to_list(kwargs.get('axes'))
            required_angles       = to_list(kwargs.get('angles'))

            returns_axes         = to_list(kwargs.get('returns_axes'))
            returns_angles       = to_list(kwargs.get('returns_angles'))
            returns_measurements = to_list(kwargs.get('returns_measurements'))

            parameters = { "markers":      required_markers,
                           "measurements": required_measurements,
                           "axes":         required_axes,
                           "angles":       required_angles }

            returns = { "axes":         returns_axes,
                        "angles":       returns_angles,
                        "measurements": returns_measurements }
             
            return Function(func, parameters, returns)
             
        return structure
