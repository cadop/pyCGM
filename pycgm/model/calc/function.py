import numpy as np


class Function:
    def __init__(self, func, parameters=None, returns=None):
        self.func = func
        self.name = func.__name__

        self.parameter_values = {}
        self.parameters = parameters
        self.returns = returns

    @staticmethod
    def _get_key(parameter):
        if parameter != [] and parameter[0] is not None:
            return parameter
        return []

    @property
    def required_markers(self):
        markers = self.parameters.get('markers')
        return self._get_key(markers)

    @property
    def required_measurements(self):
        measurements = self.parameters.get('measurements')
        return self._get_key(measurements)

    @property
    def required_axes(self):
        axes = self.parameters.get('axes')
        return self._get_key(axes)

    @property
    def required_angles(self):
        angles = self.parameters.get('angles')
        return self._get_key(angles)

    @property
    def returned_measurements(self):
        measurements = self.returns.get('measurements')
        return self._get_key(measurements)

    @property
    def returned_axes(self):
        axes = self.returns.get('axes')
        return self._get_key(axes)

    @property
    def returned_angles(self):
        angles = self.returns.get('angles')
        return self._get_key(angles)

    def run(self, trial_name):
        params = []
        for value in self.parameter_values[trial_name]:
            if type(value) == np.ndarray:
                if value.shape == (1,) and value.dtype in [float, np.bool_]:
                    params.append(value[0])
                else:
                    params.append(value)
            else:
                params.append(value)
        return np.asarray(self.func(*params))

    def info(*args, **kwargs):
        def structure(func):
            to_list        = lambda value: [value] if not isinstance(value, list) else value
            to_dtype_tuple = lambda item: (item, None) if not isinstance(item, tuple) else item
            set_dtype      = lambda item_list, default_dtype: [(name, dtype if dtype is not None else default_dtype) for name, dtype in item_list]

            def process_items(key, default_dtype):
                items = [to_dtype_tuple(x) for x in to_list(kwargs.get(key, []))]
                return set_dtype(items, default_dtype)

            required_items = {
                'markers': np.dtype((float, (3))),
                'measurements': np.dtype(float),
                'axes': np.dtype((float, (3, 4))),
                'angles': np.dtype((float, (3)))
            }

            return_items = {
                'returns_axes': np.dtype((float, (3, 4))),
                'returns_angles': np.dtype((float, (3))),
                'returns_measurements': np.dtype(float)
            }

            parameters = {key: process_items(key, dtype) for key, dtype in required_items.items()}
            returns    = {key.replace('returns_', ''): process_items(key, dtype) for key, dtype in return_items.items()}

            return Function(func, parameters, returns)
        return structure
