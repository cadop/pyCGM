
def bad_docstring(a):
    """Really bad docstring poc

    bad description

    Parameters
    ----------
    a: int
        A number

    Returns
    -------
    b : int
            Another number greater than a
    References
    ----------
    .. [1] nothing

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import bad_docstring
    >>> bad_docstring(1)
    3


    """

    return a + 2