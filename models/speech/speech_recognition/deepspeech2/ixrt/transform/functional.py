import inspect


def check_kwargs(func, kwargs, name=None):
    """check kwargs are valid for func

    If kwargs are invalid, raise TypeError as same as python default
    :param function func: function to be validated
    :param dict kwargs: keyword arguments for func
    :param str name: name used in TypeError (default is func name)
    """
    try:
        params = inspect.signature(func).parameters
    except ValueError:
        return
    if name is None:
        name = func.__name__
    for k in kwargs.keys():
        if k not in params:
            raise TypeError(
                f"{name}() got an unexpected keyword argument '{k}'")


class TransformInterface:
    """Transform Interface"""

    def __call__(self, x):
        raise NotImplementedError("__call__ method is not implemented")

    @classmethod
    def add_arguments(cls, parser):
        return parser

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Identity(TransformInterface):
    """Identity Function"""

    def __call__(self, x):
        return x


class FuncTrans(TransformInterface):
    """Functional Transformation

    WARNING:
        Builtin or C/C++ functions may not work properly
        because this class heavily depends on the `inspect` module.

    Usage:

    >>> def foo_bar(x, a=1, b=2):
    ...     '''Foo bar
    ...     :param x: input
    ...     :param int a: default 1
    ...     :param int b: default 2
    ...     '''
    ...     return x + a - b


    >>> class FooBar(FuncTrans):
    ...     _func = foo_bar
    ...     __doc__ = foo_bar.__doc__
    """

    _func = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        check_kwargs(self.func, kwargs)

    def __call__(self, x):
        return self.func(x, **self.kwargs)

    @classmethod
    def add_arguments(cls, parser):
        fname = cls._func.__name__.replace("_", "-")
        group = parser.add_argument_group(fname + " transformation setting")
        for k, v in cls.default_params().items():
            # TODO(karita): get help and choices from docstring?
            attr = k.replace("_", "-")
            group.add_argument(f"--{fname}-{attr}", default=v, type=type(v))
        return parser

    @property
    def func(self):
        return type(self)._func

    @classmethod
    def default_params(cls):
        try:
            d = dict(inspect.signature(cls._func).parameters)
        except ValueError:
            d = dict()
        return {
            k: v.default
            for k, v in d.items() if v.default != inspect.Parameter.empty
        }

    def __repr__(self):
        params = self.default_params()
        params.update(**self.kwargs)
        ret = self.__class__.__name__ + "("
        if len(params) == 0:
            return ret + ")"
        for k, v in params.items():
            ret += "{}={}, ".format(k, v)
        return ret[:-2] + ")"
