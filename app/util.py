import logging
import toml
import os

from . import module
from .module import Simple
from collections import namedtuple


def make_model(Model, **kwargs):
    """Make model. Parameters are specified by keyword arguments.

    Example:
        >>> model = make_model(Simple, foo='bar')
        >>> print(model.config)
        Config(foo='bar')
    """
    config = namedtuple('Config', kwargs.keys())(*kwargs.values())
    return Model(config)


def save_config(obj, workspace):
    r"""Save model configuration to ``workspace``."""
    f = open(os.path.join(workspace, 'config.toml'), 'w')
    toml.dump({'model': obj.__class__.__name__,
               'config': obj.config._asdict()}, f)
    f.close()


def load_config(workspace):
    r"""Load model configuration from ``workspace``."""
    config = toml.load(open(os.path.join(workspace, 'config.toml'), 'r'))
    Model = getattr(module, config['model'])
    return make_model(Model, **config['config'])


class _BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _colored(text, color, bold=False):
    if bold:
        return _BColors.BOLD + color + text + _BColors.ENDC
    else:
        return color + text + _BColors.ENDC


#: Log level to color mapping.
LOG_COLORS = {
    'WARNING': _BColors.WARNING,
    'INFO': _BColors.OKGREEN,
    'DEBUG': _BColors.OKBLUE,
    'CRITICAL': _BColors.WARNING,
    'ERROR': _BColors.FAIL
}


class ColoredFormatter(logging.Formatter):
    r"""Log formatter that provides colored output."""

    def __init__(self, fmt, datefmt, use_color=True):
        r"""
        Args:
            fmt (str): message format string
            datefmt (str): date format string
            use_color (bool): whether to use colored_output. Default: ``True``
        """
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record):
        r"""Format the specified record as text.

        If ``self.use_color`` is ``True``, format log messages according to
        :data:`~app.utils.LOG_COLORS`.
        """
        levelname = record.levelname
        if self.use_color and levelname in LOG_COLORS:
            record.levelname = _colored(record.levelname[0],
                                        LOG_COLORS[record.levelname])
        return logging.Formatter.format(self, record)
