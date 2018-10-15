import logging
import toml
import os
import sys
import argparse
import signal

from . import models


def save_config(model_name, config, workspace):
    """Save model configuration to ``workspace``."""
    f = open(os.path.join(workspace, 'config.toml'), 'w')
    toml.dump({'model': model_name,
               'config': config}, f)
    f.close()


def load_config(workspace):
    """Load model configuration from ``workspace``."""
    config = toml.load(open(os.path.join(workspace, 'config.toml'), 'r'))
    Model = getattr(models, config['model'])
    return Model, config['config']


class DelayedKeyboardInterrupt:
    signal_received = None

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


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
    """Log formatter that provides colored output."""

    def __init__(self, fmt, datefmt):
        """
        Args:
            fmt (str): message format string
            datefmt (str): date format string
        """
        super().__init__(fmt, datefmt)

    def format(self, record):
        """Format the specified record as text.

        If ``self.use_color`` is ``True``, format log messages according to
        :data:`~app.utils.LOG_COLORS`.
        """
        levelname = record.levelname
        if levelname in LOG_COLORS:
            record.levelname = _colored(record.levelname[0],
                                        LOG_COLORS[record.levelname])
        return logging.Formatter.format(self, record)


class _ArgumentParser(argparse.ArgumentParser):
    def __init__(self, raise_error=False, **kwargs):
        super().__init__(**kwargs)
        self.raise_error = raise_error

    def error(self, message):
        if self.raise_error:
            raise ValueError(message)
        # customize error message
        self.print_usage(sys.stderr)
        err = _colored('error:', LOG_COLORS['ERROR'], True)
        self.exit(2, '%s %s\n' % (err, message))
