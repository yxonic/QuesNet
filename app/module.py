"""
Model classes that defines model parameters and architecture.
"""
import abc
from collections import namedtuple


class _Module(abc.ABC):
    """Interface for any module that wants parameters.

    Each model class should have an ``_add_argument`` class method.
    """
    @classmethod
    @abc.abstractmethod
    def _add_arguments(cls, parser):
        """Add arguments to an argparse subparser."""
        pass

    @classmethod
    def build(cls, **kwargs):
        """Build module. Parameters are specified by keyword arguments.

        Example:
            >>> model = Simple.build(foo='bar')
            >>> print(model.config)
            Config(foo='bar')
        """
        config = namedtuple('Config', kwargs.keys())(*kwargs.values())
        return cls(config)

    def __init__(self, config):
        """
        Args:
            config (namedtuple): model configuration
        """
        self.config = config


class Simple(_Module):
    """A toy class to demonstrate how to add model arguments."""

    @classmethod
    def _add_arguments(cls, parser):
        parser.add_argument('-foo', default=10, help='dumb param')

    def __init__(self, config):
        super().__init__(config)
