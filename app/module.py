"""
Model classes that defines model parameters and architecture.
"""
from abc import ABC, abstractmethod


class _Module(ABC):
    """Interface for any module that wants parameters.

    Each model class should have an ``_add_argument`` static method.
    """
    @classmethod
    @abstractmethod
    def _add_arguments(cls, parser):
        """Add arguments to an argparse subparser"""
        pass

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
