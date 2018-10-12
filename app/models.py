"""
Model classes that defines model parameters and architecture.
"""
import abc
import argparse
from collections import namedtuple
from . import util


class Model(abc.ABC):
    """Interface for model that can save/load parameters.

    Each model class should have an ``_add_argument`` class method to define
    model arguments along with their types, default values, etc.
    """
    @classmethod
    @abc.abstractmethod
    def _add_arguments(cls, parser: argparse.ArgumentParser):
        """Add arguments to an argparse subparser."""
        raise NotImplementedError

    @classmethod
    def build(cls, **kwargs):
        """Build model. Parameters are specified by keyword arguments."""
        config = namedtuple('Config', kwargs.keys())(*kwargs.values())
        return cls(config)

    @classmethod
    def parse(cls, args):
        """Parse command-line options and build model."""
        parser = util._ArgumentParser(prog='', add_help=False,
                                      raise_error=True)
        cls._add_arguments(parser)
        args = parser.parse_args(args)
        return cls.build(**dict(args._get_kwargs()))

    def __init__(self, config):
        """
        Args:
            config (namedtuple): model configuration
        """
        self.config = config


class RNN(Model):
    """Sequence-to-sequence models based on RNN. Supports different input
    forms (by word / by char), different RNN types (LSTM/GRU), """
    @classmethod
    def _add_arguments(cls, parser):
        parser.add_argument('-vocab', '-v', required=True, help='vocab file')
        parser.add_argument('-emb', '-e', help='pretrained vectors')
        parser.add_argument('-rnn', '-r', default='LSTM',
                            choices=['LSTM', 'GRU'], help='RNN type')
        parser.add_argument('-rnn_size', '-s', default=500, type=int,
                            help='size of rnn hidden states')
        parser.add_argument('-layers', '-l', default=1, type=int,
                            help='number of layers')
        parser.add_argument('-bi_enc', '-b', action='store_true',
                            help='use bi-directional encoder')


class ELMo(Model):
    @classmethod
    def _add_arguments(cls, parser):
        pass


class ULMFiT(Model):
    @classmethod
    def _add_arguments(cls, parser):
        pass


class TransformerLM(Model):
    @classmethod
    def _add_arguments(cls, parser):
        pass
