"""
Model classes that defines model parameters and architecture.
"""
import abc
import argparse
import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from collections import namedtuple
from . import util


class Model(abc.ABC, torch.nn.Module):
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
        super().__init__()
        self.config = config


class RNN(Model):
    """Sequence-to-sequence models based on RNN. Supports different input
    forms (by word / by char), different RNN types (LSTM/GRU), """

    @classmethod
    def _add_arguments(cls, parser):
        parser.add_argument('-vocab', '-v', required=True, help='vocab file')
        parser.add_argument('-emb', '-e', help='pretrained vectors')
        parser.add_argument('-emb_size', '-es', default=200, type=int,
                            help='size of embedding vectors')
        parser.add_argument('-rnn', '-r', default='LSTM',
                            choices=['LSTM', 'GRU'], help='RNN type')
        parser.add_argument('-rnn_size', '-rs', default=500, type=int,
                            help='size of rnn hidden states')
        parser.add_argument('-layers', '-l', default=1, type=int,
                            help='number of layers')
        parser.add_argument('-bi_enc', '-b', action='store_true',
                            help='use bi-directional encoder')

    def __init__(self, config):
        super().__init__(config)
        vocab = torch.load(config.vocab)
        vocab_size = len(vocab.stoi)
        emb_size = 200

        self.embedding = torch.nn.Embedding(vocab_size, emb_size)
        if config.rnn == 'GRU':
            self.rnn = torch.nn.GRU(emb_size, config.rnn_size,
                                    config.layers)
            self.h0 = torch.nn.Parameter(torch.rand(config.layers, 1,
                                                    config.rnn_size))
        else:
            self.rnn = torch.nn.LSTM(emb_size, config.rnn_size,
                                     config.layers)
            self.h0 = torch.nn.Parameter(torch.rand(config.layers, 1,
                                                    config.rnn_size))
        self.output = torch.nn.Linear(config.rnn_size, vocab_size)

    def lm_loss(self, batch):
        x, lens = batch
        lens -= 1
        input = x[:-1, :]
        h = self.init_h(x)

        emb = self.embedding(input)
        y, _ = self.rnn(pack_padded_sequence(emb, lens), h)

        y_pred = self.output(y.data)
        y_true = pack_padded_sequence(x[1:, :], lens).data

        loss = F.cross_entropy(y_pred, y_true)
        return loss

    def init_h(self, batch):
        size = list(self.h0.size())
        size[1] = batch.size(1)
        return self.h0.expand(size)

    def forward(self, *input):
        pass


class ELMo(Model):
    @classmethod
    def _add_arguments(cls, parser):
        pass

    def forward(self, *input):
        pass


class ULMFiT(Model):
    @classmethod
    def _add_arguments(cls, parser):
        pass

    def forward(self, *input):
        pass


class TransformerLM(Model):
    @classmethod
    def _add_arguments(cls, parser):
        pass

    def forward(self, *input):
        pass
