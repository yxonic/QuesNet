import fret

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class RNN(fret.Module, torch.nn.Module):
    """Sequence-to-sequence models based on RNN. Supports different input
    forms (by word / by char), different RNN types (LSTM/GRU), """

    @classmethod
    def add_arguments(cls, parser):
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

    def __init__(self, **config):
        super().__init__(**config)
        torch.nn.Module.__init__(self)
        config = self.config
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
            self.c0 = torch.nn.Parameter(torch.rand(config.layers, 1,
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
        if self.config.rnn == 'GRU':
            return self.h0.expand(size)
        else:
            return self.h0.expand(size), self.c0.expand(size)

    def forward(self, *input):
        pass


class ELMo(fret.Module):
    @classmethod
    def add_arguments(cls, parser):
        pass

    def forward(self, *input):
        pass


class ULMFiT(fret.Module):
    @classmethod
    def add_arguments(cls, parser):
        pass

    def forward(self, *input):
        pass


class TransformerLM(fret.Module):
    @classmethod
    def add_arguments(cls, parser):
        pass

    def forward(self, *input):
        pass
