import io
import logging
import signal

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


sigint_handler = signal.getsignal(signal.SIGINT)


def critical(f):
    it = iter(f)
    signal_received = ()

    def handler(sig, frame):
        nonlocal signal_received
        signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    while True:
        try:
            signal.signal(signal.SIGINT, handler)
            yield next(it)
            signal.signal(signal.SIGINT, sigint_handler)
            if signal_received:
                sigint_handler(*signal_received)
        except StopIteration:
            break


def stateful(states):

    def wrapper(cls):
        def state_dict(self):
            return {s: getattr(self, s) for s in states}

        def load_state_dict(self, state):
            for s in states:
                setattr(self, s, state[s])

        cls.state_dict = state_dict
        cls.load_state_dict = load_state_dict
        return cls

    return wrapper


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


class SeqBatch:
    def __init__(self, seqs, dtype=None, device=None):
        self.seqs = [torch.tensor(s, dtype=dtype, device=device) for s in seqs]
        self.lens = [len(x) for x in seqs]
        self.ind = torch.tensor(argsort(self.lens)[::-1], dtype=torch.long)
        self.inv = torch.tensor(argsort(self.ind), dtype=torch.long)

    def packed(self):
        padded = self.padded()
        return pack_padded_sequence(padded.index_select(1, self.ind),
                                     sorted(self.lens, reverse=True))

    def padded(self):
        return pad_sequence(self.seqs)

    def invert(self, batch, dim=0):
        return batch.index_select(dim, self.inv)


class TableBuider:
    def column(self, *headers):
        self.headers = headers

    def row(self, *rows):
        self.rows = rows

    def data(self, data):
        self.data = data

    def to_latex(self, decimal=2, percentage=False):
        table = io.StringIO()
        table.write('\\begin{table*}\n')
        table.write('  \\begin{tabular}{l|%s}\n'
                    '    \\toprule\n' % ('c' * (len(self.headers) - 1)))
        table.write('    ' + ' & '.join(self.headers) + '\\\\\n')

        table.write('    \\midrule\n')

        for i, row in enumerate(self.data):
            m = max(row)
            if percentage:
                f = '%%.%df\\%%%%' % decimal
                data = [f % (x * 100) if x < m
                        else '\\textbf{%s}' % f % (x * 100)
                        for x in row]
            else:
                f = '%%.%df' % decimal
                data = [f % x if x < m else '\\textbf{%s}' % f % x
                        for x in row]
            table.write('    ' + ' & '.join([self.rows[i]] + data) +
                        '\\\\\n')

        table.write('    \\bottomrule\n'
                    '  \\end{tabular}\n'
                    '\\end{table*}\n')
        return table.getvalue()


if __name__ == '__main__':
    table = TableBuider()
    table.column('Task', 'ELMo', 'BERT', 'QuesNet')
    table.row('KP', 'DP', 'SP')
    table.data([[0.34, 0.422, 0.31416],
                [0.11, 0.222, 0.618],
                [0.152, 0.134, 0.12341341]])
    print(table.to_latex())
