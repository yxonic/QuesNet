import io
import linecache
import logging
import math
import queue
import random
import signal
import subprocess
import threading

import torch
from torch.nn.utils.rnn import pack_padded_sequence

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


def stateful(*states):

    def wrapper(cls):
        orig_state_dict = orig_load_state_dict = None
        if hasattr(cls, 'state_dict'):
            orig_state_dict = cls.state_dict
        if hasattr(cls, 'load_state_dict'):
            orig_load_state_dict = cls.load_state_dict

        def state_dict(self):
            state = {s: getattr(self, s) for s in states}
            if orig_state_dict is not None:
                state.update(orig_state_dict(self))
            return state

        def load_state_dict(self, state):
            for s in states:
                setattr(self, s, state[s])
            if orig_load_state_dict is not None:
                orig_load_state_dict(self, state)

        cls.state_dict = state_dict
        cls.load_state_dict = load_state_dict
        return cls

    return wrapper


def clip(v, low, high):
    if v < low:
        v = low
    if v > high:
        v = high
    return v


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


# noinspection PyPep8Naming
class lines:
    def __init__(self, filename, skip=0, preserve_newline=False):
        self.filename = filename
        with open(filename):
            pass
        output = subprocess.check_output(('wc -l ' + filename).split())
        self.length = int(output.split()[0]) - skip
        self.skip = skip
        self.preserve_newline = preserve_newline

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        d = self.skip + 1
        if isinstance(item, int):
            if item < self.length:
                line = linecache.getline(self.filename,
                                         item % len(self) + d)
                if self.preserve_newline:
                    return line
                else:
                    return line.strip('\r\n')

        elif isinstance(item, slice):
            low = 0 if item.start is None else item.start
            low = clip(low, -len(self), len(self) - 1)
            if low < 0:
                low += len(self)
            high = len(self) if item.stop is None else item.stop
            high = clip(high, -len(self), len(self))
            if high < 0:
                high += len(self)
            ls = []
            for i in range(low, high):
                line = linecache.getline(self.filename, i + d)
                if not self.preserve_newline:
                    line = line.strip('\r\n')
                ls.append(line)

            return ls

        raise IndexError('index must be int or slice')


@stateful('batch_size', 'index', 'pos')
class PrefetchIter:
    """Iterator on data and labels, with states for save and restore."""

    def __init__(self, data, *label, length=None, batch_size=1, shuffle=True):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=8)
        self.length = length if length is not None else len(data)

        assert all(self.length == len(lab) for lab in label), \
            'data and label must have same lengths'

        self.index = list(range(len(self)))
        if shuffle:
            random.shuffle(self.index)
        self.thread = None
        self.pos = 0

    def __len__(self):
        return math.ceil(self.length / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.produce, daemon=True)
            self.thread.start()

        if self.pos >= len(self.index):
            raise StopIteration

        item = self.queue.get()
        if isinstance(item, Exception):
            raise item
        else:
            self.pos += 1
            return item

    def produce(self):
        for i in range(self.pos, len(self.index)):
            try:
                index = self.index[i]

                bs = self.batch_size

                if callable(self.data):
                    data_batch = self.data(index * bs, (index + 1) * bs)
                else:
                    data_batch = self.data[index * bs:(index + 1) * bs]

                label_batch = [label[index * bs:(index + 1) * bs]
                               for label in self.label]
                if label_batch:
                    self.queue.put([data_batch] + label_batch)
                else:
                    self.queue.put(data_batch)
            except Exception as e:
                self.queue.put(e)
                return


class SeqBatch:
    def __init__(self, seqs, dtype=None, device=None):
        self.dtype = dtype
        self.device = device
        self.seqs = seqs
        self.lens = [len(x) for x in seqs]

        self.ind = argsort(self.lens)[::-1]
        self.inv = argsort(self.ind)
        self.lens.sort(reverse=True)
        self._prefix = [0]
        self._index = {}
        c = 0
        for i in range(self.lens[0]):
            for j in range(len(self.lens)):
                if self.lens[j] <= i:
                    break
                self._index[i, j] = c
                c += 1

    def packed(self):
        ind = torch.tensor(self.ind, dtype=torch.long, device=self.device)
        padded = self.padded()[0].index_select(1, ind)
        return pack_padded_sequence(padded, self.lens)

    def padded(self, max_len=None, batch_first=False):
        seqs = [torch.tensor(s, dtype=self.dtype, device=self.device)
                if not isinstance(s, torch.Tensor) else s
                for s in self.seqs]
        if max_len is None:
            max_len = self.lens[0]
        seqs = [s[:max_len] for s in seqs]
        mask = [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs]

        trailing_dims = seqs[0].size()[1:]
        if batch_first:
            out_dims = (len(seqs), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(seqs)) + trailing_dims

        padded = seqs[0].new(*out_dims).fill_(0)
        for i, tensor in enumerate(seqs):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                padded[i, :length, ...] = tensor
            else:
                padded[:length, i, ...] = tensor
        return padded, torch.tensor(mask).byte().to(self.device)

    def index(self, item):
        return self._index[item[0], self.inv[item[1]]]

    def invert(self, batch, dim=0):
        return batch.index_select(dim, torch.tensor(self.inv))


class TableBuilder:
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
                data = [f % (v * 100) if v < m
                        else '\\textbf{%s}' % f % (v * 100)
                        for v in row]
            else:
                f = '%%.%df' % decimal
                data = [f % v if v < m else '\\textbf{%s}' % f % v
                        for v in row]
            table.write('    ' + ' & '.join([self.rows[i]] + data) +
                        '\\\\\n')

        table.write('    \\bottomrule\n'
                    '  \\end{tabular}\n'
                    '\\end{table*}\n')
        return table.getvalue()


if __name__ == '__main__':
    b = SeqBatch([[1, 2], [1, 2, 3, 4, 5, 6], [1], [1, 2, 3], [1, 2, 3]])
    print(b.index((2, 3)), b.packed().data[b.index((2, 3))])
    print(b.padded()[0].size())
    print(b.padded(max_len=20, batch_first=True)[0].size())
    print(b.padded(max_len=20, batch_first=True)[1])
