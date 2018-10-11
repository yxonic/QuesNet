"""Pre-process input text, tokenizing, building vocabs, and pre-train word
level vectors."""

import torchtext as tt
import csv
import os


def _cut(x):
    if all(ord(c) > 127 for c in x):
        return list(x)
    else:
        return [x]


class CharTokenizer:
    def __init__(self, max_len=400):
        self.max_len = max_len

    def __call__(self, s):
        rv = []
        for x in s.split():
            rv.extend(_cut(x))
        return rv[:self.max_len]


class WordTokenizer:
    def __init__(self, max_len=400):
        self.max_len = max_len

    def __call__(self, s):
        return s.split()[:self.max_len]


class MultipleField(tt.data.RawField):
    def __init__(self, *args):
        super().__init__()
        self.fields = args
        self._vocab = None

    def preprocess(self, x):
        return [f.preprocess(x) for f in self.fields]

    def process(self, batch, *args, **kwargs):
        rv = []
        for i in range(len(self.fields)):
            rv.append(self.fields[i].process([x[i] for x in batch],
                                             **kwargs))
        return rv

    def build_vocab(self, *args):
        for i, f in enumerate(self.fields):
            sources = []
            for arg in args:
                if isinstance(arg, tt.data.Dataset):
                    sources += [getattr(arg, name) for name, field in
                                arg.fields.items() if field is self]
                else:
                    sources.append(arg)
            f.build_vocab(*[[x[i] for x in source] for source in sources])
        self._vocab = [f.vocab for f in self.fields]

    @property
    def vocab(self):
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        for v, f in zip(vocab, self.fields):
            f.vocab = v


class DataLoader:
    """Data loader class for vocab building, data splitting, loading/saving,
    etc."""
    def __init__(self, dirname, raw_file=None,
                 input_type='char', max_len=400):
        if raw_file is None:
            # load previous splits
            return

        # build text field
        if input_type == 'char':
            self.text_field = tt.data.Field(init_token='<sos>',
                                            eos_token='<eos>',
                                            tokenize=CharTokenizer(max_len),
                                            include_lengths=True)
        elif input_type == 'word':
            self.text_field = tt.data.Field(init_token='<sos>',
                                            eos_token='<eos>',
                                            tokenize=WordTokenizer(max_len),
                                            include_lengths=True)
        elif input_type == 'both':
            self.char_field = tt.data.Field(init_token='<sos>',
                                            eos_token='<eos>',
                                            tokenize=CharTokenizer(max_len),
                                            include_lengths=True)
            self.word_field = tt.data.Field(init_token='<sos>',
                                            eos_token='<eos>',
                                            tokenize=WordTokenizer(max_len),
                                            include_lengths=True)
            self.text_field = MultipleField(self.char_field, self.word_field)
        else:
            raise ValueError('input type not recognized')

        # load raw data
        fields = {'content': ('content', self.text_field)}
        reader = csv.reader(open(raw_file), quoting=csv.QUOTE_NONE,
                            delimiter='\t')
        header = next(reader)
        field_to_index = {f: header.index(f) for f in fields.keys()}
        examples = [tt.data.Example.fromCSV(line, fields, field_to_index)
                    for line in reader]

        field_list = []
        for field in fields.values():
            if isinstance(field, list):
                field_list.extend(field)
            else:
                field_list.append(field)

        self.dataset = tt.data.Dataset(examples, field_list)

        # split data

        # build vocab
        self.text_field.build_vocab(self.dataset)

        # save to data dir
        fn = os.path.join(dirname, os.path.basename(raw_file))
        print(fn)


if __name__ == '__main__':
    import time
    then = time.time()
    _ = DataLoader('', raw_file='data/questions.tsv',
                   input_type='char')
    now = time.time()
    print(now - then)
    then = now

    _ = DataLoader('', raw_file='data/questions.tsv',
                   input_type='word')
    now = time.time()
    print(now - then)
    then = now

    data = DataLoader('', raw_file='data/questions.tsv',
                      input_type='both')
    now = time.time()
    print(now - then)
    then = now

    train_iter = tt.data.BucketIterator(dataset=data.dataset, batch_size=4)
    b = next(iter(train_iter))
    print('lens:', [x.item() for x in b.content[0][1]])

    vocab = data.char_field.vocab
    print(' '.join(vocab.itos[x] for x in b.content[0][0].t()[0]))
