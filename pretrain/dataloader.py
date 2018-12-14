"""Pre-process input text, tokenizing, building vocabs, and pre-train word
level vectors."""

import csv

import torchtext as tt


def _cut(x):
    if ord(x[0]) > 127:
        return list(x)
    else:
        return [x]


class CharTokenizer:
    """Split text into characters."""

    def __init__(self, max_len=400):
        self.max_len = max_len

    def __call__(self, s):
        rv = []
        for x in s.split():
            if ord(x[0]) > 127:
                # cut chinese word into chars
                rv.extend(list(x))
            else:
                rv.append(x)
        return rv[:self.max_len]


class WordTokenizer:
    """Split text into words."""

    def __init__(self, max_len=400):
        self.max_len = max_len

    def __call__(self, s):
        return s.split()[:self.max_len]


class MultipleField(tt.data.RawField):
    """A field with multiple sub-fields."""

    def __init__(self, *args):
        super().__init__()
        self.fields = list(args)
        self._vocab = None

    def preprocess(self, x):
        return [f.preprocess(x) for f in self.fields]

    def process(self, batch, *args, **kwargs):
        rv = []
        for i in range(len(self.fields)):
            rv.append(self.fields[i].process([x[i] for x in batch],
                                             **kwargs))
        return rv

    def build_vocab(self, *args, **kwargs):
        for i, f in enumerate(self.fields):
            sources = []
            for arg in args:
                if isinstance(arg, tt.data.Dataset):
                    sources += [getattr(arg, name) for name, field in
                                arg.fields.items() if field is self]
                else:
                    sources.append(arg)
            f.build_vocab(*[[x[i] for x in source] for source in sources],
                          **kwargs)
        self._vocab = [f.vocab for f in self.fields]

    @property
    def vocab(self):
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        self._vocab = vocab
        for v, f in zip(vocab, self.fields):
            f.vocab = v


class DataLoader:
    """Data loader class for vocab building, data splitting, and data
    loading."""

    def __init__(self, raw_file=None, input_type='char', max_len=400,
                 split_ratio=(0.8, 0.1, 0.1), split_rand_seed=None):
        if raw_file is None:
            # return empty loader
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
        self.examples = examples

        field_list = []
        for field in fields.values():
            if isinstance(field, list):
                field_list.extend(field)
            else:
                field_list.append(field)
        self.field_list = field_list

        dataset = tt.data.Dataset(examples, field_list)

        # split data
        self.split_ratio = list(split_ratio)
        self.split_rand_seed = split_rand_seed
        train_set, valid_set, test_set = dataset.split(
            split_ratio=self.split_ratio,
            random_state=self.split_rand_seed
        )

        self.splits = [train_set, valid_set, test_set]

    def build_vocab(self, **kwargs):
        self.text_field.build_vocab(self.splits[0], **kwargs)

    def state_dict(self):
        return {
            'examples': self.examples,
            'field_list': self.field_list,
            'split_ratio': self.split_ratio,
            'split_rand_seed': self.split_rand_seed
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        rv = cls()
        rv.examples = state_dict['examples']
        rv.field_list = state_dict['field_list']
        rv.text_field = rv.field_list[0]
        rv.split_ratio = state_dict['split_ratio']
        rv.split_rand_seed = state_dict['split_rand_seed']
        dataset = tt.data.Dataset(rv.examples, rv.field_list)
        train_set, valid_set, test_set = dataset.split(
            split_ratio=rv.split_ratio,
            random_state=rv.split_rand_seed
        )
        rv.splits = [train_set, valid_set, test_set]
        return rv


if __name__ == '__main__':
    data = DataLoader(raw_file='data/questions.head.tsv',
                      input_type='both')

    train_iter, valid_iter, test_iter = \
        tt.data.BucketIterator.splits(data.splits, batch_sizes=[8, 32, 32],
                                      sort_key=lambda x: len(x.content[0]),
                                      sort_within_batch=True)

    b = next(iter(train_iter))
    print('lens:', [x.item() for x in b.content[0][1]])

    cv = data.char_field.vocab
    print(' '.join(cv.itos[x] for x in b.content[0][0].t()[0]))
