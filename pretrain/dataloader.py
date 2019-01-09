"""Pre-process input text, tokenizing, building vocabs, and pre-train word
level vectors."""

import csv
import os
from collections import namedtuple
from pathlib import Path

import torchtext as tt
from PIL import Image
from torchvision.transforms.functional import to_grayscale

from .util import lines

Question = namedtuple('Question',
                      ['id', 'content', 'answer', 'false_options', 'labels'])


class QuestionLoader:
    def __init__(self, ques_file, word_file, img_dir,
                 *label_file, pipeline=None):
        """Read question file as data list. Same behavior on same file."""
        self.ques = lines(ques_file, skip=1)
        self.img_dir = img_dir
        self.labels = []

        self.itos = dict()
        self.stoi = dict()
        self.itos['word'] = lines(word_file)
        self.stoi['word'] = {s: i for i, s in enumerate(self.itos['word'])}

        for filename in label_file:
            f = lines(filename)
            label_name = f[0].split('\t')[1]
            map = {}
            for l in f[1:]:
                qid, v = l.split('\t')
                map[qid] = v if label_name.startswith('[') else float(v)
            if label_name.startswith('['):  # [label]
                label_name = label_name[1:-1]
                f = str(Path(filename).parent / (label_name + '_list.txt'))
                self.itos[label_name] = lines(f)
                self.stoi[label_name] = {s: i for i, s in
                                         enumerate(self.itos[label_name])}
            self.labels.append((label_name, map))

        self.pipeline = pipeline

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, item):
        qs = []
        for line in self.ques[item]:
            fields = line.split('\t')
            qid, content, answer = fields[0], fields[1], fields[2]
            false_options = fields[4]
            content = content.split()
            for i in range(len(content)):
                if content[i].startswith('{img:'):
                    try:
                        im = Image.open(os.path.join(self.img_dir,
                                                     content[i][5:-1]))
                        im = im.resize((56, 56))
                        content[i] = to_grayscale(im)
                    except Exception:
                        content[i] = self.stoi['word']['{img}']
                else:
                    content[i] = self.stoi['word'].get(content[i]) or 0

            answer = [self.stoi['word'].get(a) or 0 for a in answer.split()]

            if len(false_options):
                false_options = [[self.stoi['word'].get(x) or 0
                                  for x in o.split()]
                                 for o in false_options.split('::')]

            else:
                false_options = None

            labels = {}
            for name, map in self.labels:
                if qid in map:
                    v = map[qid]
                    if isinstance(v, float):
                        labels[name] = v
                    else:
                        labels[name] = [self.stoi[name].get(k) or 0
                                        for k in v.split(',')]

            qs.append(Question(qid, content, answer, false_options, labels))

        if callable(self.pipeline):
            return self.pipeline(qs)
        else:
            return qs


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
