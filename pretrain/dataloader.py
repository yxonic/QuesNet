"""Pre-process input text, tokenizing, building vocabs, and pre-train word
level vectors."""

import os
from collections import namedtuple
from pathlib import Path
from copy import copy

import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_grayscale

from .util import lines

Question = namedtuple('Question',
                      ['id', 'content', 'answer', 'false_options', 'labels'])


class QuestionLoader:
    def __init__(self, ques_file, word_file, img_dir,
                 *label_file, pipeline=None, range=None):
        """Read question file as data list. Same behavior on same file."""
        self.range = None
        self.ques = lines(ques_file, skip=1)
        self.range = range or slice(0, len(self), 1)
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

    def split_(self, split_ratio):
        first_size = int(len(self) * split_ratio)
        other = copy(self)
        self.range = slice(0, first_size, 1)
        other.range = slice(first_size, len(other), 1)
        return other

    def __len__(self):
        return len(self.ques) if self.range is None \
            else self.range.stop - self.range.start

    def __getitem__(self, item):
        if isinstance(item, int):
            item += self.range.start
            item = slice(item, item + 1, 1)
        else:
            item = slice(item.start + self.range.start,
                         item.stop + self.range.start, 1)
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


def load_word2vec(size):
    emb_file = Path('data/emb_%d.txt' % size)
    if not emb_file.exists():
        return None

    f = emb_file.open()
    next(f)

    words = []
    embs = []
    for line in f:
        fields = line.strip().split(' ')
        word = fields[0]
        emb = np.array([float(x) for x in fields[1:]])
        words.append(word)
        embs.append(emb)

    embs = np.asarray(embs)
    return embs
