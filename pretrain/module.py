import math
from functools import partial

import fret

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torchvision.transforms.functional import to_grayscale, to_tensor
from tqdm import tqdm

from .dataloader import QuestionLoader
from .util import PrefetchIter, SeqBatch, critical, device


@fret.configurable
class Trainer:
    def __init__(self, feature_extractor):
        self.ques = QuestionLoader('data/ques.txt', 'data/words.txt',
                                   'data/imgs')

        self.feature_extractor = \
            feature_extractor(_stoi=self.ques.stoi).to(device)

        self.pretrain_pipeline = partial(self.feature_extractor.make_batch,
                                         pretrain=True)
        self.eval_pipeline = self.feature_extractor.make_batch

    def pretrain(self, args):
        self.feature_extractor.train()
        self.ques.pipeline = self.pretrain_pipeline

        optim = torch.optim.Adam(self.feature_extractor.parameters())

        for epoch in range(args.n_epochs):
            train_iter = PrefetchIter(self.ques, batch_size=args.batch_size)

            try:
                for i, batch in critical(enumerate(tqdm(train_iter))):
                    loss = self.feature_extractor.pretrain_loss(batch)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
            except KeyboardInterrupt:
                raise

    def eval(self, args):
        pass

    def save_state(self):
        pass

    def load_state(self):
        pass

    def save_model(self, tag):
        pass

    def load_model(self, tag):
        pass


@fret.configurable
class FeatureExtractor(nn.Module):
    def __init__(self, feat_size=512):
        super(FeatureExtractor, self).__init__()
        self.feat_size = feat_size

    def make_batch(self, data, pretrain=False):
        """Make batch from input data (python data / np arrays -> tensors)"""
        pass

    def pretrain_loss(self, batch):
        """Returns pretraining loss on a batch of data"""
        pass

    def forward(self, *input):
        pass


class _Predictor:
    def __init__(self, out_dim):
        pass

    def forward(self, features):
        pass


class _ScorePrediction:
    def __init__(self):
        pass

    def forward(self):
        pass


@fret.configurable
class RNN(FeatureExtractor):
    """Sequence-to-sequence feature extractor based on RNN. Supports different
    input forms and different RNN types (LSTM/GRU), """

    def __init__(self, _stoi,
                 emb_size=(256, 'size of embedding vectors'),
                 rnn=('LSTM', 'size of rnn hidden states', ['LSTM', 'GRU']),
                 layers=(1, 'number of layers'), **kwargs):
        super(RNN, self).__init__(**kwargs)

        rnn_size = self.feat_size

        self.stoi = _stoi
        vocab_size = len(_stoi)

        self.embedding = nn.Embedding(vocab_size, emb_size)
        if rnn == 'GRU':
            self.rnn = nn.GRU(emb_size, rnn_size, layers)
            self.h0 = nn.Parameter(torch.rand(layers, 1, rnn_size))
        else:
            self.rnn = nn.LSTM(emb_size, rnn_size, layers)
            self.h0 = nn.Parameter(torch.rand(layers, 1, rnn_size))
            self.c0 = nn.Parameter(torch.rand(layers, 1, rnn_size))
        self.output = nn.Linear(rnn_size, vocab_size)

    def make_batch(self, data, pretrain=False):
        qs = [[x if isinstance(x, int) else self.stoi.get('{img}') or 0
               for x in q.content] for q in data]
        if pretrain:
            inputs = [[0] + q for q in qs]
            outputs = [q + [0] for q in qs]
            return SeqBatch(inputs, device=device), \
                SeqBatch(outputs, device=device)
        return SeqBatch(qs, device=device)

    def forward(self, batch: SeqBatch):
        packed = batch.packed()
        emb = self.embedding(packed.data)
        h = self.init_h(packed.batch_sizes[0])
        y, h = self.rnn(PackedSequence(emb, packed.batch_sizes), h)
        if self.config['rnn'] == 'GRU':
            return y, batch.invert(h, 1)
        else:
            return y, batch.invert(h[0], 1)

    def pretrain_loss(self, batch):
        input, output = batch
        y_true = output.packed().data
        y = self(input)
        y_pred = self.output(y[0].data)
        loss = F.cross_entropy(y_pred, y_true.data)
        return loss

    def init_h(self, batch_size):
        size = list(self.h0.size())
        size[1] = batch_size
        if self.config['rnn'] == 'GRU':
            return self.h0.expand(size)
        else:
            return self.h0.expand(size), self.c0.expand(size)


@fret.configurable
class HRNN(FeatureExtractor):
    """Sequence-to-sequence feature extractor based on RNN. Supports different
    input forms and different RNN types (LSTM/GRU), """
    def __init__(self, _stoi,
                 emb_size=(256, 'size of embedding vectors'),
                 rnn=('LSTM', 'size of rnn hidden states', ['LSTM', 'GRU']),
                 layers=(1, 'number of layers'), **kwargs):
        super(HRNN, self).__init__(**kwargs)

        feat_size = self.feat_size

        self.stoi = _stoi
        vocab_size = len(_stoi)

        self.we = nn.Embedding(vocab_size, emb_size)
        self.ie = ImageAE(emb_size)
        self.me = MetaAE(emb_size)

        if rnn == 'GRU':
            self.rnn = nn.GRU(emb_size, feat_size, layers)
            self.h0 = nn.Parameter(torch.rand(layers, 1, feat_size))
        else:
            self.rnn = nn.LSTM(emb_size, feat_size, layers)
            self.h0 = nn.Parameter(torch.rand(layers, 1, feat_size))
            self.c0 = nn.Parameter(torch.rand(layers, 1, feat_size))

        self.woutput = nn.Linear(feat_size, vocab_size)
        self.ioutput = nn.Linear(feat_size, emb_size)
        self.moutput = nn.Linear(feat_size, emb_size)

    def make_batch(self, data, pretrain=False):
        """Returns embeddings"""
        embs = []
        gt = []
        for q in data:
            _embs = [self.we(torch.tensor([0], device=device))]
            _gt = []

            for w in q.content:
                if isinstance(w, int):
                    _embs.append(self.we(torch.tensor([w], device=device)))
                    _gt.append(torch.tensor([w], device=device))
                else:
                    _embs.append(self.ie.enc(to_tensor(w).to(device)))
                    _gt.append(to_tensor(w).to(device))
            _gt.append(torch.tensor([0], device=device))

            embs.append(torch.cat(_embs, dim=0))
            gt.append(_gt)

        embs = SeqBatch(embs)

        length = sum(embs.lens)
        words = []
        ims = []
        metas = []
        wmask = torch.zeros(length, device=device).byte()
        imask = torch.zeros(length, device=device).byte()
        mmask = torch.zeros(length, device=device).byte()
        for i, _gt in enumerate(gt):
            for j, v in enumerate(_gt):
                ind = embs.index((j, i))
                if v.size() == torch.Size([1]):  # word
                    words.append(v)
                    wmask[ind] = 1
                elif len(v.size()) == 1:
                    metas.append(v.unsqueeze(0))
                    mmask[ind] = 1
                else:
                    ims.append(v.unsqueeze(0))
                    imask[ind] = 1
        words = torch.cat(words, dim=0) if words else None
        ims = torch.cat(ims, dim=0) if ims else None
        metas = torch.cat(metas, dim=0) if metas else None

        if pretrain:
            return embs, words, ims, metas, wmask, imask, mmask
        else:
            return SeqBatch(embs)

    def forward(self, batch: SeqBatch):
        packed = batch.packed()
        h = self.init_h(packed.batch_sizes[0])
        y, h = self.rnn(packed, h)
        if self.config['rnn'] == 'GRU':
            return y, batch.invert(h, 1)
        else:
            return y, batch.invert(h[0], 1)

    def pretrain_loss(self, batch):
        input, words, ims, metas, wmask, imask, mmask = batch
        y = self(input)
        y_pred = y[0].data

        wfea = torch.masked_select(y_pred, wmask.unsqueeze(1)) \
            .view(-1, self.feat_size)
        ifea = torch.masked_select(y_pred, imask.unsqueeze(1)) \
            .view(-1, self.feat_size)
        mfea = torch.masked_select(y_pred, imask.unsqueeze(1)) \
            .view(-1, self.feat_size)

        out = self.woutput(wfea)
        wloss = F.cross_entropy(out, words)

        out = self.ioutput(ifea)
        iloss = F.mse_loss(self.ie.dec(out), ims)

        loss = wloss + iloss
        return loss

    def init_h(self, batch_size):
        size = list(self.h0.size())
        size[1] = batch_size
        if self.config['rnn'] == 'GRU':
            return self.h0.expand(size)
        else:
            return self.h0.expand(size), self.c0.expand(size)


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


@fret.configurable
class BERT(torch.nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self,
                 vocab_size=(0, 'size of vocabulary'),
                 dim=(768, 'dimension of hidden layer in transformer encoder'),
                 n_layers=(12, 'numher of hidden Layers'),
                 n_heads=(12, 'numher of heads in multi-headed attn layers'),
                 dim_ff=(768 * 4, 'dimension of intermediate layers in '
                                  'positionwise feedforward net'),
                 p_drop_hidden=(0.1, 'probability of dropout of hid Layers'),
                 p_drop_attn=(0.1, 'probability of dropout of attn Layers'),
                 max_len=(512, 'maximum length for positional embeddings'),
                 n_segments=2):
        super(BERT, self).__init__()
        cfg = self.config
        self.embed = Embeddings(cfg)
        self.blocks = \
            torch.nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."

    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim)
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim)

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (S,) -> (B, S)

        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None  # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D)-proj->(B, S, D)-split->(B, S, H, W)-trans->(B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """

    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


def split_last(x, shape):
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    s = x.size()
    assert 1 < n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class ImageAE:
    def __init__(self, emb_size):
        self.emb_size = emb_size

    def enc(self, im):
        return torch.zeros(im.size(0), self.emb_size)

    def dec(self, fea):
        return torch.zeros(fea.size(0), 1, 56, 56)


class MetaAE:
    def __init__(self, emb_size):
        pass
