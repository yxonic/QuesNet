import copy
import math
import random

import fret

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torchvision.transforms.functional import to_tensor

from .dataloader import load_word2vec
from .optim import BertAdam
from .util import SeqBatch
from . import device


@fret.configurable
class FeatureExtractor(nn.Module):
    def __init__(self, feat_size=512):
        super(FeatureExtractor, self).__init__()
        self.feat_size = feat_size

    def make_batch(self, data, pretrain=False):
        """Make batch from input data (python data / np arrays -> tensors)"""
        return torch.tensor(data)

    def load_emb(self, emb):
        pass

    def pretrain_loss(self, batch):
        """Returns pretraining loss on a batch of data"""
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError


class SP(nn.Module):
    def __init__(self, feat_size, wcnt, emb_size=50, seq_h_size=50,
                 n_layers=1, attn_k=10):
        super().__init__()
        self.wcnt = wcnt
        self.emb_size = emb_size
        self.ques_h_size = feat_size
        self.seq_h_size = seq_h_size
        self.n_layers = n_layers
        self.attn_k = attn_k

        self.seq_net = EERNNSeqNet(self.ques_h_size, seq_h_size,
                                   n_layers, attn_k)

    def forward(self, ques_h, score, hidden=None):
        s, h = self.seq_net(ques_h, score, hidden)

        if hidden is None:
            hidden = ques_h, h
        else:
            # concat all qs and hs for attention
            qs, hs = hidden
            qs = torch.cat([qs, ques_h])
            hs = torch.cat([hs, h])
            hidden = qs, hs

        return s, hidden


class EERNNSeqNet(nn.Module):
    def __init__(self, ques_size, seq_hidden_size, n_layers, attn_k):
        super(EERNNSeqNet, self).__init__()

        self.initial_h = nn.Parameter(torch.zeros(n_layers *
                                                  seq_hidden_size))
        self.ques_size = ques_size  # exercise size
        self.seq_hidden_size = seq_hidden_size
        self.n_layers = n_layers
        self.attn_k = attn_k

        # initialize network
        self.seq_net = nn.GRU(ques_size * 2, seq_hidden_size, n_layers)
        self.score_net = nn.Linear(ques_size + seq_hidden_size, 1)

    def forward(self, question, score, hidden):
        if hidden is None:
            h = self.initial_h.view(self.n_layers, 1, self.seq_hidden_size)
            attn_h = self.initial_h
        else:
            questions, hs = hidden
            h = hs[-1:]
            alpha = torch.mm(questions, question.view(-1, 1)).view(-1)
            alpha, idx = alpha.topk(min(len(alpha), self.attn_k), sorted=False)
            alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

            # flatten each h
            hs = hs.view(-1, self.n_layers * self.seq_hidden_size)
            attn_h = torch.mm(alpha, torch.index_select(hs, 0, idx)).view(-1)

        # prediction
        pred_v = torch.cat([question, attn_h]).view(1, -1)
        pred = self.score_net(pred_v)[0]

        if score is None:
            score = pred

        # update seq_net
        x = torch.cat([question * (score >= 0.5).float(),
                       question * (score < 0.5).float()])

        _, h_ = self.seq_net(x.view(1, 1, -1), h)
        return pred, h_


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
        vocab_size = len(_stoi['word'])

        self.embedding = nn.Embedding(vocab_size, emb_size)
        embs = load_word2vec(emb_size)
        if embs is not None:
            self.load_emb(embs)

        if rnn == 'GRU':
            self.rnn = nn.GRU(emb_size, rnn_size, layers)
            self.h0 = nn.Parameter(torch.rand(layers, 1, rnn_size))
        else:
            self.rnn = nn.LSTM(emb_size, rnn_size, layers)
            self.h0 = nn.Parameter(torch.rand(layers, 1, rnn_size))
            self.c0 = nn.Parameter(torch.rand(layers, 1, rnn_size))
        self.output = nn.Linear(rnn_size, vocab_size)

    def load_emb(self, emb):
        self.embedding.weight.data.copy_(torch.from_numpy(emb))

    def make_batch(self, data, pretrain=False):
        qs = [[x if isinstance(x, int) else self.stoi['word'].get('{img}') or 0
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
            return y, batch.invert(h, 1).squeeze(0)
        else:
            return y, batch.invert(h[0], 1).squeeze(0)

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
                 i_lambda=5., m_lambda=5.,
                 layers=(1, 'number of layers'), **kwargs):
        super(HRNN, self).__init__(**kwargs)

        feat_size = self.feat_size

        self.stoi = _stoi
        vocab_size = len(_stoi['word'])
        self.itos = {v: k for k, v in self.stoi['word'].items()}

        self.we = nn.Embedding(vocab_size, emb_size)
        embs = load_word2vec(emb_size)
        if embs is not None:
            self.load_emb(embs)

        self.ie = ImageAE(emb_size)
        self.me = MetaAE(len(_stoi['grade']), emb_size)

        self.i_lambda = i_lambda
        self.m_lambda = m_lambda

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

    def load_emb(self, emb):
        self.we.weight.data.copy_(torch.from_numpy(emb))

    def make_batch(self, data, pretrain=False):
        """Returns embeddings"""
        embs = []
        gt = []
        for q in data:
            meta = torch.zeros(len(self.stoi['grade'])).to(device)
            meta[q.labels.get('grade') or []] = 1
            _embs = [self.we(torch.tensor([0], device=device)),
                     self.me.enc(meta.unsqueeze(0))]
            _gt = [meta]
            for w in q.content:
                if isinstance(w, int):
                    word = torch.tensor([w], device=device)
                    _embs.append(self.we(word))
                    _gt.append(word)
                else:
                    im = to_tensor(w).to(device)
                    _embs.append(self.ie.enc(im.unsqueeze(0)))
                    _gt.append(im)
            _gt.append(torch.tensor([0], device=device))

            embs.append(torch.cat(_embs, dim=0))
            gt.append(_gt)

        embs = SeqBatch(embs)

        length = sum(embs.lens)
        words = []
        ims = []
        metas = []
        p = embs.packed().data
        wmask = torch.zeros(length, device=device).byte()
        imask = torch.zeros(length, device=device).byte()
        mmask = torch.zeros(length, device=device).byte()

        for i, _gt in enumerate(gt):
            for j, v in enumerate(_gt):
                ind = embs.index((j, i))
                if v.size() == torch.Size([1]):  # word
                    words.append((v, ind))
                    wmask[ind] = 1
                elif v.dim() == 1:  # meta
                    metas.append((v.unsqueeze(0), ind))
                    mmask[ind] = 1
                else:  # img
                    ims.append((v.unsqueeze(0), ind))
                    imask[ind] = 1
        words = [x[0] for x in sorted(words, key=lambda x: x[1])]
        ims = [x[0] for x in sorted(ims, key=lambda x: x[1])]
        metas = [x[0] for x in sorted(metas, key=lambda x: x[1])]

        words = torch.cat(words, dim=0) if words else None
        ims = torch.cat(ims, dim=0) if ims else None
        metas = torch.cat(metas, dim=0) if metas else None

        if pretrain:
            return embs, words, ims, metas, wmask, imask, mmask
        else:
            return embs

    def forward(self, batch: SeqBatch):
        packed = batch.packed()
        h = self.init_h(packed.batch_sizes[0])
        y, h = self.rnn(packed, h)
        if self.config['rnn'] == 'GRU':
            return y, batch.invert(h, 1).squeeze(0)
        else:
            return y, batch.invert(h[0], 1).squeeze(0)

    def pretrain_loss(self, batch):
        input, words, ims, metas, wmask, imask, mmask = batch
        y = self(input)
        y_pred = y[0].data

        wloss = iloss = mloss = None

        if words is not None:
            wfea = torch.masked_select(y_pred, wmask.unsqueeze(1)) \
                .view(-1, self.feat_size)
            out = self.woutput(wfea)
            wloss = F.cross_entropy(out, words)

        if ims is not None:
            ifea = torch.masked_select(y_pred, imask.unsqueeze(1)) \
                .view(-1, self.feat_size)
            out = self.ioutput(ifea)
            iloss = (self.ie.loss(ims, out) +
                     self.ie.loss(ims)) * self.i_lambda

        if metas is not None:
            mfea = torch.masked_select(y_pred, mmask.unsqueeze(1)) \
                .view(-1, self.feat_size)

            out = self.moutput(mfea)
            mloss = (self.me.loss(metas, out) +
                     self.me.loss(metas)) * self.m_lambda

        return {
            'word_loss': wloss,
            'image_vae_loss': iloss,
            'meta_vae_loss': mloss
        }

    def init_h(self, batch_size):
        size = list(self.h0.size())
        size[1] = batch_size
        if self.config['rnn'] == 'GRU':
            return self.h0.expand(size)
        else:
            return self.h0.expand(size), self.c0.expand(size)


@fret.configurable
class BERT(FeatureExtractor):
    """ Transformer with Self-Attentive Blocks"""

    # noinspection PyUnusedLocal
    def __init__(self, _stoi,
                 n_layers=(12, 'number of hidden Layers'),
                 n_heads=(12, 'number of heads in multi-headed attn layers'),
                 dim_ff=(768 * 4, 'dimension of intermediate layers in '
                                  'position-wise feed-forward net'),
                 p_drop_hidden=(0.1, 'probability of dropout of hid Layers'),
                 p_drop_attn=(0.1, 'probability of dropout of attn Layers'),
                 max_len=(512, 'maximum length for positional embeddings'),
                 **args):
        super(BERT, self).__init__(**args)
        cfg = self.config
        self.vocab_size = cfg['vocab_size'] = len(_stoi['word'])
        cfg['dim'] = self.feat_size

        self.stoi = _stoi
        self.max_len = max_len

        self.embed = Embeddings(cfg)
        self.blocks = \
            torch.nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # modules for pretrain
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.dim, cfg.dim)
        self.activ2 = gelu
        self.norm = LayerNorm(cfg)
        # decoder is shared with embedding layer
        embed_weight = self.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

        self.optim_cls = BertAdam

    def make_batch(self, data, pretrain=False):
        qs = [[0] + [x if isinstance(x, int)
                     else self.stoi['word'].get('{img}') or 0
                     for x in q.content] for q in data]

        if not pretrain:
            return SeqBatch([self.embed(torch.tensor(q).long().to(device))
                             for q in qs], device=device).padded(
                max_len=self.max_len, batch_first=True)

        masks = []
        target = []
        embs = []
        for q in qs:
            _m = []
            _embs = self.embed(torch.tensor(q).long().to(device))
            for i, w in enumerate(q):
                if random.random() < 0.8:  # 80%
                    _embs[i] *= 0.
                    _m.append(1)
                    target.append(w)
                elif random.random() < 0.5:  # 10%
                    _w = random.choice(range(0, self.vocab_size))
                    _embs[i] = self.embed(
                        torch.tensor([_w]).long().to(device),
                        torch.tensor([i]).long().to(device))
                    _m.append(1)
                    target.append(w)
                else:
                    _m.append(0)
            masks.append(_m)
            embs.append(_embs)

        input = SeqBatch(embs, device=device).padded(self.max_len, True)[0]

        masks = SeqBatch(masks, device=device)
        mask = masks.padded(self.max_len, batch_first=True)[0].byte()
        target = torch.tensor(target).long().to(device)
        batch = input, mask

        return batch, mask, target

    def load_emb(self, emb):
        self.embed.tok_embed.weight.data.copy_(torch.from_numpy(emb))

    def pretrain_loss(self, batch):
        batch, mask, target = batch
        h, _ = self(batch)

        h_masked = torch.masked_select(h, mask[:, :, None]).view(-1, h.size(2))
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        loss = F.cross_entropy(logits_lm, target)
        return loss

    def forward(self, batch):
        h, mask = batch
        for block in self.blocks:
            h = block(h, mask)
        hs = self.activ1(self.fc(h) + h)
        return hs, hs.max(1)[0]


@fret.configurable
class HBERT(BERT):
    def __init__(self, **kwargs):
        super(HBERT, self).__init__(**kwargs)
        cfg = self.config

        self.stoi['word']['<sep>'] = self.vocab_size
        self.vocab_size += 1

        self.ie = ImageAE(cfg.dim)
        self.me = MetaAE(len(self.stoi['grade']), cfg.dim)
        self.woutput = nn.Linear(cfg.dim, self.vocab_size)
        self.ioutput = nn.Linear(cfg.dim, cfg.dim)
        self.moutput = nn.Linear(cfg.dim, cfg.dim)

    def make_batch(self, data, pretrain=False):
        full_batch = []
        full_wm, full_wt, full_im, full_it, full_mm, full_mt = \
            ([] for _ in range(6))
        for q in data:
            meta = torch.zeros(len(self.stoi['grade'])).to(device)
            meta[q.labels.get('grade') or []] = 1
            _embs = [self.me.enc(meta.unsqueeze(0))]
            for item in q.content:
                if isinstance(item, int):  # word
                    word = torch.tensor([item], device=device)
                    _embs.append(self.embed(word))
                else:
                    im = to_tensor(item).to(device)
                    _embs.append(self.ie.enc(im.unsqueeze(0)))
            full_batch.append(torch.cat(_embs, dim=0))
        full_batch, seq_mask = SeqBatch(full_batch).padded(self.max_len, True)

        if not pretrain:
            return full_batch, seq_mask

        full_wm, full_wt, full_im, full_it, full_mm, full_mt = \
            (torch.cat(v, dim=0) for v in
             [full_wm, full_wt, full_im, full_it, full_mm, full_mt])

        w_batch = []
        w_mask = []
        w_target = []
        for q in data:
            for item in q:
                pass
        w_batch, _ = SeqBatch(w_batch).padded(self.max_len, True)
        w_mask, _ = SeqBatch(w_mask).padded(self.max_len, True)
        w_target = torch.cat(w_target, dim=0)

        i_batch = []
        i_mask = []
        i_target = []
        for q in data:
            for item in q:
                pass
        i_batch, _ = SeqBatch(i_batch).padded(self.max_len, True)
        i_mask, _ = SeqBatch(i_mask).padded(self.max_len, True)
        i_target = torch.cat(i_target, dim=0)

        m_batch = []
        m_mask = []
        m_target = []
        for q in data:
            for item in q:
                pass
        m_batch, _ = SeqBatch(m_batch).padded(self.max_len, True)
        m_mask, _ = SeqBatch(m_mask).padded(self.max_len, True)
        m_target = torch.cat(m_target, dim=0)

        pos_batch = []
        for q in data:
            for item in q:
                pass
        pos_batch, pos_mask = SeqBatch(pos_batch).padded(self.max_len, True)

        neg_batch = []
        for q in data:
            for item in q:
                pass
        neg_batch, neg_mask = SeqBatch(neg_batch).padded(self.max_len, True)

        return (
            (full_batch, full_wm, full_wt, full_im, full_it, full_mm, full_mt),
            (w_batch, w_mask, w_target),
            (i_batch, i_mask, i_target),
            (m_batch, m_mask, m_target),
            (pos_batch, pos_mask),
            (neg_batch, neg_mask),
        )

    def pretrain_loss(self, batch):
        pass


def gelu(x):
    """Implementation of the gelu activation function by Hugging Face"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    """A layernorm module in the TF style (epsilon inside the square root)."""

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
    """The embedding module from word, position and token_type embeddings."""

    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim)

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, pos=None):
        if pos is None:
            seq_len = x.size(-1)
            pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
            if len(x.size()) > 1:
                pos = pos.unsqueeze(0).expand_as(x)  # (S,) -> (B, S)

        e = self.tok_embed(x) + self.pos_embed(pos)
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


class VAE(nn.Module):
    def enc(self, item):
        return self.encoder(item)

    def dec(self, emb):
        es = emb.size(1)
        mu, logvar = emb[:, :es // 2], emb[:, es // 2:]
        std = logvar.mul(0.5).exp_()
        eps = torch.empty(std.size()).float().normal_().to(device)
        z = eps.mul(std).add_(mu)
        return self.decoder(z)

    def loss(self, item, emb=None):
        if emb is None:
            out, mu, logvar = self(item)
        else:
            out = self.dec(emb)
            es = emb.size(1)
            mu, logvar = emb[:, :es // 2], emb[:, es // 2:]

        bce = self.recons_loss(out, item)

        kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(
            logvar)
        kld = kld_element.sum().mul_(-0.5)

        return bce + kld

    def forward(self, images):
        emb = self.enc(images)
        out = self.dec(emb)
        es = emb.size(1)
        mu, logvar = emb[:, :es // 2], emb[:, es // 2:]
        return out, mu, logvar


class ImageAE(VAE):
    def __init__(self, emb_size):
        super().__init__()
        self.recons_loss = nn.MSELoss()
        self._encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, emb_size, 3, stride=2)
        )
        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(emb_size // 2, 32, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encoder(self, item):
        return self._encoder(item).view(item.size(0), -1)

    def decoder(self, emb):
        return self._decoder(emb[:, :, None, None])


class MetaAE(VAE):
    def __init__(self, meta_size, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.recons_loss = nn.BCEWithLogitsLoss()
        self.encoder = nn.Sequential(nn.Linear(meta_size, emb_size),
                                     nn.ReLU(True),
                                     nn.Linear(emb_size, emb_size))
        self.decoder = nn.Sequential(nn.Linear(emb_size // 2, emb_size),
                                     nn.ReLU(True),
                                     nn.Linear(emb_size, meta_size))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(-1),
                                        dtype=torch.long,
                                        device=input_ids.device)
        if input_ids.dim() > 1:
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


@fret.configurable
class BertAlt(FeatureExtractor):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").
    Params:
        config: a BertConfig class instance with the configuration to build a new model
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, _stoi,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02, **cfg):
        super(BertAlt, self).__init__(**cfg)
        config = self.config
        self.stoi = _stoi
        config['hidden_size'] = self.feat_size
        self.vocab_size = config['vocab_size'] = len(_stoi['word'])
        self.max_len = max_position_embeddings
        self.emb = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        self.cls = BertOnlyMLMHead(config, self.emb.word_embeddings.weight)

    def init_bert_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, batch, output_all_encoded_layers=False):
        embedding_output, attention_mask = batch

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=False)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

    def make_batch(self, data, pretrain=False):
        qs = [[0] + [x if isinstance(x, int)
                     else self.stoi['word'].get('{''img}') or 0
                     for x in q.content] for q in data]
        qs = [q[:self.max_len] for q in qs]

        if not pretrain:
            return SeqBatch([self.emb(torch.tensor(q).long().to(device))
                             for q in qs], device=device).padded(
                max_len=self.max_len, batch_first=True)

        masks = []
        target = []
        embs = []
        for q in qs:
            _m = []
            _embs = self.emb(torch.tensor(q).long().to(device))
            for i, w in enumerate(q):
                if random.random() < 0.8:  # 80%
                    _embs[i] *= 0.
                    _m.append(1)
                    target.append(w)
                elif random.random() < 0.5:  # 10%
                    w = random.choice(range(0, self.vocab_size))
                    _embs[i] = self.emb(
                        torch.tensor([w]).long().to(device))
                    _m.append(1)
                    target.append(w)
                else:
                    _m.append(0)
            masks.append(_m)
            embs.append(_embs)

        input = SeqBatch(embs, device=device).padded(self.max_len, True)[0]

        masks = SeqBatch(masks, device=device)
        mask = masks.padded(self.max_len, batch_first=True)[0].byte()
        target = torch.tensor(target).long().to(device)
        batch = input, mask

        return batch, mask, target

    def load_emb(self, emb):
        self.embed.word_embeddings.weight.data.copy_(torch.from_numpy(emb))

    def pretrain_loss(self, batch):
        batch, mask, target = batch
        h, _ = self(batch)

        h_masked = torch.masked_select(h, mask[:, :, None]).view(-1, h.size(2))
        logits_lm = self.cls(h_masked)

        loss = F.cross_entropy(logits_lm, target)
        return loss
