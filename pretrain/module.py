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


class Predictor(nn.Module):
    def __init__(self, feat_size, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(True),
            nn.Linear(feat_size, out_dim))

    def forward(self, features):
        return self.model(features[1])


class SP(nn.Module):
    def __init__(self, feat_model, wcnt, emb_size=50, seq_h_size=50,
                 n_layers=1, attn_k=10):
        super().__init__()
        self.wcnt = wcnt
        self.emb_size = emb_size
        self.ques_h_size = feat_model.feat_size
        self.seq_h_size = seq_h_size
        self.n_layers = n_layers
        self.attn_k = attn_k

        self.question_net = feat_model

        self.seq_net = EERNNSeqNet(self.ques_h_size, seq_h_size,
                                   n_layers,attn_k)

    def forward(self, question, score, hidden=None):
        ques_h0 = None
        batch = self.question_net.make_batch([question])
        _, ques_h = self.question_net(batch)

        s, h = self.seq_net(ques_h[0], score, hidden)

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
    def __init__(self, ques_size=100, seq_hidden_size=50,
                 n_layers=1, attn_k=10):
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
        pred = self.score_net(pred_v)

        if score is None:
            score = pred.flatten()

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
                elif len(v.size()) == 1:  # meta
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
            return y, batch.invert(h, 1)
        else:
            return y, batch.invert(h[0], 1)

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


class ELMo:
    pass


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

    def get_optimizer(self, **kwargs):
        return BertAdam(self.parameters(), **kwargs)

    def make_batch(self, data, pretrain=False):
        qs = [[x if isinstance(x, int) else self.stoi['word'].get('{img}') or 0
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
                # TODO: change mask strategy to BERT
                if random.random() < 0.8:  # 80%
                    _embs[i] *= 0.
                    _m.append(1)
                    target.append(w)
                elif random.random() < 0.5:  # 10%
                    w = random.choice(range(0, self.vocab_size))
                    _embs[i] = self.embed(
                        torch.tensor([w]).long().to(device),
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
        return h, self.activ1(self.fc(h[:, 0]))


@fret.configurable
class QuesNet(BERT):
    def __init__(self, **kwargs):
        super(QuesNet, self).__init__(**kwargs)
        cfg = self.config

        self.stoi['word']['<sep>'] = self.vocab_size
        self.vocab_size += 1

        self.ie = ImageAE(cfg.dim_ff)
        self.me = MetaAE(len(self.stoi['grade']), cfg.dim_ff)
        self.woutput = nn.Linear(cfg.dim, self.vocab_size)
        self.ioutput = nn.Linear(cfg.dim, cfg.dim_ff)
        self.moutput = nn.Linear(cfg.dim, cfg.dim_ff)

    def make_batch(self, data, pretrain=False):
        full_batch = []
        full_wm, full_wt, full_im, full_it, full_mm, full_mt = \
            ([] for _ in range(6))
        for q in data:
            for item in q:
                pass
        full_batch, seq_mask = SeqBatch(full_batch).padded(self.max_len, True)

        if not pretrain:
            return full_batch

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
