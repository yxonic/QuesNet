"""Define commands."""
import datetime
import json
import random
import subprocess
from functools import partial

import numpy as np
import sklearn.metrics as metrics
import fret
import torch
from scipy import stats
from tensorboardX import SummaryWriter
from tqdm import tqdm

from . import device
from .dataloader import QuestionLoader
from .module import FeatureExtractor, Predictor, SP
from .util import stateful, critical, PrefetchIter, lines


@fret.command
def pretrain(ws, n_epochs=5, batch_size=8, save_every=5000, lr=0.1,
             restart=False):
    """Pretrain feature extraction model"""
    logger = ws.logger('pretrain')
    trainer = Trainer(ws)
    trainer.pretrain(pretrain.args)


@fret.command
def eval(ws, diff=False, know=False, sp=False,
         checkpoint=None, tag='test', early_stop=False, fix=False,
         split_ratio=0.8, n_epochs=5, batch_size=16, test_batch_size=32):
    """Use feature extraction model for each evaluation task"""
    trainer = Trainer(ws)
    if diff:
        trainer.eval_diff(eval.args)
    if know:
        trainer.eval_know(eval.args)
    if sp:
        trainer.eval_sp(eval.args)


@fret.command
def tb(_):
    subprocess.run('tensorboard --logdir ws', shell=True, check=True)


@stateful('epoch', 'run_id', 'n_batches')
class Trainer:
    def __init__(self, ws):
        self.ws = ws
        self.make_model = ws.build_module
        # training states
        self.epoch = 0
        self.run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.n_batches = 0
        self._cur_iter = None
        self._iter_state = None
        self._cur_optim = None
        self._optim_state = None

    def pretrain(self, args):
        logger = self.ws.logger('pretrain')

        ques = QuestionLoader('data/train/ques.txt', 'data/words.txt',
                              'data/imgs', 'data/train/id_grade.txt')
        self.model = self.make_model(_stoi=ques.stoi).to(device)

        ques.pipeline = partial(self.model.make_batch, pretrain=True)

        logger.info('[%s] model: %s, args: %s', self.ws, self.model, args)

        if not args.restart:
            self.load_state()

        self.model.train()

        optim = self.optimizer(self.model, lr=args.lr)
        writer = SummaryWriter(str(self.ws.log_path /
                                   ('pretrain_%s/' % self.run_id)))

        for self.epoch in range(self.epoch, args.n_epochs):
            train_iter = self.pretrain_iter(ques, args.batch_size)

            try:
                bar = enumerate(tqdm(train_iter, initial=train_iter.pos),
                                train_iter.pos)
                for i, batch in critical(bar):
                    self.n_batches += 1
                    loss = self.model.pretrain_loss(batch)
                    if isinstance(loss, dict):
                        total_loss = 0.
                        for k, v in loss.items():
                            if v is not None:
                                writer.add_scalar(
                                    'pretrain/%s/sub/' %
                                    self.model.__class__.__name__ + k,
                                    v.item(), self.n_batches)
                                total_loss += v
                        loss = total_loss

                    writer.add_scalar('pretrain/%s/loss' %
                                      self.model.__class__.__name__,
                                      loss.item(), self.n_batches)

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    if args.save_every > 0 and i % args.save_every == 0:
                        self.save('%d.%d' % (self.epoch, i))

                self.save('%d' % (self.epoch + 1))

            except KeyboardInterrupt:
                self.save_state()
                raise

        self.save_state()

    def pretrain_iter(self, ques, batch_size):
        self._cur_iter = PrefetchIter(ques, batch_size=batch_size)
        if self._iter_state is not None:
            self._cur_iter.load_state_dict(self._iter_state)
            self._iter_state = None
        return self._cur_iter

    def optimizer(self, *models, **kwargs):
        self._cur_optim = [m.optim_cls(m.parameters(), **kwargs)
                           if hasattr(m, 'optim_cls')
                           else torch.optim.Adam(m.parameters(), **kwargs)
                           for m in models]
        if self._optim_state is not None:
            for o, s in zip(self._cur_optim, self._optim_state):
                o.load_state_dict(s)
            self._optim_state = None
        if len(self._cur_optim) == 1:
            return self._cur_optim[0]
        else:
            return self._cur_optim

    def eval_diff(self, args):
        logger = self.ws.logger('eval')

        diff_ques = QuestionLoader('data/test/diff_ques.txt', 'data/words.txt',
                                   'data/imgs', 'data/test/id_grade.txt',
                                   'data/test/id_difficulty.txt')
        self.model: FeatureExtractor = self.make_model(_stoi=diff_ques.stoi)
        self.model.to(device)
        if args.checkpoint is not None:
            self.load(args.checkpoint)
        model = Predictor(self.model.feat_size, 1).to(device)

        def make_label(qs):
            labels = [[q.labels['difficulty']] for q in qs]
            return torch.tensor(labels).to(device)

        def make_result(y_pred, y_true):
            y_pred = y_pred.view(1, -1).numpy()
            y_true = y_true.view(1, -1).numpy()

            return {
                'diff/mae-': float(np.abs(y_pred - y_true).mean()),
                'diff/rmse-': float(np.sqrt(((y_pred - y_true) ** 2).mean())),
                'diff/pearsonr+': float(stats.pearsonr(
                    y_pred[0], y_true[0])[0])
            }

        results = list(self._eval(self.model, model, diff_ques,
                                  torch.nn.MSELoss(),
                                  make_label, make_result, args))
        if not results:
            return
        self.write_result('%s_%s_%s' % (args.tag, str(args.checkpoint),
                                        self.run_id), results)

    def eval_know(self, args):
        know_ques = QuestionLoader('data/test/know_ques.txt', 'data/words.txt',
                                   'data/imgs', 'data/test/id_grade.txt',
                                   'data/test/id_know.txt')
        know_size = len(know_ques.stoi['know'])

        self.model: FeatureExtractor = self.make_model(_stoi=know_ques.stoi)
        self.model.to(device)
        if args.checkpoint is not None:
            self.load(args.checkpoint)

        model = Predictor(self.model.feat_size, know_size).to(device)

        def make_label(qs):
            labels = [q.labels['know'] for q in qs]
            rv = torch.zeros(len(labels), know_size).to(device)
            for i in range(len(labels)):
                rv[i, labels[i]] = 1
            return rv

        def make_result(y_true, y_pred):
            n_samples = y_true.size(0)
            y_true = y_true.view(-1).numpy()

            l = 0.
            lo = 0.382
            hi = 0.618
            r = 1.
            y_pred_ = (torch.sigmoid(y_pred) > lo).view(-1).numpy()
            _, _, fl, _ = metrics.precision_recall_fscore_support(
                y_true, y_pred_, average='binary')
            y_pred_ = (torch.sigmoid(y_pred) > hi).view(-1).numpy()
            _, _, fh, _ = metrics.precision_recall_fscore_support(
                y_true, y_pred_, average='binary')

            while hi - lo > 0.01:
                if fl < fh:  # right side
                    l = lo
                    lo = hi
                    fl = fh
                    hi = l + (r - l) * 0.618
                    y_pred_ = (torch.sigmoid(y_pred) > hi).view(-1).numpy()
                    _, _, fh, _ = metrics.precision_recall_fscore_support(
                        y_true, y_pred_, average='binary')
                else:  # left side
                    r = hi
                    hi = lo
                    fh = fl
                    lo = l + (r - l) * 0.382
                    y_pred_ = (torch.sigmoid(y_pred) > lo).view(-1).numpy()
                    _, _, fl, _ = metrics.precision_recall_fscore_support(
                        y_true, y_pred_, average='binary')

            max_thresh = (lo + hi) / 2

            y_pred = (torch.sigmoid(y_pred) > max_thresh).view(-1).numpy()
            acc = [np.all(t == p)
                   for t, p in zip(y_true.reshape((n_samples, -1)),
                                   y_pred.reshape((n_samples, -1)))]
            acc = np.mean(acc)
            macc = metrics.accuracy_score(y_true, y_pred)
            p, r, f, _ = metrics.precision_recall_fscore_support(
                y_true, y_pred, average='binary')
            return {
                'know/acc+': float(acc),
                'know/micro_acc+': float(macc),
                'know/precision+': float(p),
                'know/recall+': float(r),
                'know/f1+': float(f)
            }

        results = list(self._eval(self.model, model, know_ques,
                                  torch.nn.BCEWithLogitsLoss(),
                                  make_label, make_result, args))
        if not results:
            return
        self.write_result('%s_%s_%s' % (args.tag, str(args.checkpoint),
                                        self.run_id), results)

    def eval_sp(self, args):
        diff_ques = QuestionLoader('data/test/diff_ques.txt', 'data/words.txt',
                                   'data/imgs', 'data/test/id_grade.txt',
                                   'data/test/id_difficulty.txt')
        id_ind = {}
        for i in range(len(diff_ques)):
            id_ind[diff_ques[i].id] = i

        self.model: FeatureExtractor = self.make_model(_stoi=diff_ques.stoi)
        self.model.to(device)
        if args.checkpoint is not None:
            self.load(args.checkpoint)

        sp_model = SP(self.model.feat_size, len(diff_ques.stoi['word']))
        sp_model.to(device)
        q_optim = self.optimizer(self.model)
        optim = self.optimizer(sp_model)
        loss_f = torch.nn.MSELoss()

        results = []
        try:
            for epoch in range(args.n_epochs):
                for line in tqdm(lines('data/test/records.txt')):
                    loss = 0.
                    rec = line.strip().split(' ')
                    random.shuffle(rec)
                    for r in rec:
                        qid, s = r.split(',')
                        s = torch.tensor([float(s)]).to(device)
                        q = diff_ques[id_ind[qid]]
                        qh = self.model(self.model.make_batch([q]))[1]
                        if args.fix:
                            qh = qh.detach()
                        s_pred, _ = sp_model(qh, torch.tensor([0]))
                        loss += loss_f(s_pred, s)

                    if not args.fix:
                        q_optim.zero_grad()
                    optim.zero_grad()
                    loss.backward()
                    if not args.fix:
                        q_optim.step()
                    optim.step()

                with torch.no_grad():
                    true = []
                    pred = []
                    for line in tqdm(lines('data/test/test_records.txt')):
                        for i, r in enumerate(line.strip().split(' ')):
                            qid, s = r.split(',')
                            s = torch.tensor([float(s)]).to(device)
                            q = diff_ques[id_ind[qid]]
                            qh = self.model(self.model.make_batch([q]))[1]
                            s_pred, _ = sp_model(qh, torch.tensor([0]))
                            if i > 20:
                                true.append(s[0].item())
                                pred.append(s_pred[0].item())
                    y_pred = np.asarray(pred)
                    y_true = np.asarray(true)
                    p, r, f, _ = metrics.precision_recall_fscore_support(
                        y_true > 0.5, y_pred > 0.5, average='binary')
                    result = {
                        'diff/mae-': float(np.abs(y_pred - y_true).mean()),
                        'diff/rmse-': float(
                            np.sqrt(((y_pred - y_true) ** 2).mean())),
                        'know/f1+': float(f)
                    }
                    print(result)
                    results.append(result)
        except KeyboardInterrupt:
            pass

        self.write_result('%s_%s_%s' % (args.tag, str(args.checkpoint),
                                        self.run_id), results)

    def _eval(self, ques_model, model, ques, loss_f,
              make_label, make_result, args):
        q_optim = self.optimizer(ques_model)
        optim = self.optimizer(model)

        train_ques = ques
        test_ques = train_ques.split_(args.split_ratio)

        best = None

        try:
            for epoch in range(args.n_epochs):
                train_iter = PrefetchIter(train_ques,
                                          batch_size=args.batch_size)
                for qs in tqdm(train_iter):
                    batch = ques_model.make_batch(qs)
                    labels = make_label(qs)
                    h = ques_model(batch)[1]
                    if args.fix:
                        h = h.detach()
                    loss = loss_f(model(h), labels)
                    if not args.fix:
                        q_optim.zero_grad()
                    optim.zero_grad()
                    loss.backward()
                    if not args.fix:
                        q_optim.step()
                    optim.step()

                test_iter = PrefetchIter(test_ques, shuffle=False,
                                         batch_size=args.test_batch_size)
                total_loss = 0.
                y_true = []
                y_pred = []
                with torch.no_grad():
                    for qs in tqdm(test_iter):
                        batch = ques_model.make_batch(qs)
                        labels = make_label(qs)
                        pred = model(ques_model(batch)[1])
                        loss = loss_f(pred, labels)
                        total_loss += loss.item()
                        y_true.append(labels)
                        y_pred.append(pred)
                result = make_result(torch.cat(y_true, 0),
                                     torch.cat(y_pred, 0))
                print(result)
                yield result

                if not args.early_stop:
                    continue

                if best is None:
                    best = result
                    continue

                # early stopping
                b = False
                for k, v in result.items():
                    if (best[k] < v) ^ k.endswith('-'):
                        best[k] = v
                        b = True
                if not b and epoch > 5:
                    break
        except KeyboardInterrupt:
            return

    def write_result(self, tag, result):
        ws: fret.Workspace = self.ws
        json.dump(result,
                  (ws.result_path /
                   (tag + '_' + self.run_id + '.json')).open('w'),
                  indent=4)

    def state_dict(self):
        return {
            'train_iter_state': self._cur_iter.state_dict(),
            'optim_state': [o.state_dict() for o in self._cur_optim],
            'model_state': self.model.state_dict()
        }

    def load_state_dict(self, state):
        self._iter_state = state['train_iter_state']
        self._optim_state = state['optim_state']
        self.model.load_state_dict(state['model_state'])

    def save_state(self):
        state_path = self.ws.checkpoint_path / 'trainer.state.pt'
        torch.save(self.state_dict(), state_path.open('wb'))

    def load_state(self):
        state_path = self.ws.checkpoint_path / 'trainer.state.pt'
        if state_path.exists():
            state = torch.load(state_path.open('rb'))
            self.load_state_dict(state)

    def save(self, tag):
        path = self.ws.checkpoint_path
        cp_path = path / ('%s_%s.pt' % (self.model.__class__.__name__,
                                        str(tag)))
        torch.save(self.model.state_dict(), cp_path.open('wb'))

    def load(self, tag):
        path = self.ws.checkpoint_path
        cp_path = path / ('%s_%s.pt' % (self.model.__class__.__name__,
                                        str(tag)))
        self.model.load_state_dict(
            torch.load(cp_path.open('rb'), map_location=lambda s, _: s)
        )


def _better(r1, r2):
    if r1 is None:
        return False
    if r2 is None:
        return True
    rv = []
    for k in r1:
        if k.endswith('+'):
            rv.append(r1[k] > r2[k])
        else:
            rv.append(r1[k] < r2[k])
    return np.mean(rv) >= 0.5


class _result_key:
    def __init__(self, r):
        self.r = r

    def __cmp__(self, other):
        if _better(self.r, other.r):
            return 1
        if _better(other.r, self.r):
            return -1
        return 0
