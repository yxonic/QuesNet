"""Define commands."""
import datetime
import subprocess
from functools import partial

import fret
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from . import device
from .dataloader import QuestionLoader
from .util import stateful, critical, PrefetchIter


@fret.command
def pretrain(ws, n_epochs=5, batch_size=8, save_every=5000, lr=0.1,
             restart=False):
    """Pretrain feature extraction model"""
    logger = ws.logger('pretrain')
    trainer = Trainer(ws)
    trainer.pretrain(pretrain.args)


@fret.command
def eval(ws):
    """Use feature extraction model for each evaluation task"""
    trainer = Trainer(ws)
    trainer.eval(eval.args)


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
                              'data/imgs', 'data/train/id_area.txt')
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

                self.save('%d' % self.epoch)

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
        self._cur_optim = [m.get_optimizer(**kwargs)
                           if hasattr(m, 'get_optimizer')
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

    def eval(self, args):
        pass

    def eval_sp(self, args):
        pass

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
