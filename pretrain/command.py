"""Define commands."""
import logging
import os
import sys
import datetime
from itertools import islice
from tqdm import tqdm

import dill
import fret

import torch
from statistics import mean
from torchtext import data
from tensorboardX import SummaryWriter

from . import dataloader
from .module import Trainer
from .util import critical, device


@fret.command
def prep(_):
    """Generate inputs for each task, filtering out some test data"""
    pass


@fret.command
def pretrain(ws, n_epochs=5, batch_size=16):
    """Pretrain feature extraction model"""
    logger = ws.logger('pretrain')
    trainer: Trainer = ws.build_module()
    logger.info('[%s] model: %s, args: %s', ws, trainer, pretrain.args)
    trainer.pretrain(pretrain.args)


@fret.command
def eval(ws):
    """Use feature extraction model for each evaluation task"""
    trainer: Trainer = ws.build_module()
    trainer.eval(eval.args)


@fret.command
def train(ws, epochs=10, resume=False, batch_size=16,
          log_every=16, save_every=-1):
    logger = ws.logger('train')

    logger.info('Training...')

    model = ws.build_module()
    model.to(device)

    logger.info('[%s] model: %s, args: %s', ws, model, train.args)

    loader = dataloader.DataLoader.load_state_dict(
        torch.load(model.config.vocab.replace('vocab', 'loader'),
                   pickle_module=dill))

    train_iter, valid_iter = \
        data.BucketIterator.splits(datasets=loader.splits[:-1],
                                   batch_size=batch_size,
                                   sort_key=lambda x: len(x.content),
                                   sort_within_batch=True,
                                   device=device)
    epoch_size = len(train_iter)

    debug = logger.level == logging.DEBUG
    optim = torch.optim.Adam(model.parameters())

    state_path = ws.checkpoint_path / 'state.int.pt'
    if resume and state_path.exists():
        cp_path = ws.checkpoint_path / 'model.int.pt'
        model.load_state_dict(torch.load(str(cp_path),
                                         map_location=lambda s, _: s))

        state = torch.load(str(state_path))
        train_iter.load_state_dict(state['train_iter_state'])
        optim.load_state_dict(state['optim_state'])
        current_run = state['current_run']
        loss_avg = state['loss_avg']
        start_epoch = train_iter.epoch
        n_samples = state['n_samples']
        initial = train_iter._iterations_this_epoch
    else:
        if resume:
            logger.warning('nothing to resume, starting from scratch')
        elif state_path.exists():
            print('has previous training state, overwrite? (y/N) ', end='')
            c = input()
            if c.lower() not in ['y', 'yes']:
                logger.warning('cancelled (add -r to resume training)')
                sys.exit(1)

        n_samples = 0  # track total #samples for plotting
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_run = ws.log_path / ('run-%s/' % now)
        loss_avg = []
        start_epoch = 0
        initial = 0

    model.train()
    writer = SummaryWriter(str(current_run))

    for epoch in range(start_epoch, epochs):
        epoch_iter = iter(tqdm(islice(train_iter, epoch_size - initial),
                               total=epoch_size,
                               initial=initial,
                               desc=f'Epoch {epoch+1:3d}: ',
                               unit='bz',
                               smoothing=0.1, disable=debug))
        initial = 0

        try:
            # training
            for batch in critical(epoch_iter):
                # critical section on one batch

                i = train_iter._iterations_this_epoch
                n_samples += len(batch)

                # backprop on one batch
                optim.zero_grad()
                loss = model.pretrain_loss(batch.content)
                loss.backward()
                optim.step()

                # log loss
                loss_avg.append(loss.item())
                if log_every == len(loss_avg):
                    writer.add_scalar('train/loss', mean(loss_avg),
                                      n_samples)
                    loss_avg = []

                # save model
                if save_every > 0 and i % save_every == 0:
                    cp_path = ws / f'model.{epoch}.{i}.pt'
                    torch.save(model.state_dict(), str(cp_path))

            # save after one epoch
            cp_path = ws.checkpoint_path / f'model.{epoch+1}.pt'
            torch.save(model.state_dict(), str(cp_path))

            # validation
            model.eval()
            with torch.no_grad():
                loss = 0.
                for batch in tqdm(valid_iter, desc='    Valid: ',
                                  unit='bz', disable=debug):
                    loss += model.lm_loss(batch.content).item()
                # todo: change to epoch + 1 after this run
                writer.add_scalar('train/eval_loss',
                                  loss / len(valid_iter), epoch)

        except KeyboardInterrupt:
            _save_state(ws, current_run, model, optim,
                        train_iter, n_samples, loss_avg)
            raise


def _save_state(ws, current_run, model, optim, train_iter,
                n_samples, loss_avg):
    snapshot_path = ws.checkpoint_path / 'model.int.pt'
    state_path = ws.checkpoint_path / 'state.int.pt'
    torch.save(model.state_dict(), str(snapshot_path))
    torch.save({
        'optim_state': optim.state_dict(),
        'train_iter_state': train_iter.state_dict(),
        'loss_avg': loss_avg,
        'current_run': current_run,
        'n_samples': n_samples
    }, str(state_path))


@fret.command
def test(ws, snapshots):
    """Test the model. See :class:`~app.run.Test` for ``args``."""
    logger = ws.logger('test')
    logger.info('Testing...')
    model = ws.build_module()
    logger.info('[%s] model: %s', ws, model)


@fret.command
def prep_(_, input_file,
          input_type=('char', 'input type', ['char', 'word', 'both']),
          split_ratio=([0.8, 0.2, 0.2], 'train/valid/test ratio'),
          split_rand_seed=fret.arg(help='random state for splitting',
                                   type=int),
          max_len=(400, 'maximum length'),
          max_size=fret.arg(help='random state for splitting', type=int),
          min_freq=(1, 'minimum frequency of word'),
          output_dir=fret.arg(required=True, help='output directory')):

    if split_ratio is not None and len(split_ratio) != 3:
        logging.error("split_ratio must be of length 3 (train/valid/test)")
        sys.exit(1)

    loader = dataloader.DataLoader(raw_file=input_file,
                                   input_type=input_type,
                                   split_ratio=split_ratio,
                                   split_rand_seed=split_rand_seed,
                                   max_len=max_len)
    print('In [{}]:'.format(output_dir))
    print('  #train: {}\n  #valid: {}\n  #test: {}'
          .format(*(len(d) for d in loader.splits)))
    loader.build_vocab(max_size=max_size, min_freq=min_freq)
    vocab = loader.text_field.vocab
    print('  vocab size: {}'.format(len(vocab.stoi)))

    try:
        os.makedirs(output_dir)
    except OSError:
        pass
    torch.save(loader.state_dict(), os.path.join(output_dir, 'loader.pt'),
               pickle_module=dill)
    torch.save(vocab, os.path.join(output_dir, 'vocab.pt'))


if __name__ == '__main__':
    prep(fret.workspace('ws/test'))
