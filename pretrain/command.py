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

from .dataloader import DataLoader
from .util import critical


class Train(fret.Command):
    def __init__(self, parser):
        parser.add_argument('-epochs', '-N', type=int, default=10,
                            help='number of epochs to train')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume training')
        parser.add_argument('-batch_size', '-bs', type=int, default=16,
                            help='batch size')
        parser.add_argument('-log_every', type=int, default=16,
                            help='write stats every # samples')
        parser.add_argument('-save_every', type=int, default=-1,
                            help='save model every # batches')

    def run(self, ws, args):
        return train(ws, args)


def train(ws, args):
    logger = ws.logger('train')

    logger.info('Training...')

    model = ws.build_module()

    logger.info('[%s] model: %s, args: %s', ws, model, args)

    loader = DataLoader.load_state_dict(
        torch.load(model.config.vocab.replace('vocab', 'loader'),
                   pickle_module=dill))

    train_iter, valid_iter = \
        data.BucketIterator.splits(datasets=loader.splits[:-1],
                                   batch_size=args.batch_size,
                                   sort_key=lambda x: len(x.content),
                                   sort_within_batch=True)
    epoch_size = len(train_iter)

    debug = logger.level == logging.DEBUG
    optim = torch.optim.Adam(model.parameters())

    state_path = ws.checkpoint_path / 'state.int.pt'
    if args.resume and state_path.exists():
        cp_path = ws.checkpoint_path / 'model.int.pt'
        model.load_state_dict(torch.load(str(cp_path)))

        state = torch.load(str(state_path))
        train_iter.load_state_dict(state['train_iter_state'])
        optim.load_state_dict(state['optim_state'])
        current_run = state['current_run']
        loss_avg = state['loss_avg']
        start_epoch = train_iter.epoch
        n_samples = state['n_samples']
        initial = train_iter._iterations_this_epoch
    else:
        if args.resume:
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

    writer = SummaryWriter(str(current_run))

    for epoch in range(start_epoch, args.epochs):
        epoch_iter = iter(tqdm(islice(train_iter, epoch_size - initial),
                               total=epoch_size,
                               initial=initial,
                               desc=f'Epoch {epoch+1:3d}: ',
                               unit='bz', disable=debug))
        initial = 0

        try:
            # training
            model.train()
            for batch in critical(epoch_iter):
                # critical section on one batch

                i = train_iter._iterations_this_epoch
                n_samples += len(batch)

                # backprop on one batch
                optim.zero_grad()
                loss = model.loss(batch.content)
                loss.backward()
                optim.step()

                # log loss
                loss_avg.append(loss.item())
                if args.log_every == len(loss_avg):
                    writer.add_scalar('train/loss', mean(loss_avg),
                                      n_samples)
                    loss_avg = []

                # save model
                if args.save_every > 0 and i % args.save_every == 0:
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


class Prep(fret.Command):
    def __init__(self, parser):
        parser.add_argument('-input', '-i', help='raw input file',
                            default='data/questions.head.tsv')
        parser.add_argument('-input_type', '-t', default='char',
                            choices=['char', 'word', 'both'],
                            help='input type')
        parser.add_argument('-split_ratio', '-s', default=[0.8, 0.2, 0.2],
                            nargs='+', type=float,
                            help='ratio of train/valid/test dataset')
        parser.add_argument('-split_rand_seed', type=int,
                            help='random state for splitting')
        parser.add_argument('-max_len', type=int, default=400,
                            help='maximum length')
        parser.add_argument('-max_size', type=int,
                            help='maximum vocab size')
        parser.add_argument('-min_freq', type=int, default=1,
                            help='minimum frequency of word')
        parser.add_argument('-output_dir', '-o', required=True,
                            help='output directory')

    def run(self, ws, args):
        return prep(ws, args)


def prep(ws, args):
    if args.split_ratio is not None and len(args.split_ratio) != 3:
        logging.error("split_ratio must be of length 3 (train/valid/test)")
        sys.exit(1)

    loader = DataLoader(raw_file=args.input,
                        input_type=args.input_type,
                        split_ratio=args.split_ratio,
                        split_rand_seed=args.split_rand_seed,
                        max_len=args.max_len)
    print('In [{}]:'.format(args.output_dir))
    print('  #train: {}\n  #valid: {}\n  #test: {}'
          .format(*(len(d) for d in loader.splits)))
    loader.build_vocab(max_size=args.max_size, min_freq=args.min_freq)
    vocab = loader.text_field.vocab
    print('  vocab size: {}'.format(len(vocab.stoi)))

    try:
        os.makedirs(args.output_dir)
    except OSError:
        pass
    torch.save(loader.state_dict(), os.path.join(args.output_dir, 'loader.pt'),
               pickle_module=dill)
    torch.save(vocab, os.path.join(args.output_dir, 'vocab.pt'))
