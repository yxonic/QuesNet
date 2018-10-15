"""Define commands."""

import logging
import os
import sys
import time
from itertools import islice
from tqdm import tqdm

import torch
from statistics import mean
from torchtext import data
from tensorboardX import SummaryWriter

from .dataloader import DataLoader
from .util import DelayedKeyboardInterrupt


def train(model, args):
    """Train the model. See :class:`~app.run.Train` for ``args``."""
    logger = logging.getLogger(__name__)
    logger.setLevel(args.logging_level)
    logger.info('Training...')
    logger.info(model.config)
    logger.info(args)

    loader = DataLoader.load_state_dict(
        torch.load(model.config.vocab.replace('vocab', 'loader')))

    train_iter, valid_iter = \
        data.BucketIterator.splits(datasets=loader.splits[:-1],
                                   batch_size=args.batch_size,
                                   sort_key=lambda x: len(x.content),
                                   sort_within_batch=True)
    epoch_size = len(train_iter)

    debug = logger.level == logging.DEBUG
    optim = torch.optim.Adam(model.parameters())

    state_path = os.path.join(args.workspace, 'snapshots/state.int.pt')
    if args.resume and os.path.exists(state_path):
        snapshot_path = os.path.join(args.workspace, 'snapshots/model.int.pt')
        model.load_state_dict(torch.load(snapshot_path))

        state = torch.load(state_path)
        train_iter.load_state_dict(state['train_iter_state'])
        optim.load_state_dict(state['optim_state'])
        current_run = state['current_run']
        loss_avg = state['loss_avg']
        start_epoch = train_iter.epoch
        n_samples = len(loader.splits[0]) * start_epoch + \
            train_iter._iterations_this_epoch * train_iter.batch_size
        initial = train_iter._iterations_this_epoch
    else:
        if args.resume:
            logger.warning('nothing to resume, starting from scratch')
        elif os.path.exists(state_path):
            print('has previous training state, overwrite? (y/N) ', end='')
            c = input()
            if c.lower() not in ['y', 'yes']:
                logger.warning('cancelled (add -r to resume training)')
                sys.exit(1)

        n_samples = 0  # track total #samples for plotting
        current_run = os.path.join(args.workspace,
                                   'logs/run-%d' % time.time())
        loss_avg = []
        start_epoch = 0
        initial = 0

    writer = SummaryWriter(current_run)

    for epoch in range(start_epoch, args.epochs):
        epoch_iter = iter(tqdm(islice(train_iter, epoch_size - initial),
                               total=epoch_size,
                               initial=initial,
                               desc=f'Epoch {epoch+1:3d}: ',
                               ncols=80, unit='bz', disable=debug))
        initial = 0

        try:
            # training
            model.train()
            while True:
                with DelayedKeyboardInterrupt():
                    # critical section on one batch
                    try:
                        batch = next(epoch_iter)
                    except StopIteration:
                        break
                    i = train_iter._iterations_this_epoch
                    n_samples += len(batch)

                    # backprop on one batch
                    optim.zero_grad()
                    loss = model.lm_loss(batch.content)
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
                        snapshot_path = os.path.join(
                            args.workspace,
                            f'snapshots/model.{epoch}.{i}.pt'
                        )
                        torch.save(model.state_dict(), snapshot_path)

            # save after one epoch
            snapshot_path = os.path.join(args.workspace,
                                         f'snapshots/model.{epoch+1}.pt')
            torch.save(model.state_dict(), snapshot_path)

            # validation
            model.eval()
            with torch.no_grad():
                loss = 0.
                for batch in tqdm(valid_iter, ncols=80, desc='    Valid: ',
                                  unit='bz', disable=debug):
                    loss += model.lm_loss(batch.content).item()
                writer.add_scalar('train/eval_loss',
                                  loss / len(valid_iter), epoch)

        except KeyboardInterrupt:
            _save_state(args.workspace, current_run, model, optim,
                        train_iter, loss_avg)
            raise


def _save_state(workspace, current_run, model, optim, train_iter, loss_avg):
    snapshot_path = os.path.join(workspace, 'snapshots/model.int.pt')
    state_path = os.path.join(workspace, 'snapshots/state.int.pt')
    torch.save(model.state_dict(), snapshot_path)
    torch.save({
        'optim_state': optim.state_dict(),
        'train_iter_state': train_iter.state_dict(),
        'loss_avg': loss_avg,
        'current_run': current_run
    }, state_path)


def test(model, args):
    """Test the model. See :class:`~app.run.Test` for ``args``."""
    logging.info('Testing...')
    logging.info(model.config)
    logging.info(args)


def prep(args):
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
    torch.save(loader.state_dict(), os.path.join(args.output_dir, 'loader.pt'))
    torch.save(vocab, os.path.join(args.output_dir, 'vocab.pt'))
