"""Define commands."""

import logging
import os
import sys
import torch as T
from .dataloader import DataLoader


def train(model, args):  # pragma: no cover
    """Train the model. See :class:`~app.run.Train` for ``args``."""
    logging.info('Training...')
    logging.info(model.config)
    logging.info(args)


def test(model, args):  # pragma: no cover
    """Test the model. See :class:`~app.run.Test` for ``args``."""
    logging.info('Testing...')
    logging.info(model.config)
    logging.info(args)


def prep(args):
    if args.split_ratio is not None and len(args.split_ratio) != 3:
        logging.error("split_ratio must be of length 3 (train/valid/test)")
        sys.exit(1)

    data = DataLoader(raw_file=args.input,
                      input_type=args.input_type,
                      split_ratio=args.split_ratio,
                      split_rand_seed=args.split_rand_seed,
                      max_len=args.max_len)
    print('In [{}]:'.format(args.output_dir))
    print('  #train: {}\n  #valid: {}\n  #test: {}'
          .format(*(len(d) for d in data.splits)))
    data.build_vocab(max_size=args.max_size, min_freq=args.min_freq)
    vocab = data.text_field.vocab
    print('  vocab size: {}'.format(len(vocab.stoi)))

    try:
        os.makedirs(args.output_dir)
    except OSError:
        pass
    T.save(data.state_dict(), os.path.join(args.output_dir, 'loader.pt'))
    T.save(vocab, os.path.join(args.output_dir, 'vocab.pt'))
