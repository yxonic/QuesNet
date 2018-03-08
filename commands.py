import logging


def train(model, args):
    logging.info('Training...')
    logging.info(model.config)
    logging.info(args)


def test(model, args):
    logging.info('Testing...')
    logging.info(model.config)
    logging.info(args)
