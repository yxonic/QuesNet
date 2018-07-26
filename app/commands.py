'''
Define commands.
'''
import logging


def train(model, args):
    """Train the model. See :class:`~app.run.Train` for ``args``."""
    logging.info('Training...')
    logging.info(model.config)
    logging.info(args)


def test(model, args):
    """Test the model. See :class:`~app.run.Test` for ``args``."""
    logging.info('Testing...')
    logging.info(model.config)
    logging.info(args)
