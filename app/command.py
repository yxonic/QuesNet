"""Define commands."""

import logging


def train(model, args):  # pragma: no cover
    """Train the model. See :class:`~app.run.Train` for ``args``."""
    logger = logging.getLogger('train')
    logger.info('Training...')
    logger.info(model.config)
    logger.info(args)


def test(model, args):  # pragma: no cover
    """Test the model. See :class:`~app.run.Test` for ``args``."""
    logger = logging.getLogger('test')
    logger.info('Testing...')
    logger.info(model.config)
    logger.info(args)
