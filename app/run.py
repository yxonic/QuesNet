'''
Main program parsing arguments and running commands.
'''
from __future__ import print_function
import argparse
import os
import sys
import shutil
import logging
import inspect as ins
from collections import namedtuple

from . import models
from . import commands
from . import utils

_parser_formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=_parser_formatter)
parser.add_argument('-w', '--workspace',
                    help='workspace dir', default='ws/test')
subparsers = parser.add_subparsers(title='supported commands', dest='command')
subparsers.required = True


class _WorkspaceCommand:
    """Base class for commands that requires to run in a workspace."""

    def _run(self, args):
        if os.path.exists(os.path.join(args.workspace, 'config.toml')):
            utils.load_config(args.workspace)
        else:
            print('you must run config first!')
            sys.exit(1)

        model = utils.load_config(args.workspace)
        args = {name: value for (name, value) in args._get_kwargs()
                if name != 'command' and name != 'func'}
        args = namedtuple('Args', args.keys())(*args.values())
        self.run(model, args)


class Train(_WorkspaceCommand):
    r"""Command ``train``. See :func:`~app.commands.train`."""

    def __init__(self, parser):
        r"""
        Args:
            -N,--epochs (int): number of epochs to train. Default: 10
        """
        parser.add_argument('-N', '--epochs', type=int, default=10,
                            help='number of epochs to train')

    def run(self, model, args):
        commands.train(model, args)


class Test(_WorkspaceCommand):
    r"""Command ``test``. See :func:`~app.commands.test`."""

    def __init__(self, parser):
        r"""
        Args:
            -s,--snapshot (str): model snapshot to test with
        """
        parser.add_argument('-s', '--snapshot',
                            help='model snapshot to test with')

    def run(self, model, args):
        commands.test(model, args)


class Config:
    r"""Command ``config``,

    Configure a model and its parameters for a workspace.
    
    Example:
        .. code-block:: bash
            
            $ python app.run -w ws/test config Simple -foo=5
            In [ws/test]: configured Simple with Config(foo='5')
    """

    def __init__(self, parser):
        subs = parser.add_subparsers(title='models available', dest='model')
        subs.required = True
        group_options = set()
        model_names = [m[0] for m in ins.getmembers(models, ins.isclass)
                       if not m[0].startswith('_')]
        for model in model_names:
            sub = subs.add_parser(model, formatter_class=_parser_formatter)
            group = sub.add_argument_group('config')
            Model = getattr(models, model)
            Model._add_arguments(group)
            for action in group._group_actions:
                group_options.add(action.dest)

            def save(args):
                model = args.model
                Model = getattr(models, model)
                config = {name: value for (name, value) in args._get_kwargs()
                          if name in group_options}
                m = utils.make_model(Model, **config)
                print('In [%s]: configured %s with %s' %
                      (args.workspace, model, str(m.config)))
                utils.save_config(m, args.workspace)

            sub.set_defaults(func=save)

    def _run(self, args):
        pass


class Clean:
    r"""Command ``clean``.

    Remove all snapshots in specific workspace. If ``--all`` is specified,
    clean the entire workspace
    """

    def __init__(self, parser):
        parser.add_argument('--all', action='store_true',
                            help='clean the entire workspace')

    def _run(self, args):
        if args.all:
            shutil.rmtree(args.workspace)
        else:
            for file in os.scandir(os.path.join(args.workspace, 'snapshots')):
                os.remove(file.path)


if __name__ == '__main__':
    cmds = {m[0].lower(): m[1]
            for m in ins.getmembers(sys.modules[__name__], ins.isclass)
            if not m[0].startswith('_')}
    for command in cmds:
        sub = subparsers.add_parser(command, formatter_class=_parser_formatter)
        subcommand = cmds[command](sub)
        sub.set_defaults(func=subcommand._run)

    args = parser.parse_args()
    workspace = args.workspace
    try:
        os.makedirs(os.path.join(workspace, 'snapshots'))
        os.makedirs(os.path.join(workspace, 'results'))
        os.makedirs(os.path.join(workspace, 'logs'))
    except OSError:
        pass

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logFormatter = utils.ColoredFormatter(
        '%(levelname)s %(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logFormatter = utils.ColoredFormatter(
        '%(levelname)s %(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fileFormatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

    if issubclass(cmds[args.command], _WorkspaceCommand):
        fileHandler = logging.FileHandler(os.path.join(workspace, 'logs',
                                                       args.command + '.log'))
        fileHandler.setFormatter(fileFormatter)
        logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    try:
        args.func(args)
    except KeyboardInterrupt:
        logging.warn('cancelled by user')
    except Exception as e:
        import traceback
        sys.stderr.write(traceback.format_exc())
        logging.error('exception occurred: %s', e)
