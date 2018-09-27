"""
Main program parsing arguments and running commands.
"""
from __future__ import print_function
import argparse
import os
import sys
import shutil
import logging
import inspect as ins
from abc import ABC, abstractmethod
from collections import namedtuple

from . import command
from . import module
from . import util

_parser_formatter = argparse.ArgumentDefaultsHelpFormatter
_parser = argparse.ArgumentParser(formatter_class=_parser_formatter)
_parser.add_argument('-w', '--workspace',
                     help='workspace dir', default='ws/test')
_subparsers = _parser.add_subparsers(title='supported commands',
                                     dest='command')
_subparsers.required = True


class _Command(ABC):
    """Command interface."""
    @abstractmethod
    def _run(self, args):
        pass


class _WorkspaceCommand(_Command):
    """Base class for commands that requires to run in a workspace."""

    def _run(self, args):
        if os.path.exists(os.path.join(args.workspace, 'config.toml')):
            util.load_config(args.workspace)
        else:
            print('you must run config first!')
            sys.exit(1)

        model = util.load_config(args.workspace)
        args = {name: value for (name, value) in args._get_kwargs()
                if name != 'command' and name != 'func'}
        args = namedtuple('Args', args.keys())(*args.values())
        self.run(model, args)

    def run(self, model, args):
        # to be overridden
        pass


class Train(_WorkspaceCommand):
    """Command ``train``. See :func:`~app.commands.train`."""

    def __init__(self, parser):
        r"""
        Args:
            -N,--epochs (int): number of epochs to train. Default: 10
        """
        parser.add_argument('-N', '--epochs', type=int, default=10,
                            help='number of epochs to train')

    def run(self, model, args):
        command.train(model, args)


class Test(_WorkspaceCommand):
    """Command ``test``. See :func:`~app.commands.test`."""

    def __init__(self, parser):
        r"""
        Args:
            -s,--snapshot (str): model snapshot to test with
        """
        parser.add_argument('-s', '--snapshot',
                            help='model snapshot to test with')

    def run(self, model, args):
        command.test(model, args)


class Config(_Command):
    """Command ``config``,

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
        model_names = [m[0] for m in ins.getmembers(module, ins.isclass)
                       if not m[0].startswith('_')]
        for model in model_names:
            sub = subs.add_parser(model, formatter_class=_parser_formatter)
            group = sub.add_argument_group('config')
            Model = getattr(module, model)
            Model._add_arguments(group)
            for action in group._group_actions:
                group_options.add(action.dest)

            def save(args):
                _model = args.model
                _Model = getattr(module, _model)
                config = {name: value for (name, value) in args._get_kwargs()
                          if name in group_options}
                m = util.make_model(_Model, **config)
                print('In [%s]: configured %s with %s' %
                      (args.workspace, _model, str(m.config)))
                util.save_config(m, args.workspace)

            sub.set_defaults(func=save)

    def _run(self, args):
        pass


class Clean(_Command):
    """Command ``clean``.

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
        _sub = _subparsers.add_parser(command,
                                      formatter_class=_parser_formatter)
        subcommand = cmds[command](_sub)
        _sub.set_defaults(func=subcommand._run)

    _args = _parser.parse_args()
    workspace = _args.workspace
    try:
        os.makedirs(os.path.join(workspace, 'snapshots'))
        os.makedirs(os.path.join(workspace, 'results'))
        os.makedirs(os.path.join(workspace, 'logs'))
    except OSError:
        pass

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logFormatter = util.ColoredFormatter(
        '%(levelname)s %(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fileFormatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

    if issubclass(cmds[_args.command], _WorkspaceCommand):
        fileHandler = logging.FileHandler(
            os.path.join(workspace, 'logs', _args.command + '.log'))
        fileHandler.setFormatter(fileFormatter)
        logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    try:
        _args.func(_args)
    except KeyboardInterrupt:
        logging.warning('cancelled by user')
    except Exception as e:
        import traceback
        sys.stderr.write(traceback.format_exc())
        logging.error('exception occurred: %s', e)
