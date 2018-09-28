"""
Main program parsing arguments and running commands.
"""
from __future__ import print_function
import abc
import argparse
import os
import sys
import shutil
import logging
import inspect as ins
from collections import namedtuple, defaultdict

from . import command
from . import util
from . import models as mm

logging.getLogger().setLevel(logging.INFO)


def _sub_class_checker(cls):
    def rv(obj):
        if ins.isclass(obj) and not ins.isabstract(obj) \
                and issubclass(obj, cls):
            return True
        else:
            return False
    return rv


models = [m[0] for m in ins.getmembers(mm, _sub_class_checker(mm.Model))
          if not m[0].startswith('_')]

_parser_formatter = argparse.ArgumentDefaultsHelpFormatter
main_parser = util._ArgumentParser(formatter_class=_parser_formatter,
                                   prog='python -m app.run')
main_parser.add_argument('-w', '--workspace',
                         help='workspace dir', default='ws/test')
_subparsers = main_parser.add_subparsers(title='supported commands',
                                         dest='command')
_subparsers.required = True


class Command(abc.ABC):
    """Command interface."""
    @abc.abstractmethod
    def run(self, args):
        pass


class WorkspaceCommand(Command):
    """Base class for commands that requires to run in a workspace."""

    def run(self, args):
        if os.path.exists(os.path.join(args.workspace, 'config.toml')):
            util.load_config(args.workspace)
        else:
            print('you must run config first!', file=sys.stderr)
            sys.exit(1)

        model = util.load_config(args.workspace)
        args = {name: value for (name, value) in args._get_kwargs()
                if name != 'command' and name != 'func'}
        args = namedtuple('Args', args.keys())(*args.values())
        return self.run_with(model, args)

    @abc.abstractmethod
    def run_with(self, model, args):
        pass


class Train(WorkspaceCommand):
    """Command ``train``. See :func:`~app.command.train`."""

    def __init__(self, parser):
        r"""
        Args:
            -N,--epochs (int): number of epochs to train. Default: 10
        """
        parser.add_argument('-N', '--epochs', type=int, default=10,
                            help='number of epochs to train')

    def run_with(self, model, args):
        return command.train(model, args)


class Test(WorkspaceCommand):
    """Command ``test``. See :func:`~app.command.test`."""

    def __init__(self, parser):
        r"""
        Args:
            -s,--snapshot (str): model snapshot to test with
        """
        parser.add_argument('-s', '--snapshot',
                            help='model snapshot to test with')

    def run_with(self, model, args):
        return command.test(model, args)


class Config(Command):
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
        group_options = defaultdict(set)

        for model in models:
            sub = subs.add_parser(model, formatter_class=_parser_formatter)
            group = sub.add_argument_group('config')
            Model = getattr(mm, model)
            Model._add_arguments(group)
            for action in group._group_actions:
                group_options[model].add(action.dest)

            def save(args):
                _model = args.model
                _Model = getattr(mm, _model)
                config = {name: value for (name, value) in args._get_kwargs()
                          if name in group_options[_model]}
                m = _Model.build(**config)
                print('In [%s]: configured %s with %s' %
                      (args.workspace, _model, str(m.config)),
                      file=sys.stderr)
                util.save_config(m, args.workspace)

            sub.set_defaults(func=save)

    def run(self, args):
        pass


class Clean(Command):
    """Command ``clean``.

    Remove all snapshots in specific workspace. If ``--all`` is specified,
    clean the entire workspace
    """

    def __init__(self, parser):
        parser.add_argument('--all', action='store_true',
                            help='clean the entire workspace')

    def run(self, args):
        if args.all:
            shutil.rmtree(args.workspace)
        else:
            shutil.rmtree(os.path.join(args.workspace, 'snapshots'))
            os.makedirs(os.path.join(args.workspace, 'snapshots'))


def main(args):
    workspace = args.workspace
    try:
        os.makedirs(os.path.join(workspace, 'snapshots'))
        os.makedirs(os.path.join(workspace, 'results'))
        os.makedirs(os.path.join(workspace, 'logs'))
    except OSError:
        pass

    logger = logging.getLogger()
    logFormatter = util.ColoredFormatter(
        '%(levelname)s %(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fileFormatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

    if issubclass(commands[args.command], WorkspaceCommand):
        fileHandler = logging.FileHandler(
            os.path.join(workspace, 'logs', args.command + '.log'))
        fileHandler.setFormatter(fileFormatter)
        logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    rv = None
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning('cancelled by user')
    except Exception as e:  # pragma: no cover
        import traceback
        sys.stderr.write(traceback.format_exc())
        logger.error('exception occurred: %s', e)


commands = {m[0].lower(): m[1]
            for m in ins.getmembers(sys.modules[__name__],
                                    _sub_class_checker(Command))}
for _cmd in commands:
    _sub = _subparsers.add_parser(_cmd,
                                  formatter_class=_parser_formatter)
    _sub.set_defaults(func=commands[_cmd](_sub).run)


if __name__ == '__main__':
    main(main_parser.parse_args())
