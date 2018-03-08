'''
Main program parsing arguments and running commands
'''
from __future__ import print_function
import argparse
import os
import sys
import logging
import inspect as ins
from collections import namedtuple

import models
import commands
import utils

_parser_formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=_parser_formatter)
parser.add_argument('-w', '--workspace',
                    help='workspace dir', default='ws/test')
subparsers = parser.add_subparsers(title='supported commands', dest='command')
subparsers.required = True


class _Command:
    def run(self, args):
        if os.path.exists(os.path.join(args.workspace, 'config.toml')):
            utils.load_config(args.workspace)
        else:
            print('you must run config first!')
            sys.exit(1)

        model = utils.load_config(args.workspace)
        command = self.__class__.__name__.lower()
        args = {name: value for (name, value) in args._get_kwargs()
                if name != 'command' and name != 'func'}
        args = namedtuple('Args', args.keys())(*args.values())
        getattr(commands, command)(model, args)


class Train(_Command):
    def __init__(self, parser):
        parser.add_argument('-N', '--epochs', type=int, default=10,
                            help='number of epochs to train')


class Test(_Command):
    def __init__(self, parser):
        parser.add_argument('-s', '--snapshot',
                            help='model snapshot to test with')


class Config:
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
            Model.add_arguments(group)
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

    def run(self, args):
        pass


if __name__ == '__main__':
    cmds = {m[0].lower(): m[1]
            for m in ins.getmembers(sys.modules[__name__], ins.isclass)
            if not m[0].startswith('_')}
    for command in cmds:
        sub = subparsers.add_parser(command, formatter_class=_parser_formatter)
        subcommand = cmds[command](sub)
        sub.set_defaults(func=subcommand.run)

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

    if args.command != 'config':
        fileHandler = logging.FileHandler(os.path.join(workspace, 'logs',
                                                       args.command + '.log'))
        fileHandler.setFormatter(logFormatter)
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
