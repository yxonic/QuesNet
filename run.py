'''
Main program parsing arguments and running commands
'''

import argparse
import os
import sys
import logging
from model import *
from util import *
from commands import *

commands = ['config', 'train', 'test']
models = ['simple']


class Config:
    def __init__(self, parser):
        subs = parser.add_subparsers(title='models available', dest='model')
        subs.required = True
        group_options = set()
        for model in models:
            sub = subs.add_parser(model, formatter_class=parser_formatter)
            group = sub.add_argument_group('setup')
            Model = get_class(model)
            Model.add_arguments(group)
            for action in group._group_actions:
                group_options.add(action.dest)

            def save(args):
                for file in os.listdir(args.workspace):
                    if file.endswith('.json'):
                        os.remove(os.path.join(args.workspace, file))
                model = args.model
                Model = get_class(model)
                setup = {name: value for (name, value) in args._get_kwargs()
                         if name in group_options}
                setup = namedtuple('Setup', setup.keys())(*setup.values())
                conf = os.path.join(args.workspace,
                                    str(model) + '.json')
                m = Model(setup)
                print('model: %s, setup: %s' % (model, str(m.args)))
                save_config(m, conf)

            sub.set_defaults(func=save)

    def run(self, args):
        pass


class Train:
    def __init__(self, parser):
        parser.add_argument('-N', '--epochs', type=int, default=10,
                            help='number of epochs to train')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        train(model, args)


class Test:
    def __init__(self, parser):
        parser.add_argument('-s', '--snapshot',
                            help='model snapshot to test with')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        test(model, args)


parser_formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=parser_formatter)
parser.add_argument('-w', '--workspace',
                    help='workspace dir', default='ws/test')
subparsers = parser.add_subparsers(title='supported commands', dest='command')
subparsers.required = True


def main():
    for command in commands:
        sub = subparsers.add_parser(command, formatter_class=parser_formatter)
        subcommand = get_class(command)(sub)
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

    logFormatter = ColoredFormatter('%(levelname)s %(asctime)s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')

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


def get_class(name):
    return globals()[name[0].upper() + name[1:]]


if __name__ == '__main__':
    main()
