'''
Main program parsing arguments and running commands
'''

import argparse
import os
import sys
from model import *

commands = ['config', 'train', 'test']
models = ['simple']


class Config:
    def __init__(self, parser):
        subs = parser.add_subparsers(title='models available', dest='model')
        subs.required = True
        for model in models:
            sub = subs.add_parser(model)
            group = sub.add_argument_group('setup')
            Model = get_class(model)
            Model.add_arguments(group)
            group_options = [action.dest for action in group._group_actions]

            def save(args):
                setup = {name: value for (name, value) in args._get_kwargs()
                         if name in group_options}
                conf = os.path.join(args.workspace,
                                    str(model) + '.conf')
                Model(setup).save_config(conf)

            sub.set_defaults(func=save)

    def run(self, args):
        pass


class Train:
    def __init__(self, parser):
        parser.add_argument('-N', '--epochs',
                            help='number of epochs to train')

    def run(self, args):
        print(args)
        for name in os.listdir(args.workspace):
            if name.endswith('.conf'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = Model.load_config(config)
        print(model.args)


class Test:
    def __init__(self, parser):
        parser.add_argument('-s', '--snapshot',
                            help='model snapshot to test with')

    def run(self, args):
        print(args)
        for name in os.listdir(args.workspace):
            if name.endswith('.conf'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = Model.load_config(config)
        print(model)


parser = argparse.ArgumentParser()
parser.add_argument('-w', '--workspace',
                    help='workspace dir', default='data/test')
subparsers = parser.add_subparsers(title='supported commands', dest='command')
subparsers.required = True


def main():
    for command in commands:
        sub = subparsers.add_parser(command)
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
    args.func(args)


def get_class(name):
    return globals()[name[0].upper() + name[1:]]


if __name__ == '__main__':
    main()
