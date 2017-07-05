from __future__ import print_function
import json


class Simple:
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--output-size', help='output size')

    def __init__(self, args):
        self.args = args

    def save_config(self, path):
        f = open(path, 'w')
        json.dump(self.args, f)
        f.close()

    @staticmethod
    def load_config(path):
        f = open(path, 'r')
        return Simple(json.load(f))
