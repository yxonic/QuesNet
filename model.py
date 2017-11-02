from __future__ import print_function


class Simple:
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-foo', default=10)

    def __init__(self, args):
        self.args = args
