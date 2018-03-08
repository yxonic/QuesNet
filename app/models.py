class Simple:
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-foo', default=10, help='dumb param')

    def __init__(self, config):
        self.config = config
