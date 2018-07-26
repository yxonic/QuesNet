'''
Model classes that defines model parameters and architecture.
'''


class Simple:
    r"""A toy class to demonstrate how to add model arguments.

    Each model class should have an ``_add_argument`` static method.
    """

    @staticmethod
    def _add_arguments(parser):
        parser.add_argument('-foo', default=10, help='dumb param')

    def __init__(self, config):
        r"""
        Args:
            config (namedtuple): model configuration
        """
        self.config = config
