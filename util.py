import logging
import json
from collections import namedtuple


def save_config(obj, path):
    f = open(path, 'w')
    json.dump(obj.args._asdict(), f)
    f.close()


def load_config(Model, path):
    setup = json.load(open(path, 'r'))
    setup = namedtuple('Setup', setup.keys())(*setup.values())
    return Model(setup)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored(text, color, bold=False):
    if bold:
        return bcolors.BOLD + color + text + bcolors.ENDC
    else:
        return color + text + bcolors.ENDC


LOG_COLORS = {
    'WARNING': bcolors.WARNING,
    'INFO': bcolors.OKGREEN,
    'DEBUG': bcolors.OKBLUE,
    'CRITICAL': bcolors.WARNING,
    'ERROR': bcolors.FAIL
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, datefmt, use_color=True):
        logging.Formatter.__init__(self, msg, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in LOG_COLORS:
            record.levelname = colored(record.levelname[0],
                                       LOG_COLORS[record.levelname])
        return logging.Formatter.format(self, record)
