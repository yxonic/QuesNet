import app.util
import logging
from _pytest.capture import CaptureFixture


def test_colored():
    assert app.util._colored('hello', app.util.LOG_COLORS['ERROR']) == \
        '\x1b[91mhello\x1b[0m'
    assert app.util._colored('hello', app.util.LOG_COLORS['ERROR'], True) == \
        '\x1b[1m\x1b[91mhello\x1b[0m'


def test_logging(capsys: CaptureFixture):
    # dummy formatter for color demonstration
    logFormatter = app.util.ColoredFormatter('%(levelname)s', '')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)

    logger = logging.getLogger('dummy')
    logger.addHandler(consoleHandler)

    logger.warning("msg")

    # level name printed with color
    assert capsys.readouterr().err.startswith('\x1b[93mW')
