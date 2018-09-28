import py
import pytest
import app.models
import app.command
from app.models import Model
from _pytest.capture import CaptureFixture


class ModelTest(Model):
    @classmethod
    def _add_arguments(cls, parser):
        parser.add_argument('-x', default=10, type=int)


def dummy(model, args):
    return model, args


# insert model and command for testing
app.models.ModelTest = ModelTest
app.command.train = dummy
app.command.test = dummy


def test_main(tmpdir: py.path.local, capsys: CaptureFixture):
    """Test main workflow."""

    # temporary workspace
    ws = tmpdir.join('ws')
    ws_path = str(ws)

    import app.run

    args = app.run.main_parser.parse_args(
        ['-w', ws_path, 'config', 'ModelTest']
    )
    app.run.main(args)
    assert ws.join('config.toml').check(file=1)
    assert capsys.readouterr().err.strip().endswith(
        'configured ModelTest with Config(x=10)')

    args = app.run.main_parser.parse_args(
        ['-w', ws_path, 'config', 'ModelTest', '-x=-3']
    )
    app.run.main(args)
    assert ws.join('config.toml').check(file=1)
    assert capsys.readouterr().err.strip().endswith(
        'configured ModelTest with Config(x=-3)')

    args = app.run.main_parser.parse_args(
        ['-w', ws_path, 'train', '-N', '3']
    )
    model, args = app.run.main(args)
    assert model.config.x == -3
    assert args.epochs == 3

    args = app.run.main_parser.parse_args(
        ['-w', ws_path, 'test', '-s', '15']
    )
    model, args = app.run.main(args)
    assert model.config.x == -3
    assert args.snapshot == '15'

    args = app.run.main_parser.parse_args(
        ['-w', ws_path, 'clean']
    )
    app.run.main(args)
    assert ws.join('config.toml').check(file=1)

    args = app.run.main_parser.parse_args(
        ['-w', ws_path, 'clean', '--all']
    )
    app.run.main(args)
    assert not ws.exists()

    with pytest.raises(SystemExit) as e:
        args = app.run.main_parser.parse_args(
            ['-w', ws_path, 'train']
        )
        app.run.main(args)
    assert e.value.code == 1
    assert capsys.readouterr().err.strip() == 'you must run config first!'

    with pytest.raises(SystemExit) as e:
        args = app.run.main_parser.parse_args(
            ['-w', ws_path, 'foo']
        )
        app.run.main(args)
    assert e.value.code == 2
    assert capsys.readouterr().err.strip().startswith('usage:')
