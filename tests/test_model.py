import pytest
import app.models as models


class ModelTest(models.Model):
    @classmethod
    def _add_arguments(cls, parser):
        parser.add_argument('-x', type=int, required=True)
        parser.add_argument('-y', type=int, default=10)


def test_model():
    model1 = ModelTest.parse(['-x', '10'])
    model2 = ModelTest.build(x=10, y=10)
    assert model1.config == model2.config

    with pytest.raises(ValueError) as e:
        ModelTest.parse([])

    assert str(e.value) == 'the following arguments are required: -x'
