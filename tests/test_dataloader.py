from pretrain.util import PrefetchIter


def test_dataiter():
    _data = list(range(1000))
    it = PrefetchIter(_data, batch_size=4)
    for i, _ in enumerate(it):
        if i > 12:
            break

    state = it.state_dict()

    it2 = PrefetchIter(_data, batch_size=16)
    it2.load_state_dict(state)

    assert it.batch_size == it2.batch_size
    assert next(it) == next(it2)

    it = PrefetchIter(_data, batch_size=1)

    for i, d in enumerate(it):
        assert d[0] == it.index[i]
