import json


def save_config(obj, path):
    f = open(path, 'w')
    json.dump(obj.args, f)
    f.close()


def load_config(Model, path):
    f = open(path, 'r')
    return Model(json.load(f))
