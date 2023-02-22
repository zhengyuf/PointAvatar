import os

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def get_split_name(splits):
    # example splits: [MVI_1812, MVI_1813]
    # example output: MVI_1812+MVI_1813
    name = ''
    for s in splits:
        name += str(s)
        name += '+'
    assert len(name) > 1
    return name[:-1]
