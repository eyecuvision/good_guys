class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

    def __setitem__(self, name, value):
        return setattr(self.module, self.prefix + str(name), value)

    def __call__(self, i):
        return self.prefix + str(i)
