from omegaconf import OmegaConf


class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, config=None):
        return cls()

    def build(self, **kwargs):
        config = OmegaConf.create(kwargs)

        return self.from_config(config)