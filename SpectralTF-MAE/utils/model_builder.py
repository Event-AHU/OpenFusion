"""Create a registered SpectralTF architecture using shared constructor arguments."""

from .registry import model_entrypoint


def create_model(model_name, **kwargs):
    return model_entrypoint(model_name)(**kwargs)
