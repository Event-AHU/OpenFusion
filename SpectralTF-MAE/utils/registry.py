"""Registry of supported SpectralTF model builders."""

_MODEL_BUILDERS = {}


def register_model(builder):
    name = builder.__name__
    if name in _MODEL_BUILDERS:
        raise ValueError(f"A SpectralTF model named {name!r} is already registered")
    _MODEL_BUILDERS[name] = builder
    return builder


def model_entrypoint(name):
    try:
        return _MODEL_BUILDERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown SpectralTF model {name!r}; valid options: {list_models()}") from exc


def list_models():
    return sorted(_MODEL_BUILDERS)
