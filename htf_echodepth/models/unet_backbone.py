"""Backbone public placeholders for HTF-EchoDepth."""


class _ReleaseSoon:
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError("Backbone implementation will be released soon.")


Downsample = _ReleaseSoon
Upsample = _ReleaseSoon

__all__ = ["Downsample", "Upsample"]
