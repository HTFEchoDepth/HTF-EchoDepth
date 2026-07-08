"""LTFE public placeholders."""


class _ReleaseSoon:
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError("LTFE implementation will be released soon.")


FACCoreLiteV1 = _ReleaseSoon
DTFSpatialConv2d = _ReleaseSoon
ResFADTFTDFBlockV3V1 = _ReleaseSoon

__all__ = ["FACCoreLiteV1", "DTFSpatialConv2d", "ResFADTFTDFBlockV3V1"]
