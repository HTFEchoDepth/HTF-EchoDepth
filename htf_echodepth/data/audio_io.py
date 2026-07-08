"""Binaural waveform I/O for BatVision V2 echo files."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

LAST_LOAD_BACKEND: str = "unknown"

_TORCHCODEC_MARKERS = (
    "torchcodec",
    "libtorchcodec",
    "libavutil",
    "could not load libtorchcodec",
)


def _is_torchcodec_load_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _TORCHCODEC_MARKERS)


def load_waveform(
    audio_path: str | Path,
    *,
    channels_first: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load binaural WAV as float32 tensor (channels, samples).

    Uses torchaudio when available; falls back to soundfile if TorchCodec/FFmpeg fails.
    """
    global LAST_LOAD_BACKEND
    path = str(audio_path)
    try:
        import torchaudio

        waveform, sr = torchaudio.load(path)
        LAST_LOAD_BACKEND = "torchaudio"
        if not channels_first:
            waveform = waveform.transpose(0, 1)
        return waveform.float(), int(sr)
    except Exception as exc:
        if not _is_torchcodec_load_error(exc):
            raise

    try:
        import soundfile as sf
    except ImportError as imp_exc:
        raise RuntimeError(
            "torchaudio.load failed and soundfile is not installed. "
            "Install with: pip install soundfile"
        ) from imp_exc

    data, sr = sf.read(path, always_2d=True, dtype="float32")
    LAST_LOAD_BACKEND = "soundfile"
    waveform = torch.from_numpy(data.T)
    if not channels_first:
        waveform = waveform.transpose(0, 1)
    return waveform, int(sr)
