"""BatVision V2 (BV2) dataset — runtime STFT from binaural WAV + depth in meters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

from .audio_io import load_waveform
from .transforms import build_depth_transform, build_resize_transform

# Paper protocol defaults (see docs/METRIC_PROTOCOL.md)
BV2_DEFAULT_MAX_DEPTH_M = 30.0
BV2_DEFAULT_VALID_MIN_DEPTH_M = 0.5
BV2_IMAGE_SIZE = 256
BV2_SOUND_SPEED_M_S = 340.0

# Runtime STFT parameters when max_depth clipping is enabled (paper training path)
BV2_STFT_N_FFT = 512
BV2_STFT_WIN_LENGTH = 64
BV2_STFT_HOP_LENGTH = 16


class BV2Dataset(Dataset):
    """Load BV2 echo-depth pairs using a split index CSV and on-the-fly spectrograms.

    Parameters
    ----------
    data_root:
        Root directory containing scene folders (audio/, depth/, per-scene CSVs or unified index).
    index_file:
        CSV path relative to ``data_root`` or absolute. Expected columns:
        ``audio_relpath``, ``depth_relpath`` (and optional ``sample_id``, ``scene``).
        Legacy per-scene CSV columns ``audio path``, ``audio file name``, ``depth path``,
        ``depth file name`` are also supported when building from official scene CSVs.
    max_depth_m:
        Clip ground-truth depth above this value (meters) and normalize model targets.
    valid_min_depth_m:
        Minimum depth (meters) for valid pixels in evaluation masks. Not used for metric aggregation here.
    depth_normalize:
        If True, divide depth by ``max_depth_m`` after clipping (model input scale).
    image_size:
        Spatial size for echo spectrogram and depth (paper: 256).
    """

    def __init__(
        self,
        data_root: str | Path,
        index_file: str | Path,
        *,
        max_depth_m: float = BV2_DEFAULT_MAX_DEPTH_M,
        valid_min_depth_m: float = BV2_DEFAULT_VALID_MIN_DEPTH_M,
        depth_normalize: bool = True,
        image_size: int = BV2_IMAGE_SIZE,
    ) -> None:
        self.data_root = Path(data_root)
        index_path = Path(index_file)
        if not index_path.is_absolute():
            index_path = self.data_root / index_path
        if not index_path.is_file():
            raise FileNotFoundError(f"BV2 index file not found: {index_path}")

        self.max_depth_m = float(max_depth_m)
        self.valid_min_depth_m = float(valid_min_depth_m)
        self.depth_normalize = bool(depth_normalize)
        self.image_size = int(image_size)

        self.instances = pd.read_csv(index_path)
        self._validate_index_columns()

        self._spec_transform = T.Spectrogram(
            n_fft=BV2_STFT_N_FFT,
            win_length=BV2_STFT_WIN_LENGTH,
            hop_length=BV2_STFT_HOP_LENGTH,
            power=1.0,
        )
        self._spec_resize = build_resize_transform(self.image_size)
        self._depth_transform = build_depth_transform(
            image_size=self.image_size,
            max_depth_m=self.max_depth_m,
            normalize=self.depth_normalize,
        )

    def _validate_index_columns(self) -> None:
        cols = set(self.instances.columns)
        has_public = {"audio_relpath", "depth_relpath"}.issubset(cols)
        has_legacy = {"audio path", "audio file name", "depth path", "depth file name"}.issubset(cols)
        if not (has_public or has_legacy):
            raise ValueError(
                "Index CSV must contain either (audio_relpath, depth_relpath) "
                "or legacy (audio path, audio file name, depth path, depth file name)."
            )

    def __len__(self) -> int:
        return len(self.instances)

    def _resolve_paths(self, row: pd.Series) -> tuple[Path, Path]:
        if "audio_relpath" in row.index:
            audio_path = self.data_root / str(row["audio_relpath"])
            depth_path = self.data_root / str(row["depth_relpath"])
        else:
            audio_path = self.data_root / str(row["audio path"]) / str(row["audio file name"])
            depth_path = self.data_root / str(row["depth path"]) / str(row["depth file name"])
        return audio_path, depth_path

    def _load_depth_meters(self, depth_path: Path) -> np.ndarray:
        depth_mm = np.load(depth_path)
        depth_m = depth_mm.astype(np.float32) / 1000.0
        depth_m = np.clip(depth_m, 0.0, self.max_depth_m)
        return depth_m

    def _waveform_to_spectrogram(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        cut = int((2.0 * self.max_depth_m / BV2_SOUND_SPEED_M_S) * sample_rate)
        waveform = waveform[:, :cut]
        spec = self._spec_transform(waveform)
        return self._spec_resize(spec)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.instances.iloc[idx]
        audio_path, depth_path = self._resolve_paths(row)

        depth_m = self._load_depth_meters(depth_path)
        depth_tensor = self._depth_transform(depth_m)

        waveform, sr = load_waveform(audio_path)
        echo_spectrogram = self._waveform_to_spectrogram(waveform, sr)

        depth_meters_map = torch.from_numpy(depth_m).unsqueeze(0)
        depth_meters_resized = torch.nn.functional.interpolate(
            depth_meters_map.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0)
        valid_mask = (depth_meters_resized >= self.valid_min_depth_m).float()

        sample: dict[str, Any] = {
            "echo_spectrogram": echo_spectrogram,
            "echo": echo_spectrogram,
            "depth": depth_tensor,
            "depth_meters": depth_meters_resized,
            "valid_mask": valid_mask,
        }
        if "sample_id" in row.index and pd.notna(row["sample_id"]):
            sample["sample_id"] = str(row["sample_id"])
        if "scene" in row.index and pd.notna(row["scene"]):
            sample["scene"] = str(row["scene"])
        return sample

    @property
    def expected_output_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "echo_spectrogram": (2, self.image_size, self.image_size),
            "depth": (1, self.image_size, self.image_size),
        }
