"""SpectralTF normalization, padding, and paired CXRS/XCS dataset factory."""

import os

import numpy as np
import torch

from .dataset_folder import SpectralDataset
from .spectral_pretraining_common import FIXED_R2_RANGES, get_fixed_peak_bounds


class SpectralTFTransform(object):
    """Load, normalize, and pad spectral matrices for SpectralTF pretraining."""

    DEFAULT_INPUT_SIZES = {
        'cxrs': (12, 1024),
        'xcs': (35, 488),
    }
    DEFAULT_ORIGINAL_SIZES = {
        'cxrs': (12, 1024),
        'xcs': (35, 487),
    }

    def __init__(self, args):
        self.input_sizes = {}
        self.original_sizes = {}
        domains = getattr(args, 'all_domains', ['cxrs', 'xcs'])
        for task in domains:
            default_input = self.DEFAULT_INPUT_SIZES.get(task, getattr(args, 'input_size', (224, 224)))
            default_orig = self.DEFAULT_ORIGINAL_SIZES.get(task, default_input)
            self.input_sizes[task] = self._as_pair(getattr(args, f'{task}_input_size', default_input))
            self.original_sizes[task] = self._as_pair(getattr(args, f'{task}_original_size', default_orig))
        self.normalize = getattr(args, 'spectral_normalize', 'minmax')
        self.normalize_eps = float(getattr(args, 'spectral_normalize_eps', 1e-6))

    @staticmethod
    def _as_pair(value):
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, str):
            parts = value.replace(',', ' ').replace('x', ' ').split()
            if len(parts) == 1:
                return (int(parts[0]), int(parts[0]))
            return (int(parts[0]), int(parts[1]))
        return (int(value[0]), int(value[1]))

    @staticmethod
    def _load_array(item):
        if isinstance(item, (str, os.PathLike)):
            data = np.load(item)
            if isinstance(data, np.lib.npyio.NpzFile):
                key = data.files[0]
                data = data[key]
            return np.asarray(data)
        return np.asarray(item)

    def _reshape_if_needed(self, arr, task):
        arr = np.asarray(arr)
        if arr.ndim == 3 and 1 in (arr.shape[0], arr.shape[-1]):
            arr = np.squeeze(arr)
        if arr.ndim == 1:
            candidates = []
            for size in (
                self.original_sizes.get(task),
                self.input_sizes.get(task),
                self.DEFAULT_ORIGINAL_SIZES.get(task),
                self.DEFAULT_INPUT_SIZES.get(task),
            ):
                if size is None:
                    continue
                size = self._as_pair(size)
                if size not in candidates:
                    candidates.append(size)

            for h, w in candidates:
                if arr.size == h * w:
                    arr = arr.reshape(h, w)
                    break
            else:
                raise ValueError(
                    f"Cannot reshape 1D spectral array for {task}: size={arr.size}, "
                    f"candidate_shapes={candidates}"
                )
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D spectral matrix for {task}, got shape {arr.shape}")
        return arr

    def _pad_or_crop(self, tensor, task):
        target_h, target_w = self.input_sizes[task]
        h, w = tensor.shape[-2:]
        out = torch.zeros((target_h, target_w), dtype=tensor.dtype)
        copy_h = min(h, target_h)
        copy_w = min(w, target_w)
        out[:copy_h, :copy_w] = tensor[:copy_h, :copy_w]
        return out

    def _normalize(self, tensor, task):
        """Normalize a single spectral matrix [sampling_point, wavelength].

        Supported modes:
            none: no normalization;
            zscore: global per-sample z-score;
            minmax: global per-sample min-max;
            peak_ref_minmax / fixed_peak_ref_minmax / roi_ref_minmax:
                compute min/max only from the task-specific fixed peak interval,
                then apply that scale to the complete spectrum. Values outside
                the peak interval are intentionally not clipped and may be < 0 or > 1;
            per_wavelength_robust / per_wavelength: median/IQR are computed
                across sampling points for each wavelength;
        """
        tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mode = str(self.normalize).lower()
        eps = self.normalize_eps

        if mode == 'none':
            return tensor

        if mode == 'zscore':
            mean = tensor.mean()
            std = tensor.std().clamp_min(eps)
            return (tensor - mean) / std

        if mode in ('per_wavelength', 'per_wavelength_robust', 'column_robust'):
            median = tensor.median(dim=0).values
            q25 = torch.quantile(tensor, 0.25, dim=0)
            q75 = torch.quantile(tensor, 0.75, dim=0)
            iqr = (q75 - q25).clamp_min(eps)
            return (tensor - median.unsqueeze(0)) / iqr.unsqueeze(0)

        if mode in (
            'peak_ref_minmax',
            'fixed_peak_ref_minmax',
            'roi_ref_minmax',
        ):
            if task not in FIXED_R2_RANGES:
                min_v = tensor.min()
                max_v = tensor.max()
            else:
                start, end = get_fixed_peak_bounds(task, tensor.shape[-1])
                if end <= start:
                    raise ValueError(
                        f"Invalid fixed peak interval for {task}: "
                        f"bounds=({start}, {end}), width={tensor.shape[-1]}"
                    )
                reference = tensor[:, start:end]
                min_v = reference.min()
                max_v = reference.max()

            denom = (max_v - min_v).clamp_min(eps)
            # Scale the complete spectrum and deliberately do not clip.
            return (tensor - min_v) / denom

        min_v = tensor.min()
        max_v = tensor.max()
        denom = (max_v - min_v).clamp_min(eps)
        return (tensor - min_v) / denom

    def __call__(self, task_dict):
        output = {}
        for task, value in task_dict.items():
            arr = self._load_array(value)
            arr = self._reshape_if_needed(arr, task)
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()
            tensor = self._normalize(tensor, task)
            tensor = self._pad_or_crop(tensor, task)
            output[task] = tensor.unsqueeze(0)
        return output

    def __repr__(self):
        return (
            f"SpectralTFTransform(input_sizes={self.input_sizes}, "
            f"normalize={self.normalize}, eps={self.normalize_eps}, "
            "full_spectrum_input=True)"
        )



def build_spectral_pretraining_dataset(args, split='train'):
    """Build spectral pretraining dataset."""
    transform = SpectralTFTransform(args)
    print("Data Aug = %s" % str(transform))
    allow_missing = getattr(args, 'allow_missing_modalities', False)
    return SpectralDataset(args.data_path, args.all_domains, transform=transform, split=split, allow_missing=allow_missing)

