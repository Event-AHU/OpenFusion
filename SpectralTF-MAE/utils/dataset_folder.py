"""Paired full-spectrum CXRS/XCS data loading from .npy or .npz files."""

import os

import numpy as np
from torch.utils.data import Dataset
from typing import List


class SpectralDataset(Dataset):
    """Dataset for paired spectral matrices stored as .npy/.npz files.

    Expected layout:
        DATA_ROOT/{train,val,test}/{cxrs,xcs}/*.npy
    File names without extension are used as sample ids.
    """

    def __init__(self, data_path: str, tasks: List[str], transform=None, split='train', allow_missing=False):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.tasks = list(tasks)
        self.split = split
        self.allow_missing = allow_missing

        self._validate_data_alignment()
        self.stageids = self._get_stageids()
        if len(self.stageids) == 0:
            raise RuntimeError(
                f"No spectral samples found under {self.data_path}/{self.split} for tasks {self.tasks}."
            )
        print(f"Found {len(self.stageids)} spectral samples for tasks {self.tasks} in split {split}")

    def _task_dir(self, task: str) -> str:
        if self.split in ['train', 'val', 'test']:
            return os.path.join(self.data_path, self.split, task)
        return os.path.join(self.data_path, task)

    def _validate_data_alignment(self):
        for task in self.tasks:
            task_path = self._task_dir(task)
            if not os.path.isdir(task_path):
                raise ValueError(f"Task path does not exist: {task_path}")

    def _ids_for_task(self, task: str):
        task_path = self._task_dir(task)
        ids = set()
        for file in os.listdir(task_path):
            lower = file.lower()
            if lower.endswith('.npy') or lower.endswith('.npz'):
                ids.add(os.path.splitext(file)[0])
        return ids

    def _get_stageids(self):
        ids_by_task = {task: self._ids_for_task(task) for task in self.tasks}
        if self.allow_missing:
            stageids = set().union(*ids_by_task.values())
        else:
            stageids = set.intersection(*ids_by_task.values())
            dropped = {task: len(ids_by_task[task] - stageids) for task in self.tasks}
            if any(v > 0 for v in dropped.values()):
                print(f"Dropped unpaired spectral files per task: {dropped}")
        return sorted(stageids)

    def _get_file_path(self, task: str, stageid: str) -> str:
        for ext in ('.npy', '.npz'):
            path = os.path.join(self._task_dir(task), f"{stageid}{ext}")
            if os.path.exists(path):
                return path
        return ''

    def __getitem__(self, index: int):
        stageid = self.stageids[index]
        task_dict = {}
        for task in self.tasks:
            file_path = self._get_file_path(task, stageid)
            if file_path:
                task_dict[task] = file_path
            elif self.allow_missing:
                if task == 'cxrs':
                    task_dict[task] = np.zeros((12, 1024), dtype=np.float32)
                elif task == 'xcs':
                    task_dict[task] = np.zeros((35, 487), dtype=np.float32)
                else:
                    raise FileNotFoundError(f"Missing spectral file for task={task}, sample={stageid}")
            else:
                raise FileNotFoundError(f"Missing spectral file for task={task}, sample={stageid}")

        if self.transform is not None:
            task_dict = self.transform(task_dict)

        return task_dict, 0

    def __len__(self):
        return len(self.stageids)
