from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
 
 
class EMGIterableDataset(IterableDataset):
 
    def __init__(
        self,
        repo_id: str = "jasontchan/task123-23-12345-12_MU",
        split: str = "train",
        emg_key: str = "emg",
        expected_shape: tuple[int, int] = (100, 8),
        transform=None,
        drop_bad_rows: bool = True,
    ):
        """
        - repo_id: HF dataset repo id
        - split: split name
        - emg_key: column name for EMG
        - expected_shape: expected (T, C) shape; rows not matching are handled based on drop_bad_rows
        - transform: optional callable(tensor) -> tensor (e.g., normalization)
        - drop_bad_rows: if True, skip rows with wrong shape; if False, raise
        """
        self.repo_id = repo_id
        self.split = split
        self.emg_key = emg_key
        self.expected_shape = expected_shape
        self.transform = transform
        self.drop_bad_rows = drop_bad_rows
 
        self.ds = load_dataset(
            self.repo_id,
            split=self.split,
            streaming=True,
        )
 
    def __iter__(self):
        for row in self.ds:
            if self.emg_key not in row or row[self.emg_key] is None:
                if self.drop_bad_rows:
                    continue
                raise ValueError(f"Row missing '{self.emg_key}'")
 
            emg = row[self.emg_key]

            try:
                x = torch.tensor(emg[:50, :], dtype=torch.float32) #NOTE: window should only be last second (50 samples)
            except Exception as e:
                if self.drop_bad_rows:
                    continue
                raise e

            if x.ndim != 2:
                if self.drop_bad_rows:
                    continue
                raise ValueError(f"EMG tensor must be 2D (T,C), got shape {tuple(x.shape)}")
            if self.expected_shape and tuple(x.shape) != self.expected_shape:
                if self.drop_bad_rows:
                    continue
                raise ValueError(
                    f"EMG tensor shape {tuple(x.shape)} != expected {self.expected_shape}"
                )
 
            if self.transform is not None:
                x = self.transform(x)
 
            yield x  # shape (T, C)
 
 
def emg_collate(batch):
    return torch.stack(batch, dim=0)  # (B, T, C)