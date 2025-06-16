import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningDataModule
from pytorch_lightning.utilities.seed import isolate_rng
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader, Dataset
from functools import partial
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*does not have many workers.*")

def parity_fn(sequence: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor:
    """
    Compute the parity of the number of 0 tokens in the sequence.
    
    Args:
        sequence: A tensor of integers
        return_intermediates: If True, returns the parity at each position
    
    Returns:
        If return_intermediates is False: 1 if odd number of 0s, 0 if even
        If return_intermediates is True: Tensor of parities at each position
    """
    # Count 0s at each position (1 if the token is 0, 0 otherwise)
    zero_indicators = (sequence == 0).long()
    
    if return_intermediates:
        # Compute cumulative sum of zero indicators
        zero_counts = torch.cumsum(zero_indicators, dim=0)
        # Compute parity at each position (1 if odd, 0 if even)
        parities = zero_counts % 2
        return parities
    else:
        # Count total number of 0s
        zero_count = torch.sum(zero_indicators)
        # Return 1 if odd number of 0s, 0 if even
        return zero_count % 2


class StateTrackingDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        vocab_size: int,
        ground_truth_fn: Optional[callable] = None,
        return_intermediates: bool = False,
        sep_token: int = 99,
        pad_token: int = 100,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.ground_truth_fn = ground_truth_fn
        self.return_intermediates = ground_truth_fn.keywords["return_intermediates"] 
        self.sep_token = sep_token
        self.pad_token = pad_token

        self.data, self.task_params = self.gen_data(n_samples)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        x = self.data["x"][index]
        y = self.data["y"][index]
        return {"x": x, "y": y}

    @torch.inference_mode()
    def gen_data(
        self,
        n_samples: int,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        x_raw = torch.randint(0, self.vocab_size, (n_samples, self.seq_len))

        if self.return_intermediates:
            input_seqs = []
            target_seqs = []
            for i in range(n_samples):
                x_i = x_raw[i]
                y_full = self.ground_truth_fn(x_i, return_intermediates=True)

                input_seq = torch.cat([x_i, torch.tensor([self.sep_token]), y_full[:-1]])
                target_seq = torch.cat([
                    torch.full((self.seq_len + 1,), self.pad_token),
                    y_full
                ])
                input_seqs.append(input_seq)
                target_seqs.append(target_seq)

            max_len = max(len(seq) for seq in input_seqs)
            x = torch.stack([F.pad(seq, (0, max_len - len(seq)), value=self.pad_token) for seq in input_seqs])
            y = torch.stack([F.pad(seq, (0, max_len - len(seq)), value=self.pad_token) for seq in target_seqs])

        else:
            x = x_raw
            example_y = self.ground_truth_fn(x[0])
            y = torch.zeros((n_samples, *example_y.shape), dtype=torch.long)
            for i in range(n_samples):
                y[i] = self.ground_truth_fn(x[i])

        data_dict = {"x": x, "y": y}
        params_dict = {
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "n_samples": n_samples,
            "return_intermediates": self.return_intermediates,
            "sep_token": self.sep_token,
            "pad_token": self.pad_token,
        }

        return data_dict, params_dict



class StateTrackingDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        test_dataset: Optional[Dataset] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset if test_dataset is not None else val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=None,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=None,
        )


if __name__ == "__main__":
    x = torch.tensor([0, 1, 0, 1, 0, 1])
    parity = parity_fn(x, return_intermediates=True)
    print("Input:", x)
    print("Parity at each position:", parity)

    # Example usage of StateTrackingDataset
    dataset = StateTrackingDataset(
        n_samples=10,
        seq_len=6,
        vocab_size=2,
        ground_truth_fn= partial(parity_fn, return_intermediates=False)
    )
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample)
        print(f"Sample {i}: x={sample['x']}, y={sample['y']}")

    print(type(sample['y']))
    # Example usage of StateTrackingDataModule
    data_module = StateTrackingDataModule(
        train_dataset=dataset,
        val_dataset=dataset,
        batch_size=2,
        num_workers=0
    )
    print("Data module created with train and validation datasets.")
    print("Train dataset size:", len(data_module.train_dataset))
    print("Validation dataset size:", len(data_module.val_dataset))