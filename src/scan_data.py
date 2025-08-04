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
import random


warnings.filterwarnings("ignore", message=".*does not have many workers.*")

class ScanDataset(Dataset):
    def __init__(self, tokenizer, num_samples=10000, max_len=8):
        self.tokenizer = tokenizer
        self.data = []
        for _ in range(num_samples):
            if random.random() < 0.5:
                # SPLIT task
                bin_str = ''.join(random.choices(['0', '1'], k=random.choice(range(2, max_len + 1, 2))))
                mid = len(bin_str) // 2
                input_str = f"<split> {bin_str}"
                target_str = f"{bin_str[:mid]} {bin_str[mid:]}"
            else:
                # PARITY task
                a, b = random.choice(['0', '1']), random.choice(['0', '1'])
                input_str = f"<parity> {a} {b}"
                target_str = str(int(a) ^ int(b))
            self.data.append((input_str, target_str))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        x = self.tokenizer(src, return_tensors="pt", padding="max_length", truncation=True, max_length=16)
        y = self.tokenizer(tgt, return_tensors="pt", padding="max_length", truncation=True, max_length=16)
        return x["input_ids"].squeeze(0), y["input_ids"].squeeze(0)


class ScanDataModule(LightningDataModule):
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
    # Example usage
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "additional_special_tokens": ["<split>", "<parity>"],
    })
    #tokenizer.pad_token = tokenizer.eos_token  # Ensure padding is defined
    
    train_dataset = ScanDataset(num_samples=1000, max_len=8, tokenizer=tokenizer)
    val_dataset = ScanDataset(num_samples=200, max_len=8, tokenizer=tokenizer)

    data_module = ScanDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=3,
        num_workers=4
    )

    # Visualize a batch
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        print(batch)
        break  # Just visualize the first batch
    # Visualize the first 10 samples

    # Visualize a string
    for i in range(10):
        example_input, example_target = train_dataset[i]
        print("Input:", tokenizer.decode(example_input, skip_special_tokens=False))
        print("Target:", tokenizer.decode(example_target, skip_special_tokens=False))