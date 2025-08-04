import warnings
import re

warnings.filterwarnings(
    "ignore",
    message=".*NotOpenSSLWarning.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*urllib3 v2 only supports OpenSSL 1.1.1+.*",
)
import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal

import torch
from lightning import Callback, LightningModule
from torch import Tensor
from transformer import DecoderOnlyClassifier, DecoderOnlyTransformer
from scan_data import ScanDataset, ScanDataModule
from functools import partial
from torchmetrics import Accuracy
import torch.nn.functional as F


class ScanTask(LightningModule):
    def __init__(self, model, pad_token_id: int = 0, lr: float = 1e-4, num_classes=2):
        super().__init__()
        self.model = model
        self.lr = lr
        self.pad_token_id = pad_token_id
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=pad_token_id)
        self.save_hyperparameters(ignore=['model'])

    def forward(self, input_ids):
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        logits = self(input_ids)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_token_id
        )
        self.log("train_loss", loss)
        return loss

    def _eval_step(self, batch, stage):
        input_ids, target_ids = batch
        logits = self(input_ids)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_token_id
        )
        self.log(f"{stage}_loss", loss, prog_bar=True)

        preds = shift_logits.argmax(dim=-1)
        acc = self.accuracy(preds, shift_labels)
        self.log(f"{stage}_accuracy", acc, prog_bar=True)

        perplexity = torch.exp(loss)
        self.log(f"{stage}_perplexity", perplexity, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    import warnings

    warnings.filterwarnings(
        "ignore",
        message=".*urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL.*"
    )
    # Example usage
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "additional_special_tokens": ["<split>", "<parity>"],
    })
    #tokenizer.pad_token = tokenizer.eos_token  # Ensure padding is defined
    model = DecoderOnlyTransformer(vocab_size=len(tokenizer), d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=128)

    task = ScanTask(
        model=model,
        lr=1e-4,
        pad_token_id=0,
        num_classes = len(tokenizer)
    )
    
    train_dataset = ScanDataset(num_samples=1000, max_len=8, tokenizer=tokenizer)
    val_dataset = ScanDataset(num_samples=200, max_len=8, tokenizer=tokenizer)

    data_module = ScanDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=64,
        num_workers=4,
    )  

    from lightning import Trainer
    trainer = Trainer(
        max_epochs=5,
        accelerator="cpu",  # Change to "gpu" if you have a GPU
        devices=1,  # Number of devices to use
    )
    trainer.fit(task, datamodule=data_module)
    trainer.test(task, datamodule=data_module)