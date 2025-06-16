import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal

import torch
from lightning import Callback, LightningModule
from torch import Tensor
from transformer import DecoderOnlyClassifier, DecoderOnlyTransformer
from state_tracking_data import StateTrackingDataset, parity_fn, StateTrackingDataModule
from functools import partial


class StateTrackingTask(LightningModule):
    def __init__(
        self,
        model,
        lr: float = 1e-4,
        pad_token_id: int = 100,
        sep_token_id: int = 99,
    ):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x: Tensor) -> Any:
        """Forward pass of the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch["x"], batch["y"]
        logits = self(x)
        loss = self.loss_function(y, logits)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        x, y = batch["x"], batch["y"]

        logits = self(x)
        loss = self.loss_function(y, logits)
        perplexity = torch.exp(loss)
        accuracy = self.calculate_accuracy(y, logits)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_perplexity", perplexity, prog_bar=True, on_epoch=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx) -> Tensor:
        x, y = batch["x"], batch["y"]
        logits = self(x)
        loss = self.loss_function(y, logits)
        perplexity = torch.exp(loss)
        accuracy = self.calculate_accuracy(y, logits)
        
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_perplexity", perplexity, on_epoch=True)
        self.log("test_accuracy", accuracy, on_epoch=True)
        return loss

    def loss_function(self, target: Tensor, logits: Tensor) -> Tensor:
        """Cross entropy loss ignoring pad tokens."""
        # Reshape if needed: (B, S, V) -> (B*S, V)
        if logits.dim() > 2:
            logits = logits.view(-1, logits.size(-1))
        
        # Reshape target: (B, S) -> (B*S)
        if target.dim() > 1:
            target = target.view(-1)
            
        # Ignore pad tokens for loss calculation
        mask = target != self.pad_token_id
        filtered_logits = logits[mask]
        filtered_target = target[mask]
        
        return torch.nn.functional.cross_entropy(filtered_logits, filtered_target)
    
    def calculate_accuracy(self, target: Tensor, logits: Tensor) -> Tensor:
        """Calculate accuracy ignoring pad tokens."""
        # Get predicted tokens
        if logits.dim() > 2:
            preds = logits.argmax(dim=-1)
        else:
            preds = logits.argmax(dim=-1).reshape(target.shape)
        
        # Create mask to ignore pad tokens
        mask = target != self.pad_token_id
        
        # Calculate accuracy on non-pad tokens
        correct = (preds == target) & mask
        return correct.sum().float() / mask.sum()

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    # Example usage
    model = DecoderOnlyTransformer(vocab_size=2, d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=128)

    task = StateTrackingTask(
        model=model,
        lr=1e-4,
        pad_token_id=100,  # Example pad token ID
        sep_token_id=99,   # Example separator token ID
    )

    dataset = StateTrackingDataset(
        n_samples=10,
        seq_len=6,
        vocab_size=2,
        ground_truth_fn= partial(parity_fn, return_intermediates=True)
    )

    datamodule = StateTrackingDataModule(
        train_dataset=dataset,
        val_dataset=dataset,
        batch_size=2,
        num_workers=0
    ) 
    
    dataloader = datamodule.train_dataloader()
    for batch in dataloader:
        print(batch)  # This will print the batches of data
    # Print the model summary and hyperparameters
    print(task)  # This will print the model summary and hyperparameters

    from lightning import Trainer
    trainer = Trainer(max_epochs=1)
    trainer.fit(task, dataloader)
    trainer.test(task, dataloader)
    # This will run the training and testing loop
    # and print the results.    