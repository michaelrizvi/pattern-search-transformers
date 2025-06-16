import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal

import torch
from lightning import Callback, LightningModule
from torch import Tensor
from transformer import DecoderOnlyClassifier, DecoderOnlyTransformer
from state_tracking_data import StateTrackingDataset, parity_fn, StateTrackingDataModule
from functools import partial


class SequenceClassificationTask(LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        model=None,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch['x'], batch['y']
        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        x, y = batch['x'], batch['y']
        logits = self(x)
        loss = self.loss_function(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).long()
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx) -> Tensor:
        x, y = batch['x'], batch['y']
        logits = self(x)
        loss = self.loss_function(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).long()
        acc = (preds == y).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def loss_function(self, preds: Tensor, target: Tensor) -> Tensor:
        """Loss function to be used in the training loop."""
        return self.criterion(preds, target.float().unsqueeze(1))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    # Example usage of SequenceClassificationTask
    model = DecoderOnlyClassifier(
        vocab_size=2,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_len=100,
        n_classes=1
    )
    
    task = SequenceClassificationTask(model=model, lr=1e-4)
    
    # Create a dummy dataset
    dataset = StateTrackingDataset(
        n_samples=100,
        seq_len=10,
        vocab_size=2,
        ground_truth_fn=partial(parity_fn, return_intermediates=False)
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
    trainer = Trainer(max_epochs=100)
    trainer.fit(task, dataloader)
    trainer.test(task, dataloader)
    # This will run the training and testing loop
    # and print the results.   