import torch
import torch.nn.functional as F
from lightning import LightningModule
from functools import partial

from optimizer import PatternSearch
from transformer import DecoderOnlyTransformer
from state_tracking_data import StateTrackingDataset, parity_fn


class PatternSearchTransformer(LightningModule):
    def __init__(
        self,
        vocab_size: int = 102,  # 0-99 + sep_token (99) + pad_token (100)
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        max_len: int = 1024,
        dropout: float = 0.1,
        sep_token: int = 99,
        pad_token: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.transformer = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        self.pad_token = pad_token
        self.sep_token = sep_token
        
    def forward(self, x):
        return self.transformer(x)
    
    def compute_loss(self, batch):
        x, y = batch["x"], batch["y"]
        
        logits = self.forward(x)
        
        # Create mask to ignore padding tokens in loss computation
        mask = (y != self.pad_token)
        
        # Flatten for cross entropy
        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        mask_flat = mask.view(-1)
        
        # Compute loss only on non-padded tokens
        loss = F.cross_entropy(logits_flat[mask_flat], y_flat[mask_flat])
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        
        # Log the loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Compute accuracy for validation
        x, y = batch["x"], batch["y"]
        logits = self.forward(x)
        
        mask = (y != self.pad_token)
        predictions = torch.argmax(logits, dim=-1)
        
        # Only compute accuracy on non-padded tokens
        correct = (predictions == y) & mask
        total = mask.sum()
        
        if total > 0:
            accuracy = correct.sum().float() / total.float()
            self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Initialize PatternSearch optimizer
        optimizer = PatternSearch(
            self.parameters(),
            rho=0.05,  # SAM-style parameter for perturbation radius
            adaptive=False
        )
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Custom optimizer step for PatternSearch
        def closure():
            loss = optimizer_closure()
            return loss
        
        # PatternSearch optimizer requires a closure
        optimizer.step(closure)


def create_datasets(
    train_samples: int = 1000,
    val_samples: int = 200,
    seq_len: int = 10,
    vocab_size: int = 2,
    return_intermediates: bool = True
):
    """Create train and validation StateTrackingDatasets"""
    
    ground_truth_fn = partial(parity_fn, return_intermediates=return_intermediates)
    
    train_dataset = StateTrackingDataset(
        n_samples=train_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        ground_truth_fn=ground_truth_fn,
        return_intermediates=return_intermediates,
        sep_token=99,
        pad_token=100,
    )
    
    val_dataset = StateTrackingDataset(
        n_samples=val_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        ground_truth_fn=ground_truth_fn,
        return_intermediates=return_intermediates,
        sep_token=99,
        pad_token=100,
    )
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Example usage
    from state_tracking_data import StateTrackingDataModule
    from lightning import Trainer
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(
        train_samples=100,
        val_samples=20,
        seq_len=8,
        vocab_size=2,
        return_intermediates=True
    )
    
    # Create data module
    data_module = StateTrackingDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        num_workers=0
    )
    
    # Create model
    model = PatternSearchTransformer(
        vocab_size=102,  # 0-1 (input vocab) + 99 (sep) + 100 (pad) + output classes
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        max_len=128
    )
    
    # Create trainer
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        check_val_every_n_epoch=1
    )
    
    # Train the model
    trainer.fit(model, data_module)