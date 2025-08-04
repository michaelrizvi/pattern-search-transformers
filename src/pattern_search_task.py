import torch
from lightning import LightningModule
from torch import Tensor
from typing import Any

from optimizer import PatternSearch
from transformer import DecoderOnlyTransformer


class PatternSearchTask(LightningModule):
    def __init__(
        self,
        model,
        pad_token_id: int = 100,
        sep_token_id: int = 99,
        pattern_search_radius: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id
        self.pattern_search_radius = pattern_search_radius
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x: Tensor) -> Any:
        """Forward pass of the model."""
        return self.model(x)

    def compute_loss(self, batch):
        """Compute loss for a batch - used by both training and validation."""
        x, y = batch["x"], batch["y"]
        logits = self(x)
        return self.loss_function(y, logits)

    def training_step(self, batch, batch_idx) -> Tensor:
        loss = self.compute_loss(batch)
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
        """Configure PatternSearch optimizer."""
        optimizer = PatternSearch(
            self.parameters(),
            rho=0.05,  # SAM-style parameter for perturbation radius
            adaptive=False
        )
        # Set initial radius
        optimizer.radius = self.pattern_search_radius
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Custom optimizer step for PatternSearch."""
        def closure():
            loss = optimizer_closure()
            return loss
        
        # PatternSearch optimizer requires a closure
        optimizer.step(closure)