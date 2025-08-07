import torch
from lightning import LightningModule
from torch import Tensor
from typing import Any

from transformer import DecoderOnlyTransformer


class AdamTask(LightningModule):
    def __init__(
        self,
        model,
        pad_token_id: int = 100,
        sep_token_id: int = 99,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x: Tensor, position_ids: Tensor = None) -> Any:
        """Forward pass of the model."""
        return self.model(x, position_ids=position_ids)

    def compute_loss(self, batch):
        """Compute loss for a batch - used by both training and validation."""
        # Handle both old tensor format and new dict format with position_ids
        if isinstance(batch, dict):
            sequences = batch['input_ids']
            position_ids = batch.get('position_ids', None)
        else:
            # Legacy tensor format
            sequences = batch
            position_ids = None
            
        # For next-token prediction: input = seq[:-1], target = seq[1:]
        x = sequences[:, :-1]  # All tokens except last
        y = sequences[:, 1:]   # All tokens except first (shifted by 1)
        
        # Adjust position_ids for shifted input if present
        pos_ids = position_ids[:, :-1] if position_ids is not None else None
        
        logits = self(x, position_ids=pos_ids)
        return self.loss_function(y, logits)

    def training_step(self, batch, batch_idx) -> Tensor:
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        # Handle both old tensor format and new dict format with position_ids
        if isinstance(batch, dict):
            sequences = batch['input_ids']
            position_ids = batch.get('position_ids', None)
        else:
            # Legacy tensor format
            sequences = batch
            position_ids = None
            
        # For next-token prediction: input = seq[:-1], target = seq[1:]
        x = sequences[:, :-1]  # All tokens except last
        y = sequences[:, 1:]   # All tokens except first (shifted by 1)
        
        # Adjust position_ids for shifted input if present
        pos_ids = position_ids[:, :-1] if position_ids is not None else None

        logits = self(x, position_ids=pos_ids)
        loss = self.loss_function(y, logits)
        exact_match = self.calculate_exact_match(y, logits)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_exact_match", exact_match, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx) -> Tensor:
        # Handle both old tensor format and new dict format with position_ids
        if isinstance(batch, dict):
            sequences = batch['input_ids']
            position_ids = batch.get('position_ids', None)
        else:
            # Legacy tensor format
            sequences = batch
            position_ids = None
            
        # For next-token prediction: input = seq[:-1], target = seq[1:]
        x = sequences[:, :-1]  # All tokens except last
        y = sequences[:, 1:]   # All tokens except first (shifted by 1)
        
        # Adjust position_ids for shifted input if present
        pos_ids = position_ids[:, :-1] if position_ids is not None else None
        
        logits = self(x, position_ids=pos_ids)
        loss = self.loss_function(y, logits)
        exact_match = self.calculate_exact_match(y, logits)
        
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_exact_match", exact_match, on_epoch=True)
        return loss

    def loss_function(self, target: Tensor, logits: Tensor) -> Tensor:
        """Cross entropy loss ignoring pad tokens."""
        # Reshape if needed: (B, S, V) -> (B*S, V)
        if logits.dim() > 2:
            logits = logits.reshape(-1, logits.size(-1))
        
        # Reshape target: (B, S) -> (B*S)
        if target.dim() > 1:
            target = target.reshape(-1)
            
        # Ignore pad tokens for loss calculation
        mask = target != self.pad_token_id
        filtered_logits = logits[mask]
        filtered_target = target[mask]
        
        return torch.nn.functional.cross_entropy(filtered_logits, filtered_target)
    
    def calculate_exact_match(self, target: Tensor, logits: Tensor) -> Tensor:
        """Calculate exact match - 1 if entire sequence is correct, 0 otherwise."""
        # Get predicted tokens - always argmax on last dimension
        preds = logits.argmax(dim=-1)
        
        # Create mask to ignore pad tokens
        mask = target != self.pad_token_id
        
        # Check if all non-pad tokens match for each sequence
        correct_per_token = (preds == target) | ~mask  # True for correct or pad tokens
        exact_matches = correct_per_token.all(dim=-1)  # True if all tokens in sequence are correct
        
        return exact_matches.float().mean()  # Return proportion of sequences with exact match

    def configure_optimizers(self) -> Any:
        """Configure AdamW optimizer with cosine annealing scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Cosine annealing scheduler from lr to 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10000,  # Total steps for cosine annealing
            eta_min=0.0   # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step
            }
        }