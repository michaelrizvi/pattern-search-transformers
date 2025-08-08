import torch
from lightning import LightningModule
from torch import Tensor
from typing import Any

from optimizer import RandomGreedySearch
from transformer import DecoderOnlyTransformer


class RandomGreedyTask(LightningModule):
    def __init__(
        self,
        model,
        pad_token_id: int = 100,
        sep_token_id: int = 99,
        sigma: float = 1.0,
        max_steps_per_sigma: int = 30000,
        decay_factor: float = 2.0,
    ):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id
        self.sigma = sigma
        self.max_steps_per_sigma = max_steps_per_sigma
        self.decay_factor = decay_factor
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
        """Calculate exact match - 1 if answer portion after separator is correct, 0 otherwise."""
        # Get predicted tokens - always argmax on last dimension
        preds = logits.argmax(dim=-1)
        
        batch_size = target.size(0)
        exact_matches = []
        
        for i in range(batch_size):
            # Find separator token position in target sequence
            sep_positions = (target[i] == self.sep_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_positions) == 0:
                # No separator found, fall back to full sequence match
                mask = target[i] != self.pad_token_id
                correct_per_token = (preds[i] == target[i]) | ~mask
                exact_matches.append(correct_per_token.all().float())
            else:
                # Get position after separator - this is where the answer starts
                sep_pos = sep_positions[0].item()
                answer_start = sep_pos + 1
                
                if answer_start >= target.size(1):
                    # No answer portion, consider as no match
                    exact_matches.append(torch.tensor(0.0))
                else:
                    # Extract answer portion (everything after separator)
                    answer_target = target[i, answer_start:]
                    answer_preds = preds[i, answer_start:]
                    
                    # Create mask to ignore pad tokens in answer
                    answer_mask = answer_target != self.pad_token_id
                    
                    if answer_mask.sum() == 0:
                        # No non-pad tokens in answer, consider as match
                        exact_matches.append(torch.tensor(1.0))
                    else:
                        # Check if all non-pad answer tokens match
                        correct_answer_tokens = (answer_preds == answer_target) | ~answer_mask
                        exact_matches.append(correct_answer_tokens.all().float())
        
        return torch.stack(exact_matches).mean()

    def configure_optimizers(self) -> Any:
        """Configure RandomGreedySearch optimizer."""
        optimizer = RandomGreedySearch(
            self.parameters(),
            sigma=self.sigma,
            max_steps_per_sigma=self.max_steps_per_sigma,
            decay_factor=self.decay_factor,
            rho=0.05,  # Required by HelperOptimizer base class
            adaptive=False
        )
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Custom optimizer step for RandomGreedySearch."""
        def closure():
            loss = optimizer_closure()
            return loss
        
        # RandomGreedySearch optimizer requires a closure
        optimizer.step(closure)