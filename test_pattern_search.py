#!/usr/bin/env python3
"""
Quick test script to verify PatternSearchTransformer functionality
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from functools import partial

from pattern_search_lightning import PatternSearchTransformer, create_datasets
from state_tracking_data import StateTrackingDataModule, parity_fn
from lightning import Trainer

def test_basic_functionality():
    print("Testing PatternSearchTransformer basic functionality...")
    
    # Create small datasets for quick testing
    train_dataset, val_dataset = create_datasets(
        train_samples=20,
        val_samples=10,
        seq_len=4,
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
    
    # Create small model
    model = PatternSearchTransformer(
        vocab_size=102,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_len=64
    )
    
    print("Model created successfully!")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch = next(iter(data_module.train_dataloader()))
    with torch.no_grad():
        output = model(batch["x"])
        print(f"Forward pass successful! Input shape: {batch['x'].shape}, Output shape: {output.shape}")
    
    # Test loss computation
    loss = model.compute_loss(batch)
    print(f"Loss computation successful! Loss: {loss.item():.4f}")
    
    # Quick training test (just 1 epoch, 2 steps)
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )
    
    print("Starting quick training test...")
    trainer.fit(model, data_module)
    print("Training test completed successfully!")
    
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n✅ All tests passed! PatternSearchTransformer is working correctly.")
    else:
        print("\n❌ Tests failed!")