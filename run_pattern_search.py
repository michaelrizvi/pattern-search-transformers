#!/usr/bin/env python3
"""
Example script to run PatternSearch training with different configurations
"""
import subprocess
import sys
import os

def run_pattern_search_training(
    max_epochs: int = 10,
    batch_size: int = 4,
    model_size: str = "small",  # small, medium, large
    dataset_size: str = "small",  # small, medium, large
    wandb_offline: bool = True,
    limit_batches: int = None
):
    """
    Run PatternSearch training with specified configuration.
    
    Args:
        max_epochs: Number of training epochs
        batch_size: Batch size for training
        model_size: Model size (small/medium/large)
        dataset_size: Dataset size (small/medium/large) 
        wandb_offline: Whether to run wandb in offline mode
        limit_batches: Limit number of batches per epoch (for testing)
    """
    
    # Model configurations
    model_configs = {
        "small": {"d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 256},
        "medium": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff": 512},
        "large": {"d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 1024}
    }
    
    # Dataset configurations  
    dataset_configs = {
        "small": {"train_samples": 200, "val_samples": 50, "seq_len": 8},
        "medium": {"train_samples": 500, "val_samples": 100, "seq_len": 12},
        "large": {"train_samples": 1000, "val_samples": 200, "seq_len": 16}
    }
    
    model_config = model_configs[model_size]
    dataset_config = dataset_configs[dataset_size]
    
    # Build command
    cmd = [
        "python", "src/train.py",
        "--config-name=pattern_search_train",
        f"trainer.max_epochs={max_epochs}",
        f"datamodule.batch_size={batch_size}",
        f"model.d_model={model_config['d_model']}",
        f"model.n_layers={model_config['n_layers']}",
        f"model.n_heads={model_config['n_heads']}",
        f"model.d_ff={model_config['d_ff']}",
        f"train_dataset.n_samples={dataset_config['train_samples']}",
        f"val_dataset.n_samples={dataset_config['val_samples']}",
        f"test_dataset.n_samples={dataset_config['val_samples']}",
        f"train_dataset.seq_len={dataset_config['seq_len']}",
        f"val_dataset.seq_len={dataset_config['seq_len']}",
        f"test_dataset.seq_len={dataset_config['seq_len']}",
        f"logger.offline={str(wandb_offline).lower()}"
    ]
    
    if limit_batches:
        cmd.extend([
            f"trainer.limit_train_batches={limit_batches}",
            f"trainer.limit_val_batches={min(limit_batches, 2)}"
        ])
    
    print(f"Running PatternSearch training with {model_size} model and {dataset_size} dataset...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        print(f"\n✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        return False


def main():
    """Main function with example usage."""
    
    print("PatternSearch Training Examples")
    print("=" * 50)
    
    # Example 1: Quick test run
    print("\n1. Quick test run (2 epochs, 3 batches per epoch)")
    success1 = run_pattern_search_training(
        max_epochs=2,
        batch_size=4,
        model_size="small",
        dataset_size="small", 
        wandb_offline=True,
        limit_batches=3
    )
    
    if not success1:
        print("❌ Quick test failed!")
        sys.exit(1)
    
    # Example 2: Small full training run
    print("\n2. Small full training run (10 epochs)")
    success2 = run_pattern_search_training(
        max_epochs=10,
        batch_size=4,
        model_size="small",
        dataset_size="small",
        wandb_offline=True
    )
    
    if success2:
        print("\n🎉 All examples completed successfully!")
        print("\nTo run with different configurations, modify the parameters in this script or")
        print("run the training directly with:")
        print("cd src && python train.py --config-name=pattern_search_train [overrides...]")
    else:
        print("❌ Some training runs failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()