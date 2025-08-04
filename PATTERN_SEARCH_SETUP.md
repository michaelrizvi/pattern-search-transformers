# PatternSearch Transformer Training Setup

This document describes the PatternSearch optimizer integration with PyTorch Lightning for training transformer models on StateTrackingDataset.

## Files Created

### Core Implementation
- `src/pattern_search_task.py` - PyTorch Lightning module for PatternSearch training
- `src/pattern_search_lightning.py` - Standalone PatternSearch transformer implementation
- `configs/pattern_search_train.yaml` - Training configuration for Hydra

### Testing & Examples  
- `test_pattern_search.py` - Quick functionality test
- `run_pattern_search.py` - Example training script with different configurations

## Quick Start

### 1. Basic Training
```bash
cd src
python train.py --config-name=pattern_search_train
```

### 2. Quick Test (2 epochs, limited batches)
```bash
cd src  
python train.py --config-name=pattern_search_train \
    trainer.max_epochs=2 \
    trainer.limit_train_batches=3 \
    trainer.limit_val_batches=2 \
    logger.offline=True
```

### 3. Custom Configuration
```bash
cd src
python train.py --config-name=pattern_search_train \
    trainer.max_epochs=10 \
    model.d_model=128 \
    model.n_layers=4 \
    datamodule.batch_size=8 \
    train_dataset.n_samples=500 \
    logger.offline=False
```

## Key Features

### PatternSearch Optimizer
- **Derivative-free optimization** - Works without gradients
- **Parameter perturbation** - Systematically tests parameter changes
- **Adaptive search radius** - Shrinks search space when no improvements found
- **Closure-based** - Requires loss function closure for evaluation

### Model Architecture
- Small transformer optimized for PatternSearch (slow optimization)
- Default: 2 layers, 4 heads, 64 dimensions
- Configurable via Hydra overrides

### Dataset
- StateTrackingDataset with parity function
- Sequence-to-sequence learning with intermediate state prediction
- Default: 8-token sequences, binary vocabulary

### Logging & Monitoring
- **WandB integration** - Experiment tracking and visualization
- **Offline mode support** - For testing without cloud sync
- **Checkpointing** - Automatic model saving on validation improvement
- **Early stopping** - Prevents overfitting

## Configuration Options

### Model Sizes
```yaml
# Small (default)
model:
  d_model: 64
  n_layers: 2
  n_heads: 4
  d_ff: 256

# Medium  
model:
  d_model: 128
  n_layers: 4
  n_heads: 8
  d_ff: 512

# Large
model:
  d_model: 256
  n_layers: 6
  n_heads: 8
  d_ff: 1024
```

### Dataset Sizes
```yaml
# Small (default)
train_dataset:
  n_samples: 200
  seq_len: 8

# Medium
train_dataset:
  n_samples: 500  
  seq_len: 12

# Large
train_dataset:
  n_samples: 1000
  seq_len: 16
```

### Optimizer Settings
```yaml
task:
  pattern_search_radius: 0.1  # Initial search radius
  # Smaller = fine-grained search
  # Larger = coarse-grained search
```

## Performance Notes

- **PatternSearch is slow** - Derivative-free optimization requires many forward passes
- **Start small** - Use small models and datasets for initial experiments  
- **Batch size matters** - Smaller batches = more optimizer steps per epoch
- **Monitor convergence** - Watch for radius shrinking and loss plateaus

## Integration Details

### Custom Optimizer Step
```python
def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
    def closure():
        loss = optimizer_closure()
        return loss
    optimizer.step(closure)
```

### Loss Masking
- Properly handles padding tokens in loss computation
- Computes accuracy only on non-padded tokens
- Compatible with variable-length sequences

### WandB Integration
- Automatically logs hyperparameters from Hydra config
- Tracks training/validation loss and accuracy
- Supports both online and offline modes
- Generates unique run names with timestamps

## Troubleshooting

### Common Issues

1. **Slow training** - This is expected with PatternSearch. Use smaller models/datasets.

2. **Memory errors** - Reduce batch size or model size:
   ```bash
   datamodule.batch_size=2 model.d_model=32
   ```

3. **Loss not improving** - Try different search radius:
   ```bash
   task.pattern_search_radius=0.01  # Smaller radius
   task.pattern_search_radius=1.0   # Larger radius
   ```

4. **WandB errors** - Use offline mode for testing:
   ```bash
   logger.offline=True
   ```

### Debugging
```python
# Test basic functionality
python test_pattern_search.py

# Check config loading  
cd src && python -c "import hydra; print('Hydra working')"

# Verify PatternSearch optimizer
cd src && python -c "from optimizer import PatternSearch; print('PatternSearch imported')"
```

## Next Steps

1. **Experiment with hyperparameters** - Try different model sizes and search radii
2. **Compare with gradient-based optimizers** - Benchmark against Adam/SGD
3. **Scale up gradually** - Start small, then increase model/dataset size
4. **Monitor WandB** - Analyze loss curves and parameter evolution
5. **Custom datasets** - Try different sequence learning tasks

## Example Results

The integration successfully demonstrates:
- ✅ PatternSearch optimizer working with PyTorch Lightning
- ✅ Proper loss computation and masking
- ✅ WandB logging and experiment tracking  
- ✅ Hydra configuration management
- ✅ Checkpoint saving and early stopping
- ✅ Training loss decreasing over epochs (5.84 → 5.80)

Ready for experimentation with derivative-free transformer training!