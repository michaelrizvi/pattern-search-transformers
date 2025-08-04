# 🧠 Pattern-Search Transformers

This research project explores **derivative-free optimization of transformer models** using the PatternSearch algorithm. Instead of traditional gradient-based training (Adam, SGD), this project investigates whether transformers can be effectively trained through systematic parameter perturbation and pattern-based search strategies.

## 🎯 Project Overview

This project implements and evaluates PatternSearch optimization for training small transformer models on sequence learning tasks like state tracking and membership inference. Key research questions include:
- Can derivative-free methods effectively train neural language models?
- How does PatternSearch compare to gradient-based optimization in terms of convergence and final performance?
- What are the computational trade-offs between gradient-free and gradient-based training?

The implementation is built on top of a robust research framework that includes:

- ⚡ [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/): A framework to organize your deep learning research.
- 🔧 [Hydra](https://hydra.cc/): A powerful configuration management system.
- ✅ [Pre-commit](https://pre-commit.com/): A tool to ensure clean and formatted code.
- 🧪 [Unit Testing](https://docs.pytest.org/en/6.2.x/): For verifying that each function works as expected.
- 📊 [WandB integration](https://wandb.ai/site): For experiment tracking and visualization.
- 🤖 [CI with Github Actions](https://docs.github.com/en/actions): Continuous Integration setup to maintain project quality.

Additional utilities:
- Ready-to-use Jupyter notebook in `report/plots/notebook.ipynb` for making reproducible Seaborn plots, pulling data directly from your WandB project.
- Pre-implemented VScode debugger config file in `.vscode/launch.json` for debugging your code.

## 🚀 Key Components

### PatternSearch Optimizer (`src/optimizer.py`)
- **Derivative-free optimization** - Works without computing gradients
- **Parameter perturbation** - Systematically tests small parameter changes
- **Adaptive search radius** - Shrinks search space when no improvements found
- **Closure-based evaluation** - Requires loss function closure for parameter evaluation

### Transformer Architecture (`src/transformer.py`)
- Decoder-only transformer optimized for PatternSearch training
- Causal self-attention with configurable model dimensions
- Small-scale by default (2 layers, 4 heads, 64 dimensions) for efficient derivative-free training

### Tasks & Datasets
- **State Tracking** (`src/state_tracking_task.py`, `src/state_tracking_data.py`) - Sequence-to-sequence learning with intermediate state prediction
- **Membership Task** (`src/membership_task.py`) - Learning set membership patterns
- **SCAN Task** (`src/scan_task.py`, `src/scan_data.py`) - Compositional sequence-to-sequence learning

### Training Integration (`src/pattern_search_task.py`)
- PyTorch Lightning module with PatternSearch optimizer integration
- Custom optimizer step handling for closure-based optimization
- Proper loss masking for variable-length sequences

---

### 🔧 Hydra Configuration
The template uses Hydra for flexible configuration management. Configuration files are stored in the `configs` folder:
- `configs/train.yaml`: The main config file where you define hyperparameters.

You can also define different configurations for different experiments, overwrite configs, create nested configs etc... The configuration system is very flexible and allows you to define your own configuration structure. Use Hydra to structure your configuration system effectively. More details [here](https://hydra.cc/).

---

### ✅ Pre-commit
Pre-commit hooks ensure that your code is clean and formatted before committing any change when working with multiple collaborators. The hooks are defined in the `.pre-commit-config.yaml` file.
When the hooks are triggered, you need to re-commit any change it made. They are also automatically run by the CI pipeline on your remote repository to maintain code quality.
Install them with:
```bash
pre-commit install
```

---

### 🧪 Unit Testing
A unit test file, `test_all.py`, is included to verify that each of your functions works as expected. While not mandatory for simple projects, it is a good practice for larger or collaborative projects. The tests are automatically run by the CI pipeline on your remote repository, and notifications are sent if any test fails.

---

### 📊 WandB Integration
Log experiments and metrics seamlessly with WandB. The integration is already included in the template, and logging is as simple as using the `self.log()` function in PyTorch Lightning. To configure WandB, just edit `configs/train.yaml`:
```yaml
logger:
  _target_: lightning.pytorch.loggers.WandbLogger
  entity: # Add your WandB entity here
  project: # Add your WandB project here
```
Learn more about WandB [here](https://wandb.ai/site).

---

## ⚙️ Installation
Python 3.6 or later is required. It is recommended to use a virtual environment to avoid package conflicts.

1️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

2️⃣ Set up pre-commit hooks:
```bash
pre-commit install
```

3️⃣ Configure WandB (if applicable):
Edit `configs/train.yaml` with your WandB entity and project information.

4️⃣ You're good to go!

---

## ▶️ Usage

### Quick Start with PatternSearch Training

1. **Basic PatternSearch training**:
```bash
cd src
python train.py --config-name=pattern_search_train
```

2. **Quick test run** (2 epochs, limited batches):
```bash
cd src  
python train.py --config-name=pattern_search_train \
    trainer.max_epochs=2 \
    trainer.limit_train_batches=3 \
    trainer.limit_val_batches=2 \
    logger.offline=True
```

3. **Custom configuration**:
```bash
cd src
python train.py --config-name=pattern_search_train \
    trainer.max_epochs=10 \
    model.d_model=128 \
    model.n_layers=4 \
    datamodule.batch_size=8 \
    task.pattern_search_radius=0.1
```

### Running Different Tasks

- **State Tracking**: Use default `pattern_search_train.yaml` config
- **Membership Task**: Modify config to use membership dataset
- **SCAN Task**: Modify config to use SCAN dataset

### Performance Notes

⚠️ **PatternSearch is computationally intensive** - Each optimization step requires multiple forward passes
- Start with small models (d_model=64, n_layers=2) and datasets (n_samples=200)
- Monitor training progress with WandB logging
- Use offline mode for testing: `logger.offline=True`

For traditional gradient-based training comparison, use the standard `train.yaml` config.

---

## 📊 Research Results & Analysis

The repository includes tools for analyzing training results:
- **WandB Integration**: Experiment tracking with hyperparameter logging
- **Jupyter Notebooks**: Ready-to-use analysis in `report/plots/notebook.ipynb`
- **Comparison Framework**: Compare PatternSearch vs gradient-based training

Early results show PatternSearch can successfully train small transformers, though with different convergence characteristics compared to gradient-based methods.

## 📁 Project Structure

```
src/
├── optimizer.py              # PatternSearch implementation
├── transformer.py           # Decoder-only transformer architecture
├── pattern_search_task.py   # Lightning module for PatternSearch training
├── *_data.py               # Dataset implementations
├── *_task.py               # Task-specific training modules
└── train.py                # Main training script

configs/
├── pattern_search_train.yaml  # PatternSearch training config
└── train.yaml                 # Standard gradient-based config

tests/
└── test_pattern_search.py    # Functionality tests
```

## 🤝 Contributing

This project explores novel optimization approaches for neural language models. Contributions are welcome in:
- New derivative-free optimization algorithms
- Additional sequence learning tasks and datasets
- Performance benchmarking and analysis
- Documentation and reproducibility improvements

## 📚 References & Related Work

This research builds on derivative-free optimization literature and explores its application to modern deep learning architectures. The implementation framework is inspired by best practices from the PyTorch Lightning and Hydra communities.
