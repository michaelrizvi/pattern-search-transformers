import os
import warnings

# Suppress OpenSSL and SSL warnings
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
warnings.filterwarnings('ignore', category=UserWarning, module='torch.*')
warnings.filterwarnings('ignore', message='.*SSL.*')
warnings.filterwarnings('ignore', message='.*certificate.*')
warnings.filterwarnings('ignore', message='.*urllib3.*')

import hydra
import torch
from lightning import Trainer, seed_everything
from omegaconf import OmegaConf

from utils import ast_eval

# torch.set_float32_matmul_precision("medium")  # or 'high' based on your needs


@hydra.main(config_path="../configs/", config_name="synthetic_patternsearch_count", version_base=None)
def train(cfg):
    seed_everything(cfg.seed)
    dataset = hydra.utils.instantiate(
        cfg.datamodule
    )  # can also pass previously instantiated object : (attribute=my_attribute)!
    task = hydra.utils.instantiate(cfg.task)
    logger = hydra.utils.instantiate(cfg.logger) if cfg.logger else False
    callbacks = (
        [hydra.utils.instantiate(cfg.callbacks[cb]) for cb in cfg.callbacks] if cfg.callbacks else None
    )

    if logger:
        try:
            # Try to access the experiment config (works for online wandb)
            if hasattr(logger.experiment, 'config'):
                logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
                logger.experiment.config.update({"seed": cfg.seed})
        except (AttributeError, TypeError):
            # Handle offline mode or other logger issues
            print("Warning: Could not update logger config (likely offline mode)")

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        **cfg.trainer,
    )
    trainer.fit(model=task, datamodule=dataset)
    
    # Test on longer sequences for length generalization using best checkpoint
    trainer.test(model=task, datamodule=dataset, ckpt_path="best")


if __name__ == "__main__":
    train()
