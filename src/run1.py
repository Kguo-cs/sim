# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List

import hydra
import lightning as L
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig
import sys
import os
import torch
import numpy as np
import random

from typing import Iterable, Pattern, Union

os.environ["WANDB_SILENT"] = "true"

wandb.login(key='7eba71eb2539f241fbf502af503ea5dd098168ae')
wandb.require("service")  # forces the new service backend
# Optional: use thread start (very robust in multiprocess settings)
settings = wandb.Settings(start_method="thread")
os.environ["WANDB__SERVICE_WAIT"] = "3000"

sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/sim')
sys.path.append('/home/ke/code/sim')
sys.path.append('/home/users/ntu/ke.guo/scratch/sim')
sys.path.append('/home/zs/code/sim')
sys.path.append('/mnt/d/code/sim')
sys.path.append('/home/ke/keguo/sim')
sys.path.append('/home/guoke/sim')
working_dir=os.getcwd()

print('keguo' in working_dir or "guoke" in working_dir)

from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    print_config_tree,
)

log = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("highest")# #“highest” (default),

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.cuda.synchronize()
# print("torch.backends.cuda.matmul.allow_tf32",torch.backends.cuda.matmul.allow_tf32)
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cuda.allow_tf32 = False
# print("torch.backends.cuda.matmul.allow_tf32",torch.backends.cuda.matmul.allow_tf32)

#h800 ==4090 highest


def run(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    #model=torch.compile(model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    # setup model watching
    # for _logger in logger:
    #     if isinstance(_logger, WandbLogger):
    #         _logger.watch(model, log="all")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    log.info("Logging hyperparameters!")
    log_hyperparameters(
        {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }
    )

    log.info(f"Resuming from ckpt: cfg.ckpt_path={cfg.ckpt_path}")
    if cfg.action == "fit":
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    elif cfg.action == "finetune":
        log.info("Starting finetuning!")
        if cfg.ckpt_path is not None:
            model.load_state_dict(torch.load(cfg.ckpt_path, weights_only=False)["state_dict"], strict=False)
            if model.encoder.use_kl_penalty:
                model.bc_net.load_state_dict(model.encoder.agent_encoder.state_dict())
                if model.bc_map_net is not None:
                    model.bc_map_net.load_state_dict(model.encoder.map_encoder.state_dict())
            if model.encoder.sep_map:
                model.encoder.init_map_encoder.load_state_dict(model.encoder.map_encoder.state_dict())
        trainer.fit(model=model, datamodule=datamodule)#
    elif cfg.action == "validate":
        log.info("Starting validating!")
        trainer.validate(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )
    elif cfg.action == "test":
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


@hydra.main(config_path="../configs/", config_name="run.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_printoptions(precision=3)

    log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
    #print_config_tree(cfg, resolve=True, save_to_file=True)

    run(cfg)  # train/val/test the model

    log.info("Closing wandb!")
    wandb.finish()
    log.info(f"Output dir: {cfg.paths.output_dir}")


if __name__ == "__main__":
    main()
    log.info("run.py DONE!!!")
