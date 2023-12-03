import os
import json
import yaml
import pathlib
import argparse
import pytorch_lightning as pl
import pytorch_lightning.strategies as pl_strategies

import src.model
import src.data

pl.seed_everything(42, workers=True)

if __name__ == "__main__":
    # load config from args
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        # safeload yaml
        config = yaml.safe_load(f)

    # import model and data
    model = getattr(src.model, config["model"]["model_module"])(config)
    data_module = getattr(
        src.data, config["data"]["data_module"]
    )(config, model.tokenizer)

    # get strategy
    strategy = config["training"].get("strategy", "ddp")
    if isinstance(strategy, dict):
        print("Using custom strategy: ", strategy)
        strategy = getattr(pl_strategies, strategy["class"])(**strategy["kwargs"])

    # init trainer
    logger = pl.loggers.TensorBoardLogger(config["output_dir"], name="")
    pathlib.Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"Logging to {logger.log_dir}")
    trainer = pl.Trainer(
        accelerator='gpu',
        strategy=strategy,
        max_epochs=config["training"]["max_epochs"],
        precision=config["training"].get("precision", 32),
        logger=logger,
        accumulate_grad_batches=config["training"].get(
            "accumulate_grad_batches", 1
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(
                    logger.log_dir,
                    "checkpoints"
                ),
                every_n_train_steps=config["training"].get(
                    "checkpoint_every_n_steps", 10000
                ),
                save_top_k=-1
            )
        ],
        multiple_trainloader_mode='max_size_cycle'
    )

    # train
    checkpoint = config["model"].get("resume_from_checkpoint", None)
    if checkpoint is not None:
        print(f"Resuming training from {checkpoint}")
    trainer.fit(model, data_module, ckpt_path=checkpoint)
    # save checkpoint
    trainer.save_checkpoint(
        os.path.join(
            logger.log_dir,
            "checkpoints",
            "final-checkpoint.ckpt"
        )
    )
