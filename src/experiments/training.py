import logging
import os

import transformers
from datasets import Dataset
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.constants import CHECKPOINT_ROOT_DIR
from src.dataset.qa.utils import get_qa_dataset
from src.utils import get_absolute_path, get_tokenizer
from src.writer import Writer

logger = logging.getLogger(__name__)


def run_training(cfg: DictConfig):
    # initialize wandb with config values
    _ = Writer(cfg)

    train_dataset, train_cols = get_qa_dataset(cfg.train_dataset, cfg.local)
    eval_dataset, eval_cols = get_qa_dataset(cfg.eval_dataset, cfg.local)

    tokenizer = get_tokenizer(cfg.trainer.tokenizer_name, cfg.local)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = preprocess_training_data(
        train_dataset, train_cols["question_column"], train_cols["label_column"], tokenizer
    )
    eval_dataset = preprocess_training_data(
        eval_dataset, eval_cols["question_column"], eval_cols["label_column"], tokenizer
    )

    training_args = cfg.training_args
    training_args.output_dir = get_absolute_path(
        os.path.join(CHECKPOINT_ROOT_DIR, cfg.training_args.output_dir), cfg.local
    )
    logger.info("Checkpoints will be stored in " + training_args.output_dir)

    trainer = instantiate(
        cfg.trainer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=cfg.training_args,
        local=cfg.local,
    )

    if not isinstance(trainer, transformers.Trainer):
        raise ValueError("Trainer must be an instance of transformers.Trainer class")
    if not cfg.resume_from_checkpoint:
        logger.info("Starting training from scratch")
        trainer.train()
    else:
        assert cfg.checkpoint_dir is not None, "Specify checkpoint path to resume training from a checkpoint"

        checkpoint_dir = get_absolute_path(cfg.checkpoint_dir, cfg.local)
        logger.info("Resuming training from checkpoint in " + checkpoint_dir)
        trainer.train(resume_from_checkpoint=checkpoint_dir)


def preprocess_training_data(data: Dataset, question_col: str, answer_col: str, tokenizer) -> Dataset:
    names = data.column_names

    data = data.add_column("input_ids", tokenizer(data[question_col], padding=True).input_ids)
    data = data.add_column("labels", tokenizer(data[answer_col], padding=True).input_ids)

    data = data.remove_columns(names)

    return data
