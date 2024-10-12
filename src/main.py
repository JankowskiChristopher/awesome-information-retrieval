import logging

import hydra
import wandb

from src.experiments.beir_experiments import run_beir_experiments
from src.experiments.generator_eval import run_qa_eval
from src.experiments.training import run_training

logger = logging.getLogger(__name__)


@hydra.main(config_path="./cfgs", config_name="config", version_base=None)
def main(cfg):
    if cfg.task == "retrieval":
        logger.info("Running BEIR experiments")
        run_beir_experiments(cfg)
    elif cfg.task == "qa_eval":
        logger.info("Running QA evaluation")
        run_qa_eval(cfg)
    elif cfg.task == "train":
        logger.info("Running Training")
        run_training(cfg)
    else:
        raise ValueError(f"Invalid task: {cfg.task}. Must be 'retrieval' or 'qa_eval'.")
    wandb.finish()  # not necessary, but maybe nice to have


if __name__ == "__main__":
    main()
