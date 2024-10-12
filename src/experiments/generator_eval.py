import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.constants import QA_DATASETS, QA_METRICS
from src.evaluation.qa_eval import QAEvaluationSuite
from src.utils import get_model, get_tokenizer
from src.writer import Writer

logger = logging.getLogger(__name__)


def run_qa_eval(cfg: DictConfig) -> None:
    """
    Function to run the question answering evaluation on QA datasets.
    Creates a QAEvaluationSuite object and runs the evaluation.

    :param cfg: Hydra config object
    """
    dataset_names = QA_DATASETS.keys() if cfg.dataset_names is None else cfg.dataset_names
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    metric_names = QA_METRICS.keys() if cfg.metric_names is None else [cfg.metric_names]

    logger.info(f"Running QA benchmark on datasets {dataset_names} with metrics {metric_names}")

    device = cfg.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device {device}")

    # huggingface_tokenizer = get_tokenizer(cfg.generator.tokenizer_name, local=cfg.local)
    # huggingface_generator = get_model(cfg.generator.model_name, device=device, local=cfg.local, question_answering=True)

    # generator = instantiate(cfg.generator, tokenizer=huggingface_tokenizer, model=huggingface_generator)

    generator = instantiate(cfg.evaluator, generator=instantiate(cfg.generator, local=cfg.local, device=cfg.device))

    writer = Writer(cfg) if cfg.track else None

    qa_evaluator = QAEvaluationSuite(
        cfg,
        writer=writer,
        generator=generator,
    )

    qa_evaluator.run(dataset_names=dataset_names, metric_names=metric_names)
