import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.dataset.beir.utils import BEIR_DATASETS
from src.evaluation.beir_eval import BeirEvaluator
from src.retrievers.ensemble_retriever import EnsembleRetriever
from writer import Writer

logger = logging.getLogger(__name__)


def run_beir_experiments(cfg: DictConfig) -> None:
    """
    Function to run the retrieval experiments on BEIR datasets. Creates a BeirEvaluator object and runs the evaluation.

    :param cfg: Hydra config object
    """
    writer = Writer(cfg) if cfg.track else None

    datasets_names = BEIR_DATASETS if cfg.beir_datasets_names is None else cfg.beir_datasets_names
    logger.info(f"Running BEIR benchmark on datasets {datasets_names} with metrics@k {cfg.metrics_k_values}")

    device = cfg.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device {device}")

    # Create retriever
    if hasattr(cfg.retriever, "type") and (cfg.retriever.type == "ensemble_retriever"):
        retriever = EnsembleRetriever(cfg.retriever.configs_paths, cfg.retriever.nums_of_retrieved_passages)
    else:
        retriever = instantiate(cfg.retriever)

    # Create reranker. Object will be instantiated later to save memory.
    instantiate_reranker_func = None
    if hasattr(cfg, "reranker") and cfg.reranker is not None:
        # In simple rerankers we need to set top_k sometimes.
        if "reranker_type" not in cfg.reranker.keys():
            # to how many passages reranker should reduce
            if cfg.reranker.top_k is None:
                cfg.reranker.top_k = max(cfg.metrics_k_values)

        instantiate_reranker_func = lambda: instantiate(cfg.reranker)
        logger.info(f"Instantiating reranker.")

    beir_evaluator = BeirEvaluator(
        cfg,
        writer=writer,
        retriever=retriever,
        instantiate_reranker_func=instantiate_reranker_func,
    )

    beir_evaluator.run(
        datasets_names=datasets_names,
        metrics_k_values=cfg.metrics_k_values,
    )
