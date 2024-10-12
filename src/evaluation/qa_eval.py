import logging
import time
from typing import Dict, List

from omegaconf import DictConfig

from src.dataset.qa.utils import get_qa_dataset
from src.evaluation.hf_eval_modules.model_eval_pipeline import QAEvaluator
from src.evaluation.hf_eval_modules.qa_metric import QAEval
from writer import Writer

logger = logging.getLogger(__name__)


class QAEvaluationSuite:
    def __init__(
        self,
        cfg: DictConfig,
        writer: Writer,
        generator,
    ) -> None:
        self.local = cfg.local
        self.generator = generator
        self.writer = writer
        self.cfg = cfg

    def run(self, dataset_names: List[str], metric_names: List[str]) -> None:
        hf_metric = QAEval(metric_names)
        for dataset_name in dataset_names:
            start_time = time.perf_counter()
            self.evaluate_dataset(dataset_name, hf_metric)
            execution_time = time.perf_counter() - start_time
            logger.info(f"Execution time for {dataset_name}: {execution_time} seconds")

    def evaluate_dataset(self, dataset_name: str, hf_metric: QAEval):
        logger.info(f"Evaluating on dataset: {dataset_name}")
        logger.info("-------------------------------------")

        dataset, columns = get_qa_dataset(dataset_name, local=self.local)

        if self.cfg.zero_shot:
            columns["context_column"] = None

        results = QAEvaluator().compute(model_or_pipeline=self.generator, data=dataset, metric=hf_metric, **columns)

        self.log_evaluation_results(dataset_name, results)

    def log_evaluation_results(
        self,
        dataset_name: str,
        results: Dict[str, float],
    ) -> None:
        logger.info(f"Results for: {dataset_name}")
        if self.writer is not None:
            self.writer.log(results, dataset_name)
        logger.info(results)
        logger.info("-------------------------------------")
