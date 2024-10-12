import logging
import time
from typing import Dict, Optional

import wandb
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class Writer:
    """
    Class to write metrics to wandb.
    """

    def __init__(self, args: DictConfig):
        self._run_name = args.run_name + f"__{int(time.time())}"  # add time in case of a collision
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,  # can be None
            config=dict(args),
            name=self._run_name,
            save_code=True,
        )

    @property
    def run_name(self) -> str:
        return self._run_name

    @staticmethod
    def _add_prefix_to_dict(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
        """
        Add a prefix to each key of the dict so that it is easier to read in wandb as each dataset will be
        in a separate tab.
        :param metrics: dict of metrics for evaluated dataset
        :param prefix: prefix which will be added to each key. Usually the dataset name.
        :return: dict with prefix added to each key
        """
        return {f"{prefix}/{k}": v for k, v in metrics.items()}

    def log(self, metrics: Dict[str, float], prefix: str, step: Optional[int] = None) -> None:
        """
        Log metrics to wandb.
        :param metrics: Dictionary of metrics with key being the name of the metric and value the value of the metric.
        :param prefix: Prefix to add to each metric name. Usually the dataset name.
        :param step: Step at which to log the metrics. If None, will use the global step calculated by wandb.
        """
        logger.info(f"Dataset: {prefix}, metrics {metrics}")
        prefix_metrics = self._add_prefix_to_dict(metrics, prefix)
        wandb.log(prefix_metrics, step=step)
