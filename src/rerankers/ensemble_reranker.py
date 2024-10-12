from abc import ABC, abstractmethod
from typing import List

from omegaconf import DictConfig

from src.dataset.beir.classes import RetrievalDataset
from src.rerankers.utils import _reranker_name_to_config_and_args
from src.utils import ResultDict


class EnsembleReranker(ABC):
    """
    Base class for hybrid rerankers. Hybrid rerankers could also be a subclass of BaseReranker,
    but as they are a special case of rerankers with different behaviour and subrerankers field,
    we decided to create a separate class for them.
    """

    def __init__(self, reranker_config: DictConfig):
        # To avoid one more level of nesting in the config, we filter splits out.
        self.subrerankers_configs: List[DictConfig] = [
            _reranker_name_to_config_and_args(reranker_name, reranker_config[reranker_name])
            for reranker_name in reranker_config.keys()
            if reranker_name != "splits"
        ]

    @abstractmethod
    def rerank_results(self, results: List[ResultDict], dataset: RetrievalDataset) -> List[ResultDict]:
        """
        Function reranks the results from the retriever. Overridden by subclasses.

        :param results: Dict of queries and their respective retrieved passages and scores
        :param dataset: RetrievalDataset object
        :return: New results in the same format as the input.
        """
        pass

    def __enter__(self):
        """
        This method with __exit__ do not do anything, but must be here as they are used in rerankers and we want to
        have the same interface for all rerankers.
        """
        return self

    def __exit__(self, *args, **kwargs):
        """
        Only to have the same interface, EnsembleRerankers clean after themselves by calling __exit__ of subrerankers.
        """
        pass

    def set_automatic_batch_size(self, number_of_passages: int) -> None:
        pass
