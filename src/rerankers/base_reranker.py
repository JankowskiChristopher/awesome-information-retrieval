import logging
from abc import ABC, abstractmethod
from typing import Dict, List

from src.dataset.beir.classes import RetrievalDataset
from src.utils import ResultDict

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """
    Base class for different rerankers that can be used in the pipeline.
    """

    @abstractmethod
    def rerank_results(self, results: List[ResultDict], dataset: RetrievalDataset) -> List[ResultDict]:
        """
        Function reranks the results from the retriever. Overriden by subclasses.

        :param results: Dict of queries and their respective retrieved passages and scores
        :param dataset: RetrievalDataset object
        :return: New results in the same format as the input.
        """
        pass

    @abstractmethod
    def set_automatic_batch_size(self, number_of_passages: int) -> None:
        """
        Function sets the batch size automatically based on the number of passages.
        Used in some rerankers to set the batch size based on the number of passages.
        :param number_of_passages: Number of passages
        """
        pass
